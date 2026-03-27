"""
推理脚本。

显存占用（约）：
- 模型权重：LLaMA-8B bf16 ≈16GB，ViT + Qwen2-0.5B ≈2–4GB，合计约 18–20GB。
- 序列与 KV：L = 视觉 token 数 + prompt + max_new_tokens。
- 当 config.attention_bias_toward_visual 非 0 时，会传入 (B,1,L,L) 的 4D mask，LLaMA 无法走 Flash Attention，显存约 50–56GB（与训练一致、输出一致）。若需在 46GB 上跑，可手动将 config.attention_bias_toward_visual 置 0，但输出会与训练/原推理不同。
- 视觉 token 过多时可将 max_visual_tokens_inference 设为 1024 或 512。

128×512×512 体数据、关闭 visual bias 时的理论显存（bf16、batch=1、Flash Attention 生效）：
- 输入：volume_grid_thw=(16,32,32) → 视觉 token=512，L_prefill=512+prompt（如 544），L_max=544+max_new_tokens（如 672）。
- 权重：≈18 GB。预填：pixel_values (1,16384,6144) bf16≈0.2GB；ViT 激活约 0.3–0.5GB；Qwen2 约 0.1GB；LLaMA inputs_embeds+KV 约 0.1GB；合计预填增量约 0.7–1GB。解码 KV 增长约 0.09GB。
- 合计理论约 19–20 GB。若实测远高于此（如 48GB），多为：1) 未关 output_hidden_states/output_attentions；2) 2D attention_mask 导致退化为完整 attention；3) 部分模块仍为 fp32。

为何训练不 OOM、推理 OOM？
- use_simple_projector=True 时，视觉 token 数 N' = (D/2)*(H/2)*(W/2)，无 64~1024 上限。
- 训练：Swift 里同一 volume 可能被 resize/预处理成较小 grid，或 batch 内序列较短，N' 较小。
- 推理：直接读 nii，processor 可能输出更大 grid → N' 可达 1 万～3 万 → 序列长 → SDPA/KV 显存 O(L^2) 导致 OOM。
解决：设 max_visual_tokens_inference（如 2048/1024/512），与训练时视觉 token 量级一致，避免推理序列过长。
      processor 会在 expand placeholder 时按该值截断视觉 token 数。

加载与词表对齐（避免 infer 效果不对）：
- 必须先按 checkpoint 权重的 embed 行数对齐 config，再 from_pretrained，否则会 size mismatch → 未加载/随机 → 乱码。
- 加载后若 len(tokenizer) > vocab_size，再 resize_token_embeddings 并同步 image_3d_token_id。
"""
import math
from pathlib import Path
from typing import Optional

import torch
from architectures import vit_lamma  # noqa: F401  # register vit_lamma classes
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor


def _get_embedding_vocab_size_from_checkpoint(model_dir: str) -> Optional[int]:
    """从 checkpoint 权重文件中读取 LLM embed_tokens 的 vocab 维度，用于加载前对齐 config，避免 size mismatch 导致未加载。支持分片 safetensors。"""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return None

    def _inspect(state):
        for key in state:
            if "embed_tokens" in key and key.endswith("weight"):
                t = state[key]
                if hasattr(t, "shape") and len(t.shape) >= 1:
                    return int(t.shape[0])
        return None

    # 分片 safetensors：先查 index，找到 embed_tokens 所在 shard 再读 shape
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            import json
            from safetensors.torch import load_file
            with open(index_path, "r") as f:
                index = json.load(f)
            weight_map = index.get("weight_map") or {}
            embed_key = next((k for k in weight_map if "embed_tokens" in k and k.endswith("weight")), None)
            if embed_key:
                shard_name = weight_map[embed_key]
                state = load_file(str(model_dir / shard_name), device="meta")
                if embed_key in state:
                    return int(state[embed_key].shape[0])
        except Exception:
            pass

    st_files = sorted(model_dir.glob("*.safetensors"))
    bin_files = sorted(model_dir.glob("*.bin")) if not st_files else []

    if st_files:
        try:
            from safetensors.torch import load_file
            for path in st_files:
                state = load_file(str(path), device="meta")
                v = _inspect(state)
                if v is not None:
                    return v
        except Exception:
            pass
    elif bin_files:
        try:
            for path in bin_files:
                try:
                    state = torch.load(str(path), map_location="meta", weights_only=True)
                except TypeError:
                    state = torch.load(str(path), map_location="meta")
                if isinstance(state, dict):
                    v = _inspect(state)
                    if v is not None:
                        return v
        except Exception:
            pass
    return None


def _align_config_to_checkpoint_embedding(config, model_dir: str) -> None:
    """
    在 from_pretrained 前将 config.llm_config.vocab_size 对齐到 checkpoint 权重的 embedding 行数。
    否则 config 与权重 shape 不一致会导致加载时 size mismatch，embed/lm_head 未加载或随机 → infer 乱码。
    """
    if not getattr(config, "llm_config", None):
        return
    vocab_from_ckpt = _get_embedding_vocab_size_from_checkpoint(model_dir)
    if vocab_from_ckpt is None:
        return
    lc = config.llm_config
    current = lc.get("vocab_size") if isinstance(lc, dict) else getattr(lc, "vocab_size", None)
    if current is not None and current != vocab_from_ckpt:
        if isinstance(lc, dict):
            lc["vocab_size"] = vocab_from_ckpt
        else:
            setattr(lc, "vocab_size", vocab_from_ckpt)


def _resize_embeddings_if_needed(model, tokenizer) -> None:
    """加载后若 tokenizer 词表更大，则 resize 并保证 image_3d_token_id 一致。"""
    llm = getattr(model, "model", model)
    if hasattr(llm, "llm"):
        llm = llm.llm
    vocab = getattr(llm.config, "vocab_size", None)
    if vocab is None or len(tokenizer) <= vocab:
        return
    new_vocab = math.ceil(len(tokenizer) / 128) * 128
    llm.resize_token_embeddings(new_vocab)
    if hasattr(llm.config, "vocab_size"):
        llm.config.vocab_size = new_vocab

# -------------------------
# Config (edit as needed)
# -------------------------

model_path = "/mnt/workspace/offline/tangwen.32/trained_models/vit_lamma/v28-20260218-100641/checkpoint-7707"
# model_path = "/mnt/workspace/offline/tangwen.32/trained_models/vit_lamma/v29-20260218-201250/checkpoint-2000"
# model_path = "/mnt/workspace/offline/tangwen.32/trained_models/vit_lamma/v27-20260218-091343/checkpoint-4000"



image_path = "/mnt/workspace/offline/shared_data/Medical_Data_3D/CT-RATE/dataset/train_fixed/train_3034/train_3034_b/train_3034_b_1.nii.gz"
user_prompt = "Generate a radiology report for this medical image."

# 限制推理时视觉 token 数，避免比训练时序列更长导致 OOM（建议与训练量级一致，如 1024/2048）
# 单卡 46GB OOM 时优先改为 1024 或 512；processor 会按此值截断视觉 token
max_visual_tokens_inference = 1024  # None=不限制；2048/1024/512 按显存调整

max_new_tokens = 128
do_sample = False
temperature = 0.7
top_p = 0.9
repetition_penalty = 1.15
no_repeat_ngram_size = 4

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
# 先按 checkpoint 权重的 embedding 行数对齐 config，再加载，否则 size mismatch 会导致 embed/lm_head 未加载 → 随机 → infer 乱码
_align_config_to_checkpoint_embedding(config, model_path)
# 限制视觉 token 长度，使推理序列与训练量级接近，避免 OOM
if max_visual_tokens_inference is not None:
    setattr(config, "max_visual_tokens_inference", max_visual_tokens_inference)
    setattr(processor, "max_visual_tokens_inference", max_visual_tokens_inference)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    trust_remote_code=True,
    dtype="bfloat16",  # 新版本 transformers 使用 dtype，torch_dtype 已弃用
    device_map="auto",
    attn_implementation="flash_attention_2",
)
# 若 tokenizer 词表大于 checkpoint 的 vocab（如本地 add_special_tokens），resize 并保持 image_3d_token_id 一致
_resize_embeddings_if_needed(model, processor.tokenizer)
model.eval()
# 确保推理时全模型 bf16，避免部分子模块仍为 fp32 导致显存翻倍
if next(model.parameters()).dtype != torch.bfloat16:
    model = model.to(torch.bfloat16)
# device_map="auto" 时 model.to(bf16) 可能未递归到部分子模块，显式转 ViT 与 Qwen2
_body = getattr(model, "model", None)
if _body is not None:
    for _name, _mod in [("vision_encoder", getattr(_body, "vision_encoder", None)), ("qwen2_projector", getattr(_body, "qwen2_projector", None))]:
        if _mod is not None and next(_mod.parameters(), None) is not None and next(_mod.parameters()).dtype != torch.bfloat16:
            _mod.to(torch.bfloat16)

# 诊断 1) ViT 2) Qwen2 是否存在 fp32 参数或非 bf16（会导致显存翻倍）
def _check_module_dtypes(name: str, module: Optional[torch.nn.Module]) -> None:
    if module is None:
        print(f"[inference] dtype check {name}: module is None")
        return
    dtypes = {}
    for p in module.parameters():
        d = str(p.dtype)
        dtypes[d] = dtypes.get(d, 0) + 1
    total = sum(dtypes.values())
    non_bf16 = sum(c for d, c in dtypes.items() if "bfloat16" not in d and "float16" not in d)
    status = "OK (all bf16/fp16)" if non_bf16 == 0 else f"WARNING: {non_bf16}/{total} params NOT bf16/fp16 (may double VRAM)"
    print(f"[inference] dtype check {name}: {dtypes} -> {status}")

_check_module_dtypes("ViT (vision_encoder)", getattr(getattr(model, "model", None), "vision_encoder", None))
_check_module_dtypes("Qwen2 (projector)", getattr(getattr(model, "model", None), "qwen2_projector", None))


def _check_attn_implementation(name: str, module: Optional[torch.nn.Module]) -> None:
    """打印实际使用的 attention 实现（若为 flash_attention_2 则显存友好）."""
    if module is None:
        print(f"[inference] attn_impl {name}: module is None")
        return
    # 取第一层 decoder 的 self_attn 的类名
    impl = "unknown"
    for _name, _child in module.named_children():
        if "layer" in _name or "layers" in _name:
            if hasattr(_child, "__iter__"):
                first = next(iter(_child), None)
                if first is not None and hasattr(first, "self_attn"):
                    impl = type(first.self_attn).__name__
            break
    # 部分模型用 config 存
    cfg = getattr(module, "config", None)
    if cfg is not None:
        impl_cfg = getattr(cfg, "_attn_implementation", None) or getattr(cfg, "attn_implementation", None)
        if impl_cfg is not None:
            impl = f"{impl} (config={impl_cfg})"
    print(f"[inference] attn_impl {name}: {impl}")


_llm = getattr(getattr(model, "model", None), "llm", None)
_qw = getattr(getattr(model, "model", None), "qwen2_projector", None)
_check_attn_implementation("LLaMA", getattr(_llm, "model", _llm))
_check_attn_implementation("Qwen2 (projector)", _qw)
# 额外检查：LLaMA config._attn_implementation 是否为 flash_attention_2
_llm_inner_cfg = getattr(getattr(_llm, "model", _llm), "config", None)
if _llm_inner_cfg is not None:
    _actual = getattr(_llm_inner_cfg, "_attn_implementation", "unknown")
    print(f"[inference] LLaMA config._attn_implementation = {_actual} (should be flash_attention_2 for low VRAM)")

# Critical: model must use the same image_3d placeholder token id as the tokenizer.
# If they differ, get_placeholder_mask() finds no positions and no visual tokens are injected → gibberish.
pad_token_id_in_vocab = processor.tokenizer.convert_tokens_to_ids(
    getattr(processor, "image_3d_token", "<|image_3d_pad|>")
)
if getattr(model.config, "image_3d_token_id", None) != pad_token_id_in_vocab:
    model.config.image_3d_token_id = pad_token_id_in_vocab

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_3d",
                "image_3d": image_path,
            },
            {"type": "text", "text": user_prompt},
        ],
    },
]

# 避免 tokenizer 报 "truncate to max_length but no maximum length is provided"
if getattr(processor.tokenizer, "model_max_length", None) is None or processor.tokenizer.model_max_length > 1e6:
    processor.tokenizer.model_max_length = 131072

# 显式保证推理与 vit_lamma_utils / volume_processor 的 128*512*512 一致：apply_chat_template 会调
# volume_processor(volumes=..., do_resize=True)；内部先 fetch_volume（vit_lamma_utils 里 MAX_FRAMES=128 限制 slice），
# 再 _preprocess 里按 target_spatial 做 512*512 resize。若 checkpoint 里 volume_processor 的 max_slices 等被保存为其他值，这里强制覆盖。
_vp = getattr(processor, "volume_processor", None)
if _vp is not None:
    setattr(_vp, "max_slices", 128)
    setattr(_vp, "target_spatial_height", 512)
    setattr(_vp, "target_spatial_width", 512)

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image_3d",
#                 "image_3d": "/mnt/workspace/offline/shared_data/Medical_Data_3D/M3D-CAP/M3D_Cap_nii/ct_quizze/005322/Axial_C__portal_venous_phase.nii.gz",
#             },
#             {"type": "text", "text": "Question:\nWhere does the narrowed transition point, where the small bowel enters/exits the cluster, occur?\nOptions:\nA) Left lower quadrant\nB) Right lower quadrant\nC) Left upper quadrant\nD) Right upper quadrant"},
#         ],
#     },
# ]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
# 转成 dict 并统一移到模型所在设备（device_map='auto' 时取第一个参数的 device）
_input_data = getattr(inputs, "data", inputs) if hasattr(inputs, "data") else inputs

# 诊断：token 长度与 volume 形状（应与 128*512*512 限制一致）
_in_ids = _input_data.get("input_ids")
_pv = _input_data.get("pixel_values_volumes")
_gt = _input_data.get("volume_grid_thw")
input_seq_len = 0
visual_token_count = 0
if _in_ids is not None and isinstance(_in_ids, torch.Tensor):
    input_seq_len = _in_ids.shape[1] if _in_ids.dim() > 1 else _in_ids.numel()
    _pad_id = getattr(processor, "image_3d_token_id", None)
    visual_token_count = (_in_ids == _pad_id).sum().item() if _pad_id is not None else 0
    prompt_token_count = input_seq_len - visual_token_count
    print(f"[inference] token length: input={input_seq_len} (visual={visual_token_count}, prompt={prompt_token_count}), max_new_tokens={max_new_tokens}")
if _pv is not None and isinstance(_pv, torch.Tensor):
    _pd = _pv.shape[-1] if _pv.dim() >= 3 else 0
    print(f"[inference] pixel_values_volumes.shape={tuple(_pv.shape)} | patch_dim={_pd} (=3*8*16*16 正常，processor 3 通道；ViT 若 encoder_in_channels=1 会 3→1 通道)")
if _gt is not None and isinstance(_gt, torch.Tensor):
    print(f"[inference] volume_grid_thw={_gt.tolist() if _gt.numel() <= 12 else _gt.shape} (T,H,W 格点，下采样后 token 数≈(T/2)*(H/2)*(W/2))")

device = next(model.parameters(), None)
device = device.device if device is not None else (getattr(model, "device", None) or "cuda")
if device is not None:
    _input_data = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in _input_data.items()
    }
inputs = _input_data

# Generation config: use model's native EOS (<|eot_id|>) and let it decide when to stop.
tokenizer = processor.tokenizer
eos_id = tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id
if pad_id is None:
    pad_id = eos_id

# 抑制重复（如 "patient was included..." 循环）：开启 repetition_penalty 与 no_repeat_ngram_size
gen_kwargs = dict(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=do_sample,
    eos_token_id=eos_id,
    pad_token_id=pad_id,
    # repetition_penalty=repetition_penalty,
    # no_repeat_ngram_size=no_repeat_ngram_size,
)
if do_sample:
    gen_kwargs["temperature"] = temperature
    gen_kwargs["top_p"] = top_p

# 监控显存：生成前重置峰值统计，生成后打印最大占用
_cuda_device = device if isinstance(device, torch.device) else torch.device(device)
if _cuda_device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(_cuda_device)
    torch.cuda.synchronize(_cuda_device)
    mem_before = torch.cuda.memory_allocated(_cuda_device) / (1024**3)
    print(f"[inference] GPU memory before generate: {mem_before:.2f} GB")

with torch.inference_mode():
    generated_ids = model.generate(**gen_kwargs)

in_ids_batch = inputs["input_ids"]
generated_ids_trimmed = [
    out_row[len(in_row) :] for in_row, out_row in zip(in_ids_batch, generated_ids)
]
output_seq_len = generated_ids.shape[1] if generated_ids.dim() > 1 else generated_ids.numel()
num_generated = output_seq_len - (in_ids_batch.shape[1] if in_ids_batch.dim() > 1 else in_ids_batch.numel())
print(f"[inference] token length: output_total={output_seq_len}, generated={num_generated}")

if _cuda_device.type == "cuda":
    torch.cuda.synchronize(_cuda_device)
    mem_peak = torch.cuda.max_memory_allocated(_cuda_device) / (1024**3)
    mem_reserved = torch.cuda.max_memory_reserved(_cuda_device) / (1024**3)
    print(f"[inference] GPU memory: peak_allocated={mem_peak:.2f} GB, max_reserved={mem_reserved:.2f} GB")

output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
