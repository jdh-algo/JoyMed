# coding=utf-8
# Copyright 2025 VitQwen Team. All rights reserved.
"""
VitQwen 推理脚本。

显存与优化（与 VitLamMA 对齐）：
- 推理时关闭 output_hidden_states/output_attentions，forward 中显式传 False 以降低显存。
- Qwen2 projector 与 Qwen3-VL language_model 使用 SDPA + bf16，避免 inputs_embeds 下 Flash2 varlen 的 cu_seqlens_q 报错及 fp32 显存翻倍。
- 若仍 OOM：设 max_visual_tokens_inference=1024 或 512，与训练量级一致。

为何训练不 OOM、推理 OOM？
- use_simple_projector=True 或 projector_type=pixel_shuffle 时，视觉 token 数 N' = (D/2)*(H/2)*(W/2)，无 64~1024 上限。
- 训练：Swift 里同一 volume 可能被 resize/预处理成较小 grid，或 batch 内序列较短，N' 较小。
- 推理：直接读 nii，processor 可能输出更大 grid → N' 可达 1 万～3 万 → 序列长 → SDPA/KV 显存 O(L^2) 导致 OOM。
解决：设 max_visual_tokens_inference（如 2048/1024），与训练时视觉 token 量级一致，避免推理序列过长。

加载与词表对齐（避免 infer 效果不对）：
- 必须先按 checkpoint 权重的 embed 行数对齐 config，再 from_pretrained，否则会 size mismatch → 未加载/随机 → 乱码。
- 加载后若 len(tokenizer) > vocab_size，再 resize_token_embeddings 并同步 image_3d_token_id。
"""
import math
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

# 确保项目根目录在路径中，以便导入 architectures
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architectures import vit_qwen  # noqa: F401  # register vit_qwen classes


def _get_embedding_vocab_size_from_checkpoint(model_dir: str) -> Optional[int]:
    """从 checkpoint 权重文件中读取 LLM embed_tokens 的 vocab 维度。支持分片 safetensors。
    VitQwen 中优先使用 model.language_model.embed_tokens（LLM），排除 model.qwen2_projector.embed_tokens。
    """
    model_dir = Path(model_dir).resolve()
    if not model_dir.is_dir():
        return None

    def _inspect(state):
        for key in state:
            if "embed_tokens" in key and key.endswith("weight") and "qwen2_projector" not in key:
                t = state[key]
                if hasattr(t, "shape") and len(t.shape) >= 1:
                    return int(t.shape[0])
        return None

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            import json
            with open(index_path, "r") as f:
                index = json.load(f)
            weight_map = index.get("weight_map") or {}
            embed_key = next(
                (k for k in weight_map if "language_model" in k and "embed_tokens" in k and k.endswith("weight")),
                None,
            )
            if embed_key is None:
                embed_key = next(
                    (k for k in weight_map if "embed_tokens" in k and "qwen2_projector" not in k and k.endswith("weight")),
                    None,
                )
            if embed_key:
                shard_name = weight_map[embed_key]
                shard_path = model_dir / shard_name
                if shard_path.is_file():
                    from safetensors.torch import load_file
                    for dev in ("meta", "cpu"):
                        try:
                            state = load_file(str(shard_path), device=dev)
                            if embed_key in state and hasattr(state[embed_key], "shape"):
                                return int(state[embed_key].shape[0])
                        except Exception:
                            continue
        except Exception:
            try:
                from safetensors.torch import load_file
                weight_map = index.get("weight_map", {})
                embed_key = next((k for k in weight_map if "language_model" in k and "embed_tokens" in k and k.endswith("weight")), None)
                if embed_key:
                    state = load_file(str(model_dir / weight_map[embed_key]), device="meta")
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
    """在 from_pretrained 前将 config.text_config.vocab_size 对齐到 checkpoint 权重的 embedding 行数。"""
    tc = getattr(config, "text_config", None)
    if tc is None:
        return
    vocab_from_ckpt = _get_embedding_vocab_size_from_checkpoint(model_dir)
    if vocab_from_ckpt is None:
        return
    current = tc.get("vocab_size") if isinstance(tc, dict) else getattr(tc, "vocab_size", None)
    if current is not None and current != vocab_from_ckpt:
        if isinstance(tc, dict):
            tc["vocab_size"] = vocab_from_ckpt
        else:
            setattr(tc, "vocab_size", vocab_from_ckpt)


def _resize_embeddings_if_needed(model, tokenizer) -> None:
    """加载后若 tokenizer 词表更大，则 resize 并保证 image_3d_token_id 一致。VitQwen: model.model.language_model。"""
    inner = getattr(model, "model", model)
    lm = getattr(inner, "language_model", inner)
    vocab = getattr(lm.config, "vocab_size", None)
    if vocab is None or len(tokenizer) <= vocab:
        return
    new_vocab = math.ceil(len(tokenizer) / 128) * 128
    lm.resize_token_embeddings(new_vocab)
    if hasattr(lm.config, "vocab_size"):
        lm.config.vocab_size = new_vocab


def _load_checkpoint_state_dict(model_dir: str) -> dict:
    """加载 checkpoint 的 state dict（仅 meta），用于诊断。"""
    model_dir = Path(model_dir).resolve()
    if not model_dir.is_dir():
        return {}
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            import json
            from safetensors.torch import load_file
            with open(index_path, "r") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            state = {}
            for shard in set(weight_map.values()):
                shard_path = model_dir / shard
                if shard_path.is_file():
                    for dev in ("meta", "cpu"):
                        try:
                            state.update(load_file(str(shard_path), device=dev))
                            break
                        except Exception:
                            continue
            if state:
                return state
        except Exception:
            pass
    for ext in ("*.safetensors", "*.bin"):
        files = sorted(model_dir.glob(ext))
        if files:
            try:
                if ext == "*.safetensors":
                    from safetensors.torch import load_file
                    state = {}
                    for p in files:
                        state.update(load_file(str(p), device="meta"))
                else:
                    state = torch.load(str(files[0]), map_location="meta", weights_only=True)
                    if not isinstance(state, dict):
                        state = {}
                return state
            except Exception:
                pass
    return {}


def _run_inference_diagnostics(
    model_path: str,
    config,
    processor,
    model,
) -> None:
    """
    诊断推理时可能出现的错误：词表不一致、权重未加载（随机初始化）等。
    运行后根据输出判断是否需要开启 align_vocab_to_checkpoint / resize_embeddings_if_needed。
    """
    lines = ["\n========== [inference diagnostics] =========="]

    # 1. Vocab 对齐检查
    vocab_ckpt = _get_embedding_vocab_size_from_checkpoint(model_path)
    tc = getattr(config, "text_config", None)
    vocab_config = None
    if tc is not None:
        vocab_config = tc.get("vocab_size") if isinstance(tc, dict) else getattr(tc, "vocab_size", None)
    vocab_tok = len(processor.tokenizer)

    lines.append("\n1. Vocab size alignment:")
    lines.append(f"   checkpoint embed_tokens.shape[0]: {vocab_ckpt or 'N/A'}")
    lines.append(f"   config.text_config.vocab_size:   {vocab_config}")
    lines.append(f"   len(tokenizer):                  {vocab_tok}")

    if vocab_ckpt is not None:
        if vocab_config is not None and vocab_config != vocab_ckpt:
            lines.append(f"   >>> WARNING: config vocab != checkpoint -> embed/lm_head 可能部分随机初始化！")
            lines.append(f"   >>> 建议: align_vocab_to_checkpoint = True")
        if vocab_tok > (vocab_config or vocab_ckpt):
            lines.append(f"   >>> WARNING: tokenizer vocab > model -> token 越界或映射错误！")
            lines.append(f"   >>> 建议: resize_embeddings_if_needed = True")

    # 2. 权重加载检查（model keys vs checkpoint keys）
    ckpt_state = _load_checkpoint_state_dict(model_path)
    if ckpt_state:
        model_state = model.state_dict()
        model_keys = set(model_state.keys())
        ckpt_keys = set(ckpt_state.keys())
        missing_in_ckpt = model_keys - ckpt_keys
        unexpected_in_ckpt = ckpt_keys - model_keys

        lines.append("\n2. Weight loading:")
        lines.append(f"   model params: {len(model_keys)}, checkpoint params: {len(ckpt_keys)}")
        if missing_in_ckpt:
            lines.append(f"   >>> MISSING in checkpoint (model 随机初始化): {len(missing_in_ckpt)} keys")
            for k in sorted(missing_in_ckpt)[:15]:
                lines.append(f"      - {k}")
            if len(missing_in_ckpt) > 15:
                lines.append(f"      ... and {len(missing_in_ckpt) - 15} more")
        if unexpected_in_ckpt:
            lines.append(f"   >>> UNEXPECTED in checkpoint (未加载到 model): {len(unexpected_in_ckpt)} keys")
            for k in sorted(unexpected_in_ckpt)[:15]:
                lines.append(f"      - {k}")
            if len(unexpected_in_ckpt) > 15:
                lines.append(f"      ... and {len(unexpected_in_ckpt) - 15} more")

        # 3. 关键模块 shape 是否一致
        for mk in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
            if mk not in model_state:
                continue
            ck = mk if mk in ckpt_keys else next(
                (k for k in ckpt_keys if k == mk or (k.endswith("embed_tokens.weight") and "language_model" in k) or (k == "lm_head.weight")),
                None,
            )
            if ck and ck in ckpt_state:
                mshp = tuple(model_state[mk].shape)
                cshp = tuple(ckpt_state[ck].shape)
                if mshp != cshp:
                    lines.append(f"   >>> SHAPE MISMATCH {mk}: model {mshp} vs ckpt {cshp}")

    # 4. image_3d_token_id 一致性
    pad_id = processor.tokenizer.convert_tokens_to_ids(
        getattr(processor, "image_3d_token", "<|image_3d_pad|>")
    )
    cfg_pad = getattr(model.config, "image_3d_token_id", None)
    lines.append("\n3. image_3d_token_id:")
    lines.append(f"   tokenizer: {pad_id}, model.config: {cfg_pad}")
    if cfg_pad is not None and cfg_pad != pad_id:
        lines.append(f"   >>> WARNING: 不一致，已自动修正为 {pad_id}")

    lines.append("==========================================\n")
    print("\n".join(lines))


def _check_processor_outputs(processor, _input_data: dict, verbose: bool = True) -> None:
    """检查 processor.apply_chat_template 的输出格式，诊断 attention_mask 等兼容问题。"""
    if not verbose:
        return
    in_ids = _input_data.get("input_ids")
    seq_len = in_ids.shape[1] if (in_ids is not None and isinstance(in_ids, torch.Tensor) and in_ids.dim() > 1) else (in_ids.numel() if in_ids is not None else 0)

    lines = ["[processor check]"]
    lines.append(f"  processor type: {type(processor).__name__}")
    lines.append(f"  has volume_processor: {hasattr(processor, 'volume_processor') and processor.volume_processor is not None}")
    lines.append(f"  image_3d_token_id: {getattr(processor, 'image_3d_token_id', 'N/A')}")
    lines.append(f"  output keys: {list(_input_data.keys()) if isinstance(_input_data, dict) else []}")

    am = _input_data.get("attention_mask")
    if am is None:
        lines.append("  attention_mask: None")
    elif isinstance(am, dict):
        lines.append(f"  attention_mask: dict with keys {list(am.keys())}")
    elif isinstance(am, torch.Tensor):
        lines.append(f"  attention_mask: Tensor shape={tuple(am.shape)}")
        if am.dim() == 2:
            match = "OK" if am.shape[-1] == seq_len else f"MISMATCH (input_ids len={seq_len})"
            lines.append(f"    length vs input_ids: {match}")
            lines.append(f"    sum(valid)={am.sum().item()}")
        elif am.dim() == 4:
            lines.append("    WARNING: 4D mask (Qwen3-VL format), may cause SDPA error")
    else:
        lines.append(f"  attention_mask: {type(am)}")

    for key in ("pixel_values_volumes", "volume_grid_thw"):
        v = _input_data.get(key)
        if v is not None and isinstance(v, torch.Tensor):
            lines.append(f"  {key}: shape={tuple(v.shape)}")

    print("\n".join(lines))


# -------------------------
# Config (edit as needed)
# -------------------------

model_path = "/mnt/workspace/offline/tangwen.32/trained_models/vit_qwen/v4-20260220-164949/checkpoint-2210"
model_path = "/mnt/workspace/offline/tangwen.32/trained_models/vit_qwen/v5-20260220-165045/checkpoint-2210"

image_path = "/mnt/workspace/offline/shared_data/Medical_Data_3D/CT-RATE/dataset/train_fixed/train_3034/train_3034_b/train_3034_b_1.nii.gz"
user_prompt = "Generate a radiology report for this medical image."

# 限制推理时视觉 token 数，避免比训练时序列更长导致 OOM（建议与训练量级一致，如 1024/2048）
max_visual_tokens_inference = 2048  # None=不限制；若推理 OOM 可改为 1024

# 加载前 vocab 对齐、加载后 resize embedding（默认关闭；若 infer 乱码或 index 越界可开启）
align_vocab_to_checkpoint = False
resize_embeddings_if_needed = False

# 推理诊断：检查词表对齐、权重加载情况（建议 infer 乱码时开启）
run_inference_diagnostics = True

max_new_tokens = 128
do_sample = False
temperature = 0.7
top_p = 0.9

# 若 model.generate() 输出异常，改为 True：手动全序列 forward（与实验 6 相同）
use_manual_generation = False  # 已加视觉缓存与 decode attention_mask=None

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
if align_vocab_to_checkpoint:
    _align_config_to_checkpoint_embedding(config, model_path)
if max_visual_tokens_inference is not None:
    setattr(config, "max_visual_tokens_inference", max_visual_tokens_inference)
    setattr(processor, "max_visual_tokens_inference", max_visual_tokens_inference)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    trust_remote_code=True,
    dtype="bfloat16",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
if resize_embeddings_if_needed:
    _resize_embeddings_if_needed(model, processor.tokenizer)
model.eval()
# 确保推理时全模型 bf16，避免部分子模块仍为 fp32 导致显存翻倍
if next(model.parameters()).dtype != torch.bfloat16:
    model = model.to(torch.bfloat16)
# device_map="auto" 时 model.to(bf16) 可能未递归到部分子模块，显式转 ViT、Qwen2、language_model
_body = getattr(model, "model", None)
if _body is not None:
    for _name, _mod in [
        ("vision_encoder", getattr(_body, "vision_encoder", None)),
        ("qwen2_projector", getattr(_body, "qwen2_projector", None)),
        ("language_model", getattr(_body, "language_model", None)),
    ]:
        if _mod is not None and next(_mod.parameters(), None) is not None and next(_mod.parameters()).dtype != torch.bfloat16:
            _mod.to(torch.bfloat16)


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


def _check_attn_implementation(name: str, module: Optional[torch.nn.Module]) -> None:
    if module is None:
        print(f"[inference] attn_impl {name}: module is None")
        return
    impl = "unknown"
    for _name, _child in module.named_children():
        if "layer" in _name or "layers" in _name:
            if hasattr(_child, "__iter__"):
                first = next(iter(_child), None)
                if first is not None and hasattr(first, "self_attn"):
                    impl = type(first.self_attn).__name__
            break
    cfg = getattr(module, "config", None)
    if cfg is not None:
        impl_cfg = getattr(cfg, "_attn_implementation", None) or getattr(cfg, "attn_implementation", None)
        if impl_cfg is not None:
            impl = f"{impl} (config={impl_cfg})"
    print(f"[inference] attn_impl {name}: {impl}")


_check_module_dtypes("ViT (vision_encoder)", getattr(_body, "vision_encoder", None))
_check_module_dtypes("Qwen2 (projector)", getattr(_body, "qwen2_projector", None))
_check_module_dtypes("Qwen3-VL (language_model)", getattr(_body, "language_model", None))
_check_attn_implementation("Qwen2 (projector)", getattr(_body, "qwen2_projector", None))
_check_attn_implementation("Qwen3-VL (language_model)", getattr(_body, "language_model", None))

# Critical: model must use the same image_3d placeholder token id as the tokenizer.
pad_token_id_in_vocab = processor.tokenizer.convert_tokens_to_ids(
    getattr(processor, "image_3d_token", "<|image_3d_pad|>")
)
if getattr(model.config, "image_3d_token_id", None) != pad_token_id_in_vocab:
    model.config.image_3d_token_id = pad_token_id_in_vocab

if run_inference_diagnostics:
    _run_inference_diagnostics(model_path, config, processor, model)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_3d", "image_3d": image_path},
            {"type": "text", "text": user_prompt},
        ],
    },
]

if getattr(processor.tokenizer, "model_max_length", None) is None or processor.tokenizer.model_max_length > 1e6:
    processor.tokenizer.model_max_length = 131072

# 限制 volume 预处理：与训练时量级一致
_vp = getattr(processor, "volume_processor", None)
if _vp is not None:
    setattr(_vp, "max_slices", 128)
    setattr(_vp, "target_spatial_height", 512)
    setattr(_vp, "target_spatial_width", 512)

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
_input_data = getattr(inputs, "data", inputs) if hasattr(inputs, "data") else inputs

# 诊断信息
_in_ids = _input_data.get("input_ids")
_pv = _input_data.get("pixel_values_volumes")
_gt = _input_data.get("volume_grid_thw")
if _in_ids is not None and isinstance(_in_ids, torch.Tensor):
    _seq_len = _in_ids.shape[1] if _in_ids.dim() > 1 else _in_ids.numel()
    _pad_id = getattr(processor, "image_3d_token_id", None)
    _n_visual = (_in_ids == _pad_id).sum().item() if _pad_id is not None else 0
    _n_prompt = _seq_len - _n_visual
    print(f"[inference] token length: input={_seq_len} (visual={_n_visual}, prompt={_n_prompt}), max_new_tokens={max_new_tokens}")
if _pv is not None and isinstance(_pv, torch.Tensor):
    print(f"[inference] pixel_values_volumes.shape={tuple(_pv.shape)}")
if _gt is not None and isinstance(_gt, torch.Tensor):
    print(f"[inference] volume_grid_thw={_gt.tolist() if _gt.numel() <= 12 else _gt.shape}")

# 检查 processor 输出格式
_check_processor_outputs(processor, _input_data)

device = next(model.parameters(), None)
device = device.device if device is not None else (getattr(model, "device", None) or "cuda")
_cuda_device = device if isinstance(device, torch.device) else torch.device(device)
if device is not None:
    _input_data = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in _input_data.items()
    }
inputs = _input_data

tokenizer = processor.tokenizer
eos_id = tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id
if pad_id is None:
    pad_id = eos_id

def _manual_generate(model, inputs, max_new_tokens, eos_id, pad_id):
    """与实验 6 相同：逐 token 全序列 forward，不使用 model.generate() 的 decode 路径。"""
    device = inputs["input_ids"].device
    seq_ids = inputs["input_ids"].clone()
    for _ in range(max_new_tokens):
        inp = dict(
            input_ids=seq_ids,
            pixel_values_volumes=inputs.get("pixel_values_volumes"),
            volume_grid_thw=inputs.get("volume_grid_thw"),
        )
        with torch.no_grad():
            out = model(**inp)
        logits = out.logits[0, -1, :].float()
        next_token = logits.argmax(dim=-1).item()
        if next_token == eos_id:
            break
        seq_ids = torch.cat(
            [seq_ids, torch.tensor([[next_token]], device=device, dtype=seq_ids.dtype)], dim=1
        )
    return seq_ids


if use_manual_generation:
    print("[inference] 使用手动全序列 forward（与实验 6 相同，避开 model.generate decode 路径）")
    generated_ids = _manual_generate(model, inputs, max_new_tokens, eos_id, pad_id)
else:
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        repetition_penalty=1.1,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    # 监控显存：生成前重置峰值统计，生成后打印最大占用
    if _cuda_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(_cuda_device)
        torch.cuda.synchronize(_cuda_device)
        mem_before = torch.cuda.memory_allocated(_cuda_device) / (1024**3)
        print(f"[inference] GPU memory before generate: {mem_before:.2f} GB")
    generated_ids = model.generate(**gen_kwargs)

in_ids_batch = inputs["input_ids"]
generated_ids_trimmed = [
    out_row[len(in_row):] for in_row, out_row in zip(in_ids_batch, generated_ids)
]
output_seq_len = generated_ids.shape[1] if generated_ids.dim() > 1 else generated_ids.numel()
num_generated = output_seq_len - (in_ids_batch.shape[1] if in_ids_batch.dim() > 1 else in_ids_batch.numel())
print(f"[inference] token length: output_total={output_seq_len}, generated={num_generated}")
_cuda_device = device if isinstance(device, torch.device) else torch.device(device)
if _cuda_device.type == "cuda":
    torch.cuda.synchronize(_cuda_device)
    mem_peak = torch.cuda.max_memory_allocated(_cuda_device) / (1024**3)
    mem_reserved = torch.cuda.max_memory_reserved(_cuda_device) / (1024**3)
    print(f"[inference] GPU memory: peak_allocated={mem_peak:.2f} GB, max_reserved={mem_reserved:.2f} GB")
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
