# coding=utf-8
# Copyright 2025 VitQwen Team. All rights reserved.
"""
VitQwen 推理诊断实验：
  1) 纯文本  2) 训练配置  3) GT next-token  4) GT next-token  acc
  5) 视觉接入（teacher-forcing）  6) 视觉接入（生成模式）

用法：
  python architectures/vit_qwen/utils/eval_experiments.py

  或指定实验：
  EXPERIMENT=1 python ...  # 纯文本推理
  EXPERIMENT=3 python ...  # 训练配置检查
  EXPERIMENT=4 python ...  # Ground Truth Next-Token 评估
  EXPERIMENT=5 python ...  # 视觉 token 接入检查（teacher-forcing next-token）
  EXPERIMENT=6 python ...  # 生成模式（自回归，每步 logits/attention）
  EXPERIMENT=all python ... # 全部（默认）
"""
import json
import math
import os
import sys
from pathlib import Path

# 确保项目根目录在路径中
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from architectures import vit_qwen  # noqa: F401

# -------------------------
# Config
# -------------------------
MODEL_PATH = "/mnt/workspace/offline/tangwen.32/trained_models/vit_qwen/v5-20260220-165045/checkpoint-2210"
MAX_VISUAL_TOKENS = 2048

# 实验 4 使用的 ground truth 样本
SAMPLE_IMAGE = "/mnt/workspace/offline/shared_data/Medical_Data_3D/CT-RATE/dataset/train_fixed/train_7610/train_7610_c/train_7610_c_1.nii.gz"
SAMPLE_USER_PROMPT = "Generate a complete radiology report for this scan."
SAMPLE_ASSISTANT_GT = (
    "Findings: Mediastinal structures were evaluated as suboptimal since the examination was unenhanced. "
    "As far as can be seen; Trachea and lumen of both main bronchi are open. "
    "No occlusive pathology was detected in the trachea and lumen of both main bronchi. "
    "Calibration of thoracic main vascular structures is natural. "
    "Calcified atheromatous plaques were observed on the walls of the thoracic aorta and coronary vascular structures. "
    "Heart contour size is natural. Pericardial thickening-effusion was not detected. "
    "Thoracic esophagus calibration was normal and no significant pathological wall thickening was detected. "
    "Slinding type hiatal hernia was observed. "
    "According to the previous examination, stable millimetric lymph nodes were observed in the mediastinal – lower paratracheal, paraesophageal area. "
    "When examined in the lung parenchyma window; Interlobular and septal thickness increases, peripheral honeycomb appearance and subpleural lines – contour irregularities in the pleura were observed in both lung parenchyma. "
    "The described findings were initially considered to be compatible with pulmonary interstitial fibrosis. "
    "In the upper abdominal sections in the study area; Exophytic hypodense lesion with a diameter of 4 cm was observed in the upper pole of the right kidney (cyst?). "
    "No new findings were detected in the current examination. "
    "Impressions: Not given."
)


def _load_model_and_processor():
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if MAX_VISUAL_TOKENS is not None:
        setattr(config, "max_visual_tokens_inference", MAX_VISUAL_TOKENS)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if MAX_VISUAL_TOKENS is not None:
        setattr(processor, "max_visual_tokens_inference", MAX_VISUAL_TOKENS)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        trust_remote_code=True,
        dtype="bfloat16",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    vp = getattr(processor, "volume_processor", None)
    if vp is not None:
        setattr(vp, "max_slices", 128)
        setattr(vp, "target_spatial_height", 512)
        setattr(vp, "target_spatial_width", 512)
    if getattr(processor.tokenizer, "model_max_length", None) is None or processor.tokenizer.model_max_length > 1e6:
        processor.tokenizer.model_max_length = 131072
    return model, processor


def run_experiment_1_text_only(model, processor):
    """实验 1: 纯文本推理（无图像）
    Processor 的 apply_chat_template(tokenize=True) 要求至少一个 volume，
    故使用 tokenize=False 获取 prompt 字符串，再手动 tokenize。
    """
    print("\n" + "=" * 60)
    print("[Experiment 1] 纯文本推理（无图像）")
    print("=" * 60)
    messages = [
        {"role": "user", "content": "Generate a complete radiology report for this scan."},
    ]
    prompt_str = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if isinstance(prompt_str, list):
        prompt_str = prompt_str[0]
    tok = processor.tokenizer(
        prompt_str,
        return_tensors="pt",
        return_attention_mask=True,
        add_special_tokens=False,
    )
    inp = {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]}
    inp = {k: v.to(model.device) for k, v in inp.items()}
    gen = model.generate(
        **inp,
        max_new_tokens=128,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )
    in_ids = inp["input_ids"]
    trimmed = gen[:, in_ids.size(1) :]
    text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("Output (text-only, no image):")
    print(text[0][:800] + ("..." if len(text[0]) > 800 else ""))
    print("\n>>> 若输出仍为 import/code 风格，说明问题可能在纯文本能力或训练数据；若有改善，则可能和视觉分支相关。")
    print()


def run_experiment_3_training_config(model_path: str):
    """实验 3: 检查训练配置"""
    print("\n" + "=" * 60)
    print("[Experiment 3] 训练配置检查")
    print("=" * 60)
    ckpt_dir = Path(model_path).resolve()
    args_files = [ckpt_dir / "args.json", ckpt_dir.parent / "args.json"]
    found = False
    for ap in args_files:
        if ap.is_file():
            found = True
            with open(ap, "r") as f:
                args = json.load(f)
            print(f"Loaded: {ap}")
            keys = [
                "model", "dataset", "num_train_epochs", "per_device_train_batch_size",
                "gradient_accumulation_steps", "learning_rate", "max_steps",
                "save_steps", "logging_steps", "dataloader_num_workers",
            ]
            for k in keys:
                if k in args:
                    print(f"  {k}: {args[k]}")
            if "dataset" not in args:
                for k in list(args.keys()):
                    if "data" in k.lower() or "train" in k.lower() or "dataset" in k.lower():
                        print(f"  {k}: {args[k]}")
            break
    if not found:
        print("  args.json not found in checkpoint dir.")
    config_path = ckpt_dir / "config.json"
    if config_path.is_file():
        with open(config_path, "r") as f:
            cfg = json.load(f)
        print("\nModel config (relevant):")
        for k in ("model_type", "vision_config", "text_config", "projector_type", "image_3d_token_id"):
            if k in cfg:
                v = cfg[k]
                if isinstance(v, dict) and len(str(v)) > 200:
                    print(f"  {k}: <dict, keys={list(v.keys())[:8]}>")
                else:
                    print(f"  {k}: {v}")
    print()


def run_experiment_4_ground_truth_next_token(model, processor):
    """实验 4: Ground Truth Next-Token 生成准确性评估"""
    print("\n" + "=" * 60)
    print("[Experiment 4] Ground Truth Next-Token 评估")
    print("=" * 60)
    messages_with_gt = [
        {"role": "user", "content": [{"type": "image_3d", "image_3d": SAMPLE_IMAGE}, {"type": "text", "text": SAMPLE_USER_PROMPT}]},
        {"role": "assistant", "content": SAMPLE_ASSISTANT_GT},
    ]
    messages_prefix = [
        {"role": "user", "content": [{"type": "image_3d", "image_3d": SAMPLE_IMAGE}, {"type": "text", "text": SAMPLE_USER_PROMPT}]},
    ]
    prefix_inp = processor.apply_chat_template(
        messages_prefix,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    try:
        full_inp = processor.apply_chat_template(
            messages_with_gt,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception as e:
        print(f"  ERROR: apply_chat_template with assistant failed: {e}")
        return
    prefix_data = getattr(prefix_inp, "data", prefix_inp) if hasattr(prefix_inp, "data") else prefix_inp
    full_data = getattr(full_inp, "data", full_inp) if hasattr(full_inp, "data") else full_inp

    prefix_ids = prefix_data["input_ids"]
    full_ids = full_data["input_ids"]
    if prefix_ids.shape[1] >= full_ids.shape[1]:
        print("  WARNING: prefix >= full (template structure may differ). Skipping.")
        return
    assistant_ids = full_ids[:, prefix_ids.shape[1] :]
    n_assistant = assistant_ids.shape[1]
    print(f"  Prefix length: {prefix_ids.shape[1]}, Assistant GT length: {n_assistant} tokens")

    device = next(model.parameters()).device
    full_in = {
        "input_ids": full_ids.to(device),
        "attention_mask": full_data.get("attention_mask"),
        "pixel_values_volumes": full_data.get("pixel_values_volumes"),
        "volume_grid_thw": full_data.get("volume_grid_thw"),
    }
    if full_in["attention_mask"] is not None:
        full_in["attention_mask"] = full_in["attention_mask"].to(device)
    if full_in["pixel_values_volumes"] is not None:
        full_in["pixel_values_volumes"] = full_in["pixel_values_volumes"].to(device)
    if full_in["volume_grid_thw"] is not None:
        full_in["volume_grid_thw"] = full_in["volume_grid_thw"].to(device)

    with torch.no_grad():
        out = model(**full_in)
    logits = out.logits

    prefix_len = prefix_ids.shape[1]
    pred_logits = logits[:, prefix_len - 1 : prefix_len + n_assistant - 1, :]
    targets = assistant_ids.to(device)

    pred_ids = pred_logits.argmax(dim=-1)
    top1_correct = (pred_ids == targets).float().sum().item()
    top1_acc = top1_correct / max(1, n_assistant)

    top5 = pred_logits.topk(5, dim=-1).indices
    top5_correct = (top5 == targets.unsqueeze(-1)).any(dim=-1).float().sum().item()
    top5_acc = top5_correct / max(1, n_assistant)

    loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
    ce = loss_fct(
        pred_logits.squeeze(0).float(),
        targets.squeeze(0).long(),
    )
    ppl = math.exp(min(ce.item(), 20))

    print(f"  Top-1 accuracy: {top1_acc:.4f} ({top1_correct:.0f}/{n_assistant})")
    print(f"  Top-5 accuracy: {top5_acc:.4f} ({top5_correct:.0f}/{n_assistant})")
    print(f"  Perplexity: {ppl:.2f}")
    print("\n>>> Top-1 若明显低于 0.3，说明 next-token 预测能力弱；若 > 0.5 且生成仍乱码，可能是 decoding 或 EOS 问题。")
    print()


def run_experiment_5_visual_integration(model, processor):
    """实验 5: 视觉 token 接入检查（next-token 预测形式）

    采用实验四结构：prefix（提问+图像） + assistant 回复。仅在「预测 assistant 回复 token」的
    next-token 位置进行分析（不包含 prompt/问题部分）。
    """
    # 为获取 attention map，需使用 eager attention
    try:
        for m in [model, getattr(model, "model", None), getattr(getattr(model, "model", None), "language_model", None)]:
            if m is not None and hasattr(m, "set_attn_implementation"):
                m.set_attn_implementation("eager")
                break
    except Exception as e:
        print(f"  [WARN] 无法切换到 eager attention: {e}，attention 可能仍为 None")

    print("\n" + "=" * 60)
    print("[Experiment 5] 视觉 token 接入检查（next-token 预测形式）")
    print("=" * 60)
    messages_prefix = [
        {"role": "user", "content": [{"type": "image_3d", "image_3d": SAMPLE_IMAGE}, {"type": "text", "text": SAMPLE_USER_PROMPT}]},
    ]
    messages_full = [
        {"role": "user", "content": [{"type": "image_3d", "image_3d": SAMPLE_IMAGE}, {"type": "text", "text": SAMPLE_USER_PROMPT}]},
        {"role": "assistant", "content": SAMPLE_ASSISTANT_GT},
    ]
    prefix_inp = processor.apply_chat_template(
        messages_prefix, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    try:
        full_inp = processor.apply_chat_template(
            messages_full, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt"
        )
    except Exception as e:
        print(f"  ERROR: apply_chat_template with assistant failed: {e}")
        return
    prefix_data = getattr(prefix_inp, "data", prefix_inp) if hasattr(prefix_inp, "data") else prefix_inp
    full_data = getattr(full_inp, "data", full_inp) if hasattr(full_inp, "data") else full_inp
    prefix_len = prefix_data["input_ids"].shape[1]
    full_ids = full_data["input_ids"]
    n_assistant = full_ids.shape[1] - prefix_len
    if n_assistant <= 0:
        print("  WARNING: prefix >= full. Skipping.")
        return
    # 输出位置：预测 assistant 每个 token 的 next-token 位置（prefix_len-1 预测第1个assistant token，...）
    output_positions = list(range(prefix_len - 1, prefix_len + n_assistant - 1))

    device = next(model.parameters()).device
    inp_real = {
        "input_ids": full_ids.to(device),
        "attention_mask": full_data.get("attention_mask"),
        "pixel_values_volumes": full_data.get("pixel_values_volumes"),
        "volume_grid_thw": full_data.get("volume_grid_thw"),
    }
    if inp_real["attention_mask"] is not None:
        inp_real["attention_mask"] = inp_real["attention_mask"].to(device)
    if inp_real["pixel_values_volumes"] is not None:
        inp_real["pixel_values_volumes"] = inp_real["pixel_values_volumes"].to(device)
    if inp_real["volume_grid_thw"] is not None:
        inp_real["volume_grid_thw"] = inp_real["volume_grid_thw"].to(device)
    inp_zeros = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inp_real.items()}
    if inp_zeros.get("pixel_values_volumes") is not None:
        inp_zeros["pixel_values_volumes"] = torch.zeros_like(inp_zeros["pixel_values_volumes"], device=device)

    with torch.no_grad():
        out_real = model(**inp_real, output_attentions=True)
        out_zeros = model(**inp_zeros, output_attentions=True)

    logits_real = out_real.logits.float()
    logits_zeros = out_zeros.logits.float()
    seq_len = logits_real.size(1)
    assistant_ids = full_ids[:, prefix_len:]

    image_3d_token_id = getattr(model.config, "image_3d_token_id", 151670)
    input_ids_1d = inp_real["input_ids"][0]
    visual_mask = (input_ids_1d == image_3d_token_id)
    n_visual = int(visual_mask.sum().item())
    last_vis = int(visual_mask.nonzero(as_tuple=True)[0].max().item()) if n_visual > 0 else -1

    # 1b. Image token 是否参与损失
    print("  1b. Image token 是否参与损失（训练时应为 labels=-100，不参与 loss）")
    print("     Swift 模板 _extend_tokens 将 image placeholder 扩展位置的 labels 设为 -100。")
    if n_visual > 0:
        print(f"     视觉 token 位置 4-{last_vis}（共 {n_visual} 个）。输出位置（预测 assistant）: {output_positions[0]}-{output_positions[-1]}")

    # 1. 仅输出位置：next-token 预测概率是否因图像变化
    print("  1. 输出位置 next-token 预测概率是否因图像变化（仅 assistant 回复预测步）")
    logits_diff = (logits_real - logits_zeros).abs()
    pos_max_diff = logits_diff.max(dim=-1).values
    out_diff = pos_max_diff[0, output_positions]
    global_max = out_diff.max().item()
    global_mean = out_diff.mean().item()
    print(f"     输出位置 logits: max |diff|={global_max:.6f}, mean={global_mean:.6f}")

    tokenizer = processor.tokenizer
    key_out = sorted(set([
        output_positions[0], output_positions[1], output_positions[2], output_positions[3],
        output_positions[len(output_positions) // 2], output_positions[-1]
    ]))
    key_out = [p for p in key_out if p in output_positions]
    for pos in key_out:
        idx = output_positions.index(pos)
        gt_token = assistant_ids[0, idx].item()
        lr = logits_real[0, pos]
        lz = logits_zeros[0, pos]
        pr = lr.softmax(dim=-1)
        pz = lz.softmax(dim=-1)
        top5r = pr.topk(5)
        top5z = pz.topk(5)
        gt_str = tokenizer.decode([gt_token])
        print(f"     步骤 {idx + 1}/{n_assistant} (位置 {pos}, GT=\"{gt_str}\"):")
        print(f"       真实图 top5: {[(tokenizer.decode([int(t)]), f'{float(p):.4f}') for t, p in zip(top5r.indices.tolist(), top5r.values.tolist())]}")
        print(f"       零图   top5: {[(tokenizer.decode([int(t)]), f'{float(p):.4f}') for t, p in zip(top5z.indices.tolist(), top5z.values.tolist())]}")
        print(f"       top1 是否相同: {top5r.indices[0].item() == top5z.indices[0].item()}, |diff|_max={pos_max_diff[0, pos].item():.6f}")

    if global_max < 1e-4:
        print(f"     >>> 几乎相同！视觉信息可能未参与 next-token 生成。")
    else:
        print(f"     >>> 有显著差异，视觉信息在影响 next-token 预测。")

    # 2. Attention 是否变化
    print("  2. Attention 是否因图像变化（视觉 token 是否被 attend）")
    attn_real = getattr(out_real, "attentions", None)
    attn_zeros = getattr(out_zeros, "attentions", None)

    if attn_real is not None and attn_zeros is not None and len(attn_real) > 0:
        attn_diff_max = 0.0
        for ar, az in zip(attn_real, attn_zeros):
            if ar is not None and az is not None:
                d = (ar.float() - az.float()).abs().max().item()
                attn_diff_max = max(attn_diff_max, d)
        print(f"     attention max |diff|: {attn_diff_max:.6f}")

        # 2b. 仅输出位置：对视觉 token 的 attention
        if n_visual > 0:
            vis_mask_expand = visual_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).float()
            print("  2b. 输出位置对视觉 token 的 attention 占比（仅 assistant 回复预测步）")
            for layer_idx, (ar, az) in enumerate(zip(attn_real, attn_zeros)):
                if ar is None or az is None:
                    continue
                ar_f = ar.float()
                az_f = az.float()
                attn_to_vis_real = (ar_f * vis_mask_expand.to(ar_f)).sum(dim=-1)
                attn_to_vis_zeros = (az_f * vis_mask_expand.to(az_f)).sum(dim=-1)
                if layer_idx == 0 or layer_idx == len(attn_real) - 1:
                    layer_label = "首层" if layer_idx == 0 else "末层"
                    print(f"     [Layer {layer_idx} {layer_label}]")
                    for pos in key_out:
                        r_mean = attn_to_vis_real[0, :, pos].mean().item()
                        z_mean = attn_to_vis_zeros[0, :, pos].mean().item()
                        r_max = attn_to_vis_real[0, :, pos].max().item()
                        print(f"       步骤 {output_positions.index(pos) + 1} 位置 {pos}: 真实图 mean={r_mean:.4f} max={r_max:.4f}  零图 mean={z_mean:.4f}")
            last_ar = attn_real[-1].float()
            attn_to_vis_last = (last_ar * vis_mask_expand.to(last_ar)).sum(dim=-1)
            out_mean = attn_to_vis_last[0, :, output_positions].mean().item()
            print(f"     末层输出位置({len(output_positions)} 步) 对视觉 token attention 均值（真实图）={out_mean:.4f}")
    else:
        print(f"     (模型未返回 attentions，跳过)")

    # 3. [辅助] inputs_embeds 变化
    print("  3. [辅助] inputs_embeds 在真实/零图像下是否变化")
    captured = {}

    def _hook(_m, args, kwargs=None):
        k = kwargs if kwargs is not None else {}
        emb = k.get("inputs_embeds")
        if emb is None and args:
            for a in (args if isinstance(args, tuple) else (args,)):
                if isinstance(a, torch.Tensor) and a.dim() == 3 and a.size(1) == seq_len:
                    emb = a
                    break
        if emb is not None:
            captured["emb"] = emb.detach().clone()
        return None

    inner = getattr(model, "model", model)
    lm = getattr(inner, "language_model", inner)
    try:
        h = lm.register_forward_pre_hook(_hook, with_kwargs=True)
    except TypeError:
        h = lm.register_forward_pre_hook(lambda m, a: _hook(m, a, None))
    with torch.no_grad():
        model(**inp_real)
        emb_r = captured.get("emb")
    captured.clear()
    with torch.no_grad():
        model(**inp_zeros)
        emb_z = captured.get("emb")
    h.remove()

    if emb_r is not None and emb_z is not None:
        emb_diff = (emb_r - emb_z).abs().max().item()
        print(f"     inputs_embeds max |diff|: {emb_diff:.6f}")
    else:
        print(f"     (未捕获)")
    print()


def run_experiment_6_generation_mode(model, processor):
    """实验 6: 生成模式 - 正常自回归生成，查看每个新生成 token 的 logits 与 attention。

    与实验 5 不同：非 teacher-forcing，而是逐 token 生成（greedy），每步对比真实图 vs 零图。
    """
    try:
        for m in [model, getattr(model, "model", None), getattr(getattr(model, "model", None), "language_model", None)]:
            if m is not None and hasattr(m, "set_attn_implementation"):
                m.set_attn_implementation("eager")
                break
    except Exception as e:
        print(f"  [WARN] 无法切换到 eager attention: {e}")

    print("\n" + "=" * 60)
    print("[Experiment 6] 生成模式：每个新生成 token 的 logits 与 attention")
    print("=" * 60)

    messages_prefix = [
        {"role": "user", "content": [{"type": "image_3d", "image_3d": SAMPLE_IMAGE}, {"type": "text", "text": SAMPLE_USER_PROMPT}]},
    ]
    prefix_inp = processor.apply_chat_template(
        messages_prefix, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    prefix_data = getattr(prefix_inp, "data", prefix_inp) if hasattr(prefix_inp, "data") else prefix_inp
    device = next(model.parameters()).device
    tokenizer = processor.tokenizer
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.convert_tokens_to_ids("<|im_end|>")
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def _make_inp(ids, pv_real=True):
        d = {
            "input_ids": ids.to(device),
            "pixel_values_volumes": prefix_data["pixel_values_volumes"].to(device) if prefix_data.get("pixel_values_volumes") is not None else None,
            "volume_grid_thw": prefix_data["volume_grid_thw"].to(device) if prefix_data.get("volume_grid_thw") is not None else None,
        }
        if not pv_real and d["pixel_values_volumes"] is not None:
            d["pixel_values_volumes"] = torch.zeros_like(d["pixel_values_volumes"], device=device)
        if prefix_data.get("attention_mask") is not None:
            am = torch.ones(1, ids.size(1), dtype=torch.long, device=device)
            d["attention_mask"] = am
        return d

    image_3d_token_id = getattr(model.config, "image_3d_token_id", 151670)
    max_new_tokens = 20
    show_every = 5

    generated = []
    seq_ids = prefix_data["input_ids"].clone().to(device)
    prefix_len = seq_ids.size(1)
    input_ids_1d = seq_ids[0]
    visual_mask_prefix = (input_ids_1d == image_3d_token_id)
    n_visual = int(visual_mask_prefix.sum().item())

    def _get_vis_mask_expand(cur_len):
        if n_visual == 0:
            return None
        vm = torch.zeros(cur_len, device=device, dtype=torch.float32)
        vm[:prefix_len] = visual_mask_prefix.float()
        return vm.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    print(f"  Prefix 长度: {prefix_len}, 视觉 token: {n_visual}, 最多生成 {max_new_tokens} tokens")
    print()

    for step in range(max_new_tokens):
        inp_real = _make_inp(seq_ids, pv_real=True)
        inp_zeros = _make_inp(seq_ids, pv_real=False)
        with torch.no_grad():
            out_r = model(**inp_real, output_attentions=True)
            out_z = model(**inp_zeros, output_attentions=True)
        logits_r = out_r.logits[0, -1, :].float()
        logits_z = out_z.logits[0, -1, :].float()
        next_token = logits_r.argmax(dim=-1).item()
        generated.append(next_token)

        pr = logits_r.softmax(dim=-1)
        pz = logits_z.softmax(dim=-1)
        top5r = pr.topk(5)
        top5z = pz.topk(5)
        diff_max = (logits_r - logits_z).abs().max().item()
        gt_str = tokenizer.decode([next_token])

        if step < 5 or step % show_every == 0 or step == max_new_tokens - 1:
            print(f"  步骤 {step + 1} 生成 token: \"{gt_str}\" (id={next_token})")
            print(f"    真实图 top5: {[(tokenizer.decode([int(t)]), f'{float(p):.4f}') for t, p in zip(top5r.indices.tolist(), top5r.values.tolist())]}")
            print(f"    零图   top5: {[(tokenizer.decode([int(t)]), f'{float(p):.4f}') for t, p in zip(top5z.indices.tolist(), top5z.values.tolist())]}")
            print(f"    top1 相同: {top5r.indices[0].item() == top5z.indices[0].item()}, |diff|_max={diff_max:.6f}")

        if n_visual > 0 and out_r.attentions is not None:
            cur_len = seq_ids.size(1)
            vm = _get_vis_mask_expand(cur_len).to(out_r.attentions[-1])
            ar = out_r.attentions[-1].float()
            az = out_z.attentions[-1].float()
            attn_to_vis_r = (ar * vm).sum(dim=-1)[0, :, -1].mean().item()
            attn_to_vis_z = (az * vm).sum(dim=-1)[0, :, -1].mean().item()
            if step < 5 or step % show_every == 0 or step == max_new_tokens - 1:
                print(f"    末层对视觉 token attention: 真实图={attn_to_vis_r:.4f}  零图={attn_to_vis_z:.4f}")

        if next_token == eos_id:
            break
        next_t = torch.tensor([[next_token]], device=device, dtype=seq_ids.dtype)
        seq_ids = torch.cat([seq_ids, next_t], dim=1)

    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"\n  生成文本（前 300 字）: {text[:300]}...")
    print()


def main():
    exp = os.environ.get("EXPERIMENT", "all")
    run_1 = exp in ("1", "all")
    run_3 = exp in ("3", "all")
    run_4 = exp in ("4", "all")
    run_5 = exp in ("5", "all")
    run_6 = exp in ("6", "all")

    if run_3:
        run_experiment_3_training_config(MODEL_PATH)

    if run_1 or run_4 or run_5 or run_6:
        model, processor = _load_model_and_processor()
        if run_1:
            run_experiment_1_text_only(model, processor)
        if run_4:
            run_experiment_4_ground_truth_next_token(model, processor)
        if run_5:
            run_experiment_5_visual_integration(model, processor)
        if run_6:
            run_experiment_6_generation_mode(model, processor)


if __name__ == "__main__":
    main()
