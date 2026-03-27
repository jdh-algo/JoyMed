# VitQwen

VitQwen: **ViT (3D)** + **Qwen2-0.5B** (projector) + **Qwen3-VL** LLM.

- **Vision**: 自研预训练 3D ViT（与 VitLamMA 相同，支持 LaMed/m3d_mae_clip_seg）。
- **Projector**: Qwen2-0.5B，将下采样后的 ViT 特征 + 可学习 decode tokens 映射到 LLM 空间。
- **LLM**: Qwen3-VL 的 language_model（仅文本解码器），复用其词表与 lm_head。
- **Deepstack**: 已接入 Qwen3-VL 的 deepstack（`visual_pos_masks` + `deepstack_visual_embeds`），与 Qwen3-VL 一致。
- **位置编码**: 使用 `get_rope_index` 生成 (3, B, L) 的 position_ids，符合 Qwen3-VL LLM 规定。**Qwen2 路径**：经 Qwen2-0.5B 投影后的 image token 无原始 T/H/W 网格，视觉段使用 (1, 1, Nv) 的 3D RoPE 表示「投影后的一维视觉带」。**pixel_shuffle 路径**：对 T/H/W 做 2×2×2 下采样后保留真实 3D 网格，`get_rope_index` 对视觉段使用基于真实 (D,H,W) 的 3D RoPE。

## ViT / LaMed 写入

VitQwen 与 **vit_lamma** 解耦，仅使用本目录下的 LaMed：当 `vision_config.model_type == "lamed_vit"` 时，由 `modeling_vit_qwen._build_lamed_vit_from_config` 从 `architectures/vit_qwen/LaMed` 构建 LaMed ViT，merge 脚本从 m3d_mae_clip_seg 中只加载 `vision_encoder.*` 权重并包成 `LaMedViTWrapper`。默认 `--lamed_path` 为 `architectures/vit_qwen`。

## Projector 类型

- **projector_type="qwen2"**（默认）：ViT → 3D 下采样 → 线性 + decode tokens → Qwen2-0.5B → 线性 → LLM。
- **projector_type="pixel_shuffle"**：ViT → 3D 下采样 → 对 T/H/W 2×2×2 降采样 → 线性(8×vision_hidden→llm_hidden) → LLM；不走 Qwen2，可使用真实 3D RoPE。

## 数据读取与 citrus_v_3 的 3D 部分

- **VitQwen**: 使用本目录 `vit_qwen_utils` 与 `VitQwenVolumeProcessor`，逻辑与 VitLamMA 的 volume 处理一致：`temporal_patch_size=8`，`merge_size=2`，`min_slices=16`，`max_slices=512`，输出 `pixel_values_volumes`、(B, N, patch_dim) 与 `volume_grid_thw`。
- **citrus_v_3**: 使用 `citrus_v_utils` 和 `CitrusV3VolumeProcessor`，`temporal_patch_size=2`，`merge_size=2`，`min_slices=4`，`max_slices=768`，且 `smart_resize_volume` 的 min/max_pixels 不同。
- 若要与 citrus_v_3 的 3D 数据流程完全一致，需改用 citrus 的 volume_processor 或在其基础上对齐参数。

## 拼接预训练参数

使用脚本合并 ViT、Qwen2-0.5B、Qwen3-VL 到单一目录：

```bash
python scripts/merge_vit_qwen.py \
  --vision_path /path/to/Vit_MAE_CLIP \
  --llm_path /path/to/Qwen3-VL-4B-Instruct \
  --projector_path /path/to/Qwen2-0.5B \
  --output_dir /path/to/merged_vit_qwen
```

若 `--vision_path` 为 m3d_mae_clip_seg 格式，需能访问 LaMed 包（默认使用 `architectures/vit_qwen` 下的 LaMed，或通过 `--lamed_path` 指定）。

## 训练与推理

- 使用 Swift 时：`--model <merged_dir> --model_type vit_qwen`。
- Processor 与 VitLamMA 类似，仅支持 volume（3D），占位符为 `<|image_3d_pad|>`。
