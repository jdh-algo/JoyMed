# VitLamMA

多模态融合：预训练 ViT（3D 图像）+ LLaMA，使用 **Qwen2-0.5B** 作为融合投影器。

## 架构流程

1. **ViT（3D）**：预训练 ViT 编码 3D 图像，得到 `(B, N, vision_hidden_size)`。
2. **3D 下采样（2×2×2 PixelShuffle）**：`Conv3dDownsample` 做 space-to-channel（2×2×2）+ 1×1×1 卷积，D/H/W 各缩小为 **1/2**，输出特征维度为 **llm_hidden_size**，可直接接 LLaMA（simple_projector）或再经 Qwen2。
3. **Token 处理（Qwen2 路径）**：下采样后的特征（已是 llm 维）经线性层映射到 Qwen2 隐藏维度，与 **1024 个可学习解码 token** 拼接，输入 **Qwen2-0.5B**。
4. **最终输入**：取 Qwen2 前 1024 个位置的隐藏状态，经线性层映射到 LLaMA 维度后，作为 **LLaMA 的视觉输入**；可与文本 token 拼接后一起送入 LLaMA。

## 目录结构

- `configuration_vit_lamma.py`：VitLamMA 配置（ViT/LLaMA/Qwen2 路径与维度等）。
- `downsample_3d.py`：3D 下采样模块（2×2×2 PixelShuffle + 1×1×1 conv，输出维度 llm_hidden_size，D/H/W → 1/2）。
- `modeling_vit_lamma.py`：VitLamMAModel、VitLamMAForConditionalGeneration。
- `__init__.py`：包导出与 Auto 类注册。

## 使用方式

```python
from architectures.vit_lamma import VitLamMAConfig, VitLamMAForConditionalGeneration

config = VitLamMAConfig(
    vision_model_name_or_path="path/to/3d_vit",
    llm_model_name_or_path="path/to/llama",
    projector_model_name_or_path="Qwen/Qwen2-0.5B",
    num_decode_tokens=1024,
    vision_hidden_size=1024,
    projector_hidden_size=896,
    llm_hidden_size=4096,
    downsampling_factor=2,
)
model = VitLamMAForConditionalGeneration(config)
# 或自行传入 vision_encoder / llm 实例：VitLamMAForConditionalGeneration(config, vision_encoder=..., llm=...)
```

前向时提供 `(pixel_values_3d, volume_grid_dhw)` 或 `(vision_features, volume_grid_dhw)`；首次调用会走视觉分支并得到 1024 个视觉 token 的嵌入，与文本嵌入拼接后送入 LLaMA。解码步仅需 `input_ids`/`past_key_values`。

## 训练模式（Swift）

- **只冻 LLaMA、全量训练 ViT + Qwen2（aligner）**  
  必须使用 **全量微调**，否则可训练参数会很少、与预期不符：

  ```bash
  --train_type full --freeze_llm true --freeze_vit false --freeze_aligner false
  ```

  此时可训练的是 ViT + 整个 aligner（downsample_3d、vit_to_projector、qwen2_projector、projector_to_llm），可训练参数量会明显大于仅 LoRA 时的约 21M。

- **使用 LoRA/QLoRA 等 PEFT**  
  会得到 `PeftModelForCausalLM`，**只有 LoRA 等适配器参数可训练**（约 20M 量级，占比约 0.24%）。  
  即使 `--freeze_llm true --freeze_vit false --freeze_aligner false`，LoRA 也只会加在 ViT + aligner 上，但 ViT 与 Qwen2 的**主体权重仍被 PEFT 冻结**，不会全量更新。  
  若希望 ViT 和 Qwen2 的**全部参数**都参与训练，请使用上面的 `train_type full`。

## 依赖

- PyTorch
- transformers（含 Qwen2、LLaMA）
- 预训练 ViT（3D）与 LLaMA 权重路径需在 config 中正确配置。
