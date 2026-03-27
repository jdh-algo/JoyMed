# coding=utf-8
# Copyright 2025 VitLamMA Team. All rights reserved.
"""
VitLamMA: ViT (3D) + Qwen2-0.5B (projector) + LLaMA.

Fusion pipeline:
1. Pre-trained ViT encodes 3D images.
2. 3D convolutional downsampling reduces D/H/W to 1/2 (reference: citrus_v_3).
3. Downsampled ViT features + 1024 learnable decode tokens -> Qwen2-0.5B.
4. The 1024 decoding token outputs -> LLaMA as final input.
"""

from .configuration_vit_lamma import VitLamMAConfig
from .modeling_vit_lamma import VitLamMAModel, VitLamMAForConditionalGeneration
from .downsample_3d import Conv3dDownsample
from .volume_processing_vit_lamma import VitLamMAVolumeProcessor
from .processing_vit_lamma import VitLamMAProcessor

try:
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor
    AutoConfig.register("vit_lamma", VitLamMAConfig)
    AutoModel.register(VitLamMAConfig, VitLamMAForConditionalGeneration)
    AutoModelForCausalLM.register(VitLamMAConfig, VitLamMAForConditionalGeneration)
    AutoProcessor.register("VitLamMAProcessor", VitLamMAProcessor)
except Exception as e:
    import warnings
    warnings.warn(f"Failed to register VitLamMA Auto classes: {e}")

__all__ = [
    "VitLamMAConfig",
    "VitLamMAModel",
    "VitLamMAForConditionalGeneration",
    "Conv3dDownsample",
    "VitLamMAVolumeProcessor",
    "VitLamMAProcessor",
]
