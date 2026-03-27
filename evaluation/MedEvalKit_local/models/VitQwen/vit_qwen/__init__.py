# coding=utf-8
# Copyright 2025 VitQwen Team. All rights reserved.
"""
VitQwen: ViT (3D) + Qwen2-0.5B (projector) + Qwen3-VL LLM.
"""

from .configuration_vit_qwen import VitQwenConfig
from .modeling_vit_qwen import VitQwenModel, VitQwenForConditionalGeneration
from .downsample_3d import Conv3dDownsample
from .volume_processing_vit_qwen import VitQwenVolumeProcessor
from .processing_vit_qwen import VitQwenProcessor

try:
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor
    AutoConfig.register("vit_qwen", VitQwenConfig)
    AutoModel.register(VitQwenConfig, VitQwenForConditionalGeneration)
    AutoModelForCausalLM.register(VitQwenConfig, VitQwenForConditionalGeneration)
    AutoProcessor.register("VitQwenProcessor", VitQwenProcessor)
except Exception as e:
    import warnings
    warnings.warn(f"Failed to register VitQwen Auto classes: {e}")

__all__ = [
    "VitQwenConfig",
    "VitQwenModel",
    "VitQwenForConditionalGeneration",
    "Conv3dDownsample",
    "VitQwenVolumeProcessor",
    "VitQwenProcessor",
]
