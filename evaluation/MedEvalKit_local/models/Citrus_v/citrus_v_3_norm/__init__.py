# coding=utf-8
# Copyright 2025 The CitrusV Team. All rights reserved.
"""
CitrusV 3.0 - Medical Multimodal AI based on Qwen3-VL

This package extends Qwen3-VL with:
- NIfTI medical image support (.nii, .nii.gz)
- Video compression at patch embedding level
- Full compatibility with Qwen3-VL pretrained weights
"""

from .configuration_citrus_v_3 import (
    CitrusV3Config,
    CitrusV3VisionConfig
)

from .modeling_citrus_v_3 import (
    CitrusV3Model,
    CitrusV3ForConditionalGeneration
)

from .image_processing_citrus_v_3 import CitrusV3ImageProcessor
from .video_processing_citrus_v_3 import CitrusV3VideoProcessor
from .volume_processing_citrus_v_3 import CitrusV3VolumeProcessor
from .processing_citrus_v_3 import CitrusV3Processor

from . import citrus_v_utils

# Register all processors and models with transformers Auto classes
# This must be done after all class definitions
try:
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModel,
        AutoModelForCausalLM,
        AutoProcessor,
        AutoVideoProcessor,
    )
    
    # Register configurations
    AutoConfig.register("citrus_v_3", CitrusV3Config)
    AutoModel.register(CitrusV3Config, CitrusV3ForConditionalGeneration)
    AutoModelForCausalLM.register(CitrusV3Config, CitrusV3ForConditionalGeneration)
    
    AutoImageProcessor.register("CitrusV3ImageProcessor", CitrusV3ImageProcessor)
    AutoVideoProcessor.register("CitrusV3VideoProcessor", CitrusV3VideoProcessor)
    AutoVideoProcessor.register("CitrusV3VolumeProcessor", CitrusV3VolumeProcessor)
    AutoProcessor.register("CitrusV3Processor", CitrusV3Processor)
    
except Exception as e:
    import warnings
    warnings.warn(f"Failed to register CitrusV classes: {e}")

__all__ = [
    "CitrusV3Config",
    "CitrusV3VisionConfig",
    "CitrusV3Model",
    "CitrusV3ForConditionalGeneration",
    "CitrusV3ImageProcessor",
    "CitrusV3VideoProcessor",
    "CitrusV3VolumeProcessor",
    "CitrusV3Processor",
    "mm_utils",
]

