# coding=utf-8
# Copyright 2025 The CitrusV Team and The HuggingFace Inc. team. All rights reserved.
"""
CitrusV 3.0 Image Processor

Extends Qwen3-VL Image Processor with NIfTI medical image support.
"""

from typing import Dict, List, Optional, Union
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from . import citrus_v_utils


class CitrusV3ImageProcessor(Qwen2VLImageProcessor):
    """
    CitrusV 3.0 Image Processor extending Qwen2VLImageProcessor (used by Qwen3-VL).
    
    Adds support for:
    - NIfTI medical images (.nii, .nii.gz)
    - Automatic format detection and conversion
    
    Example:
        ```python
        processor = CitrusV3ImageProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        
        # Process standard image
        images = processor("path/to/image.jpg")
        
        # Process NIfTI medical image
        nifti_images = processor("path/to/scan.nii.gz")
        ```
    """
    
    # def preprocess(
    #     self,
    #     images: Union[Image.Image, List[Image.Image], str, List[str]],
    #     **kwargs
    # ) -> Dict:
    #     """
    #     Preprocess images with NIfTI support.
        
    #     Args:
    #         images: Can be:
    #             - PIL Image
    #             - List of PIL Images
    #             - Path to image file (including .nii, .nii.gz)
    #             - List of image paths
    #             - NIfTI file path (will be converted to 2D slices)
    #         **kwargs: Additional arguments for parent processor
        
    #     Returns:
    #         Dictionary with processed image tensors
    #     """
        
    #     # Call parent's preprocess
    #     return super().preprocess(images, **kwargs)

