# coding=utf-8
# Copyright 2025 The CitrusV Team and The HuggingFace Inc. team. All rights reserved.
"""
CitrusV 3.0 Configuration

This module extends Qwen3-VL with:
- NIfTI medical image support (.nii, .nii.gz)
- Video compression at patch embedding level  
- Full compatibility with Qwen3-VL pretrained weights
"""

from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLVisionConfig,
    Qwen3VLTextConfig
)


class CitrusV3VisionConfig(Qwen3VLVisionConfig):
    """
    CitrusV 3.0 Vision Configuration extending Qwen3-VL Vision Config.
    
    Adds medical imaging specific parameters while maintaining full
    compatibility with Qwen3-VL.
    """
    model_type = "citrus_v_3_vision"
    
    def __init__(
        self,
        support_nifti: bool = True,
        **kwargs
    ):
        """
        Args:
            support_nifti (bool): Whether to support NIfTI medical image format
            **kwargs: All Qwen3-VL vision config parameters
        """
        super().__init__(**kwargs)
        self.support_nifti = support_nifti


class CitrusV3Config(Qwen3VLConfig):
    """
    CitrusV 3.0 Configuration extending Qwen3-VL.
    
    This configuration extends Qwen3-VL with medical AI capabilities:
    - NIfTI medical image support
    - Video compression at patch embedding level
    - Full backward compatibility with Qwen3-VL
    
    Example:
        ```python
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct",
            trust_remote_code=True
        )
        
        # Convert to CitrusV 3.0 config
        citrus_config = CitrusV3Config(**config.to_dict())
        citrus_config.support_nifti = True
        citrus_config.video_compression_enabled = True
        ```
    """
    model_type = "citrus_v_3"
    processor_class = "CitrusV3Processor"
    
    def __init__(
        self,
        # CitrusV 3.0 specific parameters
        support_nifti: bool = True,
        video_compression_enabled: bool = True,
        video_compression_threshold: float = 0.1,
        video_min_tokens_per_frame: int = 1,
        image_3d_token_id = 151670,
        vision_config: dict = None,
        text_config: dict = None,
        **kwargs
    ):
        """
        Args:
            support_nifti (bool): Whether to support NIfTI medical image format (.nii, .nii.gz)
            video_compression_enabled (bool): Whether to enable video compression at patch embedding level
            video_compression_threshold (float): Threshold for inter-frame difference (0.0-1.0)
            video_min_tokens_per_frame (int): Minimum tokens to keep per frame after compression
            vision_config (dict): Vision encoder configuration
            text_config (dict): Text model configuration
            **kwargs: Other Qwen3-VL config parameters
        """
        # Initialize parent Qwen3-VL config first
        super().__init__(
            vision_config=vision_config,
            text_config=text_config,
            **kwargs
        )
        self.image_3d_token_id = image_3d_token_id
        
        # Store CitrusV 3.0 specific parameters
        self.support_nifti = support_nifti
        self.video_compression_enabled = video_compression_enabled
        self.video_compression_threshold = video_compression_threshold
        self.video_min_tokens_per_frame = video_min_tokens_per_frame
        
        # Update vision_config with CitrusV parameters if it exists
        if hasattr(self, 'vision_config') and self.vision_config is not None:
            if isinstance(self.vision_config, dict):
                self.vision_config['support_nifti'] = support_nifti
            elif hasattr(self.vision_config, 'support_nifti'):
                self.vision_config.support_nifti = support_nifti
    
    def to_dict(self):
        """
        Serialize config to dictionary, including CitrusV 3.0 specific parameters.
        """
        output = super().to_dict()
        output['support_nifti'] = self.support_nifti
        output['video_compression_enabled'] = self.video_compression_enabled
        output['video_compression_threshold'] = self.video_compression_threshold
        output['video_min_tokens_per_frame'] = self.video_min_tokens_per_frame
        return output

