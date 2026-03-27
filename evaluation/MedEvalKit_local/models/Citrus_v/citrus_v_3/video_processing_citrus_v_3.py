# coding=utf-8
# Copyright 2025 The CitrusV Team and The HuggingFace Inc. team. All rights reserved.
"""
CitrusV 3.0 Video Processor

Extends Qwen3-VL Video Processor with video compression capabilities.
"""

from typing import Dict, List, Optional, Union
from PIL import Image
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

from . import citrus_v_utils


class CitrusV3VideoProcessor(Qwen3VLVideoProcessor):
    """
    CitrusV 3.0 Video Processor extending Qwen3VLVideoProcessor.
    
    Adds support for:
    - Flexible video loading from multiple sources
    - Optional temporal sampling
    
    Note: Video compression happens at the model level (patch embeddings),
    not at the processor level.
    """
    
    config_type = "qwen3_vl"  # Match Qwen3VLVideoProcessor
    
    def __call__(
        self,
        videos: Union[str, List[str], List[List[Image.Image]]],
        fps: Optional[float] = None,
        **kwargs
    ) -> Dict:
        """
        Process videos with flexible loading.
        
        Args:
            videos: Can be:
                - Path to video file
                - List of video file paths
                - List of frame lists (List[List[PIL.Image]])
                - Directory containing video frames
            fps: Frames per second for temporal sampling
            **kwargs: Additional arguments for parent processor
        
        Returns:
            Dictionary with processed video tensors
        """
        # Note: Video compression at patch embedding level happens in the model,
        # not here. This processor focuses on loading and basic preprocessing.
        
        return super().__call__(videos, fps=fps, **kwargs)

