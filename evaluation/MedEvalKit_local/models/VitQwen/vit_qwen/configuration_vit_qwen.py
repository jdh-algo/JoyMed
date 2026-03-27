# coding=utf-8
# Copyright 2025 VitQwen Team. All rights reserved.
"""
VitQwen Configuration

Multimodal fusion: Pre-trained ViT (3D) + Qwen2-0.5B (projector) + Qwen3-VL LLM.
- ViT encodes 3D images; 3D conv downsampling reduces D/H/W to 1/2.
- Downsampled ViT features + variable decode tokens -> Qwen2-0.5B.
- Decode token outputs -> linear -> Qwen3-VL language model as final input.
"""

from typing import Any, Dict, Optional, Tuple, Union

from transformers import PretrainedConfig


class VitQwenConfig(PretrainedConfig):
    """
    Configuration for VitQwen: ViT (3D) + Qwen2-0.5B (projector) + Qwen3-VL LLM.

    Attributes:
        vision_config: Config or dict for the 3D ViT encoder.
        text_config: Config or dict for the Qwen3-VL text/language model (from Qwen3-VL).
        projector_config: Qwen2-0.5B config (fusion bridge).
        num_decode_tokens: Max number of decode tokens (default 1024).
        vision_hidden_size: ViT output hidden size.
        projector_hidden_size: Qwen2-0.5B hidden size (typically 896).
        llm_hidden_size: Qwen3-VL text hidden size (from text_config.hidden_size).
        downsampling_factor: Factor for 3D conv downsampling (default 2).
        qwen2_use_global_attention: If True, Qwen2 projector uses global attention.
        use_simple_projector: If True, 3D conv -> linear -> LLM (no Qwen2).
    """
    model_type = "vit_qwen"
    processor_class = "VitQwenProcessor"

    def __init__(
        self,
        vision_config: Optional[Union[Dict[str, Any], "PretrainedConfig"]] = None,
        text_config: Optional[Union[Dict[str, Any], "PretrainedConfig"]] = None,
        vision_model_name_or_path: Optional[str] = None,
        llm_model_name_or_path: Optional[str] = None,
        projector_config: Optional[Union[Dict[str, Any], "PretrainedConfig"]] = None,
        projector_model_name_or_path: str = "Qwen/Qwen2-0.5B",
        num_decode_tokens: int = 4096, #1024
        min_decode_tokens: int = 64,
        max_grid_after_downsample: Optional[Tuple[int, int, int]] = None,
        image_3d_token_id: int = 151670,
        image_sep_token_id: Optional[int] = 151679,
        vision_hidden_size: int = 1024,
        projector_hidden_size: int = 896,
        llm_hidden_size: Optional[int] = None,
        downsampling_factor: int = 2,
        processor_patch_size: int = 16,
        processor_temporal_patch_size: int = 8,
        processor_merge_size: int = 2,
        qwen2_use_global_attention: bool = True,
        use_simple_projector: bool = False,
        projector_type: str = "pixel_shuffle", # qwen2: Qwen2-0.5B projector; pixel_shuffle: Pixel Shuffle projector
        num_deepstack_layers: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_config = vision_config or {}
        self.text_config = text_config or {}
        self.projector_config = projector_config or {}
        self.vision_model_name_or_path = vision_model_name_or_path
        self.llm_model_name_or_path = llm_model_name_or_path
        self.projector_model_name_or_path = projector_model_name_or_path
        self.num_decode_tokens = num_decode_tokens
        self.min_decode_tokens = min_decode_tokens
        self.max_grid_after_downsample = max_grid_after_downsample or (16, 16, 16)
        self.image_3d_token_id = image_3d_token_id
        self.image_sep_token_id = image_sep_token_id
        self.vision_hidden_size = vision_hidden_size
        self.projector_hidden_size = projector_hidden_size
        self.llm_hidden_size = llm_hidden_size  # set from text_config if None
        self.downsampling_factor = downsampling_factor
        self.processor_patch_size = processor_patch_size
        self.processor_temporal_patch_size = processor_temporal_patch_size
        self.processor_merge_size = processor_merge_size
        self.qwen2_use_global_attention = qwen2_use_global_attention
        self.use_simple_projector = use_simple_projector
        # Qwen3-VL LLM deepstack: number of layers to inject visual (default 3, from vision_config.deepstack_visual_indexes)
        pt = projector_type or "qwen2"
        self.projector_type = pt.lower() if isinstance(pt, str) else pt
        self.num_deepstack_layers = num_deepstack_layers

    def get_text_config(self, decoder: bool = True) -> "PretrainedConfig":
        """Return the config for the Qwen3-VL text decoder."""
        tc = self.text_config
        if not tc:
            raise ValueError("VitQwenConfig.text_config is required for get_text_config().")
        if isinstance(tc, dict):
            from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
            return Qwen3VLTextConfig.from_dict(tc)
        return tc
