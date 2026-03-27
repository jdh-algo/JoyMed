# coding=utf-8
# Copyright 2025 VitLamMA Team. All rights reserved.
"""
VitLamMA Configuration

Multimodal fusion: Pre-trained ViT (3D) + LLaMA, with Qwen2-0.5B as fusion projector.
- ViT encodes 3D images; 3D conv downsampling reduces D/H/W to 1/2.
- Downsampled ViT features + variable decode tokens (by grid size) -> Qwen2-0.5B.
- Decode token outputs -> LLaMA as final input. Max 512*512*256 -> 1024 tokens.
"""

from typing import Any, Dict, Optional, Tuple, Union

from transformers import PretrainedConfig


class VitLamMAConfig(PretrainedConfig):
    """
    Configuration for VitLamMA: ViT (3D) + Qwen2-0.5B (projector) + LLaMA.

    Attributes:
        vision_config: Config or dict for the 3D ViT encoder.
        llm_config: Config or dict for the LLaMA language model.
        projector_model_name_or_path: Qwen2-0.5B model path (fusion bridge).
        num_decode_tokens: Max number of decode tokens (default 1024); actual count scales with input size.
        min_decode_tokens: Minimum decode tokens when input is small (default 64).
        max_grid_after_downsample: Grid (D', H', W') after downsample for max input 512*512*256 (default (32, 16, 16)).
        vision_hidden_size: ViT output hidden size.
        projector_hidden_size: Qwen2-0.5B hidden size (typically 896).
        llm_hidden_size: LLaMA hidden size.
        downsampling_factor: Factor for 3D conv downsampling on D/H/W (default 2).
        qwen2_use_global_attention: If True, Qwen2 projector uses global (bidirectional) attention;
            if False (default), uses original causal mask (each position only sees previous tokens).
        use_simple_projector: If True, use 3D conv -> single linear -> LLaMA (no Qwen2); if False, use Qwen2-0.5B projector.
        loss_weight_after_visual_n_tokens: Weight the first N text tokens after image tokens by loss_weight_after_visual_factor (0=disabled).
        loss_weight_after_visual_factor: Multiplier for loss on first N text tokens (e.g. 2.0).
        attention_bias_toward_visual: Add this value to attention scores for keys that are image tokens (0=disabled).
    """
    model_type = "vit_lamma"
    processor_class = "VitLamMAProcessor"

    def __init__(
        self,
        vision_config: Optional[Union[Dict[str, Any], "PretrainedConfig"]] = None,
        llm_config: Optional[Union[Dict[str, Any], "PretrainedConfig"]] = None,
        vision_model_name_or_path: Optional[str] = None,
        llm_model_name_or_path: Optional[str] = None,
        projector_config: Optional[Union[Dict[str, Any], "PretrainedConfig"]] = None,
        projector_model_name_or_path: str = "Qwen/Qwen2-0.5B",
        num_decode_tokens: int = 4096, # 1024
        min_decode_tokens: int = 64,
        max_grid_after_downsample: Optional[Tuple[int, int, int]] = None,
        image_3d_token_id: int = 151670,
        image_sep_token_id: Optional[int] = 151679,
        vision_hidden_size: int = 1024,
        projector_hidden_size: int = 896,
        llm_hidden_size: int = 4096,
        downsampling_factor: int = 2,
        processor_patch_size: int = 16,
        processor_temporal_patch_size: int = 8,
        processor_merge_size: int = 2,
        qwen2_use_global_attention: bool = True, # True: global attention, False: causal attention
        ############
        use_simple_projector: bool = False,  # True: 3D conv -> linear -> LLaMA; False: Qwen2-0.5B projector
        loss_weight_after_visual_n_tokens: int = 0,  # 0=disabled; first N text tokens after image get higher loss weight
        loss_weight_after_visual_factor: float = 2.0,  # loss multiplier for those tokens
        attention_bias_toward_visual: float = 0,  # 0=disabled; add to attention scores for image key positions
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_config = vision_config or {}
        self.llm_config = llm_config or {}
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
        self.llm_hidden_size = llm_hidden_size
        self.downsampling_factor = downsampling_factor
        self.processor_patch_size = processor_patch_size
        self.processor_temporal_patch_size = processor_temporal_patch_size
        self.processor_merge_size = processor_merge_size
        self.qwen2_use_global_attention = qwen2_use_global_attention
        self.use_simple_projector = use_simple_projector
        self.loss_weight_after_visual_n_tokens = loss_weight_after_visual_n_tokens
        self.loss_weight_after_visual_factor = loss_weight_after_visual_factor
        self.attention_bias_toward_visual = attention_bias_toward_visual

    def get_text_config(self, decoder: bool = False) -> "PretrainedConfig":
        """
        Return the config for the text/LLM decoder. Used by generation (e.g. DynamicCache)
        which expects a config with num_hidden_layers and other decoder attributes.
        """
        lc = self.llm_config
        if lc is None:
            raise ValueError("VitLamMAConfig.llm_config is required for get_text_config().")
        if isinstance(lc, dict):
            from transformers import LlamaConfig
            return LlamaConfig.from_dict(lc)
        return lc
