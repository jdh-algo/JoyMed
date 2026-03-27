# coding=utf-8
# Copyright 2025 The CitrusV Team and The HuggingFace Inc. team. All rights reserved.
"""
CitrusV 3.0 Model

This module extends Qwen3-VL with:
- Video compression at patch embedding level (similar to HuluMed)
- Full compatibility with Qwen3-VL pretrained weights
"""

from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel
)
from transformers.modeling_outputs import BaseModelOutputWithPast
# from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast
from transformers.utils import is_torchdynamo_compiling

from .configuration_citrus_v_3 import CitrusV3Config

from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast, Qwen3VLCausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils import is_torchdynamo_compiling
from transformers.utils.generic import check_model_inputs

class CitrusV3Model(Qwen3VLModel):
    """
    CitrusV 3.0 Model extending Qwen3-VL with video compression and volume support.
    
    Key enhancements:
    - Video compression at patch embedding level (similar to HuluMed)
    - Native support for 3D medical volumes via volume_grid_thw
    - Full backward compatibility with Qwen3-VL
    """
    config_class = CitrusV3Config
    
    def __init__(self, config: CitrusV3Config):
        super().__init__(config)
        self.config = config
        
        # Define volume token ID (reuse video token ID for visual encoder compatibility)
        # The distinction is made at the token level in input_ids
        self.image_3d_token_id = config.image_3d_token_id if hasattr(config, 'image_3d_token_id') else 151670
    
    def get_volume_features(
        self, pixel_values_volumes: torch.FloatTensor, volume_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        Encodes 3D medical volumes into continuous embeddings that can be forwarded to the language model.
        
        Args:
            pixel_values_volumes (`torch.FloatTensor` of shape `(batch_size, num_slices, num_channels, image_size, image_size)`):
                The tensors corresponding to the input volumes.
            volume_grid_thw (`torch.LongTensor` of shape `(num_volumes, 3)`, *optional*):
                The temporal (slices), height and width of feature shape of each volume in LLM.
        
        Returns:
            Volume embeddings and deepstack visual embeddings (same format as images/videos)
        """
        # Volume processing is identical to video processing (both are 4D tensors)
        # We use the same visual encoder
        return self.get_video_features(pixel_values_volumes, volume_grid_thw)
    
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
        volume_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Extended to support volume features in addition to image and video features.
        
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that
        the placeholder token count is equal to the length of multimodal features.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
            special_volume_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.image_3d_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_volume_mask = special_volume_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id
            special_volume_mask = input_ids == self.image_3d_token_id
        
        # Check image features
        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )
        
        # Check video features
        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )
        
        # Check volume features
        n_volume_tokens = special_volume_mask.sum()
        special_volume_mask = special_volume_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if volume_features is not None and inputs_embeds[special_volume_mask].numel() != volume_features.numel():
            raise ValueError(
                f"Volume features and volume tokens do not match: tokens: {n_volume_tokens}, features {volume_features.shape[0]}"
            )
        
        return special_image_mask, special_video_mask, special_volume_mask
    
    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        volume_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Override to support volume_grid_thw in addition to image_grid_thw and video_grid_thw.
        
        Volume data is treated independently from video data, with its own grid_thw parameter.
        This maintains clean separation between temporal video sequences and spatial volume slices.
        
        Args:
            input_ids: Input token IDs
            image_grid_thw: Image grid dimensions [num_images, 3]
            video_grid_thw: Video grid dimensions [num_videos, 3]
            volume_grid_thw: Volume grid dimensions [num_volumes, 3] ✨ NEW
            attention_mask: Attention mask
        
        Returns:
            Tuple of (position_ids, mrope_position_deltas)
        """
        # Merge volume_grid_thw with video_grid_thw for RoPE computation
        # This is necessary because volume and video are processed similarly
        # (both have temporal/slice dimension) but we keep them separate in the API
        merged_video_grid_thw = video_grid_thw
        if volume_grid_thw is not None:
            if merged_video_grid_thw is not None:
                # Both video and volume present: concatenate
                merged_video_grid_thw = torch.cat([merged_video_grid_thw, volume_grid_thw], dim=0)
            else:
                # Only volume present: use it as video_grid_thw
                merged_video_grid_thw = volume_grid_thw
        
        # Call parent's get_rope_index with merged video_grid_thw
        return super().get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=merged_video_grid_thw,
            attention_mask=attention_mask,
        )
    
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        pixel_values_volumes: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        volume_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Forward pass with support for 3D medical volumes.
        
        Follows the exact same pattern as Qwen3-VL: process images, videos, and volumes
        separately, then merge their visual masks.
        
        Args:
            pixel_values_volumes: 3D volume tensors (B, S, C, H, W) where S is number of slices
            volume_grid_thw: Volume grid dimensions (B, 3) for [slices, height, width]
            ... (other args same as parent)
        """
        
        # Validate inputs (same as parent)
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        
        # Initialize masks
        image_mask = None
        video_mask = None
        volume_mask = None
        
        # Process images (if present)
        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        
        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        
        if pixel_values_volumes is not None:
            volume_embeds, deepstack_volume_embeds = self.get_volume_features(pixel_values_volumes, volume_grid_thw)
            volume_embeds = torch.cat(volume_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            volume_embeds = volume_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, _, volume_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, volume_features=volume_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(volume_mask, volume_embeds)
        
        deepstack_visual_embeds = kwargs.pop('deepstack_visual_embeds', None)
        visual_pos_masks = kwargs.pop('visual_pos_masks', None)
        if deepstack_visual_embeds is None and visual_pos_masks is None: 
            masks = []
            embeds_list = []
            if image_mask is not None:
                image_mask = image_mask[..., 0]
                masks.append(image_mask)
                embeds_list.append(deepstack_image_embeds)
            if video_mask is not None:
                video_mask = video_mask[..., 0]
                masks.append(video_mask)
                embeds_list.append(deepstack_video_embeds)
            if volume_mask is not None:
                volume_mask = volume_mask[..., 0]
                masks.append(volume_mask)
                embeds_list.append(deepstack_volume_embeds)

            if len(masks) == 0:
                visual_pos_masks = None
                deepstack_visual_embeds = None
            else:
                visual_pos_masks = masks[0]
                for mask in masks[1:]:
                    visual_pos_masks = visual_pos_masks | mask
                indices = visual_pos_masks
                total_visual_tokens = visual_pos_masks.sum().item()
                modal_masks = [mask[indices] for mask in masks]
                deepstack_visual_embeds = []
                num_layers = len(embeds_list[0])
                for layer_idx in range(num_layers):
                    layer_embeds = [embeds[layer_idx] for embeds in embeds_list]
                    D = layer_embeds[0].shape[-1]
                    device = layer_embeds[0].device
                    embed_joint = layer_embeds[0].new_zeros(total_visual_tokens, D).to(device)
                    for mod_embed, mod_mask in zip(layer_embeds, modal_masks):
                        embed_joint[mod_mask, :] = mod_embed
                    deepstack_visual_embeds.append(embed_joint)
        
        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    volume_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)


        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )
        
        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


class CitrusV3ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """
    CitrusV 3.0 for Conditional Generation.
    
    Extends Qwen3VLForConditionalGeneration with:
    - Video compression capabilities
    - NIfTI image support (via processor)
    - Full backward compatibility
    """
    config_class = CitrusV3Config
    
    def __init__(self, config: CitrusV3Config):
        super().__init__(config)
        self.model = CitrusV3Model(config)
        
        # Post-init to tie weights
        self.post_init()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        pixel_values_volumes=None,
        image_grid_thw=None,
        video_grid_thw=None,
        volume_grid_thw=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        if pixel_values_volumes is not None:
            model_inputs["pixel_values_volumes"] = pixel_values_volumes
            model_inputs["volume_grid_thw"] = volume_grid_thw

        # On subsequent decode steps, drop heavy visual tensors
        if cache_position is not None and cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["pixel_values_volumes"] = None
        
        return model_inputs


    def compress_volume_tokens(self, inputs):
        """
        Compress volume tokens using patch-level compression (HuluMed style).
        
        Args:
            inputs: Encoded inputs with pixel_values_volumes and volume_grid_thw
        
        Returns:
            Updated inputs with compressed tokens and features
        """

        # step 1: get arguments
        pixel_values_volumes = inputs.get('pixel_values_volumes', None)
        volume_grid_thw = inputs.get('volume_grid_thw', None)
        input_ids = inputs['input_ids']
        labels = inputs.get('labels')
        attention_mask = inputs.get('attention_mask')
        position_ids = inputs.get('position_ids')
        if pixel_values_volumes is None:
            return inputs

        B, N = input_ids.shape
        assert B == 1, "Batch size must be 1 for volume compression"

        input_ids = input_ids.view(B * N)
        if attention_mask is not None:
            attention_mask = attention_mask.view(B * N)
        if labels is not None:
            labels = labels.view(B * N)

        # step 2: get raw volume embeds 
        batched_num_patches = volume_grid_thw.prod(dim=1).div(self.config.vision_config.spatial_merge_size ** 2).long()
        batched_volume_embeds, deepstack_volume_embeds = self.model.get_volume_features(pixel_values_volumes, volume_grid_thw)
        volume_embeds = torch.cat(batched_volume_embeds, dim=0)

        # step3: get compression mask according to volumes embeds
        compression_mask = self._get_compression_mask(
            volume_embeds,
            batched_num_patches,
            volume_grid_thw,
            merge_sizes=torch.tensor([self.config.vision_config.spatial_merge_size], dtype=torch.long, device=volume_grid_thw.device),
            threshold=0.1,
            min_tokens=1,
        )

        # step 4: modity compression mask if trucated data
        volume_embeds, compression_mask = self._maybe_truncate_volume_tokens(
            volume_embeds, deepstack_volume_embeds, compression_mask, batched_num_patches, input_ids, position_ids
        )

        # step 5: compression and return
        volume_embeds, deepstack_volume_embeds, input_ids, attention_mask, position_ids, labels = self._compress_volume_tokens(
            volume_embeds, deepstack_volume_embeds, compression_mask, input_ids, attention_mask, position_ids, labels
        )

        # update pointer
        # seq_len = position_ids.shape[-1]
        # text_position_ids = torch.arange(seq_len, device=position_ids.device).expand(1, *position_ids.shape[1:])
        # packed_params = get_packed_seq_params(text_position_ids)
        # inputs.update(packed_params)  # 覆盖

        if attention_mask is not None:
            attention_mask = attention_mask.view(B, -1)
        if labels is not None:
            labels = labels.view(B, -1)
        if input_ids is not None:
            input_ids = input_ids.view(B, -1)

        inputs['attention_mask'] = attention_mask
        inputs['position_ids'] = position_ids
        inputs['input_ids'] = input_ids 
        inputs['labels'] = labels

        return inputs, volume_embeds, deepstack_volume_embeds

    def _get_compression_mask(
        self, 
        mm_features: torch.FloatTensor, 
        batched_num_patches: torch.LongTensor,
        grid_thws: torch.LongTensor, 
        merge_sizes: torch.LongTensor, 
        threshold: float = 0.1, 
        min_tokens: int = 1,
    ) -> torch.BoolTensor:
        batched_features = mm_features.split(batched_num_patches.tolist(), dim=0)
        compression_masks = []
        for features, num_patches, grid_thw, merge_size in zip(batched_features, batched_num_patches, grid_thws, merge_sizes):
            t, h, w = grid_thw
            if t <= 1:
                compression_masks.append(torch.ones(num_patches, dtype=torch.bool, device=features.device))
            else:
                features = features.view(t, (h // merge_size) * (w // merge_size), -1)

                pixel_diff = features[1:] - features[:-1]
                pixel_diff = torch.abs(pixel_diff).mean(dim=-1) * 255
                pixel_diff = torch.cat([torch.full_like(pixel_diff[0:1], threshold + 1), pixel_diff], dim=0)
                mask = (pixel_diff / 255.0) > threshold
                padding_ids = torch.nonzero(mask.sum(dim=1) < min_tokens)[:, 0]
                mask[padding_ids, :min_tokens] = 1
                compression_masks.append(mask.flatten())
        return torch.cat(compression_masks)
    

    def _maybe_truncate_volume_tokens(
        self,
        mm_features: torch.FloatTensor,
        deepstack_visual_embeds: List,
        compression_mask: torch.BoolTensor,
        batched_num_patches: torch.LongTensor,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
    ):
        if position_ids is None or mm_features.shape[0] == input_ids.eq(self.config.image_3d_token_id).sum():
            return mm_features, compression_mask
        
        B, N = position_ids.shape[1], position_ids.shape[2]

        # Step 1: 使用 position_ids[0] (主位置) 找到每个样本的起始位置
        main_pos = position_ids[0]  # [B, N]

        # 找到每行中 position == 0 的列索引（即每条序列的起始位置）
        # nonzero 返回 (row_idx, col_idx)
        zero_positions = (main_pos == 0).nonzero(as_tuple=False)  # [B, 2] -> (b_idx, seq_start)
        
        # 提取每个样本的起始位置（在展平序列中的位置）
        seq_start_indices = (zero_positions[:, 0] * N + zero_positions[:, 1]).tolist()  # [B]
        seq_end_indices = seq_start_indices[1:] + [B * N]  # [B]

        # Step 2: 统计每个样本中实际的视觉 token 数量
        device = input_ids.device
        num_visual_tokens = []
        for start, end in zip(seq_start_indices, seq_end_indices):
            count = input_ids[start:end].eq(self.config.image_3d_token_id).sum().item()
            num_visual_tokens.append(count)

        # Step 3: 为每个样本生成截断 mask（保留前 min(actual, allowed) 个视觉 token）
        truncation_mask = []
        for actual_num, allowed_num in zip(num_visual_tokens, batched_num_patches):
            keep_num = min(actual_num, allowed_num)
            mask = torch.zeros(actual_num, dtype=torch.bool, device=device)
            mask[:keep_num] = True  # 保留前 keep_num 个
            truncation_mask.append(mask)

        # 拼接所有样本的 mask
        truncation_mask = torch.cat(truncation_mask)  # [Total_Visual_Tokens_in_input]

        # Step 4: 应用截断 mask
        deepstack_visual_embeds = [deepstack_visual_embed[truncation_mask] for deepstack_visual_embed in deepstack_visual_embeds]
        return mm_features[truncation_mask], deepstack_visual_embeds, compression_mask[truncation_mask]


    def _compress_volume_tokens(
        self,
        mm_features,
        deepstack_visual_embeds,
        compression_mask, 
        input_ids, 
        attention_mask,
        position_ids,
        labels,
    ):
        mm_features = mm_features[compression_mask]
        deepstack_visual_embeds = [deepstack_visual_embed[compression_mask] for deepstack_visual_embed in deepstack_visual_embeds]
        volume_selected = (input_ids == self.config.image_3d_token_id) # image_3d_token mask
        
        text_masks = torch.logical_not(volume_selected) # not volume mask
        text_masks[volume_selected] = compression_mask # change volume mask to compression mask to keep selected volume tokens
        input_ids = input_ids[text_masks] # keep only selected volume tokens

        if attention_mask is not None:
            attention_mask = attention_mask[text_masks]
        if labels is not None:
            labels = labels[text_masks]
        if position_ids is not None: # accomodate to rope
            _, B, N = position_ids.shape
            assert text_masks.numel() == B * N, f"Expected {B*N}, got {text_masks.numel()}"

            text_masks_2d = text_masks.view(B, N)  # [B, N]
            compressed_position_ids = []
            for b in range(B):
                mask_b = text_masks_2d[b]  # [N]
                pos_b = position_ids[:, b, :]  # [3, N]
                pos_b_compressed = pos_b[:, mask_b]  # [3, N_new]
                compressed_position_ids.append(pos_b_compressed)
            
            # 检查长度是否一致
            if len(compressed_position_ids) > 0:
                N_new = compressed_position_ids[0].shape[1]
                assert all(p.shape[1] == N_new for p in compressed_position_ids), \
                    "压缩后序列长度不一致，无法堆叠。请确保压缩策略是 batch-invariant。"
                
                position_ids = torch.stack(compressed_position_ids, dim=1)  # [3, B, N_new]
        
        return mm_features, deepstack_visual_embeds, input_ids, attention_mask, position_ids, labels


    # @check_model_inputs
    # def forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[Cache] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     pixel_values: Optional[torch.Tensor] = None,
    #     pixel_values_videos: Optional[torch.FloatTensor] = None,
    #     pixel_values_volumes: Optional[torch.FloatTensor] = None,
    #     image_grid_thw: Optional[torch.LongTensor] = None,
    #     video_grid_thw: Optional[torch.LongTensor] = None,
    #     volume_grid_thw: Optional[torch.LongTensor] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     logits_to_keep: Union[int, torch.Tensor] = 0,
    #     **kwargs: Unpack[TransformersKwargs],
    # ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
    #     r"""
    #     labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
    #         Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
    #         config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
    #         (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    #     image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
    #         The temporal, height and width of feature shape of each image in LLM.
    #     video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
    #         The temporal, height and width of feature shape of each video in LLM.

    #     Example:
    #         TODO: Add example
    #     """
    #     if pixel_values_volumes is not None:
    #         inputs = {}
    #         inputs['input_ids'] = input_ids
    #         inputs['attention_mask'] = attention_mask
    #         inputs['position_ids'] = position_ids
    #         inputs['inputs_embeds'] = inputs_embeds
    #         inputs['labels'] = labels
    #         inputs['pixel_values_volumes'] = pixel_values_volumes
    #         inputs['volume_grid_thw'] = volume_grid_thw
            
    #         inputs, volume_embeds_compressed, deepstack_volume_embeds_compressed = self.compress_volume_tokens(inputs)
    #         attention_mask = inputs['attention_mask']
    #         position_ids = inputs['position_ids']
    #         input_ids = inputs['input_ids']
    #         labels = inputs['labels']
            
    #         # print(input_ids.shape, attention_mask.shape)
    #         rope_deltas = input_ids.eq(self.config.image_3d_token_id).sum().item()
    #         # print(rope_deltas)
            
    #         if inputs_embeds is None:
    #             inputs_embeds = self.model.get_input_embeddings()(input_ids)
            
    #         volume_embeds_compressed = volume_embeds_compressed.to(inputs_embeds.device, inputs_embeds.dtype)
    #         _, _, volume_mask = self.model.get_placeholder_mask(
    #             input_ids, inputs_embeds=inputs_embeds, volume_features=volume_embeds_compressed
    #         )
    #         inputs_embeds = inputs_embeds.masked_scatter(volume_mask, volume_embeds_compressed)
    #         visual_pos_masks = volume_mask[..., 0]
            
    #         outputs = self.model.language_model(
    #             input_ids=None,
    #             position_ids=position_ids,
    #             attention_mask=attention_mask,
    #             past_key_values=past_key_values,
    #             inputs_embeds=inputs_embeds,
    #             cache_position=torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device),
    #             visual_pos_masks=visual_pos_masks,
    #             deepstack_visual_embeds=deepstack_volume_embeds_compressed,
    #             **kwargs,
    #         )
            
    #         outputs = Qwen3VLModelOutputWithPast(
    #             last_hidden_state=outputs.last_hidden_state,
    #             past_key_values=outputs.past_key_values,
    #             rope_deltas=rope_deltas,
    #         )
        
    #     else:
    #         # print(input_ids.shape, input_ids, position_ids, inputs_embeds, cache_position)
    #         outputs = self.model(
    #             input_ids=input_ids,
    #             pixel_values=pixel_values,
    #             pixel_values_videos=pixel_values_videos,
    #             pixel_values_volumes=pixel_values_volumes,
    #             image_grid_thw=image_grid_thw,
    #             video_grid_thw=video_grid_thw,
    #             volume_grid_thw=volume_grid_thw,
    #             position_ids=position_ids,
    #             attention_mask=None,
    #             past_key_values=past_key_values,
    #             inputs_embeds=inputs_embeds,
    #             cache_position=cache_position,
    #             **kwargs,
    #         )

    #     hidden_states = outputs[0]

    #     # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    #     slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    #     logits = self.lm_head(hidden_states[:, slice_indices, :])

    #     loss = None
    #     if labels is not None:
    #         loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

    #     return Qwen3VLCausalLMOutputWithPast(
    #         loss=loss,
    #         logits=logits,
    #         past_key_values=outputs.past_key_values,
    #         rope_deltas=outputs.rope_deltas,
    #     )