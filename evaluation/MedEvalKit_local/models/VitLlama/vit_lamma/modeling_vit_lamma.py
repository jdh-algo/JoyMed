# coding=utf-8
# Copyright 2025 VitLamMA Team. All rights reserved.
"""
VitLamMA Model: ViT (3D) + Qwen2-0.5B (projector) + LLaMA.

Fusion pipeline:
1. ViT encodes 3D images -> (B, N, vision_hidden_size)
2. 3D conv downsampling: D/H/W -> 1/2 -> (B, N', vision_hidden_size)
3. Downsampled ViT -> linear -> (B, N', projector_hidden_size); inject 1024 learnable decode tokens
4. [decode_tokens, vit_embeds] -> Qwen2-0.5B -> take hidden states at first 1024 positions
5. 1024 token outputs -> linear -> LLaMA as final visual input
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PreTrainedModel, PretrainedConfig

from .configuration_vit_lamma import VitLamMAConfig
from .downsample_3d import Conv3dDownsample


def _unpatchify_volume(
    patches_flat: torch.Tensor,
    volume_grid_thw: torch.LongTensor,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
) -> torch.Tensor:
    """
    Reverse of volume_processing patchify: (B, N, patch_dim) -> (B, C, D, H, W).
    patches_flat: (B, N, patch_dim), N = grid_t * grid_h * grid_w.
    volume_grid_thw: (B, 3) [grid_t, grid_h, grid_w].
    patch_dim = C * temporal_patch_size * patch_size * patch_size.
    Returns (B, C, D, H, W) with D = grid_t * temporal_patch_size, H = grid_h * patch_size, W = grid_w * patch_size.
    """
    B, N, patch_dim = patches_flat.shape
    if volume_grid_thw.dim() == 1:
        grid_t, grid_h, grid_w = volume_grid_thw.tolist()
    else:
        grid_t, grid_h, grid_w = volume_grid_thw[0].tolist()
    grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
    channel = patch_dim // (temporal_patch_size * patch_size * patch_size)
    if channel * temporal_patch_size * patch_size * patch_size != patch_dim:
        raise ValueError(
            f"patch_dim {patch_dim} is not divisible by temporal_patch_size*patch_size^2 = "
            f"{temporal_patch_size * patch_size * patch_size}; cannot infer channel."
        )
    if N != grid_t * grid_h * grid_w:
        raise ValueError(
            f"Number of tokens N={N} does not match grid_t*grid_h*grid_w = {grid_t * grid_h * grid_w}."
        )
    gt_m = grid_t // merge_size
    gh_m = grid_h // merge_size
    gw_m = grid_w // merge_size
    # Reshape (B, N, patch_dim) -> (B, gt/m, gh/m, gw/m, m, m, m, c, tp, p, p)
    x = patches_flat.view(
        B,
        gt_m,
        gh_m,
        gw_m,
        merge_size,
        merge_size,
        merge_size,
        channel,
        temporal_patch_size,
        patch_size,
        patch_size,
    )
    # Inverse of permute(0, 1, 5, 8, 2, 6, 9, 4, 3, 7, 10)
    x = x.permute(0, 1, 4, 8, 7, 2, 5, 9, 3, 6, 10)
    # x: (B, gt/m, m, tp, c, gh/m, m, p, gw/m, m, p) -> (B, C, D, H, W)
    x = x.permute(0, 4, 1, 2, 3, 5, 6, 7, 8, 9, 10)
    x = x.reshape(
        B,
        channel,
        grid_t * temporal_patch_size,
        grid_h * patch_size,
        grid_w * patch_size,
    )
    return x


def _config_from_dict(d: dict) -> PretrainedConfig:
    """Build a PretrainedConfig from a dict; AutoConfig has no from_dict, use model_type -> config class."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    model_type = d.get("model_type")
    if model_type and model_type in CONFIG_MAPPING:
        config_class = CONFIG_MAPPING[model_type]
        return config_class.from_dict(d)
    return PretrainedConfig.from_dict(d)


def _is_full_config(c: Any) -> bool:
    if not isinstance(c, dict) or not c:
        return False
    return "model_type" in c or "hidden_size" in c


class LaMedViTWrapper(nn.Module):
    """
    Wraps LaMed ViT so that forward(pixel_values_3d, volume_grid_dhw=...) returns
    (B, N, H) patch features. Drops CLS token when classification=True.
    """

    def __init__(self, vit: nn.Module, *, expected_in_channels: int):
        super().__init__()
        self.vit = vit
        self.expected_in_channels = int(expected_in_channels)

    def forward(self, pixel_values_3d: torch.Tensor, volume_grid_dhw=None):
        # LaMed/Monai ViT expects images as (B, C, D, H, W). In our pipeline it's easy to end up with
        # (B, D, C, H, W) (depth and channel swapped), which breaks PatchEmbeddingBlock(perceptron).
        if pixel_values_3d.dim() == 5:
            # Auto-fix common swap: (B, D, C, H, W) -> (B, C, D, H, W)
            if pixel_values_3d.size(1) != self.expected_in_channels and pixel_values_3d.size(2) == self.expected_in_channels:
                pixel_values_3d = pixel_values_3d.permute(0, 2, 1, 3, 4).contiguous()

            # Channel adaptation if needed (keeps training from crashing on grayscale/RGB mismatch)
            c = pixel_values_3d.size(1)
            if c != self.expected_in_channels:
                if self.expected_in_channels == 1 and c == 3:
                    pixel_values_3d = pixel_values_3d.mean(dim=1, keepdim=True)
                elif self.expected_in_channels == 3 and c == 1:
                    pixel_values_3d = pixel_values_3d.repeat(1, 3, 1, 1, 1)
                else:
                    raise ValueError(
                        f"LaMedViTWrapper expected in_channels={self.expected_in_channels}, but got C={c} "
                        f"with input shape={tuple(pixel_values_3d.shape)}."
                    )
        # print(f"pixel_values_3d.shape: {pixel_values_3d.shape}")
        out, _ = self.vit(pixel_values_3d)
        if getattr(self.vit, "classification", False):
            out = out[:, 1:, :]
        return out


def _build_lamed_vit_from_config(vc: Dict[str, Any]) -> nn.Module:
    import sys
    from pathlib import Path
    from .LaMed.src.model.multimodal_encoder.vit import ViT
    encoder_in_channels = int(vc.get("encoder_in_channels", 1))
    vit = ViT(
        in_channels=encoder_in_channels,
        img_size=tuple(vc["img_size"]) if "img_size" in vc else (32, 256, 256),
        patch_size=tuple(vc["patch_size"]) if "patch_size" in vc else (4, 16, 16),
        hidden_size=vc.get("encoder_hidden_size", 768),
        mlp_dim=vc.get("mlp_dim", 3072),
        num_layers=vc.get("num_layers", 12),
        num_heads=vc.get("num_heads", 12),
        pos_embed=vc.get("pos_embed", "perceptron"),
        dropout_rate=vc.get("dropout_rate", 0.0),
        spatial_dims=vc.get("spatial_dims", 3),
        classification=vc.get("classification", True),
        mae=True,
        use_rope=vc.get("use_rope", True),
    )
    return LaMedViTWrapper(vit, expected_in_channels=encoder_in_channels)


def _load_vision_encoder(config: VitLamMAConfig):
    vc = getattr(config, "vision_config", None)
    if vc is not None and isinstance(vc, dict) and vc.get("model_type") == "lamed_vit":
        return _build_lamed_vit_from_config(vc).to(torch.bfloat16)
    if vc is not None and _is_full_config(vc):
        cfg = _config_from_dict(vc) if isinstance(vc, dict) else vc
        return AutoModel.from_config(cfg).to(torch.bfloat16)
    path = getattr(config, "vision_model_name_or_path", None)
    if not path:
        return None
    return AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16)


def _load_qwen2_model(config: VitLamMAConfig):
    pc = getattr(config, "projector_config", None)
    if pc is not None and _is_full_config(pc):
        cfg = _config_from_dict(pc) if isinstance(pc, dict) else pc
        # 从 config 构建：用 sdpa 避免 inputs_embeds 下 Flash2 varlen 的 cu_seqlens_q 报错；加载后转 bf16
        model = AutoModel.from_config(cfg, attn_implementation="sdpa")
        return model.to(torch.bfloat16)
    path = getattr(config, "projector_model_name_or_path", "Qwen/Qwen2-0.5B")
    # 使用 sdpa + bf16：inputs_embeds 下 Flash2 会走 varlen 报 cu_seqlens_q，故用 sdpa 保稳
    return AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )


def _load_llm(config: VitLamMAConfig):
    lc = getattr(config, "llm_config", None)
    # attention_bias_toward_visual != 0 需要 4D mask，与 Flash2 不兼容，只能用 sdpa。
    # bias == 0 时默认 flash_attention_2：O(L) 显存，比 sdpa+4D mask 的 O(L^2) 省很多。
    # 注意：config._attn_implementation 是 HF property，默认返回 "sdpa"，不能用来判断用户意图。
    bias_val = getattr(config, "attention_bias_toward_visual", 0.0)
    llm_attn = "sdpa" if bias_val != 0.0 else "flash_attention_2"

    if lc is not None and _is_full_config(lc):
        cfg = _config_from_dict(lc) if isinstance(lc, dict) else lc
        return AutoModelForCausalLM.from_config(cfg, attn_implementation=llm_attn).to(torch.bfloat16)
    path = getattr(config, "llm_model_name_or_path", None)
    if not path:
        return None
    return AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation=llm_attn,
    )


class VitLamMAModel(PreTrainedModel):
    """
    VitLamMA: ViT (3D) -> 3D downsample -> Qwen2-0.5B (+ 1024 decode tokens) -> LLaMA.
    """

    config_class = VitLamMAConfig
    _supports_flash_attn_2 = True
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(
        self,
        config: VitLamMAConfig,
        vision_encoder: Optional[nn.Module] = None,
        llm: Optional[PreTrainedModel] = None,
        qwen2_projector: Optional[nn.Module] = None,
    ):
        super().__init__(config)
        self.config = config
        c = config

        self.vision_encoder = vision_encoder if vision_encoder is not None else _load_vision_encoder(config)
        # 3D conv downsampling: D/H/W -> 1/2 each, output dim = vision_hidden_size
        self.downsample_3d = Conv3dDownsample(c.vision_hidden_size, factor=c.downsampling_factor)
        # Simple path: downsample -> linear -> LLaMA
        self.simple_projector = nn.Linear(c.vision_hidden_size, c.llm_hidden_size, bias=True)

        self.vit_to_projector = nn.Linear(c.vision_hidden_size, c.projector_hidden_size, bias=True)
        # Pool of 1024 decode tokens; interpolate to variable num_decode then linear for stability
        self.decode_tokens = nn.Parameter(torch.zeros(1, c.num_decode_tokens, c.projector_hidden_size))
        nn.init.normal_(self.decode_tokens, std=0.02)
        self.decode_interp_linear = nn.Linear(c.projector_hidden_size, c.projector_hidden_size, bias=True)

        self.qwen2_projector = qwen2_projector if qwen2_projector is not None else _load_qwen2_model(config)
        self.projector_to_llm = nn.Linear(c.projector_hidden_size, c.llm_hidden_size, bias=True)
        self.llm = llm if llm is not None else _load_llm(config)
        if self.llm is None:
            raise ValueError("LLaMA is required: pass llm or set config.llm_model_name_or_path / config.llm_config.")

        self.post_init()

    def get_placeholder_mask(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: torch.Tensor,
        volume_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Obtain the placeholder mask for volume (image_3d) tokens from input_ids or inputs_embeds.
        Validates that the number of placeholder positions matches the length of volume_features.
        Returns a boolean mask of shape (B, L, 1) expanded to (B, L, D) for masked_scatter.
        """
        mm_token_id = getattr(self.config, "image_3d_token_id", 151670)
        if input_ids is not None:
            special_volume_mask = input_ids == mm_token_id
        else:
            placeholder_embed = self.llm.get_input_embeddings()(
                torch.tensor(mm_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_volume_mask = (inputs_embeds == placeholder_embed).all(dim=-1)
        n_volume_tokens = special_volume_mask.sum().item()
        special_volume_mask = special_volume_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if volume_features is not None and inputs_embeds[special_volume_mask].numel() != volume_features.numel():
            raise ValueError(
                f"Volume features and placeholder tokens do not match: "
                f"placeholders: {n_volume_tokens}, features: {volume_features.shape[0]}"
            )
        return special_volume_mask

    def get_vision_features(
        self,
        pixel_values_3d: torch.Tensor,
        volume_grid_dhw: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Encode 3D volume with ViT.
        - If pixel_values_3d is (B, N, patch_dim) (processor output): unpatchify to (B, C, D, H, W) then run ViT.
        - Else pixel_values_3d: (B, D, C, H, W) or (B, C, D, H, W).
        volume_grid_dhw: (B, 3) or (3,) as [grid_t, grid_h, grid_w] (or [D, H, W] patch grid).
        Returns: (B, N, vision_hidden_size) with N = D*H*W.
        """
        if self.vision_encoder is None:
            raise ValueError("vision_encoder is not set; pass vision_features directly to forward.")

        # 重要：processor 通常输出 float32，但我们经常以 bf16 加载模型/ViT。
        # 线性层/Conv 等要求输入 dtype 与权重一致，否则会报 mat1/mat2 dtype mismatch。
        try:
            ve_param = next(self.vision_encoder.parameters(), None)
            if ve_param is not None and pixel_values_3d.dtype != ve_param.dtype:
                pixel_values_3d = pixel_values_3d.to(dtype=ve_param.dtype)
        except Exception:
            # 若 vision_encoder 无参数或不支持 parameters()，则跳过；后续若报错再定位
            pass

        c = self.config
        if pixel_values_3d.dim() == 3:
            # Processor output (B, N, patch_dim): unpatchify to (B, C, D, H, W) for LaMed ViT
            pixel_values_3d = _unpatchify_volume(
                pixel_values_3d,
                volume_grid_dhw,
                patch_size=getattr(c, "processor_patch_size", 16),
                temporal_patch_size=getattr(c, "processor_temporal_patch_size", 8),
                merge_size=getattr(c, "processor_merge_size", 2),
            )
            return self.vision_encoder(pixel_values_3d, volume_grid_dhw=None)
        
        if hasattr(self.vision_encoder, "get_volume_features"):
            out = self.vision_encoder.get_volume_features(pixel_values_3d, volume_grid_dhw)
            if isinstance(out, (list, tuple)):
                out = out[0] if isinstance(out[0], torch.Tensor) else torch.cat(out, dim=0)
            return out
        if hasattr(self.vision_encoder, "forward"):
            return self.vision_encoder(pixel_values_3d, volume_grid_dhw=volume_grid_dhw)
        raise NotImplementedError("vision_encoder must implement get_volume_features or forward(pixel_values_3d, volume_grid_dhw).")

    def _num_decode_from_grid(self, grid_after_downsample: Tuple[int, int, int]) -> int:
        """Compute num_decode from grid (D', H', W') after downsample. Max 512*512*256 -> 1024; scale proportionally."""
        d2, h2, w2 = grid_after_downsample
        n_prime = d2 * h2 * w2
        max_grid = getattr(self.config, "max_grid_after_downsample", (16, 16, 16))
        n_max = max_grid[0] * max_grid[1] * max_grid[2]
        max_decode = getattr(self.config, "num_decode_tokens", 1024)
        min_decode = getattr(self.config, "min_decode_tokens", 64)
        num_decode = round(max_decode * n_prime / max(n_max, 1))
        # print(f"num_decode: {num_decode}")
        return max(min_decode, min(max_decode, num_decode))

    def _forward_single_volume_to_llm(
        self,
        vit_down_i: torch.Tensor,
        grid_tuple_i: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Single volume (1, N', C) -> projector -> LLaMA，用于 variable-shape 分支。"""
        c = self.config
        B_i = 1
        if getattr(c, "use_simple_projector", False):
            return self.simple_projector(vit_down_i)

        num_decode = self._num_decode_from_grid(grid_tuple_i)
        vit_proj = self.vit_to_projector(vit_down_i)
        pool = self.decode_tokens.to(vit_proj.dtype).to(vit_proj.device).expand(B_i, -1, -1)
        if num_decode >= c.num_decode_tokens:
            dec = pool[:, : c.num_decode_tokens]
            num_decode = c.num_decode_tokens
        else:
            pool_t = pool.transpose(1, 2)
            dec_t = torch.nn.functional.interpolate(
                pool_t, size=num_decode, mode="linear", align_corners=False
            )
            dec = dec_t.transpose(1, 2)
        dec = self.decode_interp_linear(dec)
        combined = torch.cat([vit_proj, dec], dim=1)
        L = combined.size(1)
        qw_kwargs = {"inputs_embeds": combined, "return_dict": True}
        if getattr(c, "qwen2_use_global_attention", False):
            qw_param = next(self.qwen2_projector.parameters(), None)
            qw_dtype = qw_param.dtype if qw_param is not None else combined.dtype
            qw_attn = torch.zeros(
                B_i,
                1,
                L,
                L,
                device=combined.device,
                dtype=qw_dtype,
            )
            qw_kwargs["attention_mask"] = qw_attn
        qw_out = self.qwen2_projector(**qw_kwargs)
        hidden = qw_out.last_hidden_state[:, -num_decode:]
        try:
            tgt_dtype = self.projector_to_llm.weight.dtype
            if hidden.dtype != tgt_dtype:
                hidden = hidden.to(dtype=tgt_dtype)
        except Exception:
            pass
        return self.projector_to_llm(hidden)

    def forward_visual_to_llm_embeds(
        self,
        vision_features: Optional[torch.Tensor] = None,
        volume_grid_dhw: Optional[torch.LongTensor] = None,
        pixel_values_3d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run ViT -> 3D downsample -> projector -> LLaMA.
        - use_simple_projector: 返回 (B, N', llm_hidden_size)，N' = 下采样后 token 数，与 placeholder 数一致。
        - Qwen2 路径: num_decode 由 grid 计算（64..1024），返回 (B, num_decode, llm_hidden_size)。
        """
        c = self.config
        if volume_grid_dhw is None:
            raise ValueError("volume_grid_dhw is required for 3D downsampling.")

        # 当同一 batch 内各 volume grid 不同（不同 shape）时，逐 volume 单独走 ViT+Qwen2 路径，再按 placeholder 顺序拼接。
        if (
            pixel_values_3d is not None
            and volume_grid_dhw.dim() == 2
            and volume_grid_dhw.size(0) > 1
        ):
            B = volume_grid_dhw.size(0)
            same_grid = all(
                (volume_grid_dhw[i] == volume_grid_dhw[0]).all().item() for i in range(1, B)
            )
            if not same_grid:
                out_list: List[torch.Tensor] = []
                for i in range(B):
                    pv_i = pixel_values_3d[i : i + 1]
                    d, h, w = (
                        int(volume_grid_dhw[i, 0].item()),
                        int(volume_grid_dhw[i, 1].item()),
                        int(volume_grid_dhw[i, 2].item()),
                    )
                    n_i = d * h * w
                    if pv_i.dim() == 3 and pv_i.size(1) > n_i:
                        pv_i = pv_i[:, :n_i, :].contiguous()
                    vf_i = self.get_vision_features(pv_i, volume_grid_dhw[i : i + 1])
                    vd_i, new_grid_i = self.downsample_3d(vf_i, volume_grid_dhw[i : i + 1])
                    grid_tuple_i = (
                        int(new_grid_i[0, 0].item()),
                        int(new_grid_i[0, 1].item()),
                        int(new_grid_i[0, 2].item()),
                    )
                    out_i = self._forward_single_volume_to_llm(vd_i, grid_tuple_i)
                    out_list.append(out_i)
                # (1, sum_N, D): flatten 后与 placeholder 顺序一一对应
                return torch.cat(out_list, dim=1)

        if vision_features is None:
            if pixel_values_3d is None or volume_grid_dhw is None:
                raise ValueError("Need either (vision_features, volume_grid_dhw) or (pixel_values_3d, volume_grid_dhw).")
            vision_features = self.get_vision_features(pixel_values_3d, volume_grid_dhw)

        vit_down, new_grid = self.downsample_3d(vision_features, volume_grid_dhw)
        B = vit_down.size(0)
        if new_grid.dim() == 2:
            grid_tuple = (int(new_grid[0, 0].item()), int(new_grid[0, 1].item()), int(new_grid[0, 2].item()))
        else:
            grid_tuple = (int(new_grid[0].item()), int(new_grid[1].item()), int(new_grid[2].item()))
        num_decode = self._num_decode_from_grid(grid_tuple)

        if getattr(c, "use_simple_projector", False):
            # Simple path: 3D conv -> linear -> LLaMA，token 数与下采样后一致，不限制为 num_decode
            x = self.simple_projector(vit_down)
            return x

        vit_proj = self.vit_to_projector(vit_down)
        # Interpolate decode token pool (1024) to num_decode, then linear for stability
        pool = self.decode_tokens.to(vit_proj.dtype).to(vit_proj.device).expand(B, -1, -1)
        if num_decode >= c.num_decode_tokens:
            dec = pool[:, : c.num_decode_tokens]
            num_decode = c.num_decode_tokens
        else:
            pool_t = pool.transpose(1, 2)
            dec_t = torch.nn.functional.interpolate(
                pool_t, size=num_decode, mode="linear", align_corners=False
            )
            dec = dec_t.transpose(1, 2)
        dec = self.decode_interp_linear(dec)

        # Qwen2 为因果 decoder：每个位置只能 attend 到左侧。故必须 [vit_proj, dec]，
        # 这样 decode 位置在右侧，能看见全部 ViT 输出；若 [dec, vit_proj] 则 decode 看不到 vit。
        combined = torch.cat([vit_proj, dec], dim=1)
        L = combined.size(1)
        qw_kwargs = {"inputs_embeds": combined, "return_dict": True}
        if getattr(c, "qwen2_use_global_attention", False):
            # 全局 attention：前后都可见。HF 中 4D mask 常为 0=attend、负无穷=mask。
            # SDPA 要求 attn_mask dtype 与 query 一致。query 来自 qwen2_projector 内部，
            # 其 dtype 与 projector 权重的 dtype 一致。训练时 projector 多为 bf16，推理时
            # 可能为 float32（取决于 checkpoint 保存/加载方式）。用 projector 权重的 dtype。
            qw_param = next(self.qwen2_projector.parameters(), None)
            qw_dtype = qw_param.dtype if qw_param is not None else combined.dtype
            qw_attn = torch.zeros(
                B,
                1,
                L,
                L,
                device=combined.device,
                dtype=qw_dtype,
            )
            qw_kwargs["attention_mask"] = qw_attn
        # 否则不传 attention_mask，使用 Qwen2 默认因果 mask
        qw_out = self.qwen2_projector(**qw_kwargs)
        hidden = qw_out.last_hidden_state[:, -num_decode:]
        # dtype 对齐：部分算子可能把 hidden 升为 fp32，导致线性层 bf16 权重报 dtype mismatch
        try:
            tgt_dtype = self.projector_to_llm.weight.dtype
            if hidden.dtype != tgt_dtype:
                hidden = hidden.to(dtype=tgt_dtype)
        except Exception:
            pass
        llm_embeds = self.projector_to_llm(hidden)
        return llm_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values_3d: Optional[torch.Tensor] = None,
        volume_grid_dhw: Optional[torch.LongTensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        # Swift/processor naming (volume-only)
        pixel_values_volumes: Optional[torch.Tensor] = None,
        volume_grid_thw: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        """
        Full forward: visual branch -> 1024 visual embeds; then [visual_embeds, text_embeds] -> LLaMA.
        Returns: LLaMA model outputs (last_hidden_state, past_key_values, ...).

        Either supply (pixel_values_3d, volume_grid_dhw) or (vision_features, volume_grid_dhw) on first call.
        On decode steps (past_key_values set), only input_ids/inputs_embeds are used.
        """
        # Accept aliases from processor/template
        if pixel_values_3d is None and pixel_values_volumes is not None:
            pixel_values_3d = pixel_values_volumes
        if volume_grid_dhw is None and volume_grid_thw is not None:
            # volume_grid_thw is [T,H,W] in vision grid; we reuse it as dhw for downsampling
            volume_grid_dhw = volume_grid_thw

        has_visual = (
            (pixel_values_3d is not None and volume_grid_dhw is not None)
            or (vision_features is not None and volume_grid_dhw is not None)
        )
        is_decode = past_key_values is not None
        if is_decode and hasattr(past_key_values, "get_seq_length"):
            is_decode = past_key_values.get_seq_length() > 0
        elif is_decode and isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
            is_decode = past_key_values[0] is not None

        _infer_debug = __import__("os").environ.get("VITLAMMA_INFER_DEBUG") == "1"
        if _infer_debug:
            print(f"[VitLamMA forward] has_visual={has_visual}, is_decode={is_decode}, "
                  f"pv3d={pixel_values_3d is not None}, grid_dhw={volume_grid_dhw is not None}")
        if is_decode or not has_visual:
            self._last_volume_mask_for_bias = None
            if input_ids is None or input_ids.numel() == 0:
                raise ValueError("input_ids is required for text-only/decode steps.")
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        else:
            if input_ids is None or input_ids.numel() == 0:
                raise ValueError("For volume multimodal training/inference, input_ids with <|image_3d_pad|> placeholders is required.")
            llm_visual = self.forward_visual_to_llm_embeds(
                vision_features=vision_features,
                volume_grid_dhw=volume_grid_dhw,
                pixel_values_3d=pixel_values_3d,
            )
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            volume_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, volume_features=None
            )
            n_placeholders_per_row = volume_mask.any(dim=-1).sum(dim=1)
            num_placeholders = n_placeholders_per_row.sum().item()
            B, Nv, D = llm_visual.size(0), llm_visual.size(1), llm_visual.size(-1)
            # llm_visual is (num_volumes, Nv, D); total visual tokens = num_volumes * Nv.
            # Placeholders may span multiple volumes per sequence, so take first num_placeholders from flatten.
            volume_flat = llm_visual.reshape(-1, D)[:num_placeholders].to(inputs_embeds.dtype)
            if volume_flat.numel() != num_placeholders * D:
                raise ValueError(
                    f"Volume features and placeholder count mismatch: "
                    f"placeholders={num_placeholders}, volume_flat has {volume_flat.numel() // D} tokens."
                )
            if __import__("os").environ.get("VITLAMMA_INFER_DEBUG") == "1":
                print(f"[VitLamMA forward] prefill visual: llm_visual.shape={llm_visual.shape}, "
                      f"num_placeholders={num_placeholders}, n_placeholders_per_row={n_placeholders_per_row.tolist()}")
            inputs_embeds = inputs_embeds.masked_scatter(volume_mask, volume_flat)
            self._last_volume_mask_for_bias = volume_mask
            if __import__("os").environ.get("VITLAMMA_INFER_DEBUG") == "1":
                emb_at_place = inputs_embeds.masked_select(volume_mask).view(num_placeholders, -1)
                rest_flat = inputs_embeds.masked_select(~volume_mask)
                rest_str = ""
                if rest_flat.numel() > 0:
                    emb_rest = rest_flat.view(-1, D)
                    rest_str = f", emb_rest mean={emb_rest.float().mean().item():.6f} std={emb_rest.float().std().item():.6f}"
                print(f"[VitLamMA forward] after inject: emb_at_placeholder mean={emb_at_place.float().mean().item():.6f} std={emb_at_place.float().std().item():.6f}{rest_str}")

        llm_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("return_dict", "output_hidden_states", "output_attentions", "attention_mask")
        }
        # On decode steps, pass attention_mask=None so the inner Llama builds causal mask from
        # cache_position (avoids length mismatch with DynamicCache / chunked logic).
        bias_val = getattr(self.config, "attention_bias_toward_visual", 0.0)
        attn_mask_for_llm = None if is_decode else attention_mask

        # 检测 LLM 实际使用的 attn 实现
        _llm_inner = getattr(self.llm, "model", self.llm)
        _llm_attn_impl = getattr(getattr(_llm_inner, "config", None), "_attn_implementation", None) or "sdpa"

        if not is_decode and bias_val == 0.0 and inputs_embeds is not None:
            if _llm_attn_impl == "flash_attention_2":
                # Flash2 内部处理因果掩码（is_causal=True），只需 2D mask 标注 padding 位置
                B, L = inputs_embeds.shape[0], inputs_embeds.shape[1]
                if attention_mask is None:
                    attn_mask_for_llm = torch.ones(B, L, device=inputs_embeds.device, dtype=torch.long)
                else:
                    attn_mask_for_llm = attention_mask  # 应为 2D (B, L)
            else:
                # SDPA / eager: 需要显式 4D 因果 mask
                B, L = inputs_embeds.shape[0], inputs_embeds.shape[1]
                device, dtype = inputs_embeds.device, inputs_embeds.dtype
                causal_4d = torch.triu(
                    torch.full((L, L), float("-inf"), dtype=torch.float32, device=device),
                    diagonal=1,
                ).unsqueeze(0).unsqueeze(0).expand(B, 1, L, L).to(dtype)
                if attention_mask is not None and (attention_mask == 0).any().item():
                    pad_key = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
                    causal_4d = causal_4d.masked_fill(pad_key, float("-inf"))
                attn_mask_for_llm = causal_4d
        # Attention bias toward visual tokens: add positive value to scores for keys that are image tokens.
        if not is_decode and has_visual and bias_val != 0.0 and getattr(self, "_last_volume_mask_for_bias", None) is not None:
            vol_mask = self._last_volume_mask_for_bias  # (B, L, D) -> (B, L)
            is_visual_pos = vol_mask.any(dim=-1)
            B, L = inputs_embeds.shape[0], inputs_embeds.shape[1]
            dtype = inputs_embeds.dtype
            device = inputs_embeds.device
            causal = torch.triu(
                torch.full((L, L), float("-inf"), dtype=torch.float32, device=device),
                diagonal=1,
            )
            mask_4d = causal.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L).clone()
            if attention_mask is not None:
                pad_key = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
                mask_4d = mask_4d.masked_fill(pad_key, float("-inf"))
            bias_add = is_visual_pos.unsqueeze(1).unsqueeze(2) * bias_val
            mask_4d = (mask_4d + bias_add).to(dtype)
            attn_mask_for_llm = mask_4d
        # 推理时关闭 output_hidden_states/output_attentions 以降低显存。LlamaForCausalLM 在
        # output_hidden_states=False 时不返回 last_hidden_state，故改调 self.llm.model（LlamaModel），
        # 其 BaseModelOutputWithPast 始终含 last_hidden_state。
        output_hidden_states = kwargs.get("output_hidden_states", False)
        output_attentions = kwargs.get("output_attentions", False)
        inner = getattr(self.llm, "model", self.llm)
        outputs = inner(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask_for_llm,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            **llm_kwargs,
        )
        return outputs


class VitLamMAForConditionalGeneration(PreTrainedModel, GenerationMixin):
    """
    VitLamMA for conditional generation (with lm_head). Wraps VitLamMAModel and uses LLaMA's lm_head.
    Explicitly inherits GenerationMixin for generate() support (PreTrainedModel no longer includes it from v4.50).
    """

    config_class = VitLamMAConfig
    _supports_flash_attn_2 = True
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(
        self,
        config: VitLamMAConfig,
        vision_encoder: Optional[nn.Module] = None,
        llm: Optional[PreTrainedModel] = None,
        qwen2_projector: Optional[nn.Module] = None,
    ):
        super().__init__(config)
        self.model = VitLamMAModel(
            config,
            vision_encoder=vision_encoder,
            llm=llm,
            qwen2_projector=qwen2_projector,
        )
        # Do not set self.llm = self.model.llm: it would register the same module twice and cause
        # shared-tensor errors in save_pretrained (duplicate keys llm.* vs model.llm.*).

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        *model_args,
        config: Optional[Union[PretrainedConfig, str, "os.PathLike"]] = None,
        ignore_mismatched_sizes: bool = True,
        **kwargs,
    ):
        """
        加载时默认 ignore_mismatched_sizes=True：与当前模型 shape 不一致的权重不加载，保留新结构下的初始化。
        可传 ignore_mismatched_sizes=False 强制严格匹配。
        """
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            **kwargs,
        )

    @staticmethod
    def _remap_patch_embedding_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remap MONAI PatchEmbeddingBlock keys to VariablePatchEmbedding3D keys so that
        checkpoint from models using patch_embeddings.0/1 (e.g. perceptron) can load
        into our patch_embedding.proj.1 (Linear only).
        """
        out = {}
        for k, v in state_dict.items():
            if "patch_embedding.patch_embeddings.1." in k:
                k = k.replace("patch_embedding.patch_embeddings.1.", "patch_embedding.proj.1.")
            out[k] = v
        return out

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = False):
        state_dict = self._remap_patch_embedding_keys(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def get_input_embeddings(self):
        return self.model.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.llm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.model.llm.set_output_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values_3d: Optional[torch.Tensor] = None,
        volume_grid_dhw: Optional[torch.LongTensor] = None,
        pixel_values_volumes: Optional[torch.Tensor] = None,
        volume_grid_thw: Optional[torch.LongTensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if pixel_values_3d is None and pixel_values_volumes is not None:
            pixel_values_3d = pixel_values_volumes
        if volume_grid_dhw is None and volume_grid_thw is not None:
            volume_grid_dhw = volume_grid_thw
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values_3d=pixel_values_3d,
            volume_grid_dhw=volume_grid_dhw,
            vision_features=vision_features,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        # CausalLMOutputWithPast has logits but not last_hidden_state; BaseModelOutputWithPast has last_hidden_state.
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None and getattr(outputs, "hidden_states", None) is not None:
            hidden_states = outputs.hidden_states[-1]
        if hidden_states is None:
            raise AttributeError(
                "Inner LLM returned no last_hidden_state or hidden_states. "
                "VitLamMAModel.forward must call llm() which returns last_hidden_state."
            )
        logits = self.model.llm.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)
            n_tok = getattr(self.config, "loss_weight_after_visual_n_tokens", 0)
            factor = getattr(self.config, "loss_weight_after_visual_factor", 2.0)
            if n_tok > 0 and factor != 1.0:
                per_token_loss = nn.functional.cross_entropy(
                    shift_logits, shift_labels, reduction="none", ignore_index=-100
                )
                B, L = labels.shape[0], labels.shape[1]
                weights = torch.ones_like(shift_labels, dtype=per_token_loss.dtype, device=per_token_loss.device)
                for b in range(B):
                    valid = (labels[b, 1:] != -100).nonzero(as_tuple=True)[0]
                    if valid.numel() == 0:
                        continue
                    first_pred_j = valid[0].item()
                    start_j = first_pred_j
                    end_j = min(L - 1, start_j + n_tok)
                    for j in range(start_j, end_j):
                        if shift_labels[b * (L - 1) + j] != -100:
                            weights[b * (L - 1) + j] = factor
                valid_mask = (shift_labels != -100).float()
                w_sum = (weights * valid_mask).sum()
                if w_sum > 0:
                    loss = (per_token_loss * weights * valid_mask).sum() / w_sum
                else:
                    loss = per_token_loss.sum()
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits, shift_labels)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        pixel_values_3d: Optional[torch.Tensor] = None,
        volume_grid_dhw: Optional[torch.LongTensor] = None,
        pixel_values_volumes: Optional[torch.Tensor] = None,
        volume_grid_thw: Optional[torch.LongTensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # HuggingFace 新版 generate 可能会在首步就传入空的 DynamicCache（past_key_values 非 None 但 seq_len=0）。
        # 这里必须用 cache 的 seq_len 来判断 decode/prefill，否则会把首步误当作 decode，从而丢掉视觉输入，导致乱码。
        is_decode = False
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                try:
                    is_decode = past_key_values.get_seq_length() > 0
                except Exception:
                    is_decode = True
            elif isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
                is_decode = past_key_values[0] is not None
            else:
                # unknown cache type: be conservative
                is_decode = True

        if pixel_values_3d is None and pixel_values_volumes is not None:
            pixel_values_3d = pixel_values_volumes
        if volume_grid_dhw is None and volume_grid_thw is not None:
            volume_grid_dhw = volume_grid_thw
        if __import__("os").environ.get("VITLAMMA_INFER_DEBUG") == "1":
            print(f"[prepare_inputs_for_generation] is_decode={is_decode}, past_key_values is not None={past_key_values is not None}, "
                  f"pixel_values_3d is not None={pixel_values_3d is not None}, volume_grid_dhw is not None={volume_grid_dhw is not None}")
        if is_decode:
            # 与 VitQwen 一致：decode 时只传新 token，避免 key 长度 = cache + 当前序列 导致 SDPA 维度错误
            input_ids = input_ids[:, -1:] if input_ids.size(-1) > 1 else input_ids
            # 关键：decode 时仍须传 pixel_values/volume_grid 给下一 forward，否则 has_visual=False，
            # 可能导致状态异常；同时保证 model_kwargs 在生成循环中保留
            ret = dict(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=False,
                **kwargs,
            )
            if pixel_values_3d is not None:
                ret["pixel_values_3d"] = pixel_values_3d
                ret["pixel_values_volumes"] = pixel_values_volumes
            if volume_grid_dhw is not None:
                ret["volume_grid_dhw"] = volume_grid_dhw
                ret["volume_grid_thw"] = volume_grid_thw
            return ret
        need_visual = (pixel_values_3d is not None and volume_grid_dhw is not None) or vision_features is not None
        if need_visual:
            kwargs["pixel_values_3d"] = pixel_values_3d
            kwargs["volume_grid_dhw"] = volume_grid_dhw
            kwargs["vision_features"] = vision_features
        if __import__("os").environ.get("VITLAMMA_INFER_DEBUG") == "1" and need_visual:
            print(f"[prepare_inputs_for_generation] passing visual to next forward (prefill step)")
        return dict(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
            **kwargs,
        )
