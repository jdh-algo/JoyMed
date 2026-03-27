

# coding=utf-8
# Copyright 2025 VitQwen Team. All rights reserved.
"""
VitQwen Model: ViT (3D) + Qwen2-0.5B (or pixel_shuffle) projector + Qwen3-VL LLM.

Fusion pipeline:
1. ViT encodes 3D images -> (B, N, vision_hidden_size)
2. 3D conv downsampling D/H/W -> 1/2 -> (B, N', vision_hidden_size)
3. Projector: Qwen2 path [vit_proj, decode_tokens]->Qwen2->linear; pixel_shuffle path: 2x2x2 conv->linear
4. Visual embeds replace placeholders in input_ids -> Qwen3-VL language_model (3D RoPE + deepstack)
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers import PreTrainedModel, PretrainedConfig

from .configuration_vit_qwen import VitQwenConfig
from .downsample_3d import Conv3dDownsample


def _unpatchify_volume(
    patches_flat: torch.Tensor,
    volume_grid_thw: torch.LongTensor,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
) -> torch.Tensor:
    """Reverse of volume_processing patchify: (B, N, patch_dim) -> (B, C, D, H, W)."""
    B, N, patch_dim = patches_flat.shape
    if volume_grid_thw.dim() == 1:
        grid_t, grid_h, grid_w = volume_grid_thw.tolist()
    else:
        grid_t, grid_h, grid_w = volume_grid_thw[0].tolist()
    grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
    channel = patch_dim // (temporal_patch_size * patch_size * patch_size)
    if channel * temporal_patch_size * patch_size * patch_size != patch_dim:
        raise ValueError(
            f"patch_dim {patch_dim} is not divisible by temporal_patch_size*patch_size^2; cannot infer channel."
        )
    if N != grid_t * grid_h * grid_w:
        raise ValueError(
            f"Number of tokens N={N} does not match grid_t*grid_h*grid_w = {grid_t * grid_h * grid_w}."
        )
    gt_m = grid_t // merge_size
    gh_m = grid_h // merge_size
    gw_m = grid_w // merge_size
    x = patches_flat.view(
        B, gt_m, gh_m, gw_m,
        merge_size, merge_size, merge_size,
        channel, temporal_patch_size, patch_size, patch_size,
    )
    x = x.permute(0, 1, 4, 8, 7, 2, 5, 9, 3, 6, 10)
    x = x.permute(0, 4, 1, 2, 3, 5, 6, 7, 8, 9, 10)
    x = x.reshape(
        B, channel,
        grid_t * temporal_patch_size,
        grid_h * patch_size,
        grid_w * patch_size,
    )
    return x


def _config_from_dict(d: dict) -> PretrainedConfig:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    model_type = d.get("model_type")
    if model_type and model_type in CONFIG_MAPPING:
        return CONFIG_MAPPING[model_type].from_dict(d)
    return PretrainedConfig.from_dict(d)


def _is_full_config(c: Any) -> bool:
    if not isinstance(c, dict) or not c:
        return False
    return "model_type" in c or "hidden_size" in c


class LaMedViTWrapper(nn.Module):
    """Wraps LaMed ViT so that forward(pixel_values_3d, volume_grid_dhw=...) returns (B, N, H) patch features."""

    def __init__(self, vit: nn.Module, *, expected_in_channels: int):
        super().__init__()
        self.vit = vit
        self.expected_in_channels = int(expected_in_channels)

    def forward(self, pixel_values_3d: torch.Tensor, volume_grid_dhw=None):
        if pixel_values_3d.dim() == 5:
            if pixel_values_3d.size(1) != self.expected_in_channels and pixel_values_3d.size(2) == self.expected_in_channels:
                pixel_values_3d = pixel_values_3d.permute(0, 2, 1, 3, 4).contiguous()
            c = pixel_values_3d.size(1)
            if c != self.expected_in_channels:
                if self.expected_in_channels == 1 and c == 3:
                    pixel_values_3d = pixel_values_3d.mean(dim=1, keepdim=True)
                elif self.expected_in_channels == 3 and c == 1:
                    pixel_values_3d = pixel_values_3d.repeat(1, 3, 1, 1, 1)
                else:
                    raise ValueError(
                        f"LaMedViTWrapper expected in_channels={self.expected_in_channels}, but got C={c}."
                    )
        out, _ = self.vit(pixel_values_3d)
        if getattr(self.vit, "classification", False):
            out = out[:, 1:, :]
        return out


def _build_lamed_vit_from_config(vc: Dict[str, Any]) -> nn.Module:
    from pathlib import Path
    lamed_path = vc.get("lamed_repo_path", "")
    if lamed_path:
        lp = Path(lamed_path).resolve()
        if str(lp) not in __import__("sys").path:
            __import__("sys").path.insert(0, str(lp))
    try:
        from .LaMed.src.model.multimodal_encoder.vit import ViT
    except ImportError:
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


def _load_qwen2_projector(config: VitQwenConfig):
    """Build Qwen2 projector from config (for loading from checkpoint)."""
    pc = getattr(config, "projector_config", None)
    if pc is not None and _is_full_config(pc):
        cfg = _config_from_dict(pc) if isinstance(pc, dict) else pc
        # 从 config 构建：sdpa + bf16（inputs_embeds 下 Flash2 会走 varlen 需 cu_seqlens_q，易报错）
        model = AutoModel.from_config(cfg, attn_implementation="sdpa")
        return model.to(torch.bfloat16)
    path = getattr(config, "projector_model_name_or_path", "Qwen/Qwen2-0.5B")
    return AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )


def _load_qwen3_vl_llm(config: VitQwenConfig):
    """Load or build Qwen3-VL language model. When loading from checkpoint, build from text_config."""
    path = getattr(config, "llm_model_name_or_path", None)
    if path:
        qwen3 = __import__("transformers", fromlist=["Qwen3VLForConditionalGeneration"]).Qwen3VLForConditionalGeneration
        # bf16 + flash_attention_2
        model = qwen3.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        return model.model.language_model
    tc = config.get_text_config(decoder=True)
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    # 从 config 构建时也强制使用 flash_attention_2（仅 LLM 部分）
    if hasattr(tc, "attn_implementation"):
        tc.attn_implementation = "flash_attention_2"
    if hasattr(tc, "_attn_implementation"):
        tc._attn_implementation = "flash_attention_2"
    return Qwen3VLTextModel._from_config(tc).to(torch.bfloat16)


class VitQwenModel(PreTrainedModel):
    """VitQwen: ViT (3D) -> 3D downsample -> projector -> Qwen3-VL language_model (3D RoPE + deepstack)."""

    config_class = VitQwenConfig
    _supports_flash_attn_2 = True
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(
        self,
        config: VitQwenConfig,
        vision_encoder: Optional[nn.Module] = None,
        language_model: Optional[nn.Module] = None,
        qwen2_projector: Optional[nn.Module] = None,
    ):
        super().__init__(config)
        c = config

        self.vision_encoder = vision_encoder
        if vision_encoder is None and getattr(c, "vision_config", None):
            vc = c.vision_config
            if isinstance(vc, dict) and vc.get("model_type") == "lamed_vit":
                self.vision_encoder = _build_lamed_vit_from_config(vc).to(torch.bfloat16)
            elif _is_full_config(vc):
                cfg = _config_from_dict(vc) if isinstance(vc, dict) else vc
                self.vision_encoder = AutoModel.from_config(cfg).to(torch.bfloat16)

        self.downsample_3d = Conv3dDownsample(c.vision_hidden_size, factor=c.downsampling_factor)
        self.simple_projector = nn.Linear(c.vision_hidden_size, c.llm_hidden_size, bias=True)

        use_pixel_shuffle = (getattr(c, "projector_type", "qwen2") or "qwen2").lower() == "pixel_shuffle"
        self.qwen2_projector = None
        self.projector_to_llm = None
        self.pixel_shuffle_to_llm = None

        if use_pixel_shuffle:
            self.pixel_shuffle_conv = nn.Conv3d(
                c.vision_hidden_size, c.vision_hidden_size * 8,
                kernel_size=2, stride=2, padding=0,
            )
            self.pixel_shuffle_to_llm = nn.Linear(c.vision_hidden_size * 8, c.llm_hidden_size, bias=True)
        else:
            self.vit_to_projector = nn.Linear(c.vision_hidden_size, c.projector_hidden_size, bias=True)
            self.decode_tokens = nn.Parameter(torch.zeros(1, c.num_decode_tokens, c.projector_hidden_size))
            nn.init.normal_(self.decode_tokens, std=0.02)
            self.decode_interp_linear = nn.Linear(c.projector_hidden_size, c.projector_hidden_size, bias=True)
            self.qwen2_projector = qwen2_projector if qwen2_projector is not None else _load_qwen2_projector(config)
            self.projector_to_llm = nn.Linear(c.projector_hidden_size, c.llm_hidden_size, bias=True)

        self.language_model = language_model if language_model is not None else _load_qwen3_vl_llm(config)

        self.rope_deltas = None
        self._last_volume_grid_3d = None
        self._cached_pixel_values_3d = None
        self._cached_volume_grid_dhw = None
        self.post_init()

    def get_placeholder_mask(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: torch.Tensor,
        volume_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mm_token_id = getattr(self.config, "image_3d_token_id", 151670)
        if input_ids is not None:
            special_volume_mask = input_ids == mm_token_id
        else:
            placeholder_embed = self.language_model.get_input_embeddings()(
                torch.tensor(mm_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_volume_mask = (inputs_embeds == placeholder_embed).all(dim=-1)
        n_volume_tokens = special_volume_mask.sum().item()
        special_volume_mask = special_volume_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if volume_features is not None and inputs_embeds[special_volume_mask].numel() != volume_features.numel():
            raise ValueError(
                f"Volume features and placeholder count mismatch: "
                f"placeholders: {n_volume_tokens}, volume_features: {volume_features.shape[0]}"
            )
        return special_volume_mask

    def get_vision_features(
        self,
        pixel_values_3d: torch.Tensor,
        volume_grid_dhw: torch.LongTensor,
    ) -> torch.Tensor:
        if self.vision_encoder is None:
            raise ValueError("vision_encoder is not set.")
        try:
            ve_param = next(self.vision_encoder.parameters(), None)
            if ve_param is not None and pixel_values_3d.dtype != ve_param.dtype:
                pixel_values_3d = pixel_values_3d.to(dtype=ve_param.dtype)
        except Exception:
            pass
        c = self.config
        if pixel_values_3d.dim() == 3:
            pixel_values_3d = _unpatchify_volume(
                pixel_values_3d, volume_grid_dhw,
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
        return self.vision_encoder(pixel_values_3d, volume_grid_dhw=volume_grid_dhw)

    def _num_decode_from_grid(self, grid_after_downsample: Tuple[int, int, int]) -> int:
        d2, h2, w2 = grid_after_downsample
        n_prime = d2 * h2 * w2
        max_grid = getattr(self.config, "max_grid_after_downsample", (16, 16, 16))
        n_max = max_grid[0] * max_grid[1] * max_grid[2]
        max_decode = getattr(self.config, "num_decode_tokens", 1024)
        min_decode = getattr(self.config, "min_decode_tokens", 64)
        num_decode = round(max_decode * n_prime / max(n_max, 1))
        return max(min_decode, min(max_decode, num_decode))

    def _forward_single_volume_to_llm(
        self, vit_down_i: torch.Tensor, grid_tuple_i: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Single volume (1, N', C) through qwen2 path -> (1, num_decode, D). 用于 variable shape 分支。"""
        c = self.config
        d2, h2, w2 = grid_tuple_i
        B_i = 1
        if self.pixel_shuffle_to_llm is not None:
            c_in = vit_down_i.size(-1)
            x = vit_down_i.view(B_i, d2, h2, w2, c_in).permute(0, 4, 1, 2, 3)
            x = self.pixel_shuffle_conv(x)
            _, c_dim, d4, h4, w4 = x.shape
            x = x.permute(0, 2, 3, 4, 1).reshape(B_i, -1, c_dim)
            return self.pixel_shuffle_to_llm(x)
        num_decode = self._num_decode_from_grid(grid_tuple_i)
        vit_proj = self.vit_to_projector(vit_down_i)
        pool = self.decode_tokens.to(vit_proj.dtype).to(vit_proj.device).expand(B_i, -1, -1)
        if num_decode >= c.num_decode_tokens:
            dec = pool[:, : c.num_decode_tokens]
            num_decode = c.num_decode_tokens
        else:
            pool_t = pool.transpose(1, 2)
            dec_t = torch.nn.functional.interpolate(pool_t, size=num_decode, mode="linear", align_corners=False)
            dec = dec_t.transpose(1, 2)
        dec = self.decode_interp_linear(dec)
        combined = torch.cat([vit_proj, dec], dim=1)
        L = combined.size(1)
        qw_kwargs = {"inputs_embeds": combined, "return_dict": True}
        if getattr(c, "qwen2_use_global_attention", False):
            qw_param = next(self.qwen2_projector.parameters(), None)
            qw_dtype = qw_param.dtype if qw_param is not None else combined.dtype
            qw_kwargs["attention_mask"] = torch.zeros(B_i, 1, L, L, device=combined.device, dtype=qw_dtype)
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
        c = self.config
        if vision_features is None:
            if pixel_values_3d is None or volume_grid_dhw is None:
                raise ValueError("Need (vision_features, volume_grid_dhw) or (pixel_values_3d, volume_grid_dhw).")
            vision_features = self.get_vision_features(pixel_values_3d, volume_grid_dhw)
        if volume_grid_dhw is None:
            raise ValueError("volume_grid_dhw is required.")

        # 不同 volume 不同 shape 时逐条处理（从 pixel 开始），保持 RoPE 与 placeholder 顺序一致
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
                out_list = []
                self._volume_grids_3d = []
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
                    self._volume_grids_3d.append(grid_tuple_i)
                    out_i = self._forward_single_volume_to_llm(vd_i, grid_tuple_i)
                    out_list.append(out_i)
                self._last_volume_grid_3d = self._volume_grids_3d[0]
                return torch.cat(out_list, dim=1)

        B = vision_features.size(0)
        # 同 shape 或单图：原有 batch 路径
        vit_down, new_grid = self.downsample_3d(vision_features, volume_grid_dhw)
        B = vit_down.size(0)
        if new_grid.dim() == 2:
            self._volume_grids_3d = [
                (int(new_grid[i, 0].item()), int(new_grid[i, 1].item()), int(new_grid[i, 2].item()))
                for i in range(B)
            ]
            grid_tuple = self._volume_grids_3d[0]
        else:
            grid_tuple = (int(new_grid[0].item()), int(new_grid[1].item()), int(new_grid[2].item()))
            self._volume_grids_3d = [grid_tuple] if B > 0 else []
        self._last_volume_grid_3d = grid_tuple

        if self.pixel_shuffle_to_llm is not None:
            d2, h2, w2 = grid_tuple
            c_in = vit_down.size(-1)
            x = vit_down.view(B, d2, h2, w2, c_in).permute(0, 4, 1, 2, 3)
            x = self.pixel_shuffle_conv(x)
            _, c_dim, d4, h4, w4 = x.shape
            x = x.permute(0, 2, 3, 4, 1).reshape(B, -1, c_dim)
            return self.pixel_shuffle_to_llm(x)

        num_decode = self._num_decode_from_grid(grid_tuple)
        vit_proj = self.vit_to_projector(vit_down)
        pool = self.decode_tokens.to(vit_proj.dtype).to(vit_proj.device).expand(B, -1, -1)
        if num_decode >= c.num_decode_tokens:
            dec = pool[:, : c.num_decode_tokens]
            num_decode = c.num_decode_tokens
        else:
            pool_t = pool.transpose(1, 2)
            dec_t = torch.nn.functional.interpolate(pool_t, size=num_decode, mode="linear", align_corners=False)
            dec = dec_t.transpose(1, 2)
        dec = self.decode_interp_linear(dec)
        combined = torch.cat([vit_proj, dec], dim=1)
        L = combined.size(1)
        qw_kwargs = {"inputs_embeds": combined, "return_dict": True}
        if getattr(c, "qwen2_use_global_attention", False):
            qw_param = next(self.qwen2_projector.parameters(), None)
            qw_dtype = qw_param.dtype if qw_param is not None else combined.dtype
            qw_kwargs["attention_mask"] = torch.zeros(B, 1, L, L, device=combined.device, dtype=qw_dtype)
        qw_out = self.qwen2_projector(**qw_kwargs)
        hidden = qw_out.last_hidden_state[:, -num_decode:]
        try:
            tgt_dtype = self.projector_to_llm.weight.dtype
            if hidden.dtype != tgt_dtype:
                hidden = hidden.to(dtype=tgt_dtype)
        except Exception:
            pass
        return self.projector_to_llm(hidden)

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute position_ids (3, B, L) and rope_deltas (B, 1) for Qwen3-VL LLM 3D RoPE.
        Uses per-volume grid from _volume_grids_3d when available (multi-image / variable shape).
        Treats image_sep_token_id as a single text token (1D position).
        """
        mm_token_id = getattr(self.config, "image_3d_token_id", 151670)
        sep_token_id = getattr(self.config, "image_sep_token_id", None)
        B, L = input_ids.shape
        device = input_ids.device
        dtype = input_ids.dtype
        volume_grids = getattr(self, "_volume_grids_3d", None) or []
        position_ids = torch.zeros(3, B, L, device=device, dtype=dtype)
        if attention_mask is None:
            attention_mask = torch.ones(B, L, device=device, dtype=torch.long)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0).expand(B, -1)
        valid = (attention_mask != 0)
        rope_deltas_list = []
        for b in range(B):
            seq = input_ids[b]
            v = valid[b]
            pos_3_list = []
            volume_idx = 0
            i = 0
            while i < L:
                if not v[i].item():
                    i += 1
                    continue
                if seq[i].item() == mm_token_id:
                    run_end = i
                    while run_end < L and v[run_end].item() and seq[run_end].item() == mm_token_id:
                        run_end += 1
                    nv = run_end - i
                    grid_3d = (
                        volume_grids[volume_idx]
                        if volume_idx < len(volume_grids) and len(volume_grids[volume_idx]) == 3
                        else getattr(self, "_last_volume_grid_3d", None)
                    )
                    volume_idx += 1
                    if grid_3d is not None and len(grid_3d) == 3:
                        Dg, Hg, Wg = grid_3d[0], grid_3d[1], grid_3d[2]
                        if Dg * Hg * Wg == nv:
                            t_ids = torch.zeros(nv, device=device, dtype=dtype)
                            h_ids = torch.zeros(nv, device=device, dtype=dtype)
                            w_ids = torch.zeros(nv, device=device, dtype=dtype)
                            for j in range(nv):
                                w_ids[j] = j % Wg
                                j_ = j // Wg
                                h_ids[j] = j_ % Hg
                                t_ids[j] = j_ // Hg
                            pos_3_list.append(torch.stack([t_ids, h_ids, w_ids]))
                        else:
                            t_ids = torch.zeros(nv, device=device, dtype=dtype)
                            h_ids = torch.zeros(nv, device=device, dtype=dtype)
                            w_ids = torch.arange(nv, device=device, dtype=dtype)
                            pos_3_list.append(torch.stack([t_ids, h_ids, w_ids]))
                    else:
                        t_ids = torch.zeros(nv, device=device, dtype=dtype)
                        h_ids = torch.zeros(nv, device=device, dtype=dtype)
                        w_ids = torch.arange(nv, device=device, dtype=dtype)
                        pos_3_list.append(torch.stack([t_ids, h_ids, w_ids]))
                    i = run_end
                elif sep_token_id is not None and seq[i].item() == sep_token_id:
                    st_idx = sum(p.shape[1] for p in pos_3_list)
                    text_pos = torch.tensor([st_idx], device=device, dtype=dtype)
                    pos_3_list.append(text_pos.unsqueeze(0).expand(3, -1))
                    i += 1
                else:
                    text_end = i
                    while text_end < L and v[text_end].item():
                        tok = seq[text_end].item()
                        if tok == mm_token_id or (sep_token_id is not None and tok == sep_token_id):
                            break
                        text_end += 1
                    text_len = text_end - i
                    st_idx = sum(p.shape[1] for p in pos_3_list)
                    text_pos = torch.arange(text_len, device=device, dtype=dtype) + st_idx
                    pos_3_list.append(text_pos.unsqueeze(0).expand(3, -1))
                    i = text_end
            if not pos_3_list:
                pos_3_list.append(torch.zeros(3, 0, device=device, dtype=dtype))
            cat_3 = torch.cat(pos_3_list, dim=1)
            num_valid = v.sum().item()
            if num_valid != cat_3.shape[1]:
                raise ValueError(f"get_rope_index: valid count {num_valid} != built positions {cat_3.shape[1]}")
            position_ids[:, b, v] = cat_3
            max_pos = cat_3.max().item() + 1
            rope_deltas_list.append(max_pos - L)
        rope_deltas = torch.tensor(rope_deltas_list, device=device, dtype=dtype).unsqueeze(1)
        return position_ids, rope_deltas

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values_3d: Optional[torch.Tensor] = None,
        volume_grid_dhw: Optional[torch.LongTensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        pixel_values_volumes: Optional[torch.Tensor] = None,
        volume_grid_thw: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if pixel_values_3d is None and pixel_values_volumes is not None:
            pixel_values_3d = pixel_values_volumes
        if volume_grid_dhw is None and volume_grid_thw is not None:
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

        visual_pos_masks = None
        deepstack_visual_embeds = None
        num_deepstack = getattr(self.config, "num_deepstack_layers", 3)

        if is_decode or not has_visual:
            if input_ids is None or input_ids.numel() == 0:
                raise ValueError("input_ids is required for text-only/decode steps.")
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        else:
            self._cached_pixel_values_3d = pixel_values_3d
            self._cached_volume_grid_dhw = volume_grid_dhw
            if input_ids is None or input_ids.numel() == 0:
                raise ValueError("For volume multimodal, input_ids with image_3d placeholders is required.")
            llm_visual = self.forward_visual_to_llm_embeds(
                vision_features=vision_features,
                volume_grid_dhw=volume_grid_dhw,
                pixel_values_3d=pixel_values_3d,
            )
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            volume_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, volume_features=None
            )
            n_placeholders_per_row = volume_mask.any(dim=-1).sum(dim=1)
            num_placeholders = n_placeholders_per_row.sum().item()
            B, Nv, D = llm_visual.size(0), llm_visual.size(1), llm_visual.size(-1)
            volume_flat = llm_visual.reshape(-1, D)[:num_placeholders].to(inputs_embeds.dtype)
            if volume_flat.numel() != num_placeholders * D:
                raise ValueError(
                    f"Volume features and placeholder count mismatch: "
                    f"placeholders={num_placeholders}, volume_flat has {volume_flat.numel() // D} tokens."
                )
            inputs_embeds = inputs_embeds.masked_scatter(volume_mask, volume_flat)
            visual_pos_masks = volume_mask.any(dim=-1)
            deepstack_visual_embeds = [volume_flat for _ in range(num_deepstack)]

        seq_len = inputs_embeds.shape[1]
        if cache_position is None:
            past_seen = 0
            if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
                past_seen = past_key_values.get_seq_length()
            cache_position = torch.arange(
                past_seen, past_seen + seq_len, device=inputs_embeds.device, dtype=torch.long
            )

        if attention_mask is not None and attention_mask.ndim == 2 and attention_mask.shape[-1] != seq_len:
            am_len = attention_mask.shape[-1]
            if am_len < seq_len:
                attention_mask = torch.nn.functional.pad(
                    attention_mask, (seq_len - am_len, 0), value=0
                ).to(device=inputs_embeds.device)
            else:
                attention_mask = attention_mask[..., -seq_len:].contiguous()

        if has_visual and not is_decode:
            attention_mask = None
        if is_decode:
            attention_mask = None  # decode 时传 None，让 LM 从 cache_position 构建 causal mask（与 VitLamMA 一致）

        # Qwen2 projector 路径：投影后为 1D 序列，无真实 3D 网格。传 position_ids=None，
        # 让 Qwen3-VL LM 用 cache_position 生成 1D 位置（与 VitLamMA 类似）。
        # pixel_shuffle 路径：有真实 3D 网格，使用 get_rope_index。
        use_qwen2_projector = self.pixel_shuffle_to_llm is None
        prefill_stage = (
            cache_position is not None and cache_position.numel() > 0 and cache_position[0].item() == 0
        ) or (past_key_values is None or (hasattr(past_key_values, "get_seq_length") and past_key_values.get_seq_length() == 0))
        if position_ids is None and not use_qwen2_projector:
            if has_visual and not is_decode and (prefill_stage or self.rope_deltas is None):
                position_ids, self.rope_deltas = self.get_rope_index(input_ids, attention_mask=attention_mask)
            elif is_decode and self.rope_deltas is not None and cache_position is not None:
                batch_size, seq_length = inputs_embeds.shape[0], inputs_embeds.shape[1]
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if delta.dim() == 0:
                    delta = delta.unsqueeze(0)
                position_ids = torch.arange(seq_length, device=inputs_embeds.device, dtype=torch.long)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                delta = delta.repeat_interleave(batch_size // max(1, delta.shape[0]), dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            elif not has_visual and not is_decode:
                self.rope_deltas = None

        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            return_dict=True,
            output_hidden_states=kwargs.get("output_hidden_states", False),
            output_attentions=kwargs.get("output_attentions", False),
            **{k: v for k, v in kwargs.items() if k not in ("return_dict", "output_hidden_states", "output_attentions")},
        )
        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            attentions=getattr(outputs, "attentions", None),
            hidden_states=getattr(outputs, "hidden_states", None),
        )


class VitQwenForConditionalGeneration(PreTrainedModel, GenerationMixin):
    """VitQwen for conditional generation (with lm_head from Qwen3-VL)."""

    config_class = VitQwenConfig
    _supports_flash_attn_2 = True
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(
        self,
        config: VitQwenConfig,
        vision_encoder: Optional[nn.Module] = None,
        language_model: Optional[nn.Module] = None,
        qwen2_projector: Optional[nn.Module] = None,
        lm_head: Optional[nn.Module] = None,
    ):
        super().__init__(config)
        self.model = VitQwenModel(
            config,
            vision_encoder=vision_encoder,
            language_model=language_model,
            qwen2_projector=qwen2_projector,
        )
        if lm_head is None:
            lm_head = nn.Linear(
                config.llm_hidden_size,
                getattr(config.get_text_config(), "vocab_size", 152064),
                bias=False,
            )
        self.lm_head = lm_head

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        *model_args,
        config: Optional[Union[PretrainedConfig, str, "os.PathLike"]] = None,
        ignore_mismatched_sizes: bool = True,
        **kwargs,
    ):
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            **kwargs,
        )

    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

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
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
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
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        lm_dtype = self.lm_head.weight.dtype
        if hidden_states.dtype != lm_dtype:
            hidden_states = hidden_states.to(lm_dtype)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits, shift_labels)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
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
                is_decode = True

        if pixel_values_3d is None and pixel_values_volumes is not None:
            pixel_values_3d = pixel_values_volumes
        if volume_grid_dhw is None and volume_grid_thw is not None:
            volume_grid_dhw = volume_grid_thw
        if is_decode and pixel_values_3d is None:
            inner = getattr(self, "model", None)
            if inner is not None and getattr(inner, "_cached_pixel_values_3d", None) is not None:
                pixel_values_3d = inner._cached_pixel_values_3d
                volume_grid_dhw = inner._cached_volume_grid_dhw
                volume_grid_thw = volume_grid_dhw
                pixel_values_volumes = pixel_values_3d
                if __import__("os").environ.get("VITQWEN_INFER_DEBUG") == "1":
                    print("[prepare_inputs_for_generation] decode: 使用缓存的 pixel_values")

        if is_decode:
            # 必须只传新 token，否则 key 长度 = cache(531) + 当前序列(532) = 1063，
            # 与因果 mask 基于 seq_len=532 构建不一致，导致 SDPA 维度错误
            input_ids = input_ids[:, -1:] if input_ids.size(-1) > 1 else input_ids
            # 关键：decode 时仍须传 pixel_values/volume_grid，否则 has_visual=False，
            # 会清空 self.rope_deltas，导致 position_ids 错误、3D RoPE 失效、输出异常
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
                ret["pixel_values_volumes"] = pixel_values_volumes  # 可能为 None，model 会回退到 pixel_values_3d
            if volume_grid_dhw is not None:
                ret["volume_grid_dhw"] = volume_grid_dhw
                ret["volume_grid_thw"] = volume_grid_thw
            return ret
        need_visual = (pixel_values_3d is not None and volume_grid_dhw is not None) or vision_features is not None
        if need_visual:
            kwargs["pixel_values_3d"] = pixel_values_3d
            kwargs["volume_grid_dhw"] = volume_grid_dhw
            kwargs["vision_features"] = vision_features
        return dict(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
            **kwargs,
        )
