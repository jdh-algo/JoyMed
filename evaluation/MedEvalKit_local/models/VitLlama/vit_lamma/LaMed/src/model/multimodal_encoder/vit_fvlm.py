"""
Fvlm-style 3D ViT backbone.

This mirrors the ViT used in fVLM (lavis blip_models) so that checkpoints from
`fvlm/pretrain_ckpts/model.pth` load without the missing/unexpected key noise
seen with the standard Monai ViT3D tower.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.mlp import MLPBlock
from monai.networks.blocks.pos_embed_utils import build_sincos_position_embedding
from monai.networks.layers import Conv, trunc_normal_
from monai.utils import ensure_tuple_rep, optional_import

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")


class SABlock(nn.Module):
    """
    Self-attention block (matches fvlm lavis selfattention.SABlock).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
        dim_head: Optional[int] = None,
    ) -> None:
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.dim_head = hidden_size // num_heads if dim_head is None else dim_head
        self.inner_dim = self.dim_head * num_heads

        self.out_proj = nn.Linear(self.inner_dim, hidden_size)
        self.qkv = nn.Linear(hidden_size, self.inner_dim * 3, bias=qkv_bias)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate
        self.scale = self.dim_head ** -0.5
        self.save_attn = save_attn
        self.att_mat = torch.Tensor()
        self.use_flash_attention = False

    def forward(self, x):
        output = self.input_rearrange(self.qkv(x))
        q, k, v = output[0], output[1], output[2]

        if self.use_flash_attention:
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                scale=self.scale,
                dropout_p=self.dropout_rate,
                is_causal=False,
            )
        else:
            att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale)
            att_mat = att_mat.softmax(dim=-1)
            att_mat = self.drop_weights(att_mat)
            x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)

        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block (matches fvlm lavis transformerblock.TransformerBlock).
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias, save_attn)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbeddingBlock(nn.Module):
    """
    3D patch embedding (matches fvlm lavis patchembedding.PatchEmbeddingBlock).
    Uses a single conv patch embedding + learnable/sincos positional embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int,
        num_heads: int,
        proj_type: str = "conv",
        pos_embed_type: str = "learnable",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden size {hidden_size} should be divisible by num_heads {num_heads}.")

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.n_patches = int(torch.tensor([im // p for im, p in zip(img_size, patch_size)]).prod().item())

        self.patch_embeddings = Conv[Conv.CONV, spatial_dims](
            in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)

        if pos_embed_type == "learnable":
            trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        elif pos_embed_type == "sincos":
            grid_size = [im // p for im, p in zip(img_size, patch_size)]
            self.position_embeddings = build_sincos_position_embedding(grid_size, hidden_size, spatial_dims)
        elif pos_embed_type == "none":
            pass
        else:
            raise ValueError(f"pos_embed_type {pos_embed_type} not supported.")

    def forward(self, x):
        x = self.patch_embeddings(x)
        d, h, w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # [B, N, H]

        # Interpolate PE if input shape differs (keep parity with fvlm impl)
        if self.position_embeddings.shape[1] == x.shape[1]:
            pos = self.position_embeddings
        else:
            pos = self.position_embeddings.permute(0, 2, 1)
            # assume original grid is (7,16,11) per fvlm; infer cube root fallback
            n = int(round(self.position_embeddings.shape[1] ** (1 / 3)))
            orig_grid = (7, 16, 11) if self.position_embeddings.shape[1] == 7 * 16 * 11 else (n, n, n)
            pos = pos.view(1, -1, *orig_grid)
            pos = F.interpolate(pos, size=(d, h, w), mode="trilinear", align_corners=False)
            pos = pos.flatten(2).transpose(1, 2)

        x = x + pos
        x = self.dropout(x)
        return x


class ViTFVLM(nn.Module):
    """
    fVLM ViT encoder (no cls token, qkv_bias=True).
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        pos_embed_type: str = "learnable",
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type="conv",
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.patch_embedding(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


class FvlmViTTower(nn.Module):
    """
    Wrapper to align with LaMed vision tower interface.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_tower = ViTFVLM(
            in_channels=config.image_channel,
            img_size=config.img_size,
            patch_size=config.patch_size,
            hidden_size=getattr(config, "mm_hidden_size", 768),
            mlp_dim=getattr(config, "mm_mlp_dim", 3072),
            num_layers=getattr(config, "mm_num_layers", 12),
            num_heads=getattr(config, "mm_num_heads", 12),
            dropout_rate=getattr(config, "mm_dropout", 0.0),
            spatial_dims=len(config.patch_size),
            pos_embed_type=getattr(config, "pos_embed_type", "learnable"),
            qkv_bias=True,
        )

    def forward(self, images):
        # returns patch tokens [B, N, H]
        return self.vision_tower(images)

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

