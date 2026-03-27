# Copyright (c) MONAI Consortium
# RoPE (Rotary Position Embedding) for variable-length patch sequences.
# 2D RoPE: use (height, width) positions for spatial tokens (first half of head_dim for h, second for w).

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    head_dim: int,
    base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply 1D Rotary Position Embedding to Q and K.

    Args:
        q: (B, num_heads, L, head_dim)
        k: (B, num_heads, L, head_dim)
        positions: (L,) or (B, L) position indices (0, 1, ..., L-1)
        head_dim: must be even
        base: RoPE base

    Returns:
        q_rot, k_rot with same shape as q, k
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(half, device=q.device, dtype=q.dtype) / head_dim))
    if positions.dim() == 1:
        positions = positions.unsqueeze(0)
    freqs = torch.einsum("...l,d->...ld", positions.float(), inv_freq)
    cos_t = freqs.cos()
    sin_t = freqs.sin()
    if cos_t.dim() == 3:
        cos_t = cos_t.unsqueeze(1)
        sin_t = sin_t.unsqueeze(1)
    q_even, q_odd = q[..., 0::2], q[..., 1::2]
    q_rot_even = q_even * cos_t - q_odd * sin_t
    q_rot_odd = q_even * sin_t + q_odd * cos_t
    q_rot = torch.stack([q_rot_even, q_rot_odd], dim=-1).flatten(-2)
    k_even, k_odd = k[..., 0::2], k[..., 1::2]
    k_rot_even = k_even * cos_t - k_odd * sin_t
    k_rot_odd = k_even * sin_t + k_odd * cos_t
    k_rot = torch.stack([k_rot_even, k_rot_odd], dim=-1).flatten(-2)
    return q_rot, k_rot


def apply_rope_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_h: torch.Tensor,
    pos_w: torch.Tensor,
    head_dim: int,
    base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply 2D Rotary Position Embedding to Q and K.
    First half of head_dim uses pos_h (height), second half uses pos_w (width).

    Args:
        q: (B, num_heads, L, head_dim)
        k: (B, num_heads, L, head_dim)
        pos_h: (L,) or (B, L) height position indices
        pos_w: (L,) or (B, L) width position indices
        head_dim: must be even
        base: RoPE base

    Returns:
        q_rot, k_rot with same shape as q, k
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for 2D RoPE")
    half = head_dim // 2
    # Each spatial dim gets half//2 (even,odd) pairs; need half//2 frequencies per dim
    dim_rope = half // 2
    inv_freq = 1.0 / (base ** (torch.arange(dim_rope, device=q.device, dtype=q.dtype) / head_dim))
    for dim, pos in enumerate((pos_h, pos_w)):
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)
        freqs = torch.einsum("...l,d->...ld", pos.float(), inv_freq)
        cos_t = freqs.cos()
        sin_t = freqs.sin()
        if cos_t.dim() == 3:
            cos_t = cos_t.unsqueeze(1)
            sin_t = sin_t.unsqueeze(1)
        # Apply to half of the head_dim: dim 0 -> [0:half], dim 1 -> [half:head_dim]
        start = dim * half
        end = start + half
        q_h = q[..., start:end]
        k_h = k[..., start:end]
        q_even, q_odd = q_h[..., 0::2], q_h[..., 1::2]
        q_rot_even = q_even * cos_t - q_odd * sin_t
        q_rot_odd = q_even * sin_t + q_odd * cos_t
        q_rot_h = torch.stack([q_rot_even, q_rot_odd], dim=-1).flatten(-2)
        k_even, k_odd = k_h[..., 0::2], k_h[..., 1::2]
        k_rot_even = k_even * cos_t - k_odd * sin_t
        k_rot_odd = k_even * sin_t + k_odd * cos_t
        k_rot_h = torch.stack([k_rot_even, k_rot_odd], dim=-1).flatten(-2)
        if dim == 0:
            q_rot = q_rot_h
            k_rot = k_rot_h
        else:
            q_rot = torch.cat([q_rot, q_rot_h], dim=-1)
            k_rot = torch.cat([k_rot, k_rot_h], dim=-1)
    return q_rot, k_rot


class SABlockWithRoPE(nn.Module):
    """Self-attention block that applies 2D RoPE (h, w) to Q and K when positions_2d is given, else 1D RoPE."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.rope_base = rope_base
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        positions_2d: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if positions_2d is not None:
            pos_h, pos_w = positions_2d
            q, k = apply_rope_2d(q, k, pos_h, pos_w, self.head_dim, base=self.rope_base)
        else:
            positions = torch.arange(L, device=x.device, dtype=torch.long)
            q, k = apply_rope(q, k, positions, self.head_dim, base=self.rope_base)
        # SDPA: PyTorch 自动选择 Flash/memory-efficient 内核，O(L) 显存而非 O(L²)
        # 对 16384 patches 场景（ViT 3D），从 ~24GB 降到 ~数百 MB
        # RoPE 内部用 float() 计算，可能把 q/k 升为 fp32 而 v 仍为 bf16；SDPA 要求三者同 dtype
        if q.dtype != v.dtype:
            q = q.to(v.dtype)
            k = k.to(v.dtype)
        dropout_p = self.dropout.p if self.training else 0.0
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=False,
        )
        x = x.transpose(1, 2).reshape(B, L, C)
        return self.dropout(self.out_proj(x))


class TransformerBlockWithRoPE(nn.Module):
    """Transformer block with RoPE in self-attention (2D when positions_2d given)."""

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlockWithRoPE(hidden_size, num_heads, dropout_rate, qkv_bias)
        self.norm2 = nn.LayerNorm(hidden_size)
        from monai.networks.blocks import MLPBlock
        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        positions_2d: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), positions_2d=positions_2d)
        x = x + self.mlp(self.norm2(x))
        return x
