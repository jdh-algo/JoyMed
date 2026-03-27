"""
Vision resampling and bridging modules to connect a vision encoder to an LLM.

Components:
- Resampler: Perceiver-style cross-attention that compresses variable-length vision tokens to a fixed K.
- Bridge: LN + projection + MLP to map resampled tokens from encoder dim to LLM dim.
- ResamplerBridge: convenience wrapper chaining Resampler then Bridge.
- DirectConnection: simple linear projector used to bypass mm_projector when desired.
"""


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Resampler(nn.Module):
    """
    Perceiver-style resampler: converts variable-length encoder tokens [B, N, D_enc]
    into fixed K tokens [B, K, D_enc] using learnable queries and cross attention.
    """
    def __init__(
        self,
        d_enc: int,
        k: int,
        num_heads: int = 8,
    ):
        super().__init__()
        self.k = k
        self.query = nn.Parameter(torch.randn(1, k, d_enc))
        # Separate norms for attention path and MLP path to allow pre-norm residual blocks.
        self.ln_q_attn = nn.LayerNorm(d_enc)
        self.ln_q_mlp = nn.LayerNorm(d_enc)
        self.ln_kv = nn.LayerNorm(d_enc)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_enc,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.mlp_fc1 = nn.Linear(d_enc, 2 * d_enc)
        self.mlp_fc2 = nn.Linear(2 * d_enc, d_enc)

    def forward(self, x_enc: torch.Tensor, enc_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x_enc: [B, N, D_enc]
            enc_mask: [B, N] with 1 for valid, 0 for padded. If None, assumes all valid.
        Returns:
            z: [B, K, D_enc]
        """
        b = x_enc.size(0)
        z = self.query.expand(b, -1, -1)  # [B, K, D_enc]

        z_attn_in = self.ln_q_attn(z)
        kv = self.ln_kv(x_enc)

        key_padding_mask = None
        if enc_mask is not None:
            key_padding_mask = (enc_mask == 0)

        z_attn, _ = self.cross_attn(
            query=z_attn_in,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        z = z + z_attn

        z_mlp_in = self.ln_q_mlp(z)
        h = self.mlp_fc1(z_mlp_in)
        h = F.gelu(h)
        h = self.mlp_fc2(h)
        z = z + h
        return z


class Bridge(nn.Module):
    def __init__(self, d_enc, d_llm, mlp_ratio=1.0):
        super().__init__()
        self.ln_in = nn.LayerNorm(d_enc)
        self.proj = nn.Linear(d_enc, d_llm)

        self.ln_mlp = nn.LayerNorm(d_llm)   # <--- add this
        hidden = int(d_llm * mlp_ratio)
        self.mlp_fc1 = nn.Linear(d_llm, hidden)
        self.mlp_fc2 = nn.Linear(hidden, d_llm)

        self.ln_out = nn.LayerNorm(d_llm)

        # optional but recommended "safe start"
        nn.init.zeros_(self.mlp_fc2.weight)
        nn.init.zeros_(self.mlp_fc2.bias)

    def forward(self, z_enc):
        # project encoder tokens to LLM dimension
        z = self.proj(self.ln_in(z_enc))            # [B, K, d_llm]

        # residual MLP block
        mlp_in = self.ln_mlp(z)
        mlp_hidden = self.mlp_fc1(mlp_in)
        mlp_hidden = F.gelu(mlp_hidden)
        residual = self.mlp_fc2(mlp_hidden)

        return self.ln_out(z + residual)



class ResamplerBridge(nn.Module):
    """
    Convenience wrapper: Resampler -> Bridge.
    """

    def __init__(
        self,
        d_enc: int,
        d_llm: int,
        k: int,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.k = k
        self.resampler = Resampler(d_enc=d_enc, k=k, num_heads=num_heads)
        self.bridge = Bridge(d_enc=d_enc, d_llm=d_llm, mlp_ratio=mlp_ratio)

    def forward(self, x_enc: torch.Tensor, enc_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.resampler(x_enc, enc_mask)
        x_llm = self.bridge(z)
        return x_llm

    @property
    def proj_out_num(self) -> int:
        return self.k



# Create a direct connection: identity if dimensions match, else simple linear projection
# This bypasses the mm_projector compression, connecting vision encoder directly to LLM
class DirectConnection(nn.Module):
    def __init__(self, in_dim, out_dim, num_tokens):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.zeros_(self.proj.bias)
        # Initialize to an identity block on the overlapping dimensions for a pass-through start
        with torch.no_grad():
            self.proj.weight.data.zero_()
            k = min(in_dim, out_dim)
            self.proj.weight.data[:k, :k] = torch.eye(k)
            
    def forward(self, x):
        return self.proj(x)
            
    @property
    def proj_out_num(self):
        return self.num_tokens