# coding=utf-8
# Copyright 2025 VitQwen Team. All rights reserved.
"""
3D Convolutional Downsampling for ViT (3D) output.
Reduces D/H/W to 1/2 via 3D convolution (stride 2). Output dim = hidden_size.
"""

import torch
import torch.nn as nn
from typing import Tuple


class Conv3dDownsample(nn.Module):
    """
    3D convolutional downsampling: D, H, W -> D/2, H/2, W/2.
    Input: features (B, N, C), grid (D, H, W). Output: (B, N', C), N' = (D/2)*(H/2)*(W/2).
    """

    def __init__(self, hidden_size: int, factor: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.factor = factor
        self.conv = nn.Conv3d(
            hidden_size, hidden_size, kernel_size=factor, stride=factor, padding=0, bias=True
        )

    def forward(
        self,
        features: torch.Tensor,
        volume_grid_dhw: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        B, N, C = features.shape
        if volume_grid_dhw.dim() == 1:
            d, h, w = volume_grid_dhw.tolist()
            volume_grid_dhw = volume_grid_dhw.unsqueeze(0).expand(B, -1)
        else:
            d, h, w = volume_grid_dhw[0].tolist()
        assert N == d * h * w, f"features length N={N} must equal D*H*W={d*h*w}"
        x = features.view(B, d, h, w, C).permute(0, 4, 1, 2, 3)
        x = self.conv(x)
        _, c, d2, h2, w2 = x.shape
        downsampled = x.permute(0, 2, 3, 4, 1).reshape(B, -1, c)
        new_grid = torch.tensor([[d2, h2, w2]] * B, device=features.device, dtype=torch.long)
        return downsampled, new_grid
