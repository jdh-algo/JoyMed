# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as torch_checkpoint

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.utils import optional_import

from .rope_utils import TransformerBlockWithRoPE

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")


class VariablePatchEmbedding3D(nn.Module):
    """
    Patch embedding for 3D volume without position embeddings.
    Supports variable input size (D, H, W); output is (B, N, hidden_size).
    Input: (B, C, D, H, W). patch_size: (pd, ph, pw). D,H,W must be divisible by pd,ph,pw.
    """
    def __init__(
        self,
        in_channels: int,
        patch_size: Sequence[int],
        hidden_size: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        patch_size = tuple(patch_size)
        if len(patch_size) != 3:
            raise ValueError("patch_size must be length 3 for 3D")
        pd, ph, pw = patch_size
        self.patch_size = patch_size
        self.patch_dim = in_channels * pd * ph * pw
        self.proj = nn.Sequential(
            Rearrange(
                "b c (d pd) (h ph) (w pw) -> b (d h w) (pd ph pw c)",
                pd=pd, ph=ph, pw=pw,
            ),
            nn.Linear(self.patch_dim, hidden_size),
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.dropout(x)


class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
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
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,

        # NEW for MAE features:
        mae: bool = False,
        use_rope: bool = False,
    ) -> None:
        """
        Args:
            in_channels (int): dimension of input channels.
            img_size (Union[Sequence[int], int]): dimension of input image.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            pos_embed (str, optional): position embedding layer type. Defaults to "conv".
            classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
            num_classes (int, optional): number of classes if classification is used. Defaults to 2.
            dropout_rate (float, optional): faction of the input units to drop. Defaults to 0.0.
            spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
            post_activation (str, optional): add a final acivation function to the classification head
                when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
                Set to other values to remove this function.
            qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.classification = classification
        self.mae = mae
        self.use_rope = use_rope
        # When True, wrap transformer blocks with torch checkpointing to save memory.
        # This trades extra compute (recompute activations in backward) for lower peak memory.
        self.gradient_checkpointing = False

        patch_size_tuple = tuple(patch_size) if not isinstance(patch_size, int) else (patch_size,) * spatial_dims
        if mae and use_rope:
            self.patch_embedding = VariablePatchEmbedding3D(
                in_channels=in_channels,
                patch_size=patch_size_tuple,
                hidden_size=hidden_size,
                dropout_rate=dropout_rate,
            )
            self.blocks = nn.ModuleList([
                TransformerBlockWithRoPE(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias)
                for _ in range(num_layers)
            ])
        else:
            _kwargs = dict(
                in_channels=in_channels,
                img_size=img_size,
                patch_size=patch_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                spatial_dims=spatial_dims,
            )
            try:
                self.patch_embedding = PatchEmbeddingBlock(
                    proj_type=pos_embed,
                    pos_embed_type="learnable",
                    **_kwargs,
                )
            except TypeError:
                self.patch_embedding = PatchEmbeddingBlock(pos_embed=pos_embed, **_kwargs)
            self.blocks = nn.ModuleList([
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ])
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            # if post_activation == "Tanh":
            #     self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            # else:
            #     self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore

    def set_gradient_checkpointing(self, enable: bool = True) -> None:
        """Enable/disable gradient checkpointing for transformer blocks."""
        self.gradient_checkpointing = bool(enable)

    def _checkpoint_block(self, blk: nn.Module, x: torch.Tensor, positions_2d=None) -> torch.Tensor:
        """
        Run a transformer block under activation checkpointing.

        Important for DDP: use non-reentrant checkpointing when available to avoid
        'Expected to mark a variable ready only once' errors.
        """
        def _run_blk(_x):
            return blk(_x, positions_2d=positions_2d)
        try:
            return torch_checkpoint.checkpoint(_run_blk, x, use_reentrant=False)
        except TypeError:
            return torch_checkpoint.checkpoint(_run_blk, x)

    def forward(self, x):
        if not self.mae:
            # regular ViT path, for CLIP
            x = self.patch_embedding(x)
            if hasattr(self, "cls_token"):
                cls_token = self.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls_token, x), dim=1)
            hidden_states_out = []
            for blk in self.blocks:
                if self.training and self.gradient_checkpointing:
                    x = self._checkpoint_block(blk, x)
                else:
                    x = blk(x)
                hidden_states_out.append(x)
            x = self.norm(x)
            # if hasattr(self, "classification_head"):
            #     x = self.classification_head(x[:, 0])
            return x, hidden_states_out

        # MAE path
        positions_2d = None
        if x.dim() == 5:
            # images -> tokens; compute 2D RoPE positions (h, w) from spatial grid
            B, C, D, H, W = x.shape
            pd, ph, pw = self.patch_embedding.patch_size
            grid_d, grid_h, grid_w = D // pd, H // ph, W // pw
            L = grid_d * grid_h * grid_w
            # Flatten order is d, h, w -> i = d*(grid_h*grid_w) + h*grid_w + w
            idx = torch.arange(L, device=x.device, dtype=torch.long)
            pos_h = (idx // grid_w) % grid_h
            pos_w = idx % grid_w
            positions_2d = (pos_h, pos_w)
            x = self.patch_embedding(x)  # [B, N, H]
        elif x.dim() == 3:
            if x.size(-1) != self.hidden_size:
                raise ValueError(
                    f"MAE tokens last dim must equal hidden_size={self.hidden_size}, got {x.size(-1)}"
                )
        else:
            raise ValueError("MAE mode expects images [B,C,D,H,W] or tokens [B,N,H].")

        # add CLS if requested (same flag for MAE/ViT)
        if self.classification:
            cls = self.cls_token.expand(x.size(0), 1, -1)    # [B,1,H]
            x = torch.cat([cls, x], dim=1)                   # [B,1+N,H]
            # Prepend dummy 2D positions for CLS so lengths match; CLS will get (0,0)
            if positions_2d is not None:
                pos_h, pos_w = positions_2d
                pos_h = torch.cat([torch.zeros(1, device=pos_h.device, dtype=pos_h.dtype), pos_h])
                pos_w = torch.cat([torch.zeros(1, device=pos_w.device, dtype=pos_w.dtype), pos_w])
                positions_2d = (pos_h, pos_w)
        hidden_states_out = []
        for blk in self.blocks:
            if self.training and self.gradient_checkpointing:
                x = self._checkpoint_block(blk, x, positions_2d)
            else:
                x = blk(x, positions_2d=positions_2d)
            hidden_states_out.append(x)
        x = self.norm(x)
        return x, hidden_states_out



class ViT3DTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature

        self.vision_tower = ViT(
            in_channels=self.config.image_channel,
            img_size=self.config.img_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
        )

    def forward(self, images):
        last_feature, hidden_states = self.vision_tower(images)
        if self.select_layer == -1:
            image_features = last_feature
        elif self.select_layer < -1:
            image_features = hidden_states[self.select_feature]
        else:
            raise ValueError(f'Unexpected select layer: {self.select_layer}')

        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size