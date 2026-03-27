import torch
from torch import nn
from .vit import ViT3DTower
from .vit_visd import ViSDVisionTower
from .vit_fvlm import FvlmViTTower


class DualEncoderWrapper(nn.Module):
    def __init__(self, mae_encoder, clip_encoder):
        super().__init__()
        self.mae_encoder = mae_encoder
        self.clip_encoder = clip_encoder

    def forward(self, images):
        mae_features = self.mae_encoder(images)
        clip_features = self.clip_encoder(images)
        return torch.cat([mae_features, clip_features], dim=-1)

    def __call__(self, images):
        return self.forward(images)

    @property
    def hidden_size(self):
        return self.mae_encoder.hidden_size + self.clip_encoder.hidden_size

    @property
    def dtype(self):
        return self.mae_encoder.dtype

    @property
    def device(self):
        return self.mae_encoder.device


def build_vision_tower(config, **kwargs):
    vision_tower = getattr(config, 'vision_tower', None)
    is_dual = getattr(config, "dual_vision", False) or (vision_tower and 'vit3d_dual' in vision_tower.lower())

    if is_dual:
        mae_encoder = ViT3DTower(config, **kwargs)
        clip_encoder = ViT3DTower(config, **kwargs)
        return DualEncoderWrapper(mae_encoder, clip_encoder)

    if vision_tower and "fvlm" in vision_tower.lower():
        return FvlmViTTower(config, **kwargs)

    if vision_tower and 'vit3d' in vision_tower.lower():
        return ViT3DTower(config, **kwargs)

    if vision_tower and 'visd' in vision_tower.lower():
        return ViSDVisionTower(config)

    raise ValueError(f'Unknown vision tower: {vision_tower}')