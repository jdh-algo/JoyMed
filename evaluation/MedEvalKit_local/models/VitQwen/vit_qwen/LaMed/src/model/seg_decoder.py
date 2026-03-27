"""
Segmentation Decoder Module for converting vision encoder tokens to segmentation masks.

Architecture:
1. Vision encoder outputs X: [B, N, C] where N = Hp * Wp * Dp (patch tokens)
2. Learnable organ queries Q0: [B, Q, C] where Q = num_classes (117 organs)
3. Transformer decoder with L layers of self-attention + cross-attention
4. Mask projection: project queries and tokens to mask space, compute dot product
5. UNet-style upsampling from patch resolution to full resolution
6. Add background channel and output [B, num_classes+1, H, W, D]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TransformerDecoderLayer(nn.Module):
    """
    Single transformer decoder layer with:
    1. Self-attention on queries (queries talk to each other)
    2. Cross-attention (queries attend to image tokens)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Self-attention on queries
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention: queries attend to image tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # MLP
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(
        self,
        queries: torch.Tensor,  # [B, Q, C]
        image_tokens: torch.Tensor,  # [B, N, C]
    ) -> torch.Tensor:
        """
        Args:
            queries: [B, Q, C] - organ query vectors
            image_tokens: [B, N, C] - image patch tokens from encoder
        Returns:
            queries: [B, Q, C] - updated query vectors
        """
        # Self-attention on queries
        q_norm = self.norm1(queries)
        self_attn_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries = queries + self_attn_out
        
        # Cross-attention: queries attend to image tokens
        q_norm = self.norm2(queries)
        cross_attn_out, _ = self.cross_attn(
            query=q_norm,
            key=image_tokens,
            value=image_tokens,
        )
        queries = queries + cross_attn_out
        
        # MLP
        q_norm = self.norm3(queries)
        queries = queries + self.mlp(q_norm)
        
        return queries


class TransformerDecoder(nn.Module):
    """
    Stack of transformer decoder layers.
    """
    
    def __init__(
        self,
        d_model: int,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        queries: torch.Tensor,  # [B, Q, C]
        image_tokens: torch.Tensor,  # [B, N, C]
    ) -> torch.Tensor:
        """
        Args:
            queries: [B, Q, C] - organ query vectors
            image_tokens: [B, N, C] - image patch tokens from encoder
        Returns:
            queries: [B, Q, C] - decoded query vectors
        """
        for layer in self.layers:
            queries = layer(queries, image_tokens)
        return self.norm(queries)


class UNetUpsampler3D(nn.Module):
    """
    UNet-style 3D upsampler with skip connections.
    Takes low-resolution mask logits and upsamples to full resolution.
    
    Architecture:
    - Multiple upsampling stages, each doubles the spatial resolution
    - Skip connections from encoder features (if provided)
    - Convolution blocks at each stage
    """
    
    def __init__(
        self,
        in_channels: int,  # Number of query classes
        patch_grid: Tuple[int, int, int],  # (Hp, Wp, Dp)
        target_size: Tuple[int, int, int],  # (H, W, D) full resolution
        base_channels: int = 64,
    ):
        super().__init__()
        self.patch_grid = patch_grid
        self.target_size = target_size
        
        # Calculate number of upsampling stages needed
        # From (16, 16, 16) to (128, 256, 256) requires different scales per dimension
        self.scale_factors = [
            target_size[i] // patch_grid[i] for i in range(3)
        ]
        
        # Number of stages (based on max scale factor)
        max_scale = max(self.scale_factors)
        self.num_stages = int(math.log2(max_scale))
        
        # Initial projection from num_queries to base_channels
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        
        # Build upsampling stages
        self.up_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        
        current_channels = base_channels * 4
        for i in range(self.num_stages):
            out_channels = max(base_channels, current_channels // 2)
            
            # Upsample block
            self.up_blocks.append(
                nn.ConvTranspose3d(
                    current_channels, out_channels,
                    kernel_size=2, stride=2,
                )
            )
            
            # Conv block after upsampling
            self.conv_blocks.append(nn.Sequential(
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            ))
            
            current_channels = out_channels
        
        # Final projection back to num_classes (will add background later)
        self.final_conv = nn.Conv3d(current_channels, in_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, Q, Hp, Wp, Dp] - low-resolution mask logits
        Returns:
            x: [B, Q, H, W, D] - full-resolution mask logits
        """
        # Initial convolution
        x = self.initial_conv(x)
        
        # Upsampling stages
        for up, conv in zip(self.up_blocks, self.conv_blocks):
            x = up(x)
            x = conv(x)
        
        # Final projection
        x = self.final_conv(x)
        
        # Interpolate to exact target size if needed
        if x.shape[2:] != self.target_size:
            x = F.interpolate(
                x,
                size=self.target_size,
                mode='trilinear',
                align_corners=False,
            )
        
        return x


class SegmentationModel(nn.Module):
    """
    Complete segmentation model:
    1. Vision encoder (pre-trained, can be frozen or fine-tuned)
    2. Transformer decoder with learnable organ queries
    3. Mask projection to get per-token scores
    4. UNet upsampler to full resolution
    5. Background channel addition
    """
    
    def __init__(
        self,
        vision_tower: nn.Module,
        hidden_size: int = 768,
        num_classes: int = 117,  # Number of organ classes (excluding background)
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        mask_dim: int = 1024,
        patch_grid: Tuple[int, int, int] = (16, 16, 16),
        target_size: Tuple[int, int, int] = (128, 256, 256),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vision_tower = vision_tower
        self.hidden_size = hidden_size
        self.num_classes = num_classes  # Number of organ classes (117)
        self.patch_grid = patch_grid
        self.target_size = target_size
        
        # Learnable organ queries
        self.organ_queries = nn.Parameter(torch.randn(num_classes, hidden_size))
        nn.init.normal_(self.organ_queries, std=0.02)
        
        # Transformer decoder
        self.decoder = TransformerDecoder(
            d_model=hidden_size,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Mask projection layers (project to mask space for dot product)
        self.query_proj = nn.Linear(hidden_size, mask_dim)
        self.token_proj = nn.Linear(hidden_size, mask_dim)
        
        # Layer norms for mask projection
        self.query_norm = nn.LayerNorm(mask_dim)
        self.token_norm = nn.LayerNorm(mask_dim)
        
        # UNet upsampler
        self.upsampler = UNetUpsampler3D(
            in_channels=num_classes,
            patch_grid=patch_grid,
            target_size=target_size,
        )
        
        # Background logit bias (initialized to 0.0 for better learning, learnable)
        self.bg_bias = nn.Parameter(torch.tensor(0.0))
        
        # Gradient checkpointing flag
        self._gradient_checkpointing = False
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for memory efficiency."""
        self._gradient_checkpointing = True
        # Enable on vision tower if it supports it
        if hasattr(self.vision_tower, 'gradient_checkpointing_enable'):
            self.vision_tower.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False
        if hasattr(self.vision_tower, 'gradient_checkpointing_disable'):
            self.vision_tower.gradient_checkpointing_disable()
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, D, H, W] - input CT images
        Returns:
            logits: [B, num_classes+1, D, H, W] - segmentation logits including background
        """
        B = images.shape[0]
        
        # 1. Get vision encoder features
        image_tokens = self.vision_tower(images)  # [B, N, C]
        
        # 2. Expand queries for batch
        queries = self.organ_queries.unsqueeze(0).expand(B, -1, -1)  # [B, Q, C]
        
        # 3. Transformer decoder
        decoded_queries = self.decoder(queries, image_tokens)  # [B, Q, C]

        # 4. Mask projection
        # Project to mask space
        Qm = self.query_proj(decoded_queries)  # [B, Q, M]
        Xm = self.token_proj(image_tokens)  # [B, N, M]
        
        # Normalize
        Qm = self.query_norm(Qm)
        Xm = self.token_norm(Xm)
        
        # Dot product: [B, Q, M] @ [B, M, N] -> [B, Q, N]
        mask_logits_flat = torch.bmm(Qm, Xm.transpose(1, 2))
        
        # Scale by sqrt(dim) for stable attention-like behavior
        mask_logits_flat = mask_logits_flat / math.sqrt(Qm.shape[-1])
        
        # 5. Reshape to spatial grid: [B, Q, N] -> [B, Q, Hp, Wp, Dp]
        Hp, Wp, Dp = self.patch_grid
        mask_logits_lowres = mask_logits_flat.view(B, self.num_classes, Hp, Wp, Dp)  # [B, 117, Hp, Wp, Dp]

        # 6. UNet upsample to full resolution: [B, 117, Hp, Wp, Dp] -> [B, 117, H, W, D]
        mask_logits_highres = self.upsampler(mask_logits_lowres)
        
        # 7. Add background channel
        # Background logits initialized with learnable bias (starts at -2.0, can learn during training)
        bg_logits = self.bg_bias.expand(B, 1, *self.target_size)
        
        # Concatenate: [B, 1, H, W, D] (bg) + [B, 117, H, W, D] (organs) -> [B, 118, H, W, D]
        # Note: background is channel 0, organs are channels 1-117
        logits = torch.cat([bg_logits, mask_logits_highres], dim=1)  # [B, 118, H, W, D]

        return logits
    
    def compute_dice_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100,
        smooth: float = 1e-5,
    ) -> torch.Tensor:
        """
        Compute multi-class Dice loss.
        
        Args:
            logits: [B, num_classes+1, D, H, W] - predicted logits
            targets: [B, D, H, W] - ground truth with values 0-117
            ignore_index: index to ignore in loss computation
            smooth: smoothing factor to avoid division by zero
        Returns:
            dice_loss: scalar tensor
        """
        num_classes = logits.shape[1]  # num_classes + 1 (including background)
        B, C, D, H, W = logits.shape
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)  # [B, C, D, H, W]
        
        # Create one-hot encoding for targets
        targets_long = targets.long()  # [B, D, H, W]
        
        # Create mask for valid pixels (not ignored)
        valid_mask = (targets_long != ignore_index)  # [B, D, H, W]
        
        # Track dice losses only for classes that appear in the batch
        dice_losses_present = []
        
        # Compute Dice loss for each class
        for c in range(num_classes):
            # Predicted probability for class c
            pred_c = probs[:, c, :, :, :]  # [B, D, H, W]
            
            # Ground truth for class c (one-hot)
            target_c = (targets_long == c).float()  # [B, D, H, W]
            
            # Apply valid mask
            pred_c = pred_c * valid_mask.float()
            target_c = target_c * valid_mask.float()
            
            # Flatten spatial dimensions
            pred_c_flat = pred_c.contiguous().view(B, -1)  # [B, D*H*W]
            target_c_flat = target_c.contiguous().view(B, -1)  # [B, D*H*W]
            
            # Check if this class appears in the batch (in ground truth)
            gt_sum = target_c_flat.sum(dim=1)  # [B] - pixels of class c in GT per sample
            class_present_in_batch = (gt_sum.sum() > 0)  # True if class c appears in any sample
            
            # Skip if class doesn't appear in batch
            if not class_present_in_batch:
                continue
            
            # Compute intersection and union
            intersection = (pred_c_flat * target_c_flat).sum(dim=1)  # [B]
            union = pred_c_flat.sum(dim=1) + target_c_flat.sum(dim=1)  # [B]
            
            # Compute Dice score for each sample in batch
            dice_score = (2.0 * intersection + smooth) / (union + smooth)  # [B]
            
            # Dice loss = 1 - Dice score
            dice_loss = 1.0 - dice_score  # [B]
            
            # Only average over samples where class c appears in GT
            valid_samples = (gt_sum > 0).float()  # [B] - samples where class c exists in GT
            
            if valid_samples.sum() > 0:
                # Average over samples where class c appears in GT
                dice_loss = (dice_loss * valid_samples).sum() / valid_samples.sum()
                dice_losses_present.append(dice_loss)
        
        # Average Dice loss only over classes that appear in the batch
        if len(dice_losses_present) > 0:
            dice_loss_total = torch.stack(dice_losses_present).mean()
        else:
            # Edge case: no classes present in batch (shouldn't happen, but handle gracefully)
            dice_loss_total = logits.sum() * 0.0
        
        return dice_loss_total
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100,
    ) -> dict:
        """
        Compute combined cross-entropy and Dice loss for segmentation.
        
        Args:
            logits: [B, num_classes+1, D, H, W] - predicted logits
            targets: [B, D, H, W] - ground truth with values 0-117
            ignore_index: index to ignore in loss computation
        Returns:
            dict with keys:
                - 'loss': total combined loss (scalar tensor)
                - 'ce_loss': cross-entropy loss (scalar tensor)
                - 'dice_loss': Dice loss (scalar tensor)
        """
        # Cross-entropy loss
        ce_loss = F.cross_entropy(
            logits,
            targets.long(),
            ignore_index=ignore_index,
        )
        
        # Dice loss
        dice_loss = self.compute_dice_loss(
            logits,
            targets,
            ignore_index=ignore_index,
        )
        
        # Combined loss
        total_loss = ce_loss + dice_loss
        
        return {
            'loss': total_loss,
            'ce_loss': ce_loss,
            'dice_loss': dice_loss,
        }


def build_segmentation_model(
    vision_tower: nn.Module,
    config,
) -> SegmentationModel:
    """
    Build segmentation model from config.
    
    Args:
        vision_tower: Pre-trained vision encoder
        config: Configuration object with model parameters
    """
    # Get hidden size from vision tower
    hidden_size = vision_tower.hidden_size
    
    # Calculate patch grid from img_size and patch_size
    img_size = getattr(config, 'img_size', (128, 256, 256))
    patch_size = getattr(config, 'patch_size', (8, 16, 16))
    patch_grid = tuple(img_size[i] // patch_size[i] for i in range(3))
    
    return SegmentationModel(
        vision_tower=vision_tower,
        hidden_size=hidden_size,
        num_classes=getattr(config, 'num_seg_classes', 117),
        num_decoder_layers=getattr(config, 'num_decoder_layers', 6),
        num_heads=getattr(config, 'num_decoder_heads', 8),
        mask_dim=getattr(config, 'mask_dim', 1024),
        patch_grid=patch_grid,
        target_size=img_size,
        dropout=getattr(config, 'decoder_dropout', 0.0),
    )

