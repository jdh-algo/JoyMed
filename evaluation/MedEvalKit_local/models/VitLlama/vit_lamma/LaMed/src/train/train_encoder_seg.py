"""
Training script for segmentation using pre-trained vision encoder.

This script trains a segmentation model that:
1. Uses a pre-trained vision encoder (frozen or fine-tuned)
2. Adds a transformer decoder with learnable organ queries
3. Projects to mask space and upsamples with UNet decoder
4. Trains with cross-entropy loss on CT segmentation masks

No LLM is used - this is pure vision-to-segmentation.
"""

import os
import json
import logging
import math
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import Trainer, TrainingArguments as HFTrainingArguments
from safetensors.torch import load_file as load_safetensors

from LaMed.src.model.multimodal_encoder.builder import build_vision_tower
from LaMed.src.model.seg_decoder import SegmentationModel, build_segmentation_model
from LaMed.src.dataset.seg_dataset import CTSegDataset, SegDataCollator, build_seg_datasets


local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank is None:
        print(*args)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    
    # Vision encoder
    vision_tower: str = field(
        default="vit3d",
        metadata={"help": "Type of vision tower: vit3d, visd, fvlm"}
    )
    pretrain_vision_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained vision encoder weights"}
    )
    freeze_vision_tower: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the vision encoder"}
    )
    
    # Image configuration
    image_channel: int = field(default=1)
    img_size: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Image size D H W (space-separated), default: 128 256 256"}
    )
    patch_size: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Patch size D H W (space-separated), default: 8 16 16"}
    )
    
    # Vision encoder internal config
    vision_select_layer: int = field(default=-1)
    vision_select_feature: str = field(default="patch")
    
    # Segmentation decoder configuration
    num_seg_classes: int = field(
        default=117,
        metadata={"help": "Number of organ classes (excluding background)"}
    )
    num_decoder_layers: int = field(
        default=6,
        metadata={"help": "Number of transformer decoder layers"}
    )
    num_decoder_heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads in decoder"}
    )
    mask_dim: int = field(
        default=1024,
        metadata={"help": "Dimension of mask projection space"}
    )
    decoder_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate in decoder"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    
    # CT-RATE data paths
    train_image_root: str = field(
        default="/mnt/workspace/offline/shared_data/Medical_Data_3D/CT-RATE/dataset/train_fixed",
        metadata={"help": "Root directory for training CT images"}
    )
    train_mask_root: str = field(
        default="/mnt/workspace/offline/shared_data/Medical_Data_3D/CT-RATE/dataset/ts_seg/ts_total/train_fixed",
        metadata={"help": "Root directory for training segmentation masks"}
    )
    valid_image_root: str = field(
        default="/mnt/workspace/offline/shared_data/Medical_Data_3D/CT-RATE/dataset/valid_fixed",
        metadata={"help": "Root directory for validation CT images"}
    )
    valid_mask_root: str = field(
        default="/mnt/workspace/offline/shared_data/Medical_Data_3D/CT-RATE/dataset/ts_seg/ts_total/valid_fixed",
        metadata={"help": "Root directory for validation segmentation masks"}
    )
    
    # Data loading
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum training samples (for debugging)"}
    )
    max_valid_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum validation samples (for debugging)"}
    )
    use_augmentation: bool = field(
        default=True,
        metadata={"help": "Whether to use data augmentation"}
    )


@dataclass
class TrainingArguments(HFTrainingArguments):
    """Training arguments extending HuggingFace TrainingArguments."""
    
    # Override defaults
    output_dir: str = field(default="./LaMed/output/seg_only")
    num_train_epochs: float = field(default=50)
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=4)
    
    learning_rate: float = field(default=1e-4)
    vision_learning_rate: Optional[float] = field(
        default=1e-5,
        metadata={"help": "Learning rate for vision encoder (if not frozen)"}
    )
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.1)
    lr_scheduler_type: str = field(default="cosine")
    
    logging_steps: int = field(default=10)
    eval_strategy: str = field(default="steps")
    eval_steps: float = field(default=0.1)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=1000)
    save_total_limit: int = field(default=5)
    
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)
    dataloader_pin_memory: bool = field(default=True)
    
    remove_unused_columns: bool = field(default=False)
    report_to: str = field(default="tensorboard")
    
    seed: int = field(default=42)
    ddp_backend: str = field(default="nccl")
    ddp_timeout: int = field(default=128000)
    ddp_find_unused_parameters: bool = field(default=False)


class SegmentationTrainer(Trainer):
    """
    Custom trainer for segmentation task.
    Handles:
    - Different learning rates for vision encoder vs decoder
    - Segmentation-specific loss computation
    - Dice score metrics
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss for segmentation and extract individual components for logging.
        """
        images = inputs['images']
        masks = inputs['masks']
        
        # Forward pass
        logits = model(images)  # [B, 118, D, H, W]
        
        # Compute loss (returns dict with 'loss', 'ce_loss', 'dice_loss')
        loss_dict = model.compute_loss(logits, masks)
        
        # Store individual loss components for logging
        # HuggingFace Trainer will use 'loss' for backprop, but we can log the others
        if not hasattr(self, '_loss_components'):
            self._loss_components = {}
        self._loss_components = {
            'ce_loss': loss_dict['ce_loss'].detach().item(),
            'dice_loss': loss_dict['dice_loss'].detach().item(),
        }
        
        if return_outputs:
            return loss_dict['loss'], {'logits': logits, 'masks': masks}
        return loss_dict['loss']
    
    def log(self, logs: dict) -> None:
        """
        Log metrics including individual loss components.
        """
        # Add individual loss components if available
        if hasattr(self, '_loss_components'):
            logs['train/ce_loss'] = self._loss_components.get('ce_loss', 0.0)
            logs['train/dice_loss'] = self._loss_components.get('dice_loss', 0.0)
        
        # Call parent log method
        super().log(logs)
    
    def create_optimizer(self):
        """
        Create optimizer with different learning rates for vision encoder and decoder.
        """
        if self.optimizer is not None:
            return self.optimizer
        
        # If vision_learning_rate is not set, use default behavior
        vision_lr = getattr(self.args, "vision_learning_rate", None)
        if vision_lr is None:
            return super().create_optimizer()
        
        # If vision_learning_rate equals learning_rate, no need for separate groups
        if abs(vision_lr - self.args.learning_rate) < 1e-10:
            return super().create_optimizer()
        
        opt_model = self.model
        
        # Check if vision_tower has any trainable parameters
        # If vision_tower is frozen (no trainable params), use default optimizer creation
        has_trainable_vision_params = any(
            "vision_tower" in n and p.requires_grad 
            for n, p in opt_model.named_parameters()
        )
        
        if not has_trainable_vision_params:
            # Vision tower is frozen, no need for separate learning rates
            return super().create_optimizer()
        
        decay_parameters = self.get_decay_parameter_names(opt_model)
        
        optimizer_grouped_parameters = [
            # Vision encoder (with weight decay)
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if "vision_tower" in n and n in decay_parameters and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
                "lr": vision_lr,
            },
            # Vision encoder (no weight decay)
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if "vision_tower" in n and n not in decay_parameters and p.requires_grad
                ],
                "weight_decay": 0.0,
                "lr": vision_lr,
            },
            # Everything else (with weight decay)
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if "vision_tower" not in n and n in decay_parameters and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate,
            },
            # Everything else (no weight decay)
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if "vision_tower" not in n and n not in decay_parameters and p.requires_grad
                ],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate,
            },
        ]
        
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        
        # Verification print
        if self.is_world_process_zero():
            print("\n" + "=" * 60)
            print("OPTIMIZER PARAMETER GROUPS")
            print("=" * 60)
            group_names = [
                "Vision Encoder (Decay)",
                "Vision Encoder (No Decay)",
                "Decoder (Decay)",
                "Decoder (No Decay)",
            ]
            for i, group in enumerate(self.optimizer.param_groups):
                num_params = sum(p.numel() for p in group["params"])
                print(f"Group {i} [{group_names[i]}]:")
                print(f"  - Parameters: {num_params / 1e6:.2f}M")
                print(f"  - Learning Rate: {group['lr']}")
                print(f"  - Weight Decay: {group['weight_decay']}")
            print("=" * 60 + "\n")
        
        return self.optimizer


def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 118) -> Dict[str, float]:
    """
    Compute Dice score for segmentation evaluation.
    
    Args:
        pred: [B, C, D, H, W] predicted logits
        target: [B, D, H, W] ground truth
        num_classes: number of classes including background
    
    Returns:
        Dictionary with per-class and mean Dice scores
    """
    pred_classes = pred.argmax(dim=1)  # [B, D, H, W]
    
    dice_scores = {}
    valid_classes = []
    
    for c in range(num_classes):
        pred_c = (pred_classes == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union > 0:
            dice = (2 * intersection / (union + 1e-8)).item()
            dice_scores[f"dice_class_{c}"] = dice
            valid_classes.append(dice)
    
    if valid_classes:
        dice_scores["dice_mean"] = np.mean(valid_classes)
    else:
        dice_scores["dice_mean"] = 0.0
    
    return dice_scores


def compute_metrics(eval_preds):
    """
    Compute evaluation metrics.
    """
    predictions, labels = eval_preds
    
    # predictions: logits [B, C, D, H, W]
    # labels: ground truth [B, D, H, W]
    
    pred_tensor = torch.from_numpy(predictions)
    label_tensor = torch.from_numpy(labels)
    
    # Compute Dice score
    dice_scores = compute_dice_score(pred_tensor, label_tensor)
    
    # Compute accuracy
    pred_classes = pred_tensor.argmax(dim=1)
    accuracy = (pred_classes == label_tensor).float().mean().item()
    
    return {
        "accuracy": accuracy,
        "dice_mean": dice_scores["dice_mean"],
    }


def preprocess_logits_for_metrics(logits, labels):
    """
    Preprocess logits before metrics computation.
    """
    # Just return logits as-is for now
    return logits


def load_vision_encoder_weights(vision_tower, weight_path: str):
    """
    Load pretrained weights into vision encoder.
    Handles both .safetensors and .pth/.bin formats.
    """
    if weight_path is None:
        rank0_print("No pretrained vision encoder weights specified")
        return
    
    if not os.path.exists(weight_path):
        raise ValueError(f"Vision encoder weights not found: {weight_path}")
    
    rank0_print(f"Loading vision encoder weights from: {weight_path}")
    
    if weight_path.endswith('.safetensors'):
        state_dict = load_safetensors(weight_path)
    else:
        state_dict = torch.load(weight_path, map_location='cpu')
    
    # Handle nested state dicts
    if 'model' in state_dict:
        state_dict = state_dict['model']
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # Try to load weights, allowing partial match
    # Filter for vision tower keys and remove prefixes if needed
    vision_state_dict = {}
    for k, v in state_dict.items():
        # Remove common prefixes
        new_key = k
        for prefix in ['vision_tower.', 'model.vision_tower.', 'encoder.', 'model.']:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        vision_state_dict[new_key] = v
    
    # Load with strict=False to handle any mismatches
    missing, unexpected = vision_tower.load_state_dict(vision_state_dict, strict=False)
    
    if missing:
        rank0_print(f"Missing keys: {len(missing)}")
        if len(missing) < 10:
            for k in missing:
                rank0_print(f"  - {k}")
    if unexpected:
        rank0_print(f"Unexpected keys: {len(unexpected)}")
        if len(unexpected) < 10:
            for k in unexpected:
                rank0_print(f"  - {k}")
    
    rank0_print("Vision encoder weights loaded successfully")


def main():
    global local_rank
    
    # Parse arguments
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    local_rank = training_args.local_rank
    
    rank0_print("=" * 60)
    rank0_print("SEGMENTATION TRAINING (No LLM)")
    rank0_print("=" * 60)
    
    # ==================== Image/patch sizes ====================
    # Convert from list to tuple, use defaults if not provided
    img_size = tuple(model_args.img_size) if model_args.img_size else (128, 256, 256)
    patch_size = tuple(model_args.patch_size) if model_args.patch_size else (8, 16, 16)
    
    rank0_print(f"Image size: {img_size}")
    rank0_print(f"Patch size: {patch_size}")
    
    # ==================== Build Vision Encoder ====================
    rank0_print("\n" + "=" * 20 + " Building Vision Encoder " + "=" * 20)
    
    # Create a config object for the vision tower builder
    class VisionConfig:
        pass
    
    vision_config = VisionConfig()
    vision_config.vision_tower = model_args.vision_tower
    vision_config.image_channel = model_args.image_channel
    vision_config.img_size = img_size
    vision_config.patch_size = patch_size
    vision_config.vision_select_layer = model_args.vision_select_layer
    vision_config.vision_select_feature = model_args.vision_select_feature
    
    vision_tower = build_vision_tower(vision_config)
    
    # Load pretrained weights
    load_vision_encoder_weights(vision_tower, model_args.pretrain_vision_model)
    
    rank0_print(f"Vision tower type: {model_args.vision_tower}")
    rank0_print(f"Vision hidden size: {vision_tower.hidden_size}")
    
    # ==================== Build Segmentation Model ====================
    rank0_print("\n" + "=" * 20 + " Building Segmentation Model " + "=" * 20)
    
    # Create config for segmentation model
    class SegConfig:
        pass
    
    seg_config = SegConfig()
    seg_config.img_size = img_size
    seg_config.patch_size = patch_size
    seg_config.num_seg_classes = model_args.num_seg_classes
    seg_config.num_decoder_layers = model_args.num_decoder_layers
    seg_config.num_decoder_heads = model_args.num_decoder_heads
    seg_config.mask_dim = model_args.mask_dim
    seg_config.decoder_dropout = model_args.decoder_dropout
    
    model = build_segmentation_model(vision_tower, seg_config)
    
    rank0_print(f"Number of organ classes: {model_args.num_seg_classes}")
    rank0_print(f"Decoder layers: {model_args.num_decoder_layers}")
    rank0_print(f"Mask dimension: {model_args.mask_dim}")
    
    # ==================== Configure Training ====================
    rank0_print("\n" + "=" * 20 + " Configuring Training " + "=" * 20)
    
    # Freeze vision tower if requested
    if model_args.freeze_vision_tower:
        rank0_print("Freezing vision encoder")
        for param in model.vision_tower.parameters():
            param.requires_grad = False
    else:
        rank0_print("Vision encoder is trainable")
    
    # Enable gradient checkpointing if requested
    if training_args.gradient_checkpointing:
        if hasattr(model.vision_tower, 'gradient_checkpointing_enable'):
            model.vision_tower.gradient_checkpointing_enable()
    
    # Print parameter counts
    groups = {
        "Vision Encoder": 0,
        "Transformer Decoder": 0,
        "Mask Projection": 0,
        "UNet Upsampler": 0,
        "Other": 0,
    }
    group_trainable = defaultdict(int)
    group_frozen = defaultdict(int)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        is_trainable = param.requires_grad
        
        if "vision_tower" in name:
            group = "Vision Encoder"
        elif "decoder" in name:
            group = "Transformer Decoder"
        elif "query_proj" in name or "token_proj" in name:
            group = "Mask Projection"
        elif "upsampler" in name:
            group = "UNet Upsampler"
        else:
            group = "Other"
        
        groups[group] += param_count
        if is_trainable:
            group_trainable[group] += param_count
        else:
            group_frozen[group] += param_count
    
    rank0_print("\n" + "=" * 60)
    rank0_print("PARAMETER GROUP SUMMARY (in millions)")
    rank0_print("=" * 60)
    for group in groups:
        total_m = groups[group] / 1e6
        trainable_m = group_trainable[group] / 1e6
        frozen_m = group_frozen[group] / 1e6
        rank0_print(f"{group:25} | Total: {total_m:6.1f}M | Trainable: {trainable_m:6.1f}M | Frozen: {frozen_m:6.1f}M")
    
    total_all = sum(groups.values())
    total_trainable = sum(group_trainable.values())
    total_frozen = sum(group_frozen.values())
    
    rank0_print("-" * 60)
    rank0_print(f"{'Overall TOTAL':25} | Total: {total_all / 1e6:6.1f}M | Trainable: {total_trainable / 1e6:6.1f}M | Frozen: {total_frozen / 1e6:6.1f}M")
    rank0_print("=" * 60)
    
    # ==================== Build Datasets ====================
    rank0_print("\n" + "=" * 20 + " Building Datasets " + "=" * 20)
    
    train_dataset, valid_dataset = build_seg_datasets(
        train_image_root=data_args.train_image_root,
        train_mask_root=data_args.train_mask_root,
        valid_image_root=data_args.valid_image_root,
        valid_mask_root=data_args.valid_mask_root,
        img_size=img_size,
        max_train_samples=data_args.max_train_samples,
        max_valid_samples=data_args.max_valid_samples,
        use_augmentation=data_args.use_augmentation,
    )
    
    data_collator = SegDataCollator()
    
    rank0_print(f"Training samples: {len(train_dataset)}")
    rank0_print(f"Validation samples: {len(valid_dataset)}")
    
    # ==================== Training ====================
    rank0_print("\n" + "=" * 20 + " Starting Training " + "=" * 20)
    
    trainer = SegmentationTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        # compute_metrics=compute_metrics,  # Disabled for memory efficiency
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    # Auto-resume if checkpoints exist
    resume_from_checkpoint = False
    if os.path.exists(training_args.output_dir):
        checkpoint_dirs = [
            d for d in os.listdir(training_args.output_dir)
            if os.path.isdir(os.path.join(training_args.output_dir, d))
            and d.startswith('checkpoint-')
        ]
        if checkpoint_dirs:
            resume_from_checkpoint = True
            rank0_print(f"Found checkpoints, resuming training...")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state()
    
    # ==================== Save Model ====================
    rank0_print("\n" + "=" * 20 + " Saving Model " + "=" * 20)
    
    trainer.save_model(training_args.output_dir)
    rank0_print(f"Model saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()

