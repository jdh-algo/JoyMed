"""
Segmentation Dataset for CT-RATE data.

Loads CT images and corresponding segmentation masks from the CT-RATE dataset.
Follows the same loading logic as CTRATEDataset_w_Seg in multi_dataset.py for consistency.
"""

import os
import glob
import random
from typing import Tuple, Optional, List, Dict, Any

import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from monai.transforms import Resize
from monai.data import set_track_meta
import monai.transforms as mtf


class CTSegDataset(Dataset):
    """
    Dataset for CT segmentation with 117 organ classes.
    
    Folder structure expected:
    - Images: {image_root}/train_X/train_X_a/train_X_a_Y.nii.gz
    - Masks: {mask_root}/train_X/train_X_a/train_X_a_Y.nii.gz
    
    Mask values: 0 = background, 1-117 = organ classes
    
    Image loading follows the same logic as CTRATEDataset_w_Seg in multi_dataset.py:
    1. Load with nibabel
    2. Normalize: (image - (-1000)) / 2000, clip to [0, 1]
    3. Add channel dim and transpose to (C, D, H, W)
    4. Resize with MONAI, convert back to numpy
    5. Apply transforms (same as CTRATEDataset_w_Seg)
    """
    
    def __init__(
        self,
        image_root: str,
        mask_root: str,
        img_size: Tuple[int, int, int] = (128, 256, 256),
        mode: str = 'train',
        max_samples: Optional[int] = None,
        use_augmentation: bool = True,
    ):
        """
        Args:
            image_root: Root directory containing CT images
            mask_root: Root directory containing segmentation masks
            img_size: Target size for resizing (D, H, W)
            mode: 'train' or 'valid'
            max_samples: Maximum number of samples to use (for debugging)
            use_augmentation: Whether to apply data augmentation (only for train)
        """
        super().__init__()
        self.image_root = image_root
        self.mask_root = mask_root
        self.img_size = img_size
        self.mode = mode
        self.use_augmentation = use_augmentation and (mode == 'train')
        
        # Collect all sample paths
        self.samples = self._collect_samples()
        
        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]
        
        print(f"[CTSegDataset] Loaded {len(self.samples)} {mode} samples")
        
        # Build resizers (same as CTRATEDataset_w_Seg)
        self.image_resizer = Resize(spatial_size=self.img_size)
        self.mask_resizer = Resize(spatial_size=self.img_size, mode="nearest")
        
        # Disable MONAI metadata tracking (same as CTRATEDataset_w_Seg)
        set_track_meta(False)
        
        # Build transforms (exactly same as CTRATEDataset_w_Seg)
        # Note: Using keys "A" for image and "B" for mask to match original
        if self.use_augmentation:
            self.transform = mtf.Compose([
                # spatial transforms applied to both A and B
                mtf.RandRotate90d(keys=["A", "B"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["A", "B"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["A", "B"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["A", "B"], prob=0.10, spatial_axis=2),
                # intensity transforms: only on the image (A)
                mtf.RandScaleIntensityd(keys=["A"], factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys=["A"], offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["A"], dtype=torch.float32),
                mtf.ToTensord(keys=["B"], dtype=torch.int8),
            ])
        else:
            self.transform = mtf.Compose([
                mtf.ToTensord(keys=["A"], dtype=torch.float32),
                mtf.ToTensord(keys=["B"], dtype=torch.int8),
            ])
        
    def _collect_samples(self) -> List[Dict[str, str]]:
        """
        Collect all image-mask pairs.
        """
        samples = []
        
        # Get all subdirectories in image_root
        if not os.path.exists(self.image_root):
            raise ValueError(f"Image root does not exist: {self.image_root}")
        if not os.path.exists(self.mask_root):
            raise ValueError(f"Mask root does not exist: {self.mask_root}")
        
        # Pattern: train_X/train_X_a/train_X_a_Y.nii.gz or valid_X/valid_X_a/valid_X_a_Y.nii.gz
        prefix = "train" if "train" in os.path.basename(self.image_root) else "valid"
        
        # Find all nii.gz files in image root
        image_files = glob.glob(
            os.path.join(self.image_root, f"{prefix}_*", f"{prefix}_*", "*.nii.gz"),
            recursive=True
        )
        
        for img_path in image_files:
            # Construct corresponding mask path
            # Replace image_root with mask_root in the path
            rel_path = os.path.relpath(img_path, self.image_root)
            mask_path = os.path.join(self.mask_root, rel_path)
            
            if os.path.exists(mask_path):
                samples.append({
                    'image_path': img_path,
                    'mask_path': mask_path,
                })
        
        # Sort for reproducibility
        samples = sorted(samples, key=lambda x: x['image_path'])
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and preprocess a single sample.
        Following exactly the same logic as CTRATEDataset_w_Seg._get_ct_rate_item()
        
        Returns:
            dict with:
                - 'image': [1, D, H, W] tensor, float32
                - 'mask': [D, H, W] tensor, long, values 0-117
        """
        max_attempts = 100
        original_idx = idx
        
        for attempt in range(max_attempts):
            try:
                sample = self.samples[idx]
                image_abs_path = sample['image_path']
                seg_abs_path = sample['mask_path']
                
                # Load with nibabel (same as CTRATEDataset_w_Seg)
                image = nib.load(image_abs_path).get_fdata()
                seg = nib.load(seg_abs_path).get_fdata()
                
                # Check for empty image
                if np.max(image) == 0:
                    raise ValueError("Empty image (all zeros)")
                
                # Normalize: map [-1000, 1000] HU to [0, 1] (same as CTRATEDataset_w_Seg)
                image = (image - (-1000)) / 2000
                image[image > 1.0] = 1.0
                image[image < 0.0] = 0.0
                
                # Add channel dimension: (H, W, D) -> (1, H, W, D)
                image = np.expand_dims(image, 0)
                seg = np.expand_dims(seg, 0)
                
                # Transpose: (1, H, W, D) -> (1, D, H, W) (same as CTRATEDataset_w_Seg)
                image = np.transpose(image, [0, 3, 1, 2])
                seg = np.transpose(seg, [0, 3, 1, 2])
                
                # Resize and convert back to numpy (same as CTRATEDataset_w_Seg)
                image = self.image_resizer(image).detach().cpu().numpy()
                seg = self.mask_resizer(seg).detach().cpu().numpy()
                
                # Create sample dict with keys "A" and "B" (same as CTRATEDataset_w_Seg)
                sample_dict = {
                    "A": image,  # numpy array
                    "B": seg,    # numpy array
                }
                
                # Apply transforms (augmentation + tensor conversion)
                out = self.transform(sample_dict)
                image = out["A"]  # [1, D, H, W], float32
                seg = out["B"]    # [1, D, H, W], int8
                
                # Squeeze mask to [D, H, W] and convert to long for cross-entropy
                mask = seg.squeeze(0).long()
                
                # Clamp mask values to valid range [0, 117]
                mask = torch.clamp(mask, 0, 117)
                
                return {
                    'image': image,
                    'mask': mask,
                }
                
            except Exception as e:
                print(f"[CTSegDataset] Error loading sample {idx}: {e}, trying next...")
                idx = random.randint(0, len(self.samples) - 1)
        
        raise RuntimeError(f"Failed to load sample after {max_attempts} attempts")


class SegDataCollator:
    """
    Data collator for segmentation dataset.
    Simply stacks images and masks.
    """
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        images = torch.stack([b['image'] for b in batch], dim=0)
        masks = torch.stack([b['mask'] for b in batch], dim=0)
        
        return {
            'images': images,
            'masks': masks,
        }


def build_seg_datasets(
    train_image_root: str,
    train_mask_root: str,
    valid_image_root: str,
    valid_mask_root: str,
    img_size: Tuple[int, int, int] = (128, 256, 256),
    max_train_samples: Optional[int] = None,
    max_valid_samples: Optional[int] = None,
    use_augmentation: bool = True,
) -> Tuple[CTSegDataset, CTSegDataset]:
    """
    Build training and validation datasets.
    
    Args:
        train_image_root: Path to training images
        train_mask_root: Path to training masks
        valid_image_root: Path to validation images
        valid_mask_root: Path to validation masks
        img_size: Target size for resizing
        max_train_samples: Max training samples (for debugging)
        max_valid_samples: Max validation samples (for debugging)
        use_augmentation: Whether to use data augmentation
        
    Returns:
        train_dataset, valid_dataset
    """
    train_dataset = CTSegDataset(
        image_root=train_image_root,
        mask_root=train_mask_root,
        img_size=img_size,
        mode='train',
        max_samples=max_train_samples,
        use_augmentation=use_augmentation,
    )
    
    valid_dataset = CTSegDataset(
        image_root=valid_image_root,
        mask_root=valid_mask_root,
        img_size=img_size,
        mode='valid',
        max_samples=max_valid_samples,
        use_augmentation=False,  # Never augment validation
    )
    
    return train_dataset, valid_dataset
