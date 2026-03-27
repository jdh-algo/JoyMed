from PIL import Image
import os
from transformers.image_utils import is_valid_image
import warnings
from typing import Any, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
import torch

def is_valid_video(video) -> bool:
    if isinstance(video, (list, tuple)):
        return all(is_valid_image(frame) for frame in video)
    elif isinstance(video, np.ndarray):
        return video.ndim == 4
    elif isinstance(video, torch.Tensor):
        return video.ndim == 4
    return False

def slice_nifti_to_images(
    nifti_path: str,
    num_slices: int = 180,
    slice_axis: int = 2
) -> List[Image.Image]:
    """
    Convert NIfTI 3D medical image to 2D slices.
    
    Args:
        nifti_path: Path to .nii or .nii.gz file
        num_slices: Number of slices to extract
        axis: Slicing axis - 'axial', 'coronal', or 'sagittal'
    
    Returns:
        List of PIL Images
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "nibabel is required for NIfTI support. "
            "Install it with: pip install nibabel"
        )
    
    # axis_map = {'axial': 2, 'coronal': 1, 'sagittal': 0}
    # if axis not in axis_map:
    #     raise ValueError(f"Axis must be one of {list(axis_map.keys())}, got {axis}")
    
    # slice_axis = axis_map[axis]
    
    try:
        nifti_file = nib.load(nifti_path)
        image_data = nifti_file.get_fdata()
        
        num_total_slices = image_data.shape[slice_axis]
        sampled_indices = np.linspace(0, num_total_slices - 1, num_slices, dtype=int)
        
        images = []
        for slice_index in sampled_indices:
            # Extract 2D slice
            if slice_axis == 0:
                slice_2d = image_data[slice_index, :, :]
            elif slice_axis == 1:
                slice_2d = image_data[:, slice_index, :]
            else:  # slice_axis == 2
                slice_2d = image_data[:, :, slice_index]
            
            # Normalize to 0-255
            if slice_2d.max() > slice_2d.min():
                slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min()) * 255.0
            
            slice_2d = slice_2d.astype(np.uint8)
            pil_image = Image.fromarray(slice_2d).convert('RGB')
            images.append(pil_image)
        
        return images
    
    except Exception as e:
        warnings.warn(f"Error loading NIfTI file {nifti_path}: {e}")
        return []

def load_images(
    image_path: Union[str, List[str], Image.Image, List[Image.Image]],
    nii_axis: int = 2,
    nii_num_slices: int = 180,
) -> List[Image.Image]:
    """
    Load images from various sources including NIfTI files.
    
    Args:
        image_path: Path(s) to image file(s), directory, or PIL Image(s)
        nifti_slice_axis: Axis for NIfTI slicing
        nii_num_slices: Number of slices to extract from NIfTI
    
    Returns:
        List of PIL Images
    """
    images = []
    
    def safe_open(f):
        try:
            with Image.open(f).convert('RGB') as img:
                return img.copy()
        except Exception:
            return None
    
    # Handle NIfTI files
    if isinstance(image_path, str) and (image_path.endswith('.nii') or image_path.endswith('.nii.gz')):
        return slice_nifti_to_images(image_path, nii_num_slices, nii_axis)
    
    # Handle single file
    elif isinstance(image_path, str) and os.path.isfile(image_path):
        img = safe_open(image_path)
        if img is not None:
            images.append(img)
    
    # Handle directory
    elif isinstance(image_path, str) and os.path.isdir(image_path):
        for f in sorted(os.listdir(image_path)):
            full_path = os.path.join(image_path, f)
            if os.path.isfile(full_path):
                img = safe_open(full_path)
                if img is not None:
                    images.append(img)
    
    # Handle list of paths
    elif isinstance(image_path, list) and isinstance(image_path[0], str):
        for f in image_path:
            if f.endswith('.nii') or f.endswith('.nii.gz'):
                images.extend(slice_nifti_to_images(f, nifti_num_slices, nifti_slice_axis))
            else:
                img = safe_open(f)
                if img is not None:
                    images.append(img)
    
    # Handle PIL Images
    elif isinstance(image_path, list) and isinstance(image_path[0], Image.Image):
        images = [img.convert('RGB') for img in image_path]
    
    elif isinstance(image_path, Image.Image):
        images = [image_path.convert('RGB')]
    
    else:
        raise ValueError(f"Unsupported image path type: {type(image_path)}")
    
    return images
    
def resize_video(video_inputs):
    for video_id, video_input in enumerate(video_inputs):
        size_list = []
        for image in video_input:
            size = (image.width, image.height)
            size_list.append(size)
        size_set = set(size_list)
        if len(size_set)<=1:
            continue

        #寻找出现最多次的size
        size_info = {str(id):size for id, size in enumerate(size_set)}
        size_count_dict = {}
        for id_str in size_info.keys():
            size = size_info[id_str]
            size_count = size_list.count(size)
            size_count_dict[id_str] = size_count
        
        max_id_str = sorted(size_count_dict.items(), key=lambda x: -x[1])[0][0]
        best_size = size_info[max_id_str]

        #将所有image resize 到 best_size
        for image_id, image in enumerate(video_input):
            size = (image.width, image.height)
            if size != best_size:
                video_input[image_id] = image.resize(best_size)
        
        video_inputs[video_id] = video_input
    return video_inputs
    