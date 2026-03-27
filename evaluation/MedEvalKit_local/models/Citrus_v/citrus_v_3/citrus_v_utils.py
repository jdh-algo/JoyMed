# coding=utf-8
# Copyright 2025 CitrusV Team. All rights reserved.
"""Multimodal utilities for CitrusV 2.5 - medical imaging and video processing."""

import logging
import math
import os
import time
import warnings
from functools import lru_cache
from typing import Any, Dict, List, NewType, Optional, Tuple, Union

import cv2
import imageio
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .utils.m3d_utils import load_m3d_image_folder

logger = logging.getLogger(__name__)

# Type aliases for volume inputs
Path = NewType("Path", str)
VolumeInput = Union[
    str,  # Path to volume file (.nii, .nii.gz, .dcm, .npy) or directory (for DICOM)
    Path,  # Typed path
    List[str],  # List of volume paths
    List[Path],  # List of typed paths
    np.ndarray,  # Pre-loaded volume array in shape (S, C, H, W) or (S, H, W)
    torch.Tensor,  # Pre-loaded volume tensor in shape (S, C, H, W)
    List[np.ndarray],  # List of pre-loaded volume arrays
    List[torch.Tensor],  # List of pre-loaded volume tensors
]

# Constants for volume processing (aligned with video processing)
# VOLUME_MIN_TOKEN_NUM = 128
VOLUME_MAX_TOKEN_NUM = 768
VOLUME_MIN_TOKEN_NUM = 256 #  8 * 16 * 2 = 256
# VOLUME_MAX_TOKEN_NUM = 256 # max width ~= 384 256 -- 128
SPATIAL_MERGE_SIZE = 2
FRAME_FACTOR = 6 if os.environ.get("prompt_version","v1") == "3continue" else 2
MAX_FRAMES = 192 if os.environ.get("prompt_version","v1") == "3continue" else 32
MODEL_SEQ_LEN = int(float(os.environ.get('MODEL_SEQ_LEN', 128000)))


def slice_nifti_to_images(
    nifti_path: str,
    num_slices: int = 180,
    axis: str = 'axial'
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
    
    axis_map = {'axial': 2, 'coronal': 1, 'sagittal': 0}
    if axis not in axis_map:
        raise ValueError(f"Axis must be one of {list(axis_map.keys())}, got {axis}")
    
    slice_axis = axis_map[axis]
    
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
    nifti_slice_axis: str = 'axial',
    nifti_num_slices: int = 180,
) -> List[Image.Image]:
    """
    Load images from various sources including NIfTI files.
    
    Args:
        image_path: Path(s) to image file(s), directory, or PIL Image(s)
        nifti_slice_axis: Axis for NIfTI slicing
        nifti_num_slices: Number of slices to extract from NIfTI
    
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
        return slice_nifti_to_images(image_path, nifti_num_slices, nifti_slice_axis)
    
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


def load_video(
    video_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, List[float]]:
    """
    Load video frames with optional temporal sampling.
    
    Args:
        video_path: Path to video file or directory of frames
        start_time: Start time in seconds
        end_time: End time in seconds
        fps: Target frames per second
        max_frames: Maximum number of frames to extract
    
    Returns:
        Tuple of (frames array, timestamps list)
    """
    # Handle directory of frames
    if os.path.isdir(video_path):
        frame_files = sorted(os.listdir(video_path))
        vid_fps = 1
        num_frames_of_video = len(frame_files)
    
    # Handle GIF
    elif video_path.endswith('.gif'):
        gif_reader = imageio.get_reader(video_path)
        vid_fps = 25
        num_frames_of_video = len(gif_reader)
    
    # Handle video file
    else:
        vreader = VideoReader(video_path, ctx=cpu(0), num_threads=64)
        vid_fps = vreader.get_avg_fps()
        num_frames_of_video = len(vreader)
    
    # Calculate frame range
    f_start = 0 if start_time is None else max(int(start_time * vid_fps) - 1, 0)
    f_end = num_frames_of_video - 1 if end_time is None else min(int(end_time * vid_fps) - 1, num_frames_of_video - 1)
    frame_indices = list(range(f_start, f_end + 1))
    
    # Sample frames
    duration = len(frame_indices)
    if max_frames is not None and duration > max_frames:
        sampled_indices = np.linspace(0, duration - 1, max_frames, dtype=int)
        frame_indices = [frame_indices[i] for i in sampled_indices]
    elif fps is not None:
        segment_len = max(1, int(vid_fps / fps))
        sampled_indices = list(range(0, duration, segment_len))
        frame_indices = [frame_indices[i] for i in sampled_indices]
    
    # Load frames
    if os.path.isdir(video_path):
        frames = []
        for frame_idx in frame_indices:
            filepath = os.path.join(video_path, frame_files[frame_idx])
            try:
                with Image.open(filepath).convert('RGB') as img:
                    frames.append(np.array(img))
            except Exception as e:
                warnings.warn(f"Error loading frame {filepath}: {e}")
        frames = np.array(frames) if frames else np.zeros((0, 224, 224, 3), dtype=np.uint8)
    
    elif video_path.endswith('.gif'):
        gif_reader = imageio.get_reader(video_path)
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB) for idx, frame in enumerate(gif_reader) if idx in frame_indices]
        frames = np.array(frames) if frames else np.zeros((0, 224, 224, 3), dtype=np.uint8)
    
    else:
        frames = vreader.get_batch(frame_indices).asnumpy()
    
    # Generate timestamps
    timestamps = [idx / vid_fps for idx in frame_indices]
    
    return frames, timestamps



def expand2square(pil_img: Image.Image, background_color: Tuple[int, int, int]) -> Image.Image:
    """Expand image to square by padding."""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


# ====== Volume Processing Functions (for 3D medical images) ======

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, 
    width: int, 
    factor: int, 
    min_pixels: Optional[int] = None, 
    max_pixels: Optional[int] = None
) -> Tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:
    
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    max_pixels = max_pixels if max_pixels is not None else (VOLUME_MAX_TOKEN_NUM * factor ** 2)
    min_pixels = min_pixels if min_pixels is not None else (VOLUME_MIN_TOKEN_NUM * factor ** 2)
    assert max_pixels >= min_pixels, "The max_pixels must be greater than or equal to min_pixels."
    
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    
    return h_bar, w_bar


def smart_nslices(
    ele: Dict[str, Any],
    total_slices: int,
) -> int:
    """
    Calculate the number of slices for 3D volume used for model inputs.
    
    Args:
        ele (dict): Configuration dict for the volume.
            support either `nslices` or use smart sampling:
                - min_slices: the minimum number of slices.
                - max_slices: the maximum number of slices.
        total_slices (int): The original total number of slices in the volume.
    
    Returns:
        int: The number of slices for volume used for model inputs.
    """
    if total_slices < FRAME_FACTOR:
        return FRAME_FACTOR

    min_slices = ceil_by_factor(ele.get("min_slices", max(FRAME_FACTOR, total_slices // 10)), FRAME_FACTOR)
    max_slices = floor_by_factor(ele.get("max_slices", min(MAX_FRAMES, total_slices)), FRAME_FACTOR)
    # Use a sampling strategy similar to video frames
    nslices = min(max(total_slices, min_slices), max_slices)
    nslices = floor_by_factor(nslices, FRAME_FACTOR)
    
    if not (FRAME_FACTOR <= nslices and nslices <= total_slices):
        raise ValueError(f"nslices should be in interval [{FRAME_FACTOR}, {total_slices}], but got {nslices}.")
    
    print(f"=======nslices: {nslices}")
    
    return nslices


def _load_nifti(volume_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load NIfTI file (.nii or .nii.gz).
    
    Returns:
        tuple: (volume_data, metadata)
            - volume_data: numpy array of shape (D, H, W) or (D, H, W, C)
            - metadata: dict with spacing, orientation info
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required for NIfTI support. Install: pip install nibabel")
    
    st = time.time()
    nifti_img = nib.load(volume_path)
    volume_data = nifti_img.get_fdata()

    # **
    # volume_data = np.transpose(volume_data, (2, 0, 1))
    
    # Get metadata
    header = nifti_img.header
    spacing = header.get_zooms()[:3] if hasattr(header, 'get_zooms') else (1.0, 1.0, 1.0)
    
    metadata = {
        'spacing': spacing,
        'shape': volume_data.shape,
        'dtype': str(volume_data.dtype),
    }
    
    logger.info(f"NIfTI loaded: {volume_path=}, shape={volume_data.shape}, spacing={spacing}, time={time.time() - st:.3f}s")
    
    return volume_data, metadata


def _load_dicom(volume_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load DICOM series from directory or single DICOM file.
    
    Returns:
        tuple: (volume_data, metadata)
    """
    try:
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_modality_lut
    except ImportError:
        raise ImportError("pydicom is required for DICOM support. Install: pip install pydicom")
    
    st = time.time()
    
    if os.path.isdir(volume_path):
        # Load DICOM series
        dicom_files = []
        for f in os.listdir(volume_path):
            if f.endswith('.dcm'):
                dicom_files.append(os.path.join(volume_path, f))
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in directory: {volume_path}")
        
        # Sort by instance number or slice location
        dicoms = [pydicom.dcmread(f) for f in dicom_files]
        dicoms.sort(key=lambda x: float(x.InstanceNumber) if hasattr(x, 'InstanceNumber') else 0)
        
        # Stack slices
        slices = [apply_modality_lut(dcm.pixel_array, dcm) for dcm in dicoms]
        volume_data = np.stack(slices, axis=0)
        
        # Get metadata from first slice
        dcm = dicoms[0]
        spacing = (
            float(dcm.SliceThickness) if hasattr(dcm, 'SliceThickness') else 1.0,
            float(dcm.PixelSpacing[0]) if hasattr(dcm, 'PixelSpacing') else 1.0,
            float(dcm.PixelSpacing[1]) if hasattr(dcm, 'PixelSpacing') else 1.0,
        )
    else:
        # Load single DICOM file
        dcm = pydicom.dcmread(volume_path)
        volume_data = apply_modality_lut(dcm.pixel_array, dcm)
        if volume_data.ndim == 2:
            volume_data = volume_data[np.newaxis, ...]  # Add depth dimension
        spacing = (1.0, 1.0, 1.0)
    
    metadata = {
        'spacing': spacing,
        'shape': volume_data.shape,
        'dtype': str(volume_data.dtype),
    }
    
    logger.info(f"DICOM loaded: {volume_path=}, shape={volume_data.shape}, time={time.time() - st:.3f}s")
    
    return volume_data, metadata


def _natural_key(s: str):
    """Key function for natural sorting (e.g., img2 < img10)."""
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]



def _load_image_folder(volume_dir: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a folder containing 2D slice images (png/jpg/etc.) and stack to a 3D volume.
    Sorting uses natural order to maintain slice sequence.
    """
    if "M3D" in volume_dir:
        return load_m3d_image_folder(volume_dir)
    
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    files = [f for f in os.listdir(volume_dir) if os.path.isfile(os.path.join(volume_dir, f))]
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_exts]
    if not image_files:
        raise ValueError(f"No image files found in directory: {volume_dir}")
    image_files.sort(key=_natural_key)
    slices = []
    for fname in image_files:
        path = os.path.join(volume_dir, fname)
        try:
            with Image.open(path) as img:
                # use grayscale for volume; RGB handled later if needed
                arr = np.array(img.convert('L'))
                slices.append(arr)
        except Exception as e:
            warnings.warn(f"Error loading slice image {path}: {e}")
    if not slices:
        raise ValueError(f"Failed to load any slice images from {volume_dir}")
    volume_data = np.stack(slices, axis=0)
    volume_data = np.transpose(volume_data, (1, 2, 0))
    
    metadata = {
        'spacing': (1.0, 1.0, 1.0),
        'shape': volume_data.shape,
        'dtype': str(volume_data.dtype),
    }
    logger.info(f"Image-folder volume loaded: volume_path='{volume_dir}', shape={volume_data.shape}")
    return volume_data, metadata


def _load_npy(volume_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load numpy array (.npy).
    
    Returns:
        tuple: (volume_data, metadata)
    """
    st = time.time()
    volume_data = np.load(volume_path)
    
    if volume_data.ndim < 3:
        raise ValueError(f"Expected 3D volume, got shape {volume_data.shape}")
    
    metadata = {
        'spacing': (1.0, 1.0, 1.0),
        'shape': volume_data.shape,
        'dtype': str(volume_data.dtype),
    }
    
    logger.info(f"NPY loaded: {volume_path=}, shape={volume_data.shape}, time={time.time() - st:.3f}s")
    
    return volume_data, metadata


def _normalize_volume(volume: np.ndarray) -> np.ndarray:
    """
    Normalize volume to [0, 1] range.
    
    Args:
        volume: Input volume array
    
    Returns:
        Normalized volume as float32
    """
    # Handle different intensity ranges
    v_min, v_max = volume.min(), volume.max()
    
    if v_max > v_min:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        volume = np.zeros_like(volume)
    
    return volume.astype(np.float32)


def fetch_volume(
    ele: Dict[str, Any], 
    image_patch_size: int = 16, 
    return_volume_sample_fps: bool = True,
    return_volume_metadata: bool = True, 
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    Fetch and preprocess 3D medical volume (NIfTI, DICOM, NPY).
    
    This function is analogous to fetch_video but handles 3D medical imaging data.
    It loads the volume, samples slices, normalizes intensities, and resizes spatial dimensions.
    
    Args:
        ele (dict): Configuration dict containing:
            - volume: path to volume file (.nii, .nii.gz, .dcm, .npy) or directory (for DICOM)
            - slice_axis: axis along which to extract slices (0, 1, or 2), default 0
            - nslices: number of slices to extract (optional)
            - min_slices: minimum number of slices (optional)
            - max_slices: maximum number of slices (optional)
            - min_pixels: minimum pixels per slice (optional)
            - max_pixels: maximum pixels per slice (optional)
            - resized_height: target height (optional)
            - resized_width: target width (optional)
        image_patch_size (int): Patch size for the vision encoder (default 14)
        return_volume_sample_fps (bool): Whether to return sampling info
        return_volume_metadata (bool): Whether to return volume metadata
    
    Returns:
        torch.Tensor: Volume tensor of shape (D, C, H, W) where D is number of slices
        or tuple of (tensor, metadata, sample_fps) if return flags are True
    
    Example:
        >>> ele = {'volume': 'path/to/scan.nii.gz', 'nslices': 64}
        >>> volume_tensor = fetch_volume(ele, image_patch_size=14)
        >>> print(volume_tensor.shape)  # (64, 3, 448, 448)
    """
    image_factor = image_patch_size * SPATIAL_MERGE_SIZE
    VOLUME_SLICE_MIN_PIXELS = VOLUME_MIN_TOKEN_NUM * image_factor * image_factor
    VOLUME_SLICE_MAX_PIXELS = VOLUME_MAX_TOKEN_NUM * image_factor * image_factor
    
    volume_path = ele.get("volume", ele.get("image_3d"))
    if volume_path is None:
        raise ValueError("ele must contain 'volume' or 'image_3d' key")
    
    # Load volume based on file extension or directory contents
    if isinstance(volume_path, str):
        if volume_path.endswith(('.nii', '.nii.gz')):
            volume_data, file_metadata = _load_nifti(volume_path)
        elif volume_path.endswith('.npy'):
            volume_data, file_metadata = _load_npy(volume_path)
        elif volume_path.endswith('.dcm'):
            volume_data, file_metadata = _load_dicom(volume_path)
        elif os.path.isdir(volume_path):
            # Directory could be a DICOM series or a folder of images
            exts = [os.path.splitext(f)[1].lower() for f in os.listdir(volume_path) if os.path.isfile(os.path.join(volume_path, f))]
            if any(ext == '.dcm' for ext in exts):
                volume_data, file_metadata = _load_dicom(volume_path)
            elif any(ext in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'} for ext in exts):
                volume_data, file_metadata = _load_image_folder(volume_path)
            else:
                raise ValueError(f"Unsupported directory contents for volume: {volume_path}")
        else:
            # Single 2D image as a 1-slice volume
            if os.path.isfile(volume_path) and os.path.splitext(volume_path)[1].lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}:
                try:
                    with Image.open(volume_path) as img:
                        arr = np.array(img.convert('L'))
                    volume_data = arr[None, ...]
                    file_metadata = {
                        'spacing': (1.0, 1.0, 1.0),
                        'shape': volume_data.shape,
                        'dtype': str(volume_data.dtype),
                    }
                except Exception as e:
                    raise ValueError(f"Failed to load image as volume: {volume_path}, error: {e}")
            else:
                raise ValueError(f"Unsupported volume format: {volume_path}")
    else:
        raise ValueError(f"volume_path must be a string, got {type(volume_path)}")
    
    # Normalize volume to 0 ~ 1
    volume_data = _normalize_volume(volume_data)
    
    # Select slice axis (default: axial = 2)
    slice_axis = ele.get("slice_axis", 2)
    if slice_axis not in [0, 1, 2]:
        raise ValueError(f"slice_axis must be 0, 1, or 2, got {slice_axis}")
    
    # Transpose to make slice_axis the first dimension
    if slice_axis == 1:
        volume_data = np.transpose(volume_data, (1, 0, 2))
    elif slice_axis == 2:
        volume_data = np.transpose(volume_data, (2, 0, 1))
    
    # Determine number of slices to sample, and sample slice indices uniformly
    total_slices, height, width = volume_data.shape[:3]
    nslices = smart_nslices(ele, total_slices=total_slices)
    idx = torch.linspace(0, total_slices - 1, nslices).round().long()
    sample_fps = nslices / max(total_slices, 1e-6)  # Analogous to video fps
    slices = volume_data[idx.numpy()]  # Shape: (D, H, W) or (D, H, W, C)

    if os.environ.get("prompt_version","v1") == "3continue":
        if slices.ndim == 3:
            slices = slices.reshape(-1, 3, *slices.shape[1:]).swapaxes(1, -1)
        elif slices.ndim == 4:
            slices = slice[..., 0]
            slices = slices.reshape(-1, 3, *slices.shape[1:]).swapaxes(1, -1)
        total_slices = slices.shape[0]
    else:
    # Convert grayscale to RGB if needed
        if slices.ndim == 3:
            slices = np.stack([slices] * 3, axis=-1) # Grayscale: (D, H, W) -> (D, H, W, 3)
        elif slices.ndim == 4 and slices.shape[-1] == 1:
            slices = np.concatenate([slices] * 3, axis=-1) # Single channel: (D, H, W, 1) -> (D, H, W, 3)
    
    volume = torch.from_numpy(slices).permute(0, 3, 1, 2).float()
    
    # Resize spatial dimensions
    min_pixels = ele.get("min_pixels", VOLUME_SLICE_MIN_PIXELS)
    total_pixels = ele.get("total_pixels", MODEL_SEQ_LEN * image_factor * image_factor * 0.9)
    max_pixels = max(min(VOLUME_SLICE_MAX_PIXELS, total_pixels / nslices * FRAME_FACTOR), int(min_pixels * 1.05))
    max_pixels_supposed = ele.get("max_pixels", max_pixels)
    if max_pixels_supposed > max_pixels:
        logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
    max_pixels = min(max_pixels_supposed, max_pixels)
    
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    
    volume = transforms.functional.resize(
        volume,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    
    resized_width, resized_height = volume.shape[2], volume.shape[1]

    # Prepare metadata using VolumeMetadata dataclass
    if return_volume_metadata:
        from .volume_processing_citrus_v_3 import VolumeMetadata
        
        volume_metadata = VolumeMetadata(
            total_num_slices=total_slices,
            slice_axis=slice_axis,
            spacing=file_metadata.get('spacing'),
            width=resized_width,
            height=resized_height,
            original_shape=file_metadata.get('shape'),
            slice_indices=idx.tolist(),
        )
        final_volume = (volume, volume_metadata)
    else:
        final_volume = volume
    
    if return_volume_sample_fps:
        return final_volume, sample_fps
    
    return final_volume


def is_valid_volume(volume):
    """
    Check if input is a valid volume tensor/array.
    
    A valid volume is a 4D tensor/array in shape (S, C, H, W) where:
    - S: number of slices
    - C: number of channels (1 or 3)
    - H: height
    - W: width
    """
    if isinstance(volume, (np.ndarray, torch.Tensor)):
        return volume.ndim == 4
    return False


def make_batched_volumes(volumes: VolumeInput) -> List[Union[str, torch.Tensor, np.ndarray]]:
    """
    Ensure that the input is a list of volumes. If the input is a single volume, it is converted to a list of length 1.
    If the input is a batch of volumes, it is converted to a list of 4D volume arrays.
    
    Similar to make_batched_videos, but for 3D medical volumes. Recursively flattens any nested structure.
    
    Args:
        volumes: Volume inputs to turn into a list of volumes.
    
    Returns:
        List of volumes (paths or tensors/arrays)
    """
    # Early exit for deeply nested list of volume paths. We shouldn't flatten them
    try:
        if isinstance(volumes[0][0], list) and isinstance(volumes[0][0][0], str):
            return [volume_paths for sublist in volumes for volume_paths in sublist]
    except (IndexError, TypeError):
        pass
    
    # Single volume (string path or tensor/array)
    if isinstance(volumes, str) or is_valid_volume(volumes):
        return [volumes]
    
    # Not a list - invalid input
    if not isinstance(volumes, list):
        raise ValueError(
            f"Invalid volume input. Expected either a string path, tensor/array, or list of volumes, "
            f"but got type {type(volumes)}."
        )
    
    # Recursively flatten any nested structure
    flat_volumes_list = []
    for item in volumes:
        if isinstance(item, str) or is_valid_volume(item):
            flat_volumes_list.append(item)
        elif isinstance(item, list) and item:
            flat_volumes_list.extend(make_batched_volumes(item))
    
    return flat_volumes_list


def get_volume_size(volume: Union[np.ndarray, torch.Tensor]) -> Tuple[int, int]:
    """
    Returns the (height, width) dimensions of the volume.
    
    Args:
        volume: The volume tensor/array in shape (S, C, H, W)
    
    Returns:
        A tuple of the volume's height and width.
    """
    if volume.ndim != 4:
        raise ValueError(f"Expected 4D volume, got {volume.ndim}D")
    return volume.shape[2], volume.shape[3]


def make_batched_metadata(
    volumes: VolumeInput, 
    volume_metadata: Optional[Union["VolumeMetadata", List["VolumeMetadata"], Dict]] = None
) -> List["VolumeMetadata"]:
    """
    Create batched metadata from volumes and optional metadata.
    
    Similar to make_batched_metadata for videos, but for 3D medical volumes.
    
    Args:
        volumes: Volume inputs (will be converted to list if needed)
        volume_metadata: Optional metadata for volumes (VolumeMetadata, list, or dict)
    
    Returns:
        List of VolumeMetadata instances
    """
    from .volume_processing_citrus_v_3 import VolumeMetadata
    
    # Ensure volumes is a list
    if not isinstance(volumes, list):
        volumes = [volumes]
    
    if volume_metadata is None:
        # Create default metadata and fill attributes we can infer from given volume
        volume_metadata = [
            {
                "total_num_slices": len(volume),
                "slice_axis": 0,
                "spacing": None,
                "slice_indices": list(range(len(volume))),
                "height": get_volume_size(volume)[0] if is_valid_volume(volume) else None,
                "width": get_volume_size(volume)[1] if is_valid_volume(volume) else None,
                "original_shape": volume.shape if is_valid_volume(volume) else None,
            }
            for volume in volumes
        ]
    
    if isinstance(volume_metadata, list):
        # Flatten if nested list
        if volume_metadata and isinstance(volume_metadata[0], list):
            volume_metadata = [
                VolumeMetadata(**metadata) for metadata_list in volume_metadata for metadata in metadata_list
            ]
        # Simply wrap in VolumeMetadata if simple dict
        elif volume_metadata and isinstance(volume_metadata[0], dict):
            volume_metadata = [VolumeMetadata(**metadata) for metadata in volume_metadata]
    else:
        # Create a batched list from single object
        if isinstance(volume_metadata, dict):
            volume_metadata = [VolumeMetadata(**volume_metadata)]
        elif isinstance(volume_metadata, VolumeMetadata):
            volume_metadata = [volume_metadata]
        else:
            raise ValueError(f"Invalid volume_metadata type: {type(volume_metadata)}")
    
    return volume_metadata