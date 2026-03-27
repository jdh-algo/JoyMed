# coding=utf-8
# Copyright 2025 VitLamMA Team. All rights reserved.
"""Volume utilities for VitLamMA - medical imaging and volume processing."""

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
VOLUME_MIN_TOKEN_NUM = 64
VOLUME_MAX_TOKEN_NUM = 256
SPATIAL_MERGE_SIZE = 8
FRAME_FACTOR = 2 * SPATIAL_MERGE_SIZE
MAX_FRAMES = 128
MODEL_SEQ_LEN = int(float(os.environ.get('MODEL_SEQ_LEN', 128000)))


def load_video(
    video_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, List[float]]:
    """
    Load video frames with optional temporal sampling.
    """
    if os.path.isdir(video_path):
        frame_files = sorted(os.listdir(video_path))
        vid_fps = 1
        num_frames_of_video = len(frame_files)
    elif video_path.endswith('.gif'):
        gif_reader = imageio.get_reader(video_path)
        vid_fps = 25
        num_frames_of_video = len(gif_reader)
    else:
        vreader = VideoReader(video_path, ctx=cpu(0), num_threads=64)
        vid_fps = vreader.get_avg_fps()
        num_frames_of_video = len(vreader)

    f_start = 0 if start_time is None else max(int(start_time * vid_fps) - 1, 0)
    f_end = num_frames_of_video - 1 if end_time is None else min(int(end_time * vid_fps) - 1, num_frames_of_video - 1)
    frame_indices = list(range(f_start, f_end + 1))

    duration = len(frame_indices)
    if max_frames is not None and duration > max_frames:
        sampled_indices = np.linspace(0, duration - 1, max_frames, dtype=int)
        frame_indices = [frame_indices[i] for i in sampled_indices]
    elif fps is not None:
        segment_len = max(1, int(vid_fps / fps))
        sampled_indices = list(range(0, duration, segment_len))
        frame_indices = [frame_indices[i] for i in sampled_indices]

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
    """Calculate the number of slices for 3D volume used for model inputs."""
    if total_slices < FRAME_FACTOR:
        return FRAME_FACTOR

    min_slices = ceil_by_factor(ele.get("min_slices", max(FRAME_FACTOR, total_slices // 10)), FRAME_FACTOR)
    max_slices = floor_by_factor(ele.get("max_slices", min(MAX_FRAMES, total_slices)), FRAME_FACTOR)
    nslices = min(max(total_slices, min_slices), max_slices)
    nslices = floor_by_factor(nslices, FRAME_FACTOR)

    if not (FRAME_FACTOR <= nslices and nslices <= total_slices):
        raise ValueError(f"nslices should be in interval [{FRAME_FACTOR}, {total_slices}], but got {nslices}.")

    return nslices


def _load_nifti(volume_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load NIfTI file (.nii or .nii.gz)."""
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required for NIfTI support. Install: pip install nibabel")

    st = time.time()
    nifti_img = nib.load(volume_path)
    volume_data = nifti_img.get_fdata()
    volume_data = np.transpose(volume_data, (2, 0, 1))
    header = nifti_img.header
    spacing = header.get_zooms()[:3] if hasattr(header, 'get_zooms') else (1.0, 1.0, 1.0)
    metadata = {
        'spacing': spacing,
        'shape': volume_data.shape,
        'dtype': str(volume_data.dtype),
    }
    logger.debug(f"NIfTI loaded: {volume_path=}, shape={volume_data.shape}, spacing={spacing}, time={time.time() - st:.3f}s")
    return volume_data, metadata


def _load_dicom(volume_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load DICOM series from directory or single DICOM file."""
    try:
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_modality_lut
    except ImportError:
        raise ImportError("pydicom is required for DICOM support. Install: pip install pydicom")

    st = time.time()
    if os.path.isdir(volume_path):
        dicom_files = [os.path.join(volume_path, f) for f in os.listdir(volume_path) if f.endswith('.dcm')]
        if not dicom_files:
            raise ValueError(f"No DICOM files found in directory: {volume_path}")
        dicoms = [pydicom.dcmread(f) for f in dicom_files]
        dicoms.sort(key=lambda x: float(x.InstanceNumber) if hasattr(x, 'InstanceNumber') else 0)
        slices = [apply_modality_lut(dcm.pixel_array, dcm) for dcm in dicoms]
        volume_data = np.stack(slices, axis=0)
        dcm = dicoms[0]
        spacing = (
            float(dcm.SliceThickness) if hasattr(dcm, 'SliceThickness') else 1.0,
            float(dcm.PixelSpacing[0]) if hasattr(dcm, 'PixelSpacing') else 1.0,
            float(dcm.PixelSpacing[1]) if hasattr(dcm, 'PixelSpacing') else 1.0,
        )
    else:
        dcm = pydicom.dcmread(volume_path)
        volume_data = apply_modality_lut(dcm.pixel_array, dcm)
        if volume_data.ndim == 2:
            volume_data = volume_data[np.newaxis, ...]
        spacing = (1.0, 1.0, 1.0)

    metadata = {'spacing': spacing, 'shape': volume_data.shape, 'dtype': str(volume_data.dtype)}
    logger.debug(f"DICOM loaded: {volume_path=}, shape={volume_data.shape}, time={time.time() - st:.3f}s")
    return volume_data, metadata


def _natural_key(s: str):
    """Key function for natural sorting (e.g., img2 < img10)."""
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def _load_image_folder(volume_dir: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a folder containing 2D slice images and stack to a 3D volume."""
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
                arr = np.array(img.convert('L'))
                slices.append(arr)
        except Exception as e:
            warnings.warn(f"Error loading slice image {path}: {e}")
    if not slices:
        raise ValueError(f"Failed to load any slice images from {volume_dir}")
    volume_data = np.stack(slices, axis=0)
    metadata = {'spacing': (1.0, 1.0, 1.0), 'shape': volume_data.shape, 'dtype': str(volume_data.dtype)}
    logger.debug(f"Image-folder volume loaded: volume_path='{volume_dir}', shape={volume_data.shape}")
    return volume_data, metadata


def _load_npy(volume_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load numpy array (.npy)."""
    st = time.time()
    volume_data = np.load(volume_path)
    if volume_data.ndim < 3:
        raise ValueError(f"Expected 3D volume, got shape {volume_data.shape}")
    metadata = {'spacing': (1.0, 1.0, 1.0), 'shape': volume_data.shape, 'dtype': str(volume_data.dtype)}
    logger.debug(f"NPY loaded: {volume_path=}, shape={volume_data.shape}, time={time.time() - st:.3f}s")
    return volume_data, metadata


def _normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Normalize volume to [0, 1] range."""
    volume[volume > 1000] = 1000
    volume[volume < -1000] = -1000
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
    slice_multiple: int = 3,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    Fetch and preprocess 3D medical volume (NIfTI, DICOM, NPY).
    """
    image_factor = image_patch_size * SPATIAL_MERGE_SIZE
    VOLUME_SLICE_MIN_PIXELS = VOLUME_MIN_TOKEN_NUM * image_factor * image_factor
    VOLUME_SLICE_MAX_PIXELS = VOLUME_MAX_TOKEN_NUM * image_factor * image_factor

    volume_path = ele.get("volume", ele.get("image_3d"))
    if volume_path is None:
        raise ValueError("ele must contain 'volume' or 'image_3d' key")

    if isinstance(volume_path, str):
        if volume_path.endswith(('.nii', '.nii.gz')):
            volume_data, file_metadata = _load_nifti(volume_path)
        elif volume_path.endswith('.npy'):
            volume_data, file_metadata = _load_npy(volume_path)
        elif volume_path.endswith('.dcm'):
            volume_data, file_metadata = _load_dicom(volume_path)
        elif os.path.isdir(volume_path):
            exts = [os.path.splitext(f)[1].lower() for f in os.listdir(volume_path) if os.path.isfile(os.path.join(volume_path, f))]
            if any(ext == '.dcm' for ext in exts):
                volume_data, file_metadata = _load_dicom(volume_path)
            elif any(ext in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'} for ext in exts):
                volume_data, file_metadata = _load_image_folder(volume_path)
            else:
                raise ValueError(f"Unsupported directory contents for volume: {volume_path}")
        else:
            if os.path.isfile(volume_path) and os.path.splitext(volume_path)[1].lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}:
                try:
                    with Image.open(volume_path) as img:
                        arr = np.array(img.convert('L'))
                    volume_data = arr[None, ...]
                    file_metadata = {'spacing': (1.0, 1.0, 1.0), 'shape': volume_data.shape, 'dtype': str(volume_data.dtype)}
                except Exception as e:
                    raise ValueError(f"Failed to load image as volume: {volume_path}, error: {e}")
            else:
                raise ValueError(f"Unsupported volume format: {volume_path}")
    else:
        raise ValueError(f"volume_path must be a string, got {type(volume_path)}")

    volume_data = _normalize_volume(volume_data)
    total_slices, height, width = volume_data.shape[:3]
    nslices = smart_nslices(ele, total_slices=total_slices)
    idx = torch.linspace(0, total_slices - 1, nslices).round().long()
    sample_fps = nslices / max(total_slices, 1e-6)
    slices = volume_data[idx.numpy()]

    if slices.ndim == 3:
        slices = np.stack([slices] * slice_multiple, axis=-1)
    elif slices.ndim == 4 and slices.shape[-1] == 1:
        slices = np.concatenate([slices] * slice_multiple, axis=-1)
    volume = torch.from_numpy(slices).permute(0, 3, 1, 2).float()

    min_pixels = ele.get("min_pixels", VOLUME_SLICE_MIN_PIXELS)
    total_pixels = ele.get("total_pixels", MODEL_SEQ_LEN * image_factor * image_factor * 0.9)
    max_pixels = max(min(VOLUME_SLICE_MAX_PIXELS, total_pixels / nslices * FRAME_FACTOR), int(min_pixels * 1.05))
    max_pixels_supposed = ele.get("max_pixels", max_pixels)
    if max_pixels_supposed > max_pixels:
        logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
    max_pixels = min(max_pixels_supposed, max_pixels)

    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"], ele["resized_width"], factor=image_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height, width, factor=image_factor, min_pixels=min_pixels, max_pixels=max_pixels,
        )

    volume = transforms.functional.resize(
        volume, [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC, antialias=True,
    ).float()

    if return_volume_metadata:
        from .volume_processing_vit_lamma import VolumeMetadata
        volume_metadata = VolumeMetadata(
            total_num_slices=total_slices,
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
    """Check if input is a valid volume tensor/array (4D: S, C, H, W)."""
    if isinstance(volume, (np.ndarray, torch.Tensor)):
        return volume.ndim == 4
    return False


def make_batched_volumes(volumes: VolumeInput) -> List[Union[str, torch.Tensor, np.ndarray]]:
    """Ensure that the input is a list of volumes."""
    try:
        if isinstance(volumes[0][0], list) and isinstance(volumes[0][0][0], str):
            return [volume_paths for sublist in volumes for volume_paths in sublist]
    except (IndexError, TypeError):
        pass

    if isinstance(volumes, str) or is_valid_volume(volumes):
        return [volumes]
    if not isinstance(volumes, list):
        raise ValueError(
            f"Invalid volume input. Expected either a string path, tensor/array, or list of volumes, "
            f"but got type {type(volumes)}."
        )

    flat_volumes_list = []
    for item in volumes:
        if isinstance(item, str) or is_valid_volume(item):
            flat_volumes_list.append(item)
        elif isinstance(item, list) and item:
            flat_volumes_list.extend(make_batched_volumes(item))
    return flat_volumes_list


def get_volume_size(volume: Union[np.ndarray, torch.Tensor]) -> Tuple[int, int]:
    """Returns the (height, width) dimensions of the volume."""
    if volume.ndim != 4:
        raise ValueError(f"Expected 4D volume, got {volume.ndim}D")
    return volume.shape[2], volume.shape[3]


def make_batched_metadata(
    volumes: VolumeInput,
    volume_metadata: Optional[Union["VolumeMetadata", List["VolumeMetadata"], Dict]] = None
) -> List["VolumeMetadata"]:
    """Create batched metadata from volumes and optional metadata."""
    from .volume_processing_vit_lamma import VolumeMetadata

    if not isinstance(volumes, list):
        volumes = [volumes]

    if volume_metadata is None:
        volume_metadata = [
            {
                "total_num_slices": len(volume),
                "spacing": None,
                "slice_indices": list(range(len(volume))),
                "height": get_volume_size(volume)[0] if is_valid_volume(volume) else None,
                "width": get_volume_size(volume)[1] if is_valid_volume(volume) else None,
                "original_shape": volume.shape if is_valid_volume(volume) else None,
            }
            for volume in volumes
        ]

    if isinstance(volume_metadata, list):
        if volume_metadata and isinstance(volume_metadata[0], list):
            volume_metadata = [
                VolumeMetadata(**metadata) for metadata_list in volume_metadata for metadata in metadata_list
            ]
        elif volume_metadata and isinstance(volume_metadata[0], dict):
            volume_metadata = [VolumeMetadata(**metadata) for metadata in volume_metadata]
    else:
        if isinstance(volume_metadata, dict):
            volume_metadata = [VolumeMetadata(**volume_metadata)]
        elif isinstance(volume_metadata, VolumeMetadata):
            volume_metadata = [volume_metadata]
        else:
            raise ValueError(f"Invalid volume_metadata type: {type(volume_metadata)}")

    return volume_metadata
