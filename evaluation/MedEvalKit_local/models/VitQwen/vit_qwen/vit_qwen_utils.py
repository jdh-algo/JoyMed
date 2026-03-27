# coding=utf-8
# Copyright 2025 VitQwen Team. All rights reserved.
"""Volume utilities for VitQwen - standalone copy (no dependency on vit_lamma)."""

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

Path = NewType("Path", str)
VolumeInput = Union[
    str, Path, List[str], List[Path],
    np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor],
]

VOLUME_MIN_TOKEN_NUM = 64
VOLUME_MAX_TOKEN_NUM = 256
SPATIAL_MERGE_SIZE = 2
FRAME_FACTOR = 8 * SPATIAL_MERGE_SIZE
MAX_FRAMES = 128
MODEL_SEQ_LEN = int(float(os.environ.get('MODEL_SEQ_LEN', 128000)))


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int,
    min_pixels: Optional[int] = None, max_pixels: Optional[int] = None
) -> Tuple[int, int]:
    max_pixels = max_pixels if max_pixels is not None else (VOLUME_MAX_TOKEN_NUM * factor ** 2)
    min_pixels = min_pixels if min_pixels is not None else (VOLUME_MIN_TOKEN_NUM * factor ** 2)
    assert max_pixels >= min_pixels
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


def smart_nslices(ele: Dict[str, Any], total_slices: int) -> int:
    if total_slices < FRAME_FACTOR:
        return FRAME_FACTOR
    min_slices = ceil_by_factor(ele.get("min_slices", max(FRAME_FACTOR, total_slices // 10)), FRAME_FACTOR)
    max_slices = floor_by_factor(ele.get("max_slices", min(MAX_FRAMES, total_slices)), FRAME_FACTOR)
    nslices = min(max(total_slices, min_slices), max_slices)
    nslices = floor_by_factor(nslices, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nslices <= total_slices):
        raise ValueError(f"nslices should be in [{FRAME_FACTOR}, {total_slices}], got {nslices}.")
    return nslices


def _load_nifti(volume_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required for NIfTI. pip install nibabel")
    st = time.time()
    nifti_img = nib.load(volume_path)
    volume_data = nifti_img.get_fdata()
    volume_data = np.transpose(volume_data, (2, 0, 1))
    header = nifti_img.header
    spacing = header.get_zooms()[:3] if hasattr(header, 'get_zooms') else (1.0, 1.0, 1.0)
    metadata = {'spacing': spacing, 'shape': volume_data.shape, 'dtype': str(volume_data.dtype)}
    logger.debug(f"NIfTI loaded: {volume_path=}, shape={volume_data.shape}, time={time.time()-st:.3f}s")
    return volume_data, metadata


def _load_dicom(volume_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_modality_lut
    except ImportError:
        raise ImportError("pydicom required. pip install pydicom")
    st = time.time()
    if os.path.isdir(volume_path):
        dicom_files = [os.path.join(volume_path, f) for f in os.listdir(volume_path) if f.endswith('.dcm')]
        if not dicom_files:
            raise ValueError(f"No DICOM files in {volume_path}")
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
    logger.debug(f"DICOM loaded: {volume_path=}, shape={volume_data.shape}, time={time.time()-st:.3f}s")
    return volume_data, metadata


def _natural_key(s: str):
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def _load_image_folder(volume_dir: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    if "M3D" in volume_dir:
        return load_m3d_image_folder(volume_dir)
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    files = [f for f in os.listdir(volume_dir) if os.path.isfile(os.path.join(volume_dir, f))]
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_exts]
    if not image_files:
        raise ValueError(f"No image files in {volume_dir}")
    image_files.sort(key=_natural_key)
    slices = []
    for fname in image_files:
        path = os.path.join(volume_dir, fname)
        try:
            with Image.open(path) as img:
                arr = np.array(img.convert('L'))
                slices.append(arr)
        except Exception as e:
            warnings.warn(f"Error loading {path}: {e}")
    if not slices:
        raise ValueError(f"Failed to load any slices from {volume_dir}")
    volume_data = np.stack(slices, axis=0)
    metadata = {'spacing': (1.0, 1.0, 1.0), 'shape': volume_data.shape, 'dtype': str(volume_data.dtype)}
    return volume_data, metadata


def _load_npy(volume_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    st = time.time()
    volume_data = np.load(volume_path)
    if volume_data.ndim < 3:
        raise ValueError(f"Expected 3D volume, got {volume_data.shape}")
    metadata = {'spacing': (1.0, 1.0, 1.0), 'shape': volume_data.shape, 'dtype': str(volume_data.dtype)}
    logger.debug(f"NPY loaded: {volume_path=}, shape={volume_data.shape}, time={time.time()-st:.3f}s")
    return volume_data, metadata


def _normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = volume.copy()
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
                raise ValueError(f"Unsupported directory contents: {volume_path}")
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
        logger.warning(f"max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
    max_pixels = min(max_pixels_supposed, max_pixels)

    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(ele["resized_height"], ele["resized_width"], factor=image_factor)
    else:
        resized_height, resized_width = smart_resize(height, width, factor=image_factor, min_pixels=min_pixels, max_pixels=max_pixels)

    volume = transforms.functional.resize(
        volume, [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC, antialias=True,
    ).float()

    if return_volume_metadata:
        from .volume_processing_vit_qwen import VolumeMetadata
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


def is_valid_volume(volume) -> bool:
    if isinstance(volume, (np.ndarray, torch.Tensor)):
        return volume.ndim == 4
    return False


def make_batched_volumes(volumes: VolumeInput) -> List[Union[str, torch.Tensor, np.ndarray]]:
    try:
        if isinstance(volumes[0][0], list) and isinstance(volumes[0][0][0], str):
            return [volume_paths for sublist in volumes for volume_paths in sublist]
    except (IndexError, TypeError):
        pass
    if isinstance(volumes, str) or is_valid_volume(volumes):
        return [volumes]
    if not isinstance(volumes, list):
        raise ValueError(f"Invalid volume input. Expected string, tensor/array, or list, got {type(volumes)}.")
    flat_volumes_list = []
    for item in volumes:
        if isinstance(item, str) or is_valid_volume(item):
            flat_volumes_list.append(item)
        elif isinstance(item, list) and item:
            flat_volumes_list.extend(make_batched_volumes(item))
    return flat_volumes_list


def get_volume_size(volume: Union[np.ndarray, torch.Tensor]) -> Tuple[int, int]:
    if volume.ndim != 4:
        raise ValueError(f"Expected 4D volume, got {volume.ndim}D")
    return volume.shape[2], volume.shape[3]


def make_batched_metadata(
    volumes: VolumeInput,
    volume_metadata: Optional[Union[Any, List[Any], Dict]] = None,
) -> List[Any]:
    from .volume_processing_vit_qwen import VolumeMetadata

    if not isinstance(volumes, list):
        volumes = [volumes]
    if volume_metadata is None:
        volume_metadata = [
            {
                "total_num_slices": len(vol) if not isinstance(vol, str) else 0,
                "spacing": None,
                "slice_indices": list(range(len(vol))) if is_valid_volume(vol) else [],
                "height": get_volume_size(vol)[0] if is_valid_volume(vol) else None,
                "width": get_volume_size(vol)[1] if is_valid_volume(vol) else None,
                "original_shape": vol.shape if is_valid_volume(vol) else None,
            }
            for vol in volumes
        ]
    if isinstance(volume_metadata, list):
        if volume_metadata and isinstance(volume_metadata[0], list):
            volume_metadata = [VolumeMetadata(**m) for meta_list in volume_metadata for m in meta_list]
        elif volume_metadata and isinstance(volume_metadata[0], dict):
            volume_metadata = [VolumeMetadata(**m) for m in volume_metadata]
    else:
        if isinstance(volume_metadata, dict):
            volume_metadata = [VolumeMetadata(**volume_metadata)]
        elif isinstance(volume_metadata, VolumeMetadata):
            volume_metadata = [volume_metadata]
        else:
            raise ValueError(f"Invalid volume_metadata type: {type(volume_metadata)}")
    return volume_metadata
