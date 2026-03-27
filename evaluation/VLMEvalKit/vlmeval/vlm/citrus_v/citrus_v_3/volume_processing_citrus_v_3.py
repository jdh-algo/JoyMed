# coding=utf-8
# Copyright 2025 The CitrusV Team and The HuggingFace Inc. team. All rights reserved.
"""
CitrusV 3.0 Volume Processor

Dedicated processor for 3D medical volumes (NIfTI, DICOM, NumPy).
Extends the video processing framework to handle medical imaging data.
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ChannelDimension, PILImageResampling, SizeDict, get_image_size
from transformers.processing_utils import Unpack, VideosKwargs
from transformers.utils import TensorType, logging
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import group_videos_by_shape, reorder_videos

from .citrus_v_utils import VolumeInput, fetch_volume, make_batched_metadata, make_batched_volumes, is_valid_volume

logger = logging.get_logger(__name__)


@dataclass
class VolumeMetadata:
    """
    Metadata for 3D medical volumes (analogous to VideoMetadata).
    
    Attributes:
        total_num_slices: Total number of slices in the volume
        slice_axis: Axis along which slices are extracted (0, 1, or 2)
        spacing: Physical spacing between slices (mm)
        width: Width of each slice (pixels)
        height: Height of each slice (pixels)
        original_shape: Original volume shape (D, H, W) or (D, H, W, C)
        slice_indices: Indices of sampled slices
    """
    total_num_slices: int
    slice_axis: Optional[int] = None
    spacing: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    original_shape: Optional[tuple] = None
    slice_indices: Optional[List[int]] = None
    
    def __getitem__(self, item):
        return getattr(self, item)
    
    def __setitem__(self, key, value):
        return setattr(self, key, value)


def smart_resize_volume(
    num_slices: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 32,
    min_pixels: int = 16 * 16 * 2 * 2 * 2 * 64,
    max_pixels: int = 16 * 16 * 2 * 2 * 2 * 256,
):
    """
    Smart resize for volume dimensions (analogous to smart_resize for videos).
    
    Args:
        num_slices: Number of slices in the volume
        height: Height of each slice
        width: Width of each slice
        temporal_factor: Factor for slice dimension (similar to temporal for video)
        factor: Spatial factor for rounding
        min_pixels: Minimum total pixels
        max_pixels: Maximum total pixels
    
    Returns:
        Tuple of (height, width) after smart resizing
    """
    if num_slices < temporal_factor:
        raise ValueError(f"num_slices:{num_slices} must be larger than temporal_factor:{temporal_factor}")
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    s_bar = round(num_slices / temporal_factor) * temporal_factor
    
    if s_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_slices * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif s_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_slices * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    
    return h_bar, w_bar


class CitrusV3VolumeProcessorInitKwargs(VideosKwargs):
    """Initialization kwargs for CitrusV3VolumeProcessor."""
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]
    min_slices: Optional[int]
    max_slices: Optional[int]


class CitrusV3VolumeProcessor(BaseVideoProcessor):
    """
    CitrusV 3.0 Volume Processor for 3D medical imaging data.
    
    This processor handles NIfTI (.nii, .nii.gz), DICOM (.dcm), and NumPy (.npy) volumes.
    It treats volumes as video-like 4D tensors (Slices, Channels, Height, Width) for 
    compatibility with vision transformers.
    
    Key features:
    - Smart slice sampling (analogous to frame sampling in videos)
    - Adaptive spatial resizing based on volume size
    - Intensity normalization for medical images
    - Support for different slice axes (axial, coronal, sagittal)
    
    Args:
        patch_size (int): Spatial patch size of the vision encoder (default: 16)
        temporal_patch_size (int): Temporal patch size for slice dimension (default: 2)
        merge_size (int): Merge size from vision encoder to LLM (default: 2)
        min_slices (int): Minimum number of slices to sample (default: 4)
        max_slices (int): Maximum number of slices to sample (default: 768)
        do_sample_slices (bool): Whether to sample slices (default: True)
    """
    
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 32 * 32 * 64 * 2 * 2, "longest_edge": 32 * 32 * 256 * 2 * 48}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_resize = False 
    do_rescale = True
    rescale_factor = 1.0
    do_normalize = True
    do_convert_rgb = True
    patch_size = 16
    temporal_patch_size = 2
    merge_size = 2
    min_slices = 2 
    max_slices = 96 
    do_sample_slices = False 
    return_metadata = False
    valid_kwargs = CitrusV3VolumeProcessorInitKwargs
    model_input_names = ["pixel_values_volumes", "volume_grid_thw"]
    
    def __init__(self, **kwargs: Unpack[CitrusV3VolumeProcessorInitKwargs]):
        super().__init__(**kwargs)
        if self.size is not None and (
            self.size.get("shortest_edge", None) is None or self.size.get("longest_edge", None) is None
        ):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
    
    def _further_process_kwargs(
        self,
        size: Optional[SizeDict] = None,
        **kwargs,
    ) -> dict:
        """Update kwargs before validation."""
        if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        return super()._further_process_kwargs(size=size, **kwargs)
    
    def sample_slices(
        self,
        metadata: VolumeMetadata,
        num_slices: Optional[int] = None,
        **kwargs,
    ):
        """
        Sample slices uniformly from the volume.
        
        Args:
            metadata: Volume metadata containing slice information
            num_slices: Number of slices to sample (defaults to min_slices/max_slices range)
        
        Returns:
            Array of slice indices to extract
        """
        total_num_slices = metadata.total_num_slices
        
        if num_slices is None:
            num_slices = min(max(total_num_slices, self.min_slices), self.max_slices)
        else:
            num_slices = min(min(max(num_slices, self.min_slices), self.max_slices), total_num_slices)
        
        indices = np.linspace(0, total_num_slices - 1, num_slices).round().astype(int)
        return indices
    
    def fetch_volumes(
        self,
        volumes: List[str],
        sample_indices_fn: Optional[Callable] = None,
        do_sample_slices: Optional[bool] = True,
    ) -> Tuple[List[torch.Tensor], List[VolumeMetadata]]:
        """
        Fetch volumes from paths (similar to fetch_videos for videos).
        
        Args:
            volumes: List of volume paths
            sample_indices_fn: Optional function to compute slice indices
            
        Returns:
            Tuple of (list of volume tensors, list of VolumeMetadata)
        """
        loaded_volumes = []
        loaded_metadata = []
        
        for volume_path in volumes:
            result = fetch_volume(
                {"volume": volume_path},
                image_patch_size=self.patch_size,
                return_volume_sample_fps=True,
                return_volume_metadata=True,
            )
            
            # fetch_volume returns ((volume, metadata), sample_fps) when both flags are True
            if not isinstance(result, tuple) or len(result) != 2:
                raise ValueError(
                    f"fetch_volume must return ((volume, metadata), sample_fps) when both return_volume_sample_fps and return_volume_metadata are True. "
                    f"Got: {type(result)}"
                )
            
            (volume_tensor, volume_meta), _ = result
            if not isinstance(volume_meta, VolumeMetadata):
                raise ValueError(
                    f"fetch_volume must return VolumeMetadata. Got: {type(volume_meta)}"
                )
            
            # Apply slice sampling if provided
            if sample_indices_fn is not None and do_sample_slices:
                indices = sample_indices_fn(metadata=volume_meta)
                volume_meta.slice_indices = indices.tolist() if isinstance(indices, np.ndarray) else indices
                volume_tensor = volume_tensor[indices]
            
            loaded_volumes.append(volume_tensor)
            loaded_metadata.append(volume_meta)
        
        return loaded_volumes, loaded_metadata
    
    def _decode_and_sample_volumes(
        self,
        volumes: VolumeInput,
        volume_metadata: Optional[Union[VolumeMetadata, List[VolumeMetadata], Dict]] = None,
        do_sample_slices: Optional[bool] = None,
        sample_indices_fn: Optional[Callable] = None,
    ) -> Tuple[List[torch.Tensor], List[VolumeMetadata]]:
        """
        Decode input volumes (from paths) and sample slices if needed.
        
        Similar to BaseVideoProcessor._decode_and_sample_videos, but for 3D medical volumes.
        
        Args:
            volumes: Volume path(s) or tensor(s)
            volume_metadata: Optional metadata for each volume
            do_sample_slices: Whether to sample slices
            sample_indices_fn: Function to compute slice indices
            
        Returns:
            Tuple of (list of volume tensors, list of VolumeMetadata)
        """
        print(f"==== volumes: {volumes[:3]}")
        volumes = make_batched_volumes(volumes)
        volume_metadata = make_batched_metadata(volumes, volume_metadata=volume_metadata)
        
        # Only sample slices if a tensor/array volume is passed, otherwise first decode -> then sample
        if is_valid_volume(volumes[0]) and do_sample_slices:
            sampled_volumes = []
            sampled_metadata = []
            for volume, metadata in zip(volumes, volume_metadata):
                indices = sample_indices_fn(metadata=metadata)
                metadata.slice_indices = indices.tolist() if isinstance(indices, np.ndarray) else indices
                sampled_volumes.append(volume[indices])
                sampled_metadata.append(metadata)
            volumes = sampled_volumes
            volume_metadata = sampled_metadata
        elif not is_valid_volume(volumes[0]):
            # Volumes are paths - load them
            volumes, volume_metadata = self.fetch_volumes(volumes, sample_indices_fn=sample_indices_fn, do_sample_slices=do_sample_slices)
        
        return volumes, volume_metadata
    
    def _preprocess(
        self,
        volumes: List[torch.Tensor],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: Optional[SizeDict] = None,
        interpolation: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1.0,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        """
        Preprocess volumes (resize, normalize, etc.).
        
        Volumes are expected to be in shape (S, C, H, W) where:
        - S: number of slices
        - C: number of channels (1 for grayscale, 3 for RGB)
        - H: height
        - W: width
        """
        grouped_volumes, grouped_volumes_index = group_videos_by_shape(volumes)
        resized_volumes_grouped = {}
        
        # Group volumes by shape for resizing
        for shape, stacked_volumes in grouped_volumes.items():
            B, S, C, H, W = stacked_volumes.shape
            num_slices, height, width = S, H, W
            
            # Smart resize for volumes
            print(f"=== do_resize: {do_resize}")
            if do_resize:
                resized_height, resized_width = smart_resize_volume(
                    num_slices=num_slices,
                    height=height,
                    width=width,
                    temporal_factor=temporal_patch_size,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                # Reshape for torchvision resize: (B*S, C, H, W)
                print(f"=== stacked_volumes.shape: {stacked_volumes.shape}, resized_height: {resized_height}, resized_width: {resized_width}")
                stacked_volumes = stacked_volumes.reshape(B * S, C, H, W)
                stacked_volumes = self.resize(
                    stacked_volumes,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
                stacked_volumes = stacked_volumes.view(B, S, C, resized_height, resized_width)
            resized_volumes_grouped[shape] = stacked_volumes
        resized_volumes = reorder_videos(resized_volumes_grouped, grouped_volumes_index)

        # Group volumes by size for further processing
        # Needed in case do_resize is False, or resize returns volumes with different sizes
        grouped_volumes, grouped_volumes_index = group_videos_by_shape(resized_volumes)
        processed_volumes_grouped = {}
        processed_grids = {}
        for shape, stacked_volumes in grouped_volumes.items():
            resized_height, resized_width = get_image_size(stacked_volumes[0], channel_dim=ChannelDimension.FIRST)

            # Fused rescale and normalize
            stacked_volumes = self.rescale_and_normalize(
                stacked_volumes, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches = stacked_volumes

            # Check that volumes have `num_slices` divisible by `temporal_patch_size`
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)
            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
                
            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )
            processed_volumes_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size
        
        processed_volumes = reorder_videos(processed_volumes_grouped, grouped_volumes_index)
        processed_grids = reorder_videos(processed_grids, grouped_volumes_index)
        pixel_values_volumes = torch.cat(processed_volumes, dim=0)
        volume_grid_thw = torch.tensor(processed_grids)
        
        data = {
            "pixel_values_volumes": pixel_values_volumes,
            "volume_grid_thw": volume_grid_thw,
        }
        
        return BatchFeature(data=data, tensor_type=return_tensors)
    
    def __call__(
        self,
        volumes: VolumeInput,
        **kwargs: Unpack[CitrusV3VolumeProcessorInitKwargs],
    ) -> BatchFeature:
        return self.preprocess(volumes=volumes, **kwargs)
    
    def preprocess(
        self,
        volumes: VolumeInput,
        **kwargs: Unpack[CitrusV3VolumeProcessorInitKwargs],
    ) -> BatchFeature:
        """
        Process volumes for the model.
        
        Args:
            volumes: Volume path(s) or tensor(s) in shape (S, C, H, W)
            volume_metadata: Metadata for each volume
            do_sample_slices: Whether to sample slices (default: self.do_sample_slices)
            **kwargs: Additional preprocessing arguments
        
        Returns:
            BatchFeature with processed volumes and grid information
        """
        # Set default kwargs from self
        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))
        
        # Pop special arguments
        input_data_format = kwargs.pop("input_data_format", None)
        device = kwargs.pop("device", None)
        do_sample_slices = kwargs.pop("do_sample_slices", self.do_sample_slices)
        volume_metadata = kwargs.pop("volume_metadata", None)
        return_tensors = kwargs.pop("return_tensors", None)
        return_metadata = kwargs.pop("return_metadata", True)
        
        # Decode and sample volumes (load from paths if needed)
        sample_indices_fn = None
        if do_sample_slices:
            sample_indices_fn = partial(self.sample_slices, **kwargs)
        
        volumes, volume_metadata = self._decode_and_sample_volumes(
            volumes,
            volume_metadata=volume_metadata,
            do_sample_slices=do_sample_slices,
            sample_indices_fn=sample_indices_fn,
        )
        
        # Prepare volumes (convert format if needed)
        volumes = self._prepare_input_videos(videos=volumes, input_data_format=input_data_format, device=device)
        
        # Further process kwargs
        kwargs = self._further_process_kwargs(**kwargs)
        self._validate_preprocess_kwargs(**kwargs)
        
        # Pop kwargs that are not needed in _preprocess
        kwargs.pop("data_format", None)
        
        # Call _preprocess with remaining kwargs
        preprocessed_volumes = self._preprocess(volumes=volumes, return_tensors=return_tensors, **kwargs)
        
        if return_metadata:
            preprocessed_volumes["volume_metadata"] = volume_metadata
        
        return preprocessed_volumes

