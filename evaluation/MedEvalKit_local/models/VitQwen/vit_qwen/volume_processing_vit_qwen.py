# coding=utf-8
# Copyright 2025 VitQwen Team. All rights reserved.
"""
VitQwen Volume Processor - volume (3D) data only.
Outputs: pixel_values_volumes, volume_grid_thw. Reuses logic from VitLamMA volume processor.
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import PILImageResampling, SizeDict
from transformers.processing_utils import Unpack, VideosKwargs
from transformers.utils import TensorType, logging
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import group_videos_by_shape, reorder_videos

from .vit_qwen_utils import (
    VolumeInput,
    fetch_volume,
    is_valid_volume,
    make_batched_metadata,
    make_batched_volumes,
)

logger = logging.get_logger(__name__)


@dataclass
class VolumeMetadata:
    total_num_slices: int
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
    temporal_factor: int = 16,
    factor: int = 32,
    min_pixels: int = 16 * 16 * 8 * 2 * 2 * 2 * 256,
    max_pixels: int = 16 * 16 * 8 * 2 * 2 * 2 * 2048,
) -> Tuple[int, int]:
    if num_slices < temporal_factor:
        raise ValueError(f"num_slices:{num_slices} must be larger than temporal_factor:{temporal_factor}")
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
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


class VitQwenVolumeProcessorInitKwargs(VideosKwargs):
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]
    min_slices: Optional[int]
    max_slices: Optional[int]
    target_spatial_height: Optional[int]
    target_spatial_width: Optional[int]


class VitQwenVolumeProcessor(BaseVideoProcessor):
    """Volume-only processor for VitQwen. Input volumes as (S, C, H, W) or paths."""

    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 64 * 32 * 32, "longest_edge": 32 * 32 * 256}
    target_spatial_height = 512
    target_spatial_width = 512
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_resize = False
    do_rescale = True
    rescale_factor = 1.0
    do_normalize = True
    do_convert_rgb = True
    patch_size = 16
    temporal_patch_size = 8
    merge_size = 2
    min_slices = 16
    max_slices = 512
    do_sample_slices = False
    return_metadata = False
    valid_kwargs = VitQwenVolumeProcessorInitKwargs
    model_input_names = ["pixel_values_volumes", "volume_grid_thw"]

    def __init__(self, **kwargs: Unpack[VitQwenVolumeProcessorInitKwargs]):
        super().__init__(**kwargs)
        if self.size is not None and (
            self.size.get("shortest_edge", None) is None or self.size.get("longest_edge", None) is None
        ):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

    def _further_process_kwargs(self, size: Optional[SizeDict] = None, **kwargs) -> dict:
        if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        return super()._further_process_kwargs(size=size, **kwargs)

    def sample_slices(self, metadata: VolumeMetadata, num_slices: Optional[int] = None, **kwargs):
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
        loaded_volumes = []
        loaded_metadata = []
        for volume_path in volumes:
            result = fetch_volume(
                {"volume": volume_path},
                image_patch_size=self.patch_size,
                return_volume_sample_fps=True,
                return_volume_metadata=True,
                slice_multiple=1,
            )
            if not isinstance(result, tuple) or len(result) != 2:
                raise ValueError("fetch_volume must return ((volume, metadata), sample_fps).")
            (volume_tensor, volume_meta), _ = result
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
        volume_metadata: Optional[Union[VolumeMetadata, List[VolumeMetadata], dict]] = None,
        do_sample_slices: Optional[bool] = None,
        sample_indices_fn: Optional[Callable] = None,
    ) -> Tuple[List[torch.Tensor], List[VolumeMetadata]]:
        volumes = make_batched_volumes(volumes)
        volume_metadata = make_batched_metadata(volumes, volume_metadata=volume_metadata)
        if do_sample_slices and sample_indices_fn is None:
            sample_indices_fn = partial(self.sample_slices)
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
            volumes, volume_metadata = self.fetch_volumes(
                volumes, sample_indices_fn=sample_indices_fn, do_sample_slices=do_sample_slices
            )
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
    ) -> BatchFeature:
        grouped_volumes, grouped_volumes_index = group_videos_by_shape(volumes)
        resized_volumes_grouped = {}
        for shape, stacked_volumes in grouped_volumes.items():
            B, S, C, H, W = stacked_volumes.shape
            if do_resize:
                resized_height = getattr(self, "target_spatial_height", 512)
                resized_width = getattr(self, "target_spatial_width", 512)
                if resized_height is None or resized_width is None:
                    resized_height, resized_width = smart_resize_volume(
                        num_slices=S, height=H, width=W,
                        temporal_factor=temporal_patch_size, factor=patch_size,
                        min_pixels=size.shortest_edge, max_pixels=size.longest_edge,
                    )
                stacked_volumes = stacked_volumes.reshape(B * S, C, H, W)
                stacked_volumes = self.resize(
                    stacked_volumes,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
                stacked_volumes = stacked_volumes.view(B, S, C, resized_height, resized_width)
            resized_volumes_grouped[shape] = stacked_volumes
        resized_volumes = reorder_videos(resized_volumes_grouped, grouped_volumes_index)

        grouped_volumes, grouped_volumes_index = group_videos_by_shape(resized_volumes)
        processed_volumes_grouped = {}
        processed_grids = {}
        for shape, stacked_volumes in grouped_volumes.items():
            stacked_volumes = self.rescale_and_normalize(
                stacked_volumes, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches = stacked_volumes
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)
            batch_size, num_slices, channel, height, width = patches.shape
            _, num_slices, _, height, width = patches.shape
            grid_t = num_slices // temporal_patch_size
            grid_h = height // patch_size
            grid_w = width // patch_size
            patches = patches.view(
                batch_size,
                grid_t // merge_size, merge_size, temporal_patch_size, channel,
                grid_h // merge_size, merge_size, patch_size,
                grid_w // merge_size, merge_size, patch_size,
            )
            patches = patches.permute(0, 1, 5, 8, 2, 6, 9, 4, 3, 7, 10)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )
            processed_volumes_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_volumes = reorder_videos(processed_volumes_grouped, grouped_volumes_index)
        processed_grids = reorder_videos(processed_grids, grouped_volumes_index)
        if isinstance(processed_volumes, torch.Tensor):
            if processed_volumes.dim() == 2:
                pixel_values_volumes = processed_volumes.unsqueeze(0)
            elif processed_volumes.dim() == 3:
                pixel_values_volumes = processed_volumes
            else:
                raise ValueError(f"Unexpected processed_volumes tensor shape: {tuple(processed_volumes.shape)}")
        else:
            if not processed_volumes:
                raise ValueError("No volumes were processed; got an empty batch.")
            normalized = []
            for t in processed_volumes:
                if not isinstance(t, torch.Tensor):
                    raise ValueError(f"Unexpected element type in processed_volumes: {type(t)}")
                if t.dim() == 2:
                    t = t.unsqueeze(0)
                elif t.dim() != 3:
                    raise ValueError(f"Unexpected per-sample volume tensor shape: {tuple(t.shape)}")
                normalized.append(t)
            # 允许不同 shape：按 max_N 做 padding，不再要求同 batch 内体积同 shape
            patch_dim = normalized[0].shape[-1]
            max_n = max(t.shape[1] for t in normalized)
            padded = []
            for t in normalized:
                n = t.shape[1]
                if n < max_n:
                    t = torch.nn.functional.pad(t, (0, 0, 0, max_n - n), value=0.0)
                padded.append(t)
            pixel_values_volumes = torch.cat(padded, dim=0)
        volume_grid_thw = torch.tensor(processed_grids)
        return BatchFeature(
            data={"pixel_values_volumes": pixel_values_volumes, "volume_grid_thw": volume_grid_thw},
            tensor_type=return_tensors,
        )

    def __call__(self, volumes: VolumeInput, **kwargs: Unpack[VitQwenVolumeProcessorInitKwargs]) -> BatchFeature:
        return self.preprocess(volumes=volumes, **kwargs)

    def preprocess(self, volumes: VolumeInput, **kwargs: Unpack[VitQwenVolumeProcessorInitKwargs]) -> BatchFeature:
        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))
        input_data_format = kwargs.pop("input_data_format", None)
        device = kwargs.pop("device", None)
        do_sample_slices = kwargs.pop("do_sample_slices", self.do_sample_slices)
        volume_metadata = kwargs.pop("volume_metadata", None)
        return_tensors = kwargs.pop("return_tensors", None)
        return_metadata = kwargs.pop("return_metadata", True)
        sample_indices_fn = partial(self.sample_slices, **kwargs) if do_sample_slices else None
        volumes, volume_metadata = self._decode_and_sample_volumes(
            volumes, volume_metadata=volume_metadata,
            do_sample_slices=do_sample_slices, sample_indices_fn=sample_indices_fn,
        )
        volumes = self._prepare_input_videos(videos=volumes, input_data_format=input_data_format, device=device)
        kwargs = self._further_process_kwargs(**kwargs)
        self._validate_preprocess_kwargs(**kwargs)
        kwargs.pop("data_format", None)
        preprocessed = self._preprocess(volumes=volumes, return_tensors=return_tensors, **kwargs)
        if return_metadata:
            preprocessed["volume_metadata"] = volume_metadata
        return preprocessed
