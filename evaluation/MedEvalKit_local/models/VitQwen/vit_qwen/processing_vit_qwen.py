# coding=utf-8
# Copyright 2025 VitQwen Team. All rights reserved.
"""
VitQwen Processor (volume-only). Same placeholder/template behavior as VitLamMA.
"""

from typing import Any, List, Optional, Union

import torch
from transformers.processing_utils import ProcessorMixin, BatchFeature

from .volume_processing_vit_qwen import VitQwenVolumeProcessor
from .vit_qwen_utils import VolumeInput

DEFAULT_CHAT_TEMPLATE = "Qwen style with <|image_3d_pad|> for volume; content may be list of {type: 'image_3d'|'text', ...}."


def _num_decode_from_grid_thw(grid_thw: torch.Tensor, index: int = 0) -> int:
    if grid_thw.dim() == 2:
        idx = min(index, grid_thw.size(0) - 1)
        t, h, w = int(grid_thw[idx, 0].item()), int(grid_thw[idx, 1].item()), int(grid_thw[idx, 2].item())
    else:
        t, h, w = int(grid_thw[0].item()), int(grid_thw[1].item()), int(grid_thw[2].item())
    n_prime = (t // 2) * (h // 2) * (w // 2)
    n_max = 16 * 16 * 16
    num_decode = round(1024 * n_prime / max(n_max, 1))
    return max(64, min(1024, num_decode))


def _num_visual_tokens_from_grid_thw(grid_thw: torch.Tensor, index: int = 0) -> int:
    if grid_thw.dim() == 2:
        idx = min(index, grid_thw.size(0) - 1)
        t, h, w = int(grid_thw[idx, 0].item()), int(grid_thw[idx, 1].item()), int(grid_thw[idx, 2].item())
    else:
        t, h, w = int(grid_thw[0].item()), int(grid_thw[1].item()), int(grid_thw[2].item())
    return (t // 2) * (h // 2) * (w // 2)


def _expand_placeholder_tokens(
    input_ids: torch.Tensor,
    image_3d_token_id: int,
    volume_grid_thw: torch.Tensor,
    use_simple_projector: bool = False,
) -> torch.Tensor:
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    fn = _num_visual_tokens_from_grid_thw if use_simple_projector else _num_decode_from_grid_thw
    batch_out = []
    grid_idx = 0
    for i in range(input_ids.size(0)):
        ids = input_ids[i].tolist()
        new_ids = []
        for tok in ids:
            if tok == image_3d_token_id:
                num_tokens = fn(volume_grid_thw, min(grid_idx, volume_grid_thw.size(0) - 1))
                new_ids.extend([image_3d_token_id] * num_tokens)
                grid_idx += 1
            else:
                new_ids.append(tok)
        batch_out.append(torch.tensor(new_ids, dtype=input_ids.dtype, device=input_ids.device))
    return torch.nn.utils.rnn.pad_sequence(batch_out, batch_first=True, padding_value=0)


class VitQwenProcessor(ProcessorMixin):
    """Combine tokenizer + volume_processor for VitQwen."""

    _processor_class = "VitQwenProcessor"
    attributes = ["tokenizer"]
    volume_processor_class = "VitQwenVolumeProcessor"
    tokenizer_class = ("AutoTokenizer", "PreTrainedTokenizerFast")

    def __init__(
        self,
        tokenizer=None,
        volume_processor: Optional[VitQwenVolumeProcessor] = None,
        chat_template: Optional[str] = None,
        use_simple_projector: bool = False,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.image_3d_token = "<|image_3d_pad|>"
        self.image_3d_token_id = (
            getattr(tokenizer, "image_3d_token_id", None) if tokenizer is not None else None
        ) or (tokenizer.convert_tokens_to_ids(self.image_3d_token) if tokenizer is not None else 151670)
        self.volume_processor = volume_processor or VitQwenVolumeProcessor()
        self.chat_template = chat_template or getattr(self, "chat_template", None) or DEFAULT_CHAT_TEMPLATE
        self.use_simple_projector = use_simple_projector

    def __call__(
        self,
        text: Optional[Union[str, list]] = None,
        volumes: Optional[VolumeInput] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> BatchFeature:
        data = {}
        if text is not None:
            tok = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
            data.update(tok)
        if volumes is not None:
            vol = self.volume_processor(volumes=volumes, return_tensors=return_tensors, **kwargs)
            if hasattr(vol, "data"):
                data.update(vol.data)
            else:
                data.update(vol)
        return BatchFeature(data=data, tensor_type=return_tensors)

    def apply_chat_template(
        self,
        conversation: Union[List[dict], List[List[dict]]],
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        return_dict: bool = True,
        return_tensors: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[str, List[int], torch.Tensor, BatchFeature]:
        template = chat_template or self.chat_template
        if not template:
            raise ValueError("Cannot use apply_chat_template: no chat template.")
        is_batched = isinstance(conversation, (list, tuple)) and conversation and isinstance(conversation[0], (list, tuple))
        conversations = conversation if is_batched else [conversation]
        batch_volumes: List[List[str]] = []
        for conv in conversations:
            vol_paths: List[str] = []
            for msg in conv:
                content = msg.get("content")
                if content is None:
                    continue
                if isinstance(content, str):
                    if "<image_3d>" in content and msg.get("image_3d"):
                        v = msg["image_3d"]
                        vol_paths.extend([v] if isinstance(v, str) else list(v))
                    continue
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") in ("image_3d", "volume"):
                        path = part.get("image_3d") or part.get("volume") or part.get("path") or part.get("url")
                        if path:
                            vol_paths.append(path)
            batch_volumes.append(vol_paths)
        special = getattr(self.tokenizer, "special_tokens_map", {}) or {}
        bos = special.get("bos_token", "<|endoftext|>")
        prompts: List[str] = []
        for conv in conversations:
            parts: List[str] = [bos]
            for msg in conv:
                role = msg.get("role", "user")
                content = msg.get("content")
                if content is None:
                    continue
                parts.append(f"<|im_start|>{role}\n")
                text_parts = []
                if isinstance(content, str):
                    text_parts.append(content.replace("<image_3d>", "<|image_3d_pad|>"))
                else:
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") in ("image_3d", "volume"):
                            text_parts.append("<|image_3d_pad|>")
                        else:
                            text_parts.append(part.get("text", ""))
                parts.append("".join(text_parts))
                parts.append("<|im_end|>\n")
            if add_generation_prompt:
                parts.append("<|im_start|>assistant\n")
            prompts.append("".join(parts))
        if not tokenize:
            return prompts[0] if not is_batched else prompts
        all_vol_paths = [p for vol_list in batch_volumes for p in vol_list]
        if not all_vol_paths:
            raise ValueError("apply_chat_template with tokenize=True requires at least one volume in messages.")
        out = self.tokenizer(
            prompts, padding=True, truncation=True,
            return_attention_mask=True, return_tensors=return_tensors or "pt",
        )
        input_ids = out["input_ids"]
        attention_mask = out.get("attention_mask")
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        vol_out = self.volume_processor(volumes=all_vol_paths, return_tensors=return_tensors or "pt", do_resize=True)
        if hasattr(vol_out, "data"):
            pixel_values_volumes = vol_out.data["pixel_values_volumes"]
            volume_grid_thw = vol_out.data["volume_grid_thw"]
        else:
            pixel_values_volumes = vol_out["pixel_values_volumes"]
            volume_grid_thw = vol_out["volume_grid_thw"]
        input_ids = _expand_placeholder_tokens(
            input_ids, self.image_3d_token_id, volume_grid_thw,
            use_simple_projector=getattr(self, "use_simple_projector", False),
        )
        pad_id = getattr(self.tokenizer, "pad_token_id", None) or 0
        attention_mask = (input_ids != pad_id).long().to(input_ids.device)
        if return_dict:
            return BatchFeature(
                data={
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values_volumes": pixel_values_volumes,
                    "volume_grid_thw": volume_grid_thw,
                },
                tensor_type=return_tensors or "pt",
            )
        return input_ids
