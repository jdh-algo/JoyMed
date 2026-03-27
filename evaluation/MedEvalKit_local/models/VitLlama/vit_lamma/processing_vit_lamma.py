# coding=utf-8
# Copyright 2025 VitLamMA Team. All rights reserved.
"""
VitLamMA Processor (volume-only).

Follows the style of `architectures/citrus_v_3/processing_citrus_v_3.py` but ONLY supports volumes.

<image_3d> 与 <|image_3d_pad|> 的对应关系（与 citrus_v_3 一致，训练/推理统一）：
- 训练（Swift 模板 swift/llm/template/template/vit_lamma.py）：
  - 数据格式可为字符串 "<image_3d>Provide a radiology report..." 或 list [{type:"image_3d",...},{type:"text",...}]。
  - _split_special_tokens 按 <image_3d> 切分；_pre_tokenize 中 replace_tag 将 "<image_3d>" 替换为 "<|image_3d_pad|>"；
  - _encode 里对 input_ids 中每个 image_3d_token_id 按 volume_grid_thw 展开为 num_decode 个 token，再喂入模型。
  - 模型 forward 时用 get_placeholder_mask 找到占位位置，将 ViT->Qwen2 得到的 visual tokens 注入。
- 推理（本 processor apply_chat_template）：
  - 支持 list 格式 content: [{type:"image_3d", image_3d: path}, {type:"text", text: "Provide..."}]，
    或与训练一致的字符串格式 content: "<image_3d>Provide..." 且同条 message 带 "image_3d": path。
  - 内部统一生成 "<|image_3d_pad|>" + 文本（无换行），再 tokenize 并 _expand_placeholder_tokens，
    与训练端最终序列一致。
"""

from typing import Any, List, Optional, Union

import torch
from transformers.processing_utils import ProcessorMixin, BatchFeature

from .volume_processing_vit_lamma import VitLamMAVolumeProcessor
from .vit_lamma_utils import VolumeInput

# Placeholder so processor "has" a chat template; actual prompt is built in apply_chat_template.
DEFAULT_CHAT_TEMPLATE = (
    "Llama-3 style with <|image_3d_pad|> for volume; content may be list of {type: 'image_3d'|'text', ...}."
)


def _num_decode_from_grid_thw(grid_thw: torch.Tensor, index: int = 0) -> int:
    """Same formula as model/Swift template: max 512*512*256 -> 1024."""
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
    """Simple projector: 与下采样后 token 数一致，不做 64..1024 限制。"""
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
    max_visual_tokens_inference: Optional[int] = None,
) -> torch.Tensor:
    """Replace each single <|image_3d_pad|> with num_tokens (by volume_grid_thw).
    use_simple_projector: 与下采样 token 数一致，否则用 num_decode 64..1024。
    max_visual_tokens_inference: 推理时每 volume 视觉 token 上限，超过则截断，避免 OOM（如 2048/1024/512）。
    """
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
                # if max_visual_tokens_inference is not None and num_tokens > max_visual_tokens_inference:
                #     num_tokens = max_visual_tokens_inference
                new_ids.extend([image_3d_token_id] * num_tokens)
                grid_idx += 1
            else:
                new_ids.append(tok)
        batch_out.append(torch.tensor(new_ids, dtype=input_ids.dtype, device=input_ids.device))
    return torch.nn.utils.rnn.pad_sequence(batch_out, batch_first=True, padding_value=0)


class VitLamMAProcessor(ProcessorMixin):
    """
    Combine tokenizer + volume_processor for Swift multimodal pipeline.
    """

    _processor_class = "VitLamMAProcessor"
    attributes = ["tokenizer"]
    volume_processor_class = "VitLamMAVolumeProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast", "AutoTokenizer", "PreTrainedTokenizerFast")

    def __init__(
        self,
        tokenizer=None,
        volume_processor: Optional[VitLamMAVolumeProcessor] = None,
        chat_template: Optional[str] = None,
        use_simple_projector: bool = False,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.image_3d_token = "<|image_3d_pad|>"
        self.image_3d_token_id = (
            getattr(tokenizer, "image_3d_token_id", None) if tokenizer is not None else None
        ) or (tokenizer.convert_tokens_to_ids(self.image_3d_token) if tokenizer is not None else 151670)
        self.volume_processor = volume_processor or VitLamMAVolumeProcessor()
        self.chat_template = chat_template or getattr(self, "chat_template", None) or DEFAULT_CHAT_TEMPLATE
        self.use_simple_projector = use_simple_projector

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
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
        """
        Build prompt from messages and optionally tokenize with volume processing.

        Supports messages with content as list of {"type": "image_3d", "image_3d": path}
        and {"type": "text", "text": "..."}. When tokenize=True and return_dict=True,
        returns BatchFeature with input_ids, attention_mask, pixel_values_volumes, volume_grid_thw.
        """
        template = chat_template or self.chat_template
        if not template:
            raise ValueError("Cannot use apply_chat_template because this processor does not have a chat template.")

        is_batched = isinstance(conversation, (list, tuple)) and conversation and isinstance(conversation[0], (list, tuple))
        conversations = conversation if is_batched else [conversation]

        # Collect volume paths per conversation (in order of appearance)
        # 支持两种格式：(1) list content 中 type:"image_3d" 的 path；(2) 字符串 content "<image_3d>..." 且 message 带 "image_3d": path
        batch_volumes: List[List[str]] = []
        for conv in conversations:
            vol_paths: List[str] = []
            for msg in conv:
                content = msg.get("content")
                if content is None:
                    continue
                if isinstance(content, str):
                    if "<image_3d>" in content and "image_3d" in msg:
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

        # Render prompt string per conversation (simple loop without Jinja to avoid dependency)
        special = getattr(self.tokenizer, "special_tokens_map", {}) or {}
        bos = special.get("bos_token", "<|begin_of_text|>")
        prompts: List[str] = []
        for conv in conversations:
            # Llama-3 chat 格式：BOS 只出现一次，建议包含 system 段
            parts: List[str] = [bos]
            parts.append("<|start_header_id|>system<|end_header_id|>\n\n")
            parts.append("You are a helpful assistant.")
            parts.append("<|eot_id|>")
            for msg in conv:
                role = msg.get("role", "user")
                content = msg.get("content")
                if content is None:
                    continue
                # 统一按 role 写 header（user/assistant）
                parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n")
                text_parts = []
                if isinstance(content, str):
                    # 与训练数据格式一致：字符串 "<image_3d>Provide..." 中的 <image_3d> 替换为 <|image_3d_pad|>
                    text_parts.append(content.replace("<image_3d>", "<|image_3d_pad|>"))
                else:
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") in ("image_3d", "volume"):
                            # 与 Swift 训练一致：占位符与文本之间无换行（replace_tag 返回单 token，与 query 直接拼接）
                            text_parts.append("<|image_3d_pad|>")
                        else:
                            text_parts.append(part.get("text", ""))
                parts.append("".join(text_parts))
                parts.append("<|eot_id|>")
            if add_generation_prompt:
                parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            prompts.append("".join(parts))

        if not tokenize:
            return prompts[0] if not is_batched else prompts

        # Flatten volume paths in order for this batch (one list per sample for processor)
        all_vol_paths = [p for vol_list in batch_volumes for p in vol_list]
        if not all_vol_paths:
            raise ValueError("apply_chat_template with tokenize=True requires at least one volume in messages.")

        # Tokenize prompts (single or batch)
        out = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors=return_tensors or "pt",
        )
        input_ids = out["input_ids"]
        attention_mask = out.get("attention_mask")
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # Process volumes: one batch for all volumes (processor returns stacked tensor + grid_thw)
        vol_out = self.volume_processor(volumes=all_vol_paths, return_tensors=return_tensors or "pt", do_resize=True)
        if hasattr(vol_out, "data"):
            pixel_values_volumes = vol_out.data["pixel_values_volumes"]
            volume_grid_thw = vol_out.data["volume_grid_thw"]
        else:
            pixel_values_volumes = vol_out["pixel_values_volumes"]
            volume_grid_thw = vol_out["volume_grid_thw"]

        # Expand each <|image_3d_pad|> in input_ids (num_tokens = 下采样 token 数 when use_simple_projector, else num_decode)
        input_ids = _expand_placeholder_tokens(
            input_ids,
            self.image_3d_token_id,
            volume_grid_thw,
            use_simple_projector=getattr(self, "use_simple_projector", False),
            max_visual_tokens_inference=getattr(self, "max_visual_tokens_inference", None),
        )
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = 0
        attention_mask = (input_ids != pad_id).long().to(input_ids.device)

        if return_dict:
            data = {"input_ids": input_ids, "attention_mask": attention_mask}
            if pixel_values_volumes is not None:
                data["pixel_values_volumes"] = pixel_values_volumes
            if volume_grid_thw is not None:
                data["volume_grid_thw"] = volume_grid_thw
            return BatchFeature(data=data, tensor_type=return_tensors or "pt")
        return input_ids

