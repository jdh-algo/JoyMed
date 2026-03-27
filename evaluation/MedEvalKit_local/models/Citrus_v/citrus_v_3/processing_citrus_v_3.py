# coding=utf-8
# Copyright 2025 The CitrusV Team and The HuggingFace Inc. team. All rights reserved.
"""
CitrusV 3.0 Processor

Combines image, video, and text processing for CitrusV 3.0.
"""

import json
import os
from typing import List, Optional, Union
import torch
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.processing_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.processing_utils import render_jinja_template
from transformers.processing_utils import AllKwargsForChatTemplate

from .image_processing_citrus_v_3 import CitrusV3ImageProcessor
from .video_processing_citrus_v_3 import CitrusV3VideoProcessor
from .volume_processing_citrus_v_3 import CitrusV3VolumeProcessor
from .citrus_v_utils import VolumeInput


class CitrusV3Processor(Qwen3VLProcessor):
    """
    CitrusV 3.0 Processor combining all modalities.
    
    Wraps CitrusV 3.0 image processor, video processor, volume processor, and Qwen tokenizer.
    
    Features:
    - Image processing: standard 2D images
    - Video processing: temporal sequences
    - Volume processing: 3D medical images (NIfTI, DICOM, NumPy)
    - Full Qwen3-VL compatibility
    
    The volume_processor is automatically initialized from image_processor config if not provided,
    making it seamlessly available as processor.volume_processor.
    
    Example:
        ```python
        from transformers import AutoTokenizer
        from citrus_v_3 import CitrusV3Processor, CitrusV3ImageProcessor, CitrusV3VideoProcessor
        
        image_processor = CitrusV3ImageProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        video_processor = CitrusV3VideoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        
        processor = CitrusV3Processor(
            image_processor=image_processor,
            video_processor=video_processor,
            tokenizer=tokenizer
        )
        # volume_processor is automatically created!
        
        # Now you can use all three processors:
        # - processor.image_processor (for 2D images)
        # - processor.video_processor (for videos)
        # - processor.volume_processor (for 3D medical volumes)
        
        # Process 3D medical volume
        volume_output = processor.volume_processor(
            volumes=volume_tensor,
            volume_metadata=metadata,
            return_tensors='pt'
        )
        ```
    """
    
    config_type = "qwen3_vl"  # Use same config type as Qwen3VLProcessor
    # Also support citrus_v_3 config type for AutoProcessor
    _processor_class = "CitrusV3Processor"
    attributes = ["image_processor", "tokenizer", "video_processor"]
    volume_processor_class = "CitrusV3VolumeProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        volume_processor=None,
        chat_template=None,
        **kwargs
    ):
        """
        Initialize CitrusV 3.0 Processor.
        
        Args:
            image_processor: CitrusV3ImageProcessor or Qwen2VLImageProcessor
            tokenizer: Qwen tokenizer
            video_processor: CitrusV3VideoProcessor or Qwen3VLVideoProcessor
            volume_processor: CitrusV3VolumeProcessor for 3D medical volumes
            chat_template: Chat template string
            **kwargs: Additional arguments
        """
        # Call parent __init__ first (this handles loading from pretrained)
        # volume_processor will be handled separately in from_pretrained or created here
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs
        )
        self.image_3d_token = '<|image_3d_pad|>' if not hasattr(tokenizer, 'image_3d_token') else tokenizer.image_3d_token
        self.image_3d_token_id = (
            tokenizer.image_3d_token_id if hasattr(tokenizer, 'image_3d_token_id') 
            else tokenizer.convert_tokens_to_ids(self.image_3d_token)
        )
        
        # Create volume_processor if not provided
        if volume_processor is None and image_processor is not None:
            # Create CitrusV3VolumeProcessor with same config as image_processor
            volume_processor = CitrusV3VolumeProcessor(
                patch_size=image_processor.patch_size,
                temporal_patch_size=getattr(image_processor, 'temporal_patch_size', 2),
                merge_size=image_processor.merge_size,
            )
        
        # Add volume_processor as an attribute after parent initialization
        self.volume_processor = volume_processor
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load processor from pretrained.
        
        Overrides parent to handle volume_preprocessor_config.json loading.
        """
        # Call parent to get the main processor with sub-components
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Try to load volume_processor config if it doesn't exist or if we need to reload it
        if processor.volume_processor is not None:
            # Check if volume_preprocessor_config.json exists
            pretrained_name = str(pretrained_model_name_or_path)
            if os.path.isdir(pretrained_name):
                volume_config_path = os.path.join(pretrained_name, 'volume_preprocessor_config.json')
                if os.path.exists(volume_config_path):
                    try:
                        with open(volume_config_path, 'r') as f:
                            volume_config = json.load(f)
                        processor.volume_processor = CitrusV3VolumeProcessor.from_dict(volume_config)
                    except Exception:
                        # Keep the existing volume_processor
                        pass
        
        return processor
        
        # super().__init__(
        #     image_processor=image_processor,
        #     tokenizer=tokenizer,
        #     video_processor=video_processor,
        #     chat_template=chat_template,
        #     **kwargs
        # )
        
        # Add volume_processor as an attribute
    
    def get_processor_info(self) -> dict:
        """
        Get information about processor capabilities.
        
        Returns:
            Dictionary with processor features
        """
        return {
            "supports_images": True,
            "supports_videos": True,
            "supports_volumes": True,
            "supports_nifti": True,
            "supports_dicom": True,
            "supports_numpy": True,
            "base_model": "Qwen3-VL",
            "version": "3.0",
            "processors": {
                "image_processor": self.image_processor.__class__.__name__,
                "video_processor": self.video_processor.__class__.__name__ if self.video_processor else None,
                "volume_processor": self.volume_processor.__class__.__name__ if self.volume_processor else None,
            }
        }
    
    def __call__(
        self,
        images: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        text: Optional[Union[str, List[str]]] = None,
        videos: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        volumes: Optional[VolumeInput] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Process inputs for images, videos, volumes, and text.
        
        Args:
            images: Image inputs
            text: Text input
            videos: Video inputs
            volumes: Volume inputs (3D medical images)
            **kwargs: Additional arguments
        
        Returns:
            BatchFeature with processed inputs
        """
        # Handle volumes if provided (similar to videos processing)
        if volumes is not None:
            volumes_inputs = self.volume_processor(volumes=volumes, **kwargs)
            # Check if volume_grid_thw exists (BatchFeature wraps data in .data)
            if hasattr(volumes_inputs, 'data') and "volume_grid_thw" in volumes_inputs.data:
                volume_grid_thw = volumes_inputs.data["volume_grid_thw"]
            else:
                volume_grid_thw = None
            
            # If user has not requested volume metadata, remove it
            if "return_metadata" not in kwargs and hasattr(volumes_inputs, 'data'):
                volume_metadata = volumes_inputs.data.pop("volume_metadata", None)
            elif hasattr(volumes_inputs, 'data'):
                volume_metadata = volumes_inputs.data.get("volume_metadata", None)
            else:
                volume_metadata = None
        else:
            volumes_inputs = {}
            volume_grid_thw = None
            volume_metadata = None
        
        # Prepare text with volume token replacement before calling parent
        # Follow Qwen3VL's pattern: use placeholder then replace back
        text_with_volumes = text
        if volume_grid_thw is not None and text is not None:
            text_list = text if isinstance(text, list) else [text]
            text_list = text_list.copy()
            merge_length = self.volume_processor.merge_size**2
            index = 0
            for i in range(len(text_list)):
                while self.image_3d_token in text_list[i]:
                    num_volume_tokens = volume_grid_thw[index].prod() // merge_length
                    # Replace with placeholder
                    text_list[i] = text_list[i].replace(self.image_3d_token, "<|placeholder|>" * num_volume_tokens, 1)
                    index += 1
                # Replace placeholder back with image_3d_token
                text_list[i] = text_list[i].replace("<|placeholder|>", self.image_3d_token)
            text_with_volumes = text_list
        
        # Process images, videos, and text using parent (with modified text)
        result = super().__call__(images=images, text=text_with_volumes, videos=videos, **kwargs)
        
        # Merge volumes inputs if present
        if volumes is not None:
            # volumes_inputs is a BatchFeature, merge its .data into result.data
            if hasattr(volumes_inputs, 'data'):
                result.data.update(volumes_inputs.data)
            else:
                result.data.update(volumes_inputs)
        
        return result

    def apply_chat_template(
        self,
        conversation: Union[list[dict[str, str]], list[list[dict[str, str]]]],
        chat_template: Optional[str] = None,
        **kwargs: "Unpack[AllKwargsForChatTemplate]",
    ) -> str:
        """
        Extend chat templating to support volumes in addition to images/videos.

        Accepts messages with content items of type "volume" (or "image_3d").
        When tokenize=True and return_dict=True, this returns a BatchFeature with
        text inputs plus processed visuals, including volumes.
        """
        # Largely mirrors transformers.processing_utils.ProcessorMixin.apply_chat_template
        # but adds collection/feeding of volumes.
        if chat_template is None:
            if isinstance(self.chat_template, dict) and "default" in self.chat_template:
                chat_template = self.chat_template["default"]
            elif isinstance(self.chat_template, dict):
                raise ValueError(
                    'The processor has multiple chat templates but none of them are named "default". You need to specify '
                    f"which one to use by passing the `chat_template` argument. Available templates are: {', '.join(self.chat_template.keys())}"
                )
            elif self.chat_template is not None:
                chat_template = self.chat_template
            else:
                raise ValueError("Cannot use apply_chat_template because this processor does not have a chat template.")
        else:
            if isinstance(self.chat_template, dict) and chat_template in self.chat_template:
                # It's the name of a template, not a full template string
                chat_template = self.chat_template[chat_template]
            else:
                # It's a template string, render it directly
                pass

        is_tokenizers_fast = hasattr(self, "tokenizer") and self.tokenizer.__class__.__name__.endswith("Fast")

        if kwargs.get("continue_final_message", False):
            if kwargs.get("add_generation_prompt", False):
                raise ValueError(
                    "continue_final_message and add_generation_prompt are not compatible."
                )
            if kwargs.get("return_assistant_tokens_mask", False):
                raise ValueError("continue_final_message is not compatible with return_assistant_tokens_mask.")

        if kwargs.get("return_assistant_tokens_mask", False):
            if not is_tokenizers_fast:
                raise ValueError(
                    "`return_assistant_tokens_mask` is not possible with slow tokenizers."
                )
            else:
                kwargs["return_offsets_mapping"] = True
        
        # Fill sets of kwargs that should be used by different parts of template
        processed_kwargs = {
            "mm_load_kwargs": {},
            "template_kwargs": {},
        }

        for kwarg_type in processed_kwargs:
            for key in AllKwargsForChatTemplate.__annotations__[kwarg_type].__annotations__:
                kwarg_type_defaults = AllKwargsForChatTemplate.__annotations__[kwarg_type]
                default_value = getattr(kwarg_type_defaults, key, None)
                value = kwargs.pop(key, default_value)
                if value is not None and not isinstance(value, dict):
                    processed_kwargs[kwarg_type][key] = value

        # pop unused and deprecated kwarg
        kwargs.pop("video_load_backend", None)

        # Pass unprocessed custom kwargs
        processed_kwargs["template_kwargs"].update(kwargs)

        # Batch shape detection
        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "content")
        ):
            is_batched = True
            conversations = conversation
        else:
            is_batched = False
            conversations = [conversation]

        tokenize = processed_kwargs["template_kwargs"].pop("tokenize", False)
        return_dict = processed_kwargs["template_kwargs"].pop("return_dict", False)
        mm_load_kwargs = processed_kwargs["mm_load_kwargs"]

        if tokenize:
            batch_images, batch_videos, batch_volumes = [], [], []
            batch_audios = []
            for conversation in conversations:
                images, videos, volumes = [], [], []
                for message in conversation:
                    visuals = [contect for contect in message["content"] if contect.get("type") in ["image", "video", "volume", "image_3d"]]
                    audio_fnames = [
                        contect[key]
                        for content in message["content"]
                        for key in ["audio", "url", "path"]
                        if key in content and content["type"] == "audio"
                    ]
                    image_fnames = [
                        vision_info[key]
                        for vision_info in visuals
                        for key in ["image", "url", "path", "base64"]
                        if key in vision_info and vision_info["type"] == "image"
                    ]
                    images.extend(image_fnames)
                    video_fnames = [
                        vision_info[key]
                        for vision_info in visuals
                        for key in ["video", "url", "path"]
                        if key in vision_info and vision_info["type"] == "video"
                    ]
                    videos.extend(video_fnames)

                    volume_fnames = [
                        vision_info[key]
                        for vision_info in visuals
                        for key in ["volume", "url", "path", "image_3d"]
                        if key in vision_info and vision_info["type"] in ["volume", "image_3d"]
                    ]
                    volumes.extend(volume_fnames)
                    
                    if not mm_load_kwargs["load_audio_from_video"]:
                        for fname in audio_fnames:
                            batch_audios.append(load_audio(fname, sampling_rate=mm_load_kwargs["sampling_rate"]))
                    else:
                        for fname in video_fnames:
                            batch_audios.append(load_audio(fname, sampling_rate=mm_load_kwargs["sampling_rate"]))

                batch_images.append(images)
                batch_videos.append(videos)
                batch_volumes.append(volumes)

        # Render prompt via Jinja template
        prompt, generation_indices = render_jinja_template(
            conversations=conversations,
            chat_template=chat_template,
            **processed_kwargs["template_kwargs"],
            **self.tokenizer.special_tokens_map,
        )

        if not is_batched:
            prompt = prompt[0]

        if tokenize:
            single_prompt = prompt[0] if is_batched else prompt
            if self.tokenizer.bos_token is not None and isinstance(single_prompt, str) and single_prompt.startswith(self.tokenizer.bos_token):
                kwargs["add_special_tokens"] = False

            # Always sample frames by default unless explicitly set to `False` by users. If users do not pass `num_frames`/`fps`
            # sampling should not done for BC.
            if "do_sample_frames" not in kwargs and (
                kwargs.get("fps") is not None or kwargs.get("num_frames") is not None
            ):
                kwargs["do_sample_frames"] = True

            # Call into processor with collected modalities, including volumes
            images_exist = any((im is not None) for im_list in batch_images for im in im_list)
            videos_exist = any((vi is not None) for vi_list in batch_videos for vi in vi_list)
            volumes_exist = any((vo is not None) for vo_list in batch_volumes for vo in vo_list)

            out = self(
                text=prompt,
                images=batch_images if images_exist else None,
                videos=batch_videos if videos_exist else None,
                volumes=batch_volumes if volumes_exist else None,
                **kwargs,
            )

            if return_dict:
                # For assistant mask parity with HF if requested
                if processed_kwargs["template_kwargs"].get("return_assistant_tokens_mask", False):
                    assistant_masks = []
                    offset_mapping = out.pop("offset_mapping")
                    input_ids = out["input_ids"]
                    for i in range(len(input_ids)):
                        current_mask = [0] * len(input_ids[i])
                        offsets = offset_mapping[i]
                        offset_starts = [start for start, end in offsets]
                        for assistant_start_char, assistant_end_char in generation_indices[i]:
                            import bisect as _bisect
                            start_pos = _bisect.bisect_left(offset_starts, assistant_start_char)
                            end_pos = _bisect.bisect_left(offset_starts, assistant_end_char)

                            if not (
                                start_pos >= 0
                                and offsets[start_pos][0] <= assistant_start_char < offsets[start_pos][1]
                            ):
                                continue
                            for token_id in range(start_pos, end_pos if end_pos else len(input_ids[i])):
                                current_mask[token_id] = 1
                        assistant_masks.append(current_mask)
                    out["assistant_masks"] = assistant_masks
                    out.convert_to_tensors(tensor_type=kwargs.get("return_tensors"))
                return out
            else:
                return out["input_ids"]

        return prompt