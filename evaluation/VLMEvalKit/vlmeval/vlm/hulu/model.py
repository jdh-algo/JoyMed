from __future__ import annotations

import os
import sys
import warnings
import math
import logging

import torch
# from transformers import StoppingCriteria

from ..base import BaseModel
# from .prompt import Qwen2VLPromptMixin
# from ...smp import get_gpu_memory, listinstr
# from ...dataset import DATASET_MODALITY

class HuluChat(BaseModel):
    def __init__(self, model_path: str, use_qwen3=False) -> Any:
        super().__init__()
        if use_qwen3:
            from .Hulu_qwen3 import Hulu
        else:
            from .Hulu import Hulu
            
        self.model = Hulu(model_path)


    def process_message(self, message):
        text, images = "", []
        for item in message:
            if item["type"] == "text":
                text += item["value"]
            elif item["type"] == "image":
                images.append(item["value"])
        message = {"images":images, "prompt":text}
        
        return message

    def generate_inner(self, message, dataset=None):
        message = self.process_message(message)
        output = self.model.generate_output(message)
        return output
