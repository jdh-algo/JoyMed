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

class HealthGPTChat(BaseModel):
    def __init__(self, model_path: str) -> Any:
        super().__init__()
        if "Phi-3" in model_path:
            from .HealthGPT_phi3 import HealthGPT
        elif "Phi-4" in model_path:
            from .HealthGPT_phi4 import HealthGPT
        else:
            from .HealthGPT import HealthGPT
            
        self.model = HealthGPT(model_path)


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
