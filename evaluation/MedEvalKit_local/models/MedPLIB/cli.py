"""
A model worker executes the model.
Usage:

CUDA_VISIBLE_DEVICES=0 python -m model.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 \
    --worker http://localhost:40000 --model-path checkpoints/xxx \
    --multi-modal --add_region_feature

"""
import argparse
import asyncio
import json
import time
import uuid
import base64
import os
import numpy as np
import torch
import uvicorn
import cv2
import sys

import deepspeed
from PIL import Image

from .model.medplib.constants import WORKER_HEART_BEAT_INTERVAL
from .model.medplib.utils import (build_logger, server_error_msg,
    pretty_print_semaphore)
from .model.medplib.model.builder import load_pretrained_model
from .model.medplib.mm_utils import process_images, load_image_from_base64, tokenizer_image_token, KeywordsStoppingCriteria
from .model.medplib.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from .utils.utils import REGION_TOKEN_INDEX, DEFAULT_REGION_REFER_TOKEN_0, DEFAULT_REGION_REFER_TOKEN_1

from .model.segment_anything.utils.transforms import ResizeLongestSide


deepspeed.init_distributed(dist_backend='nccl')
class MedPLIBChatbot():

    # for sam
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    # for clip
    clip_pixel_mean = (torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)*255).clamp(0, 255).to(torch.int)
    clip_pixel_std = (torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)*255).clamp(0, 255).to(torch.int)

    def __init__(self, 
                 model_path, model_base=None, model_name=None,
                 load_8bit=False, load_4bit=False, load_fp16=False,
                 temperature=0,max_new_tokens=1024,top_p=None,
                 device='cuda'):

        self.device = device
        vision_pretrained = os.path.join(os.path.dirname(model_path), 'sam-med2d_b.pth')
        # vision_pretrained = None
        vision_tower = os.path.join(os.path.dirname(model_path), '../clip-vit-large-patch14-336')
                     
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        print(f"Loading the model {self.model_name} ...")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name, load_8bit, load_4bit, load_fp16, device_map=self.device, vision_pretrained=vision_pretrained, vision_tower=vision_tower)
            
        
        self.is_multimodal = 'llava' in self.model_name.lower() or 'medplib' in self.model_name.lower()

        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.sam_img_size = 256
        self.transform = ResizeLongestSide(self.sam_img_size)
        self.clip_img_size = 336
        self.transform_clip = ResizeLongestSide(self.clip_img_size)
        self.precision = "bf16"
        self.seg_token_idx = self.tokenizer("<SEG>", add_special_tokens=False).input_ids[0]
        self.history = []


    def pad_tensor_channelwise(self, x, pad_h, pad_w, pad_values, is_mask=False):
        """
        Pad a 3-channel image tensor with different padding values for each channel,
        considering total padding length and odd padding size.

        Parameters:
        x (torch.Tensor): Input image tensor of shape (3, h, w).
        pad_h (int): Total padding size for the height.
        pad_w (int): Total padding size for the width.
        pad_values (tuple): A tuple of three elements specifying the padding value for each channel.

        Returns:
        torch.Tensor: Padded image tensor.
        """

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if is_mask:
            assert len(pad_values) == 1, "pad_values must have 1 elements, one for each channel."
            padded_tensor = torch.empty((x.shape[0] + pad_h, x.shape[1] + pad_w), dtype=x.dtype)
            padded_tensor[:, :] = pad_values[0]
            padded_tensor[pad_top:pad_top+x.shape[0], pad_left:pad_left+x.shape[1]] = x
        else:
            assert len(pad_values) == 3, "pad_values must have three elements, one for each channel."
            padded_tensor = torch.empty((3, x.shape[1] + pad_h, x.shape[2] + pad_w), dtype=x.dtype)
            for i in range(3):
                padded_tensor[i, :, :] = pad_values[i]
            padded_tensor[:, pad_top:pad_top+x.shape[1], pad_left:pad_left+x.shape[2]] = x

        return padded_tensor


    def preprocess(self, x: torch.Tensor, image_size: int, normalize: bool=True, is_mask: bool=False) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        if normalize:
            x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = image_size - h
        padw = image_size - w
        if is_mask:
            x = self.pad_tensor_channelwise(x, padh, padw, torch.zeros(1), is_mask=True)
        else:
            # for sam. pad after normalize
            if normalize:
                x = self.pad_tensor_channelwise(x, padh, padw, torch.zeros(3))
                # x = x * self.pixel_std + self.pixel_mean

            # for clip. pad before normalize
            else:
                x = self.pad_tensor_channelwise(x, padh, padw, self.clip_pixel_mean)

        return x
    def get_image_tensors(self, images):
        num_image_tokens = 0
        if images is not None and len(images) > 0:
            list_image_tensors = []
            for image in images:
                # only support one image each time
                # image_rgb = np.array(load_image_from_base64(image))
                if isinstance(image, str):
                    image_rgb = Image.open(image).convert('RGB') # make sure that the path exists
                elif isinstance(image, Image.Image):
                    image_rgb = image.convert('RGB')
                image_rgb = np.array(image_rgb)
                origin_shape = image_rgb.shape[:2]
    
                #------------ preprocess image for sam ------------
                # image_resize = self.transform.apply_image(image_rgb)
                # resize_shape = image_resize.shape[:2]
                # image_sam = self
                
                #------------preprocess image for clip  ------------
                # c, h, w -> h, w, c
                image_clip = self.transform_clip.apply_image(image_rgb)
                #c, h, w
                image_clip = self.preprocess(torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(), self.clip_img_size, normalize=False)
                #c, h, w
                image_clip = self.image_processor.preprocess(image_clip, return_tensors='pt')['pixel_values'][0]
                list_image_tensors.append(image_clip)
            image_tensors = torch.stack(list_image_tensors).to(dtype=torch.bfloat16).to(self.device)

        else:
            image_tensors = None
            
        return image_tensors

   
    def preprocess_prompt(self, prompt, images_num) -> list: # tokenize and concat the coversations
        
        blacklist = ['<image>', '<s>', '</s>']
        for b in blacklist:
            prompt = prompt.replace(b, '')

            
        for _ in range(images_num):
            prompt = "<image>\n" + prompt

        convs = self.get_conv(prompt)
        
        input_ids = None
        convs = [ conv for conv in convs if conv['value'] is not None]
        round_num = len(convs)//2

        for ind in range(round_num):
            h = convs[ind*2]['value'].strip()
            h = f"<|user|>\n{h}\n" 

            g = convs[ind*2+1]['value']
            g = f"<|assistant|>\n{g} \n"

            cur_input_ids = tokenizer_image_token(h, self.tokenizer, return_tensors='pt')

            if input_ids is None:
                input_ids = cur_input_ids
            else:
                input_ids = torch.cat([input_ids, cur_input_ids])
            
            cur_input_ids = self.tokenizer(g, add_special_tokens= False, truncation=True, return_tensors='pt').input_ids[0]
            input_ids = torch.cat([input_ids, cur_input_ids])
        
        h = convs[-1]['value'].strip()
        h = f"<|user|>\n{h}\n<|assistant|>\n"
        cur_input_ids = tokenizer_image_token(h, self.tokenizer, return_tensors='pt')

        if input_ids is None:
            input_ids = cur_input_ids
        else:
            input_ids = torch.cat([input_ids, cur_input_ids])
        

        return input_ids


    def get_conv(self, text):
        ret = []
        if self.history is None:
            self.history = []
        
        for conv in self.history:
            ret.append({'from': 'human', 'value': conv[0]})
            ret.append({'from': 'gpt', 'value': conv[1]})

        ret.append({'from': 'human', 'value': text})
        ret.append({'from': 'gpt', 'value': None})

        return ret

  
    def inference(self, prompt, images=None, region_masks=None):
        if images is None:
            images = []
        if not isinstance(images,list):
            images = [images]
            
        images_num = len(images)
        input_ids = self.preprocess_prompt(prompt, images_num).unsqueeze(0).to(self.device)
            
        image_tensors = self.get_image_tensors(images)
        
        # indices = (input_ids == 29901).nonzero(as_tuple=True)
        # input_ids = input_ids[:, :indices[1][-1]+1]
        
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensors,
                attention_mask=attention_mask,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=1,
                # no_repeat_ngram_size=3,
                max_new_tokens=self.max_new_tokens,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        
        # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        # if n_diff_input_output > 0:
        #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        answer = outputs.strip()
        return answer

if __name__ == "__main__":
    


    model = MedPLIBChatbot(model_path="/mnt/workspace/offline/shared_model/MedPLIB/MedPLIB-7b-2e")
    answer = model.inference("渗透泵片处方中的渗透压活性物质是什么")
    print(answer)

