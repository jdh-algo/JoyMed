from transformers import AutoProcessor, AutoModelForImageTextToText
# from vllm import LLM, SamplingParams
import os
import torch

from tqdm import tqdm
from PIL import Image

from ..image_process import load_images, resize_video
from .utils import encode

torch._dynamo.config.disable = True

class MedGemma_1_5:
    def __init__(self,model_path,args):
        super().__init__()
        self.llm = AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens

    def process_messages(self,messages):
        current_messages = []
        imgs = []
        if "messages" in messages:
            messages = messages["messages"]
            for message in messages:
                role = message["role"]
                content = message["content"]
                current_messages.append({"role":role,"content":[{"type":"text","text":content}]}) 

        else:
            prompt = messages["prompt"] 
            if "system" in messages:
                system_prompt = messages["system"]
                current_messages.append({"role":"system","content":[{"type":"text","text":system_prompt}]})
            if "image" in messages:
                image = messages["image"]
                if isinstance(image,str):
                    image = Image.open(image)
                imgs.append(image)
                current_messages.append({"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]})
            elif "images" in messages:
                content = []
                for i,image in enumerate(messages["images"]):
                    content.append({"type":"text","text":f"<image_{i+1}>: "})
                    content.append({"type":"image","image":image})
                    if isinstance(image,str):
                        image = Image.open(image)
                    imgs.append(image)
                content.append({"type":"text","text":messages["prompt"]})
                current_messages.append({"role":"user","content":content})
            elif "images_3d" in messages:
                
                content = []
                if isinstance(messages["images_3d"], dict):
                    loaded_images = load_images(**messages["images_3d"])
                else:
                    loaded_images = load_images(messages["images_3d"])
                MAX_SLICE = 85
                if len(loaded_images) > MAX_SLICE:
                    loaded_images = [loaded_images[int(round(i /(MAX_SLICE) * (len(loaded_images)-1)))] for i in range(1, MAX_SLICE + 1)]

     
                content.append({"type": "text", "text": "You are an instructor teaching medical students. You are analyzing a contiguous block of CT slices . Please review the slices provided below carefully."})
                for slice_id, loaded_image in enumerate(loaded_images):
                  content.append({"type": "image", "image": encode(loaded_image)})
                  content.append({"type": "text", "text": f"SLICE {slice_id}"})
                content.append({"type": "text", "text": prompt})
                
                current_messages.append({"role":"user","content":content})
            else:
                current_messages.append({"role":"user","content":[{"type":"text","text":prompt}]}) 
            
            print("===prompt messages:", prompt)
        
        
        inputs = self.processor.apply_chat_template(
            current_messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.llm.device, dtype=torch.bfloat16)

        return inputs


    def generate_output(self,messages):
        llm_inputs = self.process_messages(messages)
        input_len = llm_inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            do_sample = False if self.temperature == 0 else True
            generation = self.llm.generate(**llm_inputs,max_new_tokens=self.max_new_tokens,do_sample = do_sample, pad_token_id= self.processor.tokenizer.eos_token_id,top_k = None,top_p = None)
            generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        print("===decoded:", decoded)
        return decoded
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in tqdm(messages_list):
            result = self.generate_output(messages)
            res.append(result)
        return res
