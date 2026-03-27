import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from tqdm import tqdm
import os
import time
from ..image_process import load_images

class Hulu:

    def __init__(self, model_path, args):

        super().__init__()
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
        except:
            time.sleep(5)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
            
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
        except:
            time.sleep(5)
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
        self.tokenizer = self.processor.tokenizer
        self.model.eval()

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.max_new_tokens = args.max_new_tokens
        self.repetition_penalty = args.repetition_penalty
        
    def process_messages(self, messages):
       
        prompt = messages.get("prompt", "")
        
        conversation = [{"role": "user", "content": []}]
        
        loaded_images = None
        if "images_3d" in messages:
            if isinstance(messages["images_3d"], dict):
                loaded_images = load_images(**messages["images_3d"])
            else:
                image_paths_or_pil = messages["images_3d"]
                loaded_images = load_images(image_paths_or_pil)
            conversation[0]["content"].append({"type": "video", "num_frames": len(loaded_images)})
            
        if "3d" in messages:
            conversation[0]["content"].append({"type": "3d", "3d": messages['3d']})
            
        else:
            image_paths_or_pil = messages.get("images") or ([messages["image"]] if "image" in messages else [])
            if image_paths_or_pil:
                loaded_images = load_images(image_paths_or_pil)
                for _ in loaded_images:
                    conversation[0]["content"].append({"type": "image"})
                
        conversation[0]["content"].append({"type": "text", "text": prompt})
        try:
            inputs = self.processor(
                images=[loaded_images] if loaded_images is not None else None,
                conversation=conversation,
                add_system_prompt=False,
                add_generation_prompt=True,
                return_tensors="pt"
            )
        except:
            import ipdb; ipdb.set_trace()
        
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
            
        return inputs

    def generate_output(self, messages):

        llm_inputs = self.process_messages(messages)
        do_sample = True if self.temperature > 0 else False
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                **llm_inputs,
                do_sample=False,
                temperature=self.temperature if do_sample else 0,
                #top_p=self.top_p,
                repetition_penalty = self.repetition_penalty,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, use_think=True)[0].strip()
        # outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # print(outputs)
        return outputs
    
    def generate_outputs(self, messages_list):

        res = []
        for messages in tqdm(messages_list, desc="Generating Outputs"):
            result = self.generate_output(messages)
            # print(result)
            res.append(result)
        return res
