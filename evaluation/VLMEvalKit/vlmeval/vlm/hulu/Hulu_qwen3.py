import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(BASE_DIR)

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
# from hulumed import disable_torch_init, model_init, mm_infer
# from hulumed.mm_utils import load_images, process_images, load_video, process_video, tokenizer_multimodal_token, get_model_name_from_path, KeywordsStoppingCriteria

from hulumed_qwen3.model import load_pretrained_model
from hulumed_qwen3.mm_utils import load_images, get_model_name_from_path
from hulumed_qwen3.model.processor import HulumedProcessor

class Hulu:
    def __init__(self,model_path, 
                 temperature=0,
                 top_p = 0.001, 
                 repetition_penalty = 1,
                 max_tokens = 1024,
                ):
        super().__init__()
        model_name = get_model_name_from_path(model_path)
        tokenizer, self.llm, image_processor, context_len = load_pretrained_model(model_path, None, model_name, \
                                                                                  torch_dtype="bfloat16", device_map='auto', \
                                                                                  attn_implementation="flash_attention_2")

        self.tokenizer = tokenizer
        self.processor = HulumedProcessor(image_processor, tokenizer)

        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_tokens



    def process_messages(self,messages):
        new_messages = []
        images = []
        if "system" in messages:
            new_messages.append({"role":"system","content":messages["system"]}) 
        if "messages" in messages:
            messages = messages["messages"]
            for message in messages:
                role = message["role"]
                content = message["content"]
                new_messages.append({"role":role,"content":[{"type":"text","text":content}]})
        else:
            if "image" in messages:
                new_messages.append({"role":"user","content":[{"type":"image"},{"type":"text","text":messages["prompt"]}]})
                # new_messages.append({"role":"user","content":[{"type":"image","image":messages["image"]},{"type":"text","text":messages["prompt"]}]})
                images.append(load_images(messages["image"]))
            elif "images" in messages:
                content = []
                for i,image in enumerate(messages["images"]):
                    content.append({"type":"text","text":f"<image_{i+1}>: "})
                    content.append({"type":"image"})
                    
                images.append(load_images(messages["images"]))
                
                content.append({"type":"text","text":messages["prompt"]})
                new_messages.append({"role":"user","content":content})
            else:
                # print(messages)
                new_messages.append({"role":"user","content":[{"type":"text","text":messages["prompt"]}]})

        messages = new_messages

        inputs = self.processor(
            images=images if images != [] else None,
            text=messages,
            merge_size=1,
            return_tensors="pt",
)
        inputs = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        modal = "text" if images==[] else "image"
        return inputs, modal


    def generate_output(self,messages):
        inputs, modal = self.process_messages(messages)


        do_sample = False if self.temperature == 0 else True
        with torch.inference_mode():
            output_ids = self.llm.generate(
                **inputs,
                do_sample=do_sample,
                modals=[modal],
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        return outputs
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in tqdm(messages_list, desc="Processing", total=len(messages_list), unit="sample"):
            result = self.generate_output(messages)
            res.append(result)
        return res

      
