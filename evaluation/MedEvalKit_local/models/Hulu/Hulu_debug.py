import os
import sys
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(BASE_DIR)
import time
from tqdm import tqdm
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


class Hulu:
    def __init__(self,model_path):
        super().__init__()

        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype="bfloat16",
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
        except:
            time.sleep(10)
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype="bfloat16",
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        self.tokenizer = self.processor.tokenizer


        self.temperature = 0
        self.top_p = 0.001
        self.repetition_penalty = 1
        self.max_new_tokens = 1024


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
                new_messages.append({"role":"user","content":[{"type":"image","image":messages["image"]},{"type":"text","text":messages["prompt"]}]})
            elif "images" in messages:
                content = []
                for i,image in enumerate(messages["images"]):
                    # content.append({"type":"text","text":f"<image_{i+1}>: "})
                    content.append({"type":"image","image":image})
                content.append({"type":"text","text":messages["prompt"]})
                new_messages.append({"role":"user","content":content})
            elif "images_3d" in messages:
                video_dir = messages["images_3d"]
                new_messages.append({"role":"user","content":[{"type":"video","video":{
                                                                            "video_path":video_dir,
                                                                            "fps": 1,
                                                                            "max_frames": 1800,
                                                                        }},
                                                              {"type":"text","text":messages["prompt"]}]})
                
            else:
                # print(messages)
                new_messages.append({"role":"user","content":[{"type":"text","text":messages["prompt"]}]})

        messages = new_messages
        
        inputs = self.processor(
            conversation=messages,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        inputs = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        return inputs


    def generate_output(self,messages):
        inputs = self.process_messages(messages)


        do_sample = False if self.temperature == 0 else True
        with torch.inference_mode():
            output_ids = self.llm.generate(
                **inputs,
                do_sample=do_sample,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, use_think=False)[0].strip()
        
        return outputs
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in tqdm(messages_list, desc="Processing", total=len(messages_list), unit="sample"):
            result = self.generate_output(messages)
            res.append(result)
        return res

      
if __name__ == "__main__":
    model_path="/mnt/workspace/offline/shared_model/Hulu/Hulu-Med-32B"
    hulu = Hulu(model_path)
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "3d",
                    "3d": {
                        "image_path": "/mnt/workspace/offline/shared_data/AMOS/MM/imagesVa/amos_0128.nii.gz",
                        "nii_num_slices": 99,
                        "nii_axis": 2,  # 0=sagittal, 1=coronal, 2=axial
                    }
                },
                {
                    "type": "text",
                    "text": "Provide a detailed description of the given 3D volume, including all relevant findings and a diagnostic assessment. Return the report in the following format: Findings: {} Impression: {}."
                },
            ]
        }
    ]

    inputs = hulu.processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
              for k, v in inputs.items()}
    
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    
    output_ids = hulu.llm.generate(
        **inputs,
        do_sample=False,
        temperature=hulu.temperature,
        max_new_tokens=hulu.max_new_tokens,
        use_cache=True,
        pad_token_id=hulu.tokenizer.eos_token_id,
        top_p=hulu.top_p,
        repetition_penalty=hulu.repetition_penalty,
    )
    outputs = hulu.processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        use_think=False
    )[0].strip()
    print(outputs)