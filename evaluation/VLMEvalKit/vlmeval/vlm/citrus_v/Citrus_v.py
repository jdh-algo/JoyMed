import torch
from PIL import Image
from tqdm import tqdm
import os
from . import citrus_v_3
from transformers import AutoModelForCausalLM, AutoProcessor

class Citrus_v:

    def __init__(self, model_path, 
                 temperature=0,
                 top_p = 0.001, 
                 repetition_penalty = 1,
                 max_tokens = 1024,
                ):

        super().__init__()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model.eval()

        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_tokens
        
    def process_messages(self, messages):
        print(f"messages: {messages}")

        if isinstance(messages, dict):
            # added by xuyang 1210, AMOS-MM-report process add additional prompt
            if ('images_3d' in messages) and  "Findings" in messages['prompt'] and "AMOS" in messages['images_3d']:
                messages['prompt'] += " This imaging study may include multiple anatomical regions."
            # twen:1215, 3D-RAD and M3D open question always very short
            if ('images_3d' in messages) and ("Please answer the question concisely." in messages['prompt']) and ("3D-RAD" in messages['images_3d']):
                messages['prompt'] = messages['prompt'].replace('Please answer the question concisely.', 'Answer the question using a single word or phrase.')
            print(f"messages: {messages}")

            new_messages = []
            if "system" in messages:
                new_messages.append({"role": "system", "content": [{"type": "text", "text": messages["system"]}]})

            if "messages" in messages:
                messages = messages["messages"]
                for message in messages:
                    role = message["role"]
                    content = message["content"]
                    new_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
            else:
                if "image" in messages:
                    new_messages.append({"role": "user","content": [{"type": "image", "image": messages["image"]},{"type": "text", "text": messages["prompt"]}]})
                elif "images" in messages:
                    content = []
                    for i, image in enumerate(messages["images"]):
                        content.append({"type": "image", "image": image})
                    content.append({"type": "text", "text": messages["prompt"]})
                    new_messages.append({"role": "user", "content": content})
                elif "images_3d" in messages:
                    # print(f"images_3d: {messages['images_3d']}")
                    if isinstance(messages['images_3d'], dict):
                        volume_path = messages['images_3d']['image_path']
                    else:
                        volume_path = messages['images_3d']
                    # new_messages.append({"role": "user","content": [{"type": "video", "video": load_images_folder(volume_path)},{"type": "text", "text": messages["prompt"]}]})
                    new_messages.append({"role": "user","content": [{"type": "image_3d", "image_3d": volume_path},{"type": "text", "text": messages["prompt"]}]})
                else:
                    new_messages.append({"role": "user", "content": [{"type": "text", "text": messages["prompt"]}]})


            messages = new_messages
        elif isinstance(messages, list):
            pass
        else:
            raise ValueError(f"messages must be list or dict, but found type {type(messages)})")

        # print(f"== messages: {messages}")

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        return inputs


    def generate_output(self, messages):

        inputs = self.process_messages(messages)
        do_sample = True if self.temperature > 0 else False
        
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=self.temperature if do_sample else 0,
                top_p=self.top_p,
                repetition_penalty = self.repetition_penalty,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        
        print(outputs)
        return outputs
    
    def generate_outputs(self, messages_list):

        res = []
        for messages in tqdm(messages_list, desc="Generating Outputs"):
            result = self.generate_output(messages)
            # print(result)
            res.append(result)
        return res
