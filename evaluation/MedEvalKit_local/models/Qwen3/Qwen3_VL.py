import os
import json
import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoProcessor
from ..image_process import load_images, resize_video


class Qwen3VL:
    def __init__(self, model_path, args):
        super().__init__()
        if "B-A" in os.path.basename(model_path):
            self.llm = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        else:
            self.llm = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )

        self.processor = AutoProcessor.from_pretrained(model_path)

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens

    def process_messages(self, messages):
        if isinstance(messages, dict):
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
                    new_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": messages["image"]},
                                {"type": "text", "text": messages["prompt"]},
                            ],
                        }
                    )
                elif "images" in messages:
                    content = []
                    for i, image in enumerate(messages["images"]):
                        # content.append({"type":"text","text":f"<image_{i+1}>: "})
                        content.append({"type": "image", "image": image})
                    content.append({"type": "text", "text": messages["prompt"]})

                    new_messages.append({"role": "user", "content": content})
                elif "images_3d" in messages:
                    content = []
                    
                    if isinstance(messages["images_3d"], dict):
                        loaded_images = load_images(**messages["images_3d"])
                    else:
                        loaded_images = load_images(messages["images_3d"])
                    
                    ##注意video_process无法处理视频帧size不一致的情况,使用resize_video将同一个video中的image resize到一致
                    ##qwen3-processer要求video_frames>1
                    loaded_images = resize_video([loaded_images])[0] if len(loaded_images)>1 else [loaded_images[0],loaded_images[0]]
                    
                    image_number = len(loaded_images)
                    content.append({"type":"video","video":loaded_images})
                    content.append({"type":"text","text":messages["prompt"]})
                    new_messages.append({"role":"user","content":content})
                else:
                    new_messages.append({"role": "user", "content": [{"type": "text", "text": messages["prompt"]}]})

            messages = new_messages
        elif isinstance(messages, list):
            pass
        else:
            raise ValueError(f"messages must be list or dict, but found type {type(messages)})")

        
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        inputs = inputs.to(self.llm.device)

        return inputs

    def generate_output(self, messages):
        inputs = self.process_messages(messages)
        do_sample = False if self.temperature == 0 else True
        generated_ids = self.llm.generate(
            **inputs,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
        )
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def generate_outputs(self, messages_list):
        res = []
        for messages in messages_list:
            result = self.generate_output(messages)
            res.append(result)
        return res
