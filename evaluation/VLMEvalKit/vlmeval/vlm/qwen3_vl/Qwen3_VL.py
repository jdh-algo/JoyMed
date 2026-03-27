import os
import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoProcessor

class Qwen3VL:
    def __init__(self,model_path,
                 temperature=0,
                 top_p = 0.001, 
                 repetition_penalty = 1,
                 max_tokens = 1024,
                ):
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

        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_tokens
        

    def process_messages(self,messages):
        new_messages = []
        if "system" in messages:
            new_messages.append({"role":"system","content":messages["system"]}) 
        if "image" in messages:
            new_messages.append({"role":"user","content":[{"type":"image","image":messages["image"]},{"type":"text","text":messages["prompt"]}]})
        elif "images" in messages:
            content = []
            for i,image in enumerate(messages["images"]):
                content.append({"type":"text","text":f"<image_{i+1}>: "})
                content.append({"type":"image","image":image})
            content.append({"type":"text","text":messages["prompt"]})
            new_messages.append({"role":"user","content":content})
        else:
            new_messages.append({"role":"user","content":[{"type":"text","text":messages["prompt"]}]})
        messages = new_messages
        inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
        )

        return inputs


    def generate_output(self,messages):
        inputs = self.process_messages(messages)
        do_sample = False if self.temperature == 0 else True
        generated_ids = self.llm.generate(**inputs,temperature=self.temperature,top_p=self.top_p,repetition_penalty=self.repetition_penalty,max_new_tokens=self.max_new_tokens,do_sample = do_sample)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in messages_list:
            result = self.generate_output(messages)
            res.append(result)
        return res
