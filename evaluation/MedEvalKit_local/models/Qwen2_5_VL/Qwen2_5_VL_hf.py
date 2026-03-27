from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor,Qwen2_5_VLProcessor

from transformers import logging
from tqdm import tqdm
from ..image_process import load_images, resize_video

logging.set_verbosity_error()

class Qwen2_5_VL:
    def __init__(self,model_path,args):
        super().__init__()
        self.llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="bfloat16", device_map="auto",attn_implementation="flash_attention_2")
        
        # min_pixels = 256*28*28
        # max_pixels = 1280*28*28
        ##qwen默认temporal_patch_size=2，hulu默认temporal_patch_size=1，修改config会报错，猜测其他地方也会用到config中的temporal_patch_size,导致某些变量不一致
        self.processor = AutoProcessor.from_pretrained(model_path)#, min_pixels=min_pixels, max_pixels=max_pixels)

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens


    def process_messages(self,messages):
        new_messages = []
        if "system" in messages:
            new_messages.append({"role": "system", "content": [{"type": "text", "text": messages["system"]}]})
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
                    content.append({"type":"text","text":f"<image_{i+1}>: "})
                    content.append({"type":"image","image":image})
                content.append({"type":"text","text":messages["prompt"]})
                new_messages.append({"role":"user","content":content})
            elif "images_3d" in messages:
                
                content = []
                if isinstance(messages["images_3d"], dict):
                    loaded_images = load_images(**messages["images_3d"])
                else:
                    loaded_images = load_images(messages["images_3d"])
                    
                content.append({"type":"video","video":loaded_images})
                content.append({"type":"text","text":messages["prompt"]})
                new_messages.append({"role":"user","content":content})
            else:
                # print(messages)
                new_messages.append({"role":"user","content":[{"type":"text","text":messages["prompt"]}]})

        messages = new_messages
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        if not video_inputs is None:
            ##注意video_process无法处理视频帧size不一致的情况,使用resize_video将同一个video中的image resize到一致
            video_inputs = resize_video(video_inputs)
            
            image_number = sum([len(video_input) for video_input in video_inputs])
            self.processor.video_processor.max_pixels = 12845056/image_number #*self.processor.video_processor.temporal_patch_size
            size = {"shortest_edge": self.processor.video_processor.min_pixels, "longest_edge": self.processor.video_processor.max_pixels}
            inputs = self.processor(
                        text=[prompt],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                        size=size,
                    )
                    
        else:
                    
            inputs = self.processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        inputs = inputs.to("cuda")

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
        for messages in tqdm(messages_list, desc="Processing", total=len(messages_list), unit="sample"):
            result = self.generate_output(messages)
            res.append(result)
        return res

        