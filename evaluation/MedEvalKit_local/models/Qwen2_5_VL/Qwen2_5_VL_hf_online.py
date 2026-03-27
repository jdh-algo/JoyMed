import requests
import base64
from PIL import Image
import io
from tqdm import tqdm

import json

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class Qwen2_5_VL:
    def __init__(self, api_url, args):
        super().__init__()
        self.api_url = api_url
        self.dataset = args.eval_datasets
        self.served_model_name = args.served_model_name
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.token}",
            "Accept": "application/json"
        }
        self.sampling_params = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "max_tokens": args.max_new_tokens,
        }

    def _convert_image_to_base64(self, image, max_size=(800, 800)):
        if isinstance(image, str):
            # print(image)
            image = Image.open(image)

        # if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        #     # print(image.size[0], image.size[1])
        #     image.thumbnail(max_size, Image.Resampling.LANCZOS)
        # print(image)
        image = image.convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str


    def process_messages(self,messages):
        current_messages = []

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
                current_messages.append({"role":"system","content":system_prompt})
            if "image" in messages:
                image = self._convert_image_to_base64(messages["image"])
                current_messages.append({"role":"user","content":[{"type":"image_url", "image_url":{"url":f"data:image;base64,{image}"}},{"type":"text","text":prompt}]})
            elif "images" in messages:
                content = []
                for i,image in enumerate(messages["images"]):
                    img = self._convert_image_to_base64(image)
                    content.append({"type":"text","text":f"<image_{i+1}>: "})
                    content.append({"type":"image_url", "image_url":{"url":f"data:image;base64,{img}"}})
                    break
                content.append({"type":"text","text":messages["prompt"]})
                current_messages.append({"role":"user","content":content})
            else:
                current_messages.append({"role":"user","content":[{"type":"text","text":prompt}]}) 

        payload = {
            "model": self.served_model_name,
            "messages": current_messages,
            **self.sampling_params
        }

        return payload

    def generate_output(self, messages):
        payload = self.process_messages(messages)

        retry_strategy = Retry(
            total=5,  # retry num
            backoff_factor=1,  # retry interval
            status_forcelist=[500, 502, 503, 504]
        )
    
        session = requests.Session()
        session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
        session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
        try:
            response = session.post(
                url=self.api_url,
                headers=self.headers,
                json=payload,
                timeout=600
            )
            
            response.raise_for_status()
            # response_json = response.json()
            
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"API request failed: {e}")
            # with open(f'error_content_{self.dataset}.txt', 'a') as f:
            #     f.write(str(e) + ':')
            #     f.write(str(payload))
            #     f.write('\n')
            # if "messages" in payload:
            #     for msg in payload["messages"]:
            #         if "content" in msg:
            #             with open('error_content_slake.txt', 'a') as f:
            #                 f.write(str(e) + ':')
            #                 f.write(str(msg["content"]))
            #                 f.write('\n')
            # raise
            return 'error'

    def generate_outputs(self, messages_list):
        results = []
        for messages in tqdm(messages_list, desc="Processing", total=len(messages_list), unit="sample"):
            result = self.generate_output(messages)
            if result == 'error':
                continue
            results.append(result)
        return results
        

        