import requests
import base64
from PIL import Image
import io
from tqdm import tqdm

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class Qwen2_5_VL:
    def __init__(self, api_url, args):
        super().__init__()
        self.api_url = api_url
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
            image = Image.open(image)

        # if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        #     image.thumbnail(max_size, Image.Resampling.LANCZOS)
        # if image.mode == 'RGBA':
        # # 创建白色背景
        #     background = Image.new('RGB', image.size, (255, 255, 255))
        # # 使用alpha通道作为mask，将RGBA图片复合到白色背景上
        #     background.paste(image, mask=image.split()[3])
        #     image = background
        # elif image.mode != 'RGB':
        # # 如果是其他模式，也转换为RGB
        #     image = image.convert('RGB')
        buffered = io.BytesIO()
        image.save(buffered, format=image.format)
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
                    # content.append({"type":"text","text":f"<image_{i+1}>: "})
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

        # retry_strategy = Retry(
        #     total=3,  # retry num
        #     backoff_factor=1,  # retry interval
        #     status_forcelist=[500, 502, 503, 504]
        # )
    
        # session = requests.Session()
        # session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
        # session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
        try:
            response = requests.post(
                url=self.api_url,
                headers=self.headers,
                json=payload
                # timeout=300
            )
            
            response_json = response.json()
            # print(response_json)
            # response.raise_for_status()
            return response_json["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"API request failed: {e}")
            # raise

    def generate_outputs(self, messages_list):
        return [self.generate_output(messages) for messages in tqdm(messages_list, desc="Processing", total=len(messages_list), unit="sample")]