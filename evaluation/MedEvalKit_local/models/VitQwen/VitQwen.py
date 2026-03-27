import torch
from PIL import Image
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from ..image_process import load_images

from .vit_qwen.utils.m3d_utils import load_images_folder

from . import vit_qwen

def _align_config_vocab_size_for_checkpoint(config, tokenizer):
    """
    Align config.llm_config.vocab_size with tokenizer (for old checkpoints saved
    before we synced vocab_size at save time). Supports llm_config as dict or object.
    """
    if not getattr(config, "llm_config", None):
        return
    vocab_from_tokenizer = len(tokenizer)
    lc = config.llm_config
    llm_vocab = lc.get("vocab_size") if isinstance(lc, dict) else getattr(lc, "vocab_size", None)
    if llm_vocab is None or vocab_from_tokenizer <= llm_vocab:
        return
    new_vocab = math.ceil(vocab_from_tokenizer / 128) * 128
    if isinstance(lc, dict):
        lc["vocab_size"] = new_vocab
    else:
        setattr(lc, "vocab_size", new_vocab)
    print(f"[inference] Aligned llm_config.vocab_size for checkpoint: {llm_vocab} -> {new_vocab}")

class VitQwen:

    def __init__(self, model_path, args):

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
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        _align_config_vocab_size_for_checkpoint(self.config, self.processor.tokenizer)
        pad_token_id_in_vocab = self.processor.tokenizer.convert_tokens_to_ids(getattr(self.processor, "image_3d_token", "<|image_3d_pad|>"))
        if getattr(self.model.config, "image_3d_token_id", None) != pad_token_id_in_vocab:
            self.model.config.image_3d_token_id = pad_token_id_in_vocab
            print(f"[inference] Synced model.config.image_3d_token_id = {pad_token_id_in_vocab}")
        
        self.model.eval()

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.max_new_tokens = args.max_new_tokens
        self.repetition_penalty = args.repetition_penalty
        
    def process_messages(self, messages):
        print(f"messages: {messages}")

        if isinstance(messages, dict):
            # added by xuyang 1210, AMOS-MM-report process add additional prompt
            if ('images_3d' in messages) and  "Findings" in messages['prompt'] and "AMOS" in messages['images_3d']:
                messages['prompt'] += " This imaging study may include multiple anatomical regions."
            if ('images_3d' in messages) and  "Findings" in messages['prompt'] and "Deeptumor" in messages['images_3d']:
                messages['prompt'] += " Strict quantification of measurement parameters is required."
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