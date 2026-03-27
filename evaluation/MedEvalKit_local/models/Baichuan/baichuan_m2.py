
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging
from tqdm import tqdm

logging.set_verbosity_error()

class Baichuan:
    def __init__(self,model_path,args):
        super().__init__()

        self.llm = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="bfloat16", device_map="auto",attn_implementation="flash_attention_2", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens


    def process_messages(self,messages):
        new_messages = []
        if "system" in messages:
            new_messages.append({"role":"system","content":messages["system"]}) 
        if "messages" in messages:
            messages = messages["messages"]
            for message in messages:
                role = message["role"]
                content = message["content"]
                # new_messages.append({"role":role,"content":[{"type":"text","text":content}]})
                new_messages.append({"role":role,"content":content})
        else:
            new_messages.append({"role":"user","content":messages["prompt"]})

        messages = new_messages

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            thinking_mode='off' # on/off/auto 
        )
        
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

        return inputs


    def generate_output(self,messages):
        inputs = self.process_messages(messages)
        do_sample = False if self.temperature == 0 else True
        generated_ids = self.llm.generate(**inputs,temperature=self.temperature,top_p=self.top_p,repetition_penalty=self.repetition_penalty,max_new_tokens=self.max_new_tokens,do_sample = do_sample)
        
        output_ids = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ][0].tolist()

        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
            
        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return content
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in tqdm(messages_list, desc="Processing", total=len(messages_list), unit="sample"):
            result = self.generate_output(messages)
            res.append(result)
        return res

        