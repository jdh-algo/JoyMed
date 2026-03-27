from tqdm import tqdm
import os
from .cli import MedPLIBChatbot


class MedPLIB:
    def __init__(self,model_path,
                 temperature=0,
                 top_p = 0.001, 
                 max_tokens = 1024,):
        super().__init__()
        self.bot = MedPLIBChatbot(model_path,
                                  temperature=temperature,
                                  max_new_tokens=max_tokens,
                                  top_p=top_p,
                                 )
    

    def process_messages(self,messages):
        question = ""
        images = []
        
        if "system" in messages:
            question =  messages["system"]

        if "messages" in messages:
            messages = messages["messages"]
            text = ''
            for con_id, message in enumerate(messages):
                role = message["role"]
                content = message["content"]
                if con_id%2 ==0:
                    text = content
                else:
                    self.bot.history.append([text, content])
            question += content
        else:
            
            question += messages["prompt"]
            if "image" in messages:
                images = messages["image"]
            elif "images" in messages:
                images = messages["images"]
            
        llm_inputs = {
            "question": question,
            "images": images,
        }
        return llm_inputs

    def generate_output(self,messages):
        llm_inputs = self.process_messages(messages)
        query = llm_inputs["question"]
        images = llm_inputs["images"]
        outputs = self.bot.inference(query, images)
        self.bot.history = []
        return outputs
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in tqdm(messages_list,total=len(messages_list)):
            result = self.generate_output(messages)
            res.append(result)
        return res


