import torch
import os
import glob
import json
import gc
from tqdm import tqdm
import gc
import re
from .utils import save_json
from datetime import datetime
import time
from utils.eval_prompts import prompts_dict, dataset_type
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


import socket
ip_adress = socket.gethostbyname(socket.gethostname())

class BaseDataset:
  def __init__(self):
    self.chunk_idx = int(os.environ.get("chunk_idx",0))
    self.num_chunks = int(os.environ.get("num_chunks",1))
    self.prompt_version = os.environ.get("prompt_version","v1")
    if self.prompt_version not in prompts_dict and not os.path.exists(self.prompt_version) :
        print(f"unknown prompt version {self.prompt_version}, we only support {list(prompts_dict.keys())}, use default version v1!")
        self.prompt_version = "v1"
    if os.path.exists(self.prompt_version):
        self.prompts_dict = json.load(open(self.prompt_version,'r'))
    else:
        self.prompts_dict = prompts_dict[self.prompt_version]
    self.REASONING = self.prompts_dict.get('REASONING',False)

  def prompt_preprocess(self, sample, messages):
    dataset_mainname = os.path.basename(self.dataset_path)
    if dataset_mainname == "MMMU":
        dataset_name = dataset_mainname+'-'+self.subset+'-'+self.split
    elif dataset_mainname == "MedXpertQA":
        dataset_name = dataset_mainname+'-'+self.split
    elif dataset_mainname == "MMLU":
        dataset_name = dataset_mainname+'-'+self.subset
    elif dataset_mainname == "CT-RATE":
        dataset_name = dataset_mainname+'-'+self.task
    elif dataset_mainname == "AMOS":
        dataset_name = dataset_mainname+'-'+self.split+'-'+self.task
    elif dataset_mainname == "ORI" and "AMOS" in self.dataset_path:
        dataset_name = "AMOS-Ori-Report"
    elif dataset_mainname == "Deeptumor":
        dataset_name = dataset_mainname+'-'+self.task
    elif dataset_mainname == "M3D":
        dataset_name = dataset_mainname+'-'+self.mode
    else:
        dataset_name = dataset_mainname
        
    if dataset_name not in dataset_type:
        raise ValueError(f"unknown dataset {dataset_name} in dataset_type")
    type = dataset_type[dataset_name]
      
            
    if self.prompt_version == "v2" or self.prompt_version == "rag_bench":
        # language = sample.get('language',sample.get('lang','en'))
        language = sample.get('language','en')
        if prompts_dict[self.prompt_version][type][language].get("system_prompt"):
            messages["system"] = prompts_dict[self.prompt_version][type][language]["system_prompt"]
        if prompts_dict[self.prompt_version][type][language].get("user_prompt"):
            user_prompt = prompts_dict[self.prompt_version][type][language]["user_prompt"]
            if "messages" in messages:
                for message_id, message in enumerate(messages["messages"]):
                    role = message["role"]
                    content = message["content"]
                    if role == "user":
                        messages["messages"][message_id]["content"] = user_prompt.format(prompt=content)
            elif "prompt" in messages:
                messages["prompt"] = user_prompt.format(prompt=messages["prompt"])
                
    else:
        language = sample.get('language','en')
        Question_Type = sample.get('Question_Type','OPEN').upper() #CHOICE,CLOSED,OPEN,JUDGE,REPORT
        if self.prompts_dict[type][language].get("system_prompt"):
            messages["system"] = self.prompts_dict[type][language]["system_prompt"]
            
        user_prompt = self.prompts_dict[type][language][Question_Type]

        if "prompt" in messages:
            messages["prompt"] = user_prompt.format(prompt=messages["prompt"]) if "{prompt}" in user_prompt else user_prompt

    if self.prompts_dict.get('report_eval_method','') == "RadSEM" and sample['Question_Type'] == 'REPORT':
        sample['report_eval_method'] = "RadSEM"
      
    return messages
      
  def prompt_postprocess(self, response):
    answer_patterns = [
        "</think>",
        "</Thought>",
        "</Thoughts>",
    ]
    for answer_pattern in answer_patterns:
        if answer_pattern in response:
            response = response.split(answer_pattern)[-1].strip()
            
    if self.prompt_version == "v2":
        response = response.replace("\n","").replace("Answer","answer")
        match = re.search("<answer>(.*?)</answer>",response)
        if match is None:
            response = ''
        else:
            response = match.group(1)
        
    return response
      
  def run(self,samples,model,batch_size = 2000):
    out_samples = []
    with torch.no_grad():
        messages_list = []
        current_messages = []
        current_samples = []
        for sample in tqdm(samples):
            messages = sample["messages"]
            current_messages.append(messages)
            current_samples.append(sample)
            if len(current_messages) >= batch_size:
                messages_list.append([current_messages,current_samples])
                current_messages = []
                current_samples = []
        if current_messages:
            messages_list.append([current_messages,current_samples])
        batch_num = len(messages_list)
        for batch_id in range(batch_num):
            current_messages,current_samples = messages_list[batch_id]
            sample_num = len(current_messages)
            desc = os.path.basename(os.path.dirname(os.path.dirname(self.output_path))).split('_')[0]+'-'+os.path.basename(self.dataset_path)+f'-chunk:{self.chunk_idx}/{self.num_chunks}'+f'-batch:{batch_id}/{batch_num}'
            for sample_id in tqdm(range(sample_num), desc=desc, total=sample_num, unit="sample"):
                sample = current_samples[sample_id]
                messages = current_messages[sample_id]
                
                if os.path.basename(self.dataset_path) != "Professional":
                    messages = self.prompt_preprocess(sample, messages)
                
                try:
                    response_ori = model.generate_output(messages)
                    response = self.prompt_postprocess(response_ori)
                    
                except Exception as e:
                    response_ori = f"推理报错! {e}"
                    response = ""
                    raise ValueError(f"推理报错! {e}")
                if "prompt" in sample["messages"]:
                    sample["prompt"] = sample["messages"]["prompt"]
                del sample["messages"]
                sample["response"] = response
                sample["response_ori"] = response_ori
                out_samples.append(sample)
            
                
            gc.collect()
    return out_samples

  def cal_matrics(self):
    pass

  def init_dataset(self):
    pass

  def construct_messages(self):
    pass


  def eval(self):
      model = self.model
      dataset_path = self.dataset_path
      output_path = self.output_path
      num_chunks = self.num_chunks
      chunk_idx = self.chunk_idx
      # print(f"chunk_idx:{self.chunk_idx},num_chunks:{self.num_chunks}")
      # time.sleep(100)
      if num_chunks == 1:
          results_path = os.path.join(output_path,"results.json")
          matric_path = os.path.join(output_path,"metrics.json")
          if not os.path.exists(results_path):
              if "MedFrameQA" in os.path.basename(self.dataset_path):
                  out_samples = self.run(self.samples,model, batch_size=200)
              else:
                  out_samples = self.run(self.samples,model)
              save_json(results_path,out_samples)
              gc.collect()
          else:
              out_samples =  json.load(open(results_path,"r"))

          metrics,out_samples = self.cal_metrics(out_samples)
          save_json(matric_path,metrics)
          save_json(results_path,out_samples)
          return metrics


      elif num_chunks > 1:
        results_path = os.path.join(output_path,f"results_{chunk_idx}.json")
        final_results_path = os.path.join(output_path,"results.json")
        if os.path.exists(final_results_path):
            
            if chunk_idx != 0:
                return None
            total_results = json.load(open(final_results_path,'r'))
            desc = os.path.basename(os.path.dirname(os.path.dirname(self.output_path))).split('_')[0]+'-'+os.path.basename(self.dataset_path)
            print(f'Calculate metrics {desc}...')
            matric_path = os.path.join(output_path,"metrics.json")
            if os.path.exists(matric_path):
                metrics = json.load(open(matric_path,'r'))
                if None in metrics.values():
                    metrics,out_samples = self.cal_metrics(total_results)
                    save_json(matric_path,metrics)
                    save_json(final_results_path,out_samples)
                    print(f'Finished cal_metrics {desc}.')
            else:
                metrics,out_samples = self.cal_metrics(total_results)
                save_json(matric_path,metrics)
                save_json(final_results_path,out_samples)
                print(f'Finished cal_metrics {desc}.')
                
            return metrics
        
        else:
            has_num = len([result for result in os.listdir(output_path) if result.startswith("results_")])
              
            if not os.path.exists(results_path) or len(json.load(open(results_path,"r"))) != len(self.samples):
                if "MedFrameQA" in os.path.basename(self.dataset_path):
                    out_samples = self.run(self.samples,model, batch_size=500)
                else:
                    out_samples = self.run(self.samples,model)
                save_json(results_path,out_samples)
                gc.collect()
    
              
            total_results_path = os.listdir(output_path)
            total_results_path = [result for result in total_results_path if result.startswith("results_")]
            print(total_results_path)
          
            if len(total_results_path) == num_chunks:
                if has_num==num_chunks and chunk_idx != 0:
                    return None
                print('!'*50, has_num, num_chunks, chunk_idx)
                total_results = []
                for result in total_results_path:
                    results_path = os.path.join(output_path,result)
                    with open(results_path,"r") as f:
                        total_results.extend(json.load(f))
                    
                save_json(final_results_path,total_results)
                
                    
                desc = os.path.basename(os.path.dirname(os.path.dirname(self.output_path))).split('_')[0]+'-'+os.path.basename(self.dataset_path)
                print(f'Calculate metrics {desc}...')
                matric_path = os.path.join(output_path,"metrics.json")
                if os.path.exists(matric_path):
                    metrics = json.load(open(matric_path,'r'))
                    if None in metrics.values():
                        metrics,out_samples = self.cal_metrics(total_results)
                        save_json(matric_path,metrics)
                        save_json(final_results_path,out_samples)
                        print(f'Finished cal_metrics {desc}.')
                else:
                    metrics,out_samples = self.cal_metrics(total_results)
                    save_json(matric_path,metrics)
                    save_json(final_results_path,out_samples)
                    print(f'Finished cal_metrics {desc}.')
                    
                    
                
                #删除子文件
                # for result in total_results_path:
                #     results_path = os.path.join(output_path,result)
                #     if os.path.exists(results_path):
                #         os.remove(results_path)
                
                # if args.chunk_idx==0:
                # for old_file in glob.glob(os.path.join(eval_output_path, "*.json")):
            #         os.remove(old_file)
                return metrics
            else:
                return None
      else:
          raise ValueError("num_chunks must be greater than 0")