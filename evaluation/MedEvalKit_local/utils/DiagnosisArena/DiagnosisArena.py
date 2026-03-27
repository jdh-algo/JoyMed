import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset,load_from_disk
from collections import defaultdict
from tqdm import tqdm

from ..utils import save_json,extract,judge_multi_choice,replace_user_content
from ..base_dataset import BaseDataset

from ..question_formats import get_multiple_choice_prompt

class DiagnosisArena(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        super().__init__()
        self.model = model
        self.output_path = output_path
        if "rag_bench" in self.prompt_version:
            self.dataset_path = os.path.join("/mnt/workspace/offline/sunqintian3/mm_pipeline/data_juicer/input/RAG", self.prompt_version, "DiagnosisArena")
        else:
            self.dataset_path = dataset_path if dataset_path else "shzyk/DiagnosisArena"
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
    
    def load_data(self):
        if "rag_bench" in self.prompt_version:
            dataset = load_dataset("json", data_files=os.path.join(self.dataset_path,"data/test.jsonl"),split="train")
        else:
            dataset = load_dataset(self.dataset_path)['test']
        
        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                sample["sample_id"] = idx
                self.samples.append(sample)
        return self.samples

    def construct_messages(self,sample):
        # import pdb;pdb.set_trace()
        choices = sample["Options"]
        answer = sample["Right Option"]
        if "final_prompt" in sample and sample["final_prompt"]:
            prompt = sample["final_prompt"]
        else:
            CaseInformation = sample["Case Information"]
            PhysicalExamination = sample["Physical Examination"]
            DiagnosticTests = sample["Diagnostic Tests"]
            
            question = f"Case Information: {CaseInformation} \n Physical Examination: {PhysicalExamination} \n Diagnostic Tests: {DiagnosticTests} \n What is the final diagnosis?"
            choices = [f"{a}.{choices[a]}" for a in choices]
            is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
            prompt = get_multiple_choice_prompt(question,choices,is_reasoning)
        
        messages = {"prompt":prompt}
        if "messages" in sample and sample["messages"]:
            # rag_bench 完整的list形式
            full_messages = json.loads(sample["messages"])
            # 更新promp
            messages = replace_user_content(full_messages, [{"type": "text", "text": messages["prompt"]}])
        
        sample["prompt"] = prompt
        sample["messages"] = messages
        sample["choices"] = choices
        sample["answer"] = answer
        sample["Question_Type"] = 'CHOICE'
        return sample


    def cal_metrics(self,out_samples):
        total = 0
        right = 0
        for i,sample in enumerate(out_samples):
            response = sample["response"]
            response = extract(response,"answer")
            choices = sample["choices"]
            answer = sample["answer"]

            correct = judge_multi_choice(choices,answer,response)
            out_samples[i]["correct"] = correct
            if correct:
                right += 1
            total += 1

        metrics = {"total metrics":{"total":total,"right":right,"acc":right/total}}
        return metrics,out_samples