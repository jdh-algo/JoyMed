import torch
import os
import json
import gc
import csv

from PIL import Image
import pandas as pd
from pprint import pprint
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

import numpy as np

from ..utils import save_json,extract,judge_multi_choice
from ..base_dataset import BaseDataset

from ..question_formats import get_report_generation_prompt, get_multiple_choice_prompt

import re
from ..report_eval import Evalreport

'''
1. Plane: axial, coronal, sagittal, etc.
2. Phase: plain, enhanced, arterial phase, portal venous phase, delayed phase, etc.
3. Organ: liver, pancreas, lung, kidney, stomach, etc.
4. Abnormality: tumors, masses, cysts, stones, nodules, etc.
5. Location: left, right, upper, lower, etc.
'''
type_map = {"1":"Plane",
            "2":"Phase",
            "3":"Organ",
            "4":"Abnormality",
            "5":"Location",
           }

class M3D(BaseDataset):
    def __init__(self,model,dataset_path,output_path, task):
        super().__init__()
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path
        self.task = task
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
        self.eval_report = Evalreport()
    
    def load_data(self):
        dataset_path = self.dataset_path
        
        if self.task == "Report":
            json_path = os.path.join(dataset_path,"reports_test.json")
        else:
            # json_path = os.path.join(dataset_path,"vqa_test.json")
            json_path = os.path.join(dataset_path,"vqa_val.json")

        with open(json_path,"r") as f:
            dataset = json.load(f)

        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                if self.task == "Report":
                    sample = self.construct_messages_report(sample)
                else:
                    sample = self.construct_messages_vqa(sample)
                    
                self.samples.append(sample)
        # self.samples = [sample for sample in self.samples if sample["type"]=='1']
        print("total samples number:", len(self.samples))
        return self.samples

    def construct_messages_report(self,sample):
        image_path = sample["image_3d"] #相对路径
        image_path = os.path.join(self.dataset_path,image_path) #nii文件（绝对路径）

        reports = sample["reports"][0]
        
        nii_num_slices = int(sample["num_slices"])
        nii_axis = int(sample["axis"])
        # nii_axis = 2

        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False

        lang = "en"
        
        prompt = get_report_generation_prompt(is_reasoning, question_version="DerectReport", image_type="3D", lang=lang)
        
        image_3d = {"image_path":image_path,"nii_num_slices":nii_num_slices, "nii_axis":nii_axis}
        messages = {"prompt":prompt,"image_3d":image_3d}
        sample["messages"] = messages
        sample["language"] = lang
        sample["reports"] = reports
        sample["Question_Type"] = "REPORT"
        
        return sample
        
    def construct_messages_vqa(self,sample):
        image_path = sample["image_3d"] #相对路径
        image_path = os.path.join(self.dataset_path,image_path) #nii文件（绝对路径）
        question = sample["question"]
        # answer = sample["answer"]
        answer_idx = sample["answer_idx"]
        choices = sample["options"]
        nii_num_slices = int(sample["num_slices"])
        nii_axis = int(sample["axis"])
        # nii_axis = 2

        

        # prompt = question + '\n' + '\n'.join([f"{opt}.{choices[opt]}" for opt in choices if choices[opt]!='-'])+ "\nAnswer with the option's letter from the given choices directly." 
        
        choices = [f"{key}.{value}" for key,value in choices.items()]  
        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        prompt = get_multiple_choice_prompt(question,choices,is_reasoning, type="3D")

        image_3d = {"image_path":image_path,"nii_num_slices":nii_num_slices, "nii_axis":nii_axis}
        messages = {"prompt":prompt,"image_3d":image_3d}
        sample["messages"] = messages
        sample["choices"] = choices
        if "Question_Type" in sample and "CLOSE" in sample["Question_Type"].upper():
            sample["Question_Type"] = "CHOICE"
        return sample
   

    def cal_metrics_report(self,out_samples):        
        metrics = self.eval_report(out_samples)
        
        return metrics,out_samples


    def cal_metrics_vqa(self,out_samples):
        total = {type:0 for type in type_map.keys()}
        right = {type:0 for type in type_map.keys()}
        metrics = {}
        for i,sample in tqdm(enumerate(out_samples), desc="cal_metrics_vqa", total=len(out_samples), unit="sample"):
            response = sample["response"]
            response = extract(response,"answer")
            choices = sample["choices"]
            answer = sample["answer"]
            answer_idx = sample["answer_idx"]
            type = sample["type"]
            
            correct = judge_multi_choice(choices,answer_idx,response)
            # if answer_idx + '.' in response:
            #     correct = True
            # else:
            #     correct = False
                    
            out_samples[i]["correct"] = correct
            if correct:
                right[type] += 1
            total[type] += 1
            
        for type in type_map.keys():
            if type in total and type in right:
                metrics[type_map[type]] = {"total":total[type],"right":right[type],"acc":right[type]/total[type]}
        
        return metrics,out_samples

    def cal_metrics(self,out_samples):
        if self.task == "Report":
            metrics,out_samples = self.cal_metrics_report(out_samples)
        else:
            metrics,out_samples = self.cal_metrics_vqa(out_samples)
        return metrics,out_samples     