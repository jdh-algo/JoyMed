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

from ..utils import save_json,extract, judge_multi_choice
from ..base_dataset import BaseDataset

from ..question_formats import get_open_ended_prompt, get_multiple_choice_prompt


import re
from ..utils_3d import evaluate_open

TASK = {
        'task1': ['Anatomical_observation', 'Pathological_observation'], #Image_Observation
        'task2': ['Abnormality_feature', 'Abnormality_position', 'Abnormality_type', 'Diagnosis'], #Anomaly_Detection
        'task3': ['Diameter', 'Size', 'Thickness'], #Medical_Computation
        'task4': ['Arterial wall calcification', 'Atelectasis', 'Bronchiectasis', 'Cardiomegaly', 'Consolidation',
                  'Coronary artery wall calcification', 'Emphysema', 'Hiatal hernia', 'Interlobular septal thickening',
                  'Lung nodule', 'Lung opacity', 'Lymphadenopathy', 'Medical material', 'Mosaic attenuation pattern',
                  'Peribronchial thickening', 'Pericardial effusion', 'Pleural effusion', 'Pulmonary fibrotic sequela'], #Existence_Detection
        'task5': ['b', 'c', 'd', 'e', 'f', 'g', 'h'], #Static_Temporal_Diagnosis
        'task6': ['b', 'c', 'd', 'e', 'f', 'g', 'h'], #Longitudinal_Temporal_Diagnosis
}

class RAD_3D(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        super().__init__()
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
    
    def load_data(self):
        dataset_path = self.dataset_path
        
        # json_path = os.path.join(dataset_path,"test_vqa.json")
        json_path = os.path.join(dataset_path,"valid_vqa.json")

        with open(json_path,"r") as f:
            dataset = json.load(f)

        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        self.samples = [sample for sample in self.samples if sample["QuestionType"]=='Close']
        print("total samples number:", len(self.samples))
        return self.samples

    def construct_messages(self,sample):
        QuestionType = sample["QuestionType"]
        if QuestionType == "Close":
            sample = self.construct_messages_close(sample)
        else:
            sample = self.construct_messages_open(sample)
        return sample
            
    def construct_messages_open(self,sample):
        image_path = sample["image_3d"] #相对路径
        image_path = os.path.join(self.dataset_path,image_path) #nii文件（绝对路径）
        question = sample["question"]
        nii_num_slices = int(sample["num_slices"])
        # nii_axis = int(sample["axis"])
        nii_axis = 2

        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False

        lang = "en"
        
        # prompt = get_open_ended_prompt(question, is_reasoning,lang=lang)
        prompt = question
        
        image_3d = {"image_path":image_path,"nii_num_slices":nii_num_slices, "nii_axis":nii_axis}
        messages = {"prompt":prompt,"image_3d":image_3d}
        sample["messages"] = messages
        sample["language"] = lang
        return sample
        
    def construct_messages_close(self,sample):
        image_path = sample["image_3d"] #相对路径
        image_path = os.path.join(self.dataset_path,image_path) #nii文件（绝对路径）
        question = sample["question"]
        # answer = sample["answer"]
        answer_idx = sample["answer_idx"]
        choices = sample["options"]
        nii_num_slices = int(sample["num_slices"])
        # nii_axis = int(sample["axis"])
        nii_axis = 2

        # prompt = question + ' ' + "Choices: A. {} B. {} C. {} D. {}".format(choices["A"], choices["B"], choices["C"], choices["D"])   
        # prompt = question + '\n' + '\n'.join([f"{opt}.{choices[opt]}" for opt in choices if choices[opt]!='-'])+ "\nAnswer with the option's letter from the given choices directly." 
        
        choices = [f"{key}.{value}" for key,value in choices.items()]
        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        prompt = get_multiple_choice_prompt(question,choices,is_reasoning, type="3D")

        image_3d = {"image_path":image_path,"nii_num_slices":nii_num_slices, "nii_axis":nii_axis}
        messages = {"prompt":prompt,"image_3d":image_3d}
        sample["messages"] = messages
        sample["choices"] = choices
        return sample
   
    def evaluate_close_task(self,subsample_dict):
        metrics = {}
        total_list = []
        right_list = []
        acc_list = []
        for subtask in subsample_dict.keys():
            sample_list = subsample_dict[subtask]
            total = 0
            right = 0
            metric = {}
            for i,sample in enumerate(sample_list):
                response = sample["response"]
                response = extract(response,"answer")
                choices = sample["choices"]
                answer = sample["answer"]
                answer_idx = sample["answer_idx"]
                
                correct = judge_multi_choice(choices,answer_idx,response)
                
                sample_list[i]["correct"] = correct
                if correct:
                    right += 1
                total += 1
                
            acc = right/total
            metrics[subtask] = {"total":total,"right":right,"acc":acc}
            total_list.append(total)
            right_list.append(right)
            acc_list.append(acc)
        
        metrics["avg"] = {"total":sum(total_list),"right":sum(right_list),"acc":sum(acc_list)/len(acc_list)}
        return metrics


    def evaluate_open_task(self, subsample_dict):
        metric_sub = {}
        bleus, rouge1s, f1s = [], [], []
        for subtask in subsample_dict.keys():
            sample_list = subsample_dict[subtask]
            pred_list = [sample["response"] for sample in sample_list] if len(sample_list)>0 else []
            answer_list = [sample["answer"] for sample in sample_list] if len(sample_list)>0 else []
            if len(pred_list)==[] or len(answer_list)==[]:
                continue
            metric = evaluate_open(pred_list, answer_list)
            metric_sub[subtask] = metric
        
            bleus.append(metric["bleu"])
            rouge1s.append(metric["rouge1"])
            f1s.append(metric["f1"])
        if len(bleus)>0:
            bleu = np.mean(np.array(bleus))
            rouge1 = np.mean(np.array(rouge1s))
            f1 = np.mean(np.array(f1s))
    
            metric_sub["avg"] = {
                "bleu": bleu,
                "rouge1": rouge1,
                "f1": f1
            }
        return metric_sub
       

    def cal_metrics(self,out_samples):
        sample_dict = {task:{subtask:[] for subtask in TASK[task]} for task in TASK.keys()}
        metrics = {task:None for task in TASK.keys()}
        for i,sample in enumerate(out_samples):
            task = sample["Task"].lower().split('_')[0]
            subtask = sample["SubTask"]
            sample_dict[task][subtask].append(sample)
            
        for task in TASK.keys():
            if task in ['task1', 'task2', 'task3']:
                metrics[task] = self.evaluate_open_task(sample_dict[task])
            else:
                metrics[task] = self.evaluate_close_task(sample_dict[task])
        return metrics,out_samples     