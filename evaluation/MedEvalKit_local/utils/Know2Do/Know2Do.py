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

from ..question_formats import get_multiple_choice_prompt


import re
import jsonlines


class Know2Do(BaseDataset):
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
        
        json_path = os.path.join(dataset_path,"Know2Do-MM.jsonl")

        f=open(json_path,'r')
        dataset = []
        for item in jsonlines.Reader(f):
            dataset.append(item)
                
        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                sample["sample_id"] = idx
                self.samples.append(sample)
        print("total samples number:", len(self.samples))
        return self.samples

        
    def construct_messages(self,sample):
        image_paths = sample["images"] #相对路径列表
        image_paths = [os.path.join(self.dataset_path,image_path) for image_path in image_paths] #nii文件（绝对路径）
        images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
        
        question = sample["question"]
        # answer = sample["answer"]
        answer_idx = sample["answer_idx"]
        choices = sample["options"]

        
        choices = [f"{key}.{value}" for key,value in choices.items()]
        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        prompt = get_multiple_choice_prompt(question,choices,is_reasoning)
        
        messages = {"prompt":prompt,"images":images}
        sample["messages"] = messages
        sample["choices"] = choices
        sample["Question_Type"] = 'CHOICE'
        return sample


    def cal_metrics(self,out_samples):
        total = 0
        right = 0
        for i,sample in enumerate(out_samples):
            response = sample["response"]
            response = extract(response,"answer")
            choices = sample["choices"]
            answer_idx = sample["answer_idx"]

            correct = judge_multi_choice(choices,answer_idx,response)
            out_samples[i]["correct"] = correct
            if correct:
                right += 1
            total += 1

        metrics = {"total metrics":{"total":total,"right":right,"acc":right/total}}
        return metrics,out_samples