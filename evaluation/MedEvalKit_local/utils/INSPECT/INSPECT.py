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

from ..question_formats import get_report_generation_prompt

import re
from ..eval_report import Evalreport

class INSPECT(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        super().__init__()
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
        self.eval_report = Evalreport()
    
    def load_data(self):
        dataset_path = self.dataset_path
        
        json_path = os.path.join(dataset_path,"test.json")

        with open(json_path,"r") as f:
            dataset = json.load(f)

        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                sample["sample_id"] = idx
                self.samples.append(sample)
        
        print("total samples number:", len(self.samples))
        return self.samples

    def construct_messages(self,sample):
        image_path = sample["image_3d"] #相对路径
        image_path = os.path.join(self.dataset_path,image_path) #nii文件（绝对路径）
        # reports = sample["report"]
        
        nii_num_slices = int(sample["num_slices"])
        nii_axis = 2

        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        question_version = 'DerectReport' #'Lingshu'
        lang = "en"
        
        prompt = get_report_generation_prompt(is_reasoning, question_version=question_version, lang=lang)
        
        nii_info = {"image_path":image_path,"nii_num_slices":nii_num_slices, "nii_axis":nii_axis}
        messages = {"prompt":prompt,"images_3d":nii_info}
        sample["messages"] = messages
        sample["language"] = lang
        sample["Question_Type"] = 'REPORT'
        return sample
    

        
    def cal_metrics(self,out_samples):
        new_out_samples = []
        for sample in out_samples:
            sample["report"] = sample["report"].replace('Findings:','').replace('Impression:','')
            new_out_samples.append(sample)
        
        metrics = self.eval_report(new_out_samples)
        
        return metrics,out_samples