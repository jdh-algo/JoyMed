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

from ..utils import save_json,extract
from ..base_dataset import BaseDataset

from ..question_formats import get_report_generation_prompt, get_open_ended_prompt, get_multiple_choice_prompt

import re
from ..eval_3d import _evaluate_core
from ..eval_report import Evalreport

class Deeptumor(BaseDataset):
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
            # json_path = os.path.join(dataset_path,"OOD_test_report.json")
            json_path = os.path.join(dataset_path,"OOD_test_report.json")
        else:
            # json_path = os.path.join(dataset_path,"IID_test_vqa.json")
            json_path = os.path.join(dataset_path,"OOD_test_vqa.json") 
        
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
        if self.task == "VQA":
            question = sample["question"]
        nii_num_slices = int(sample["num_slices"])
        # nii_axis = int(sample["axis"]) #axis_map = {'2':'axial', '1':'coronal', '0':'sagittal'}
        nii_axis = 2

        
        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        question_version = 'Lingshu' #'DerectReport' #'Lingshu'
        lang = "en"

        if self.task == "Report":
            sample["Question_Type"] = "REPORT"
            prompt = get_report_generation_prompt(is_reasoning, question_version=question_version, lang=lang)
        else:
            if "answer_idx" in sample and "options" in sample: 
                sample["Question_Type"] = "CHOICE"
                choices = sample["options"]
                choices = [f"{key}.{value}" for key,value in choices.items()]
                sample["choices"] = choices
                prompt = get_multiple_choice_prompt(question,choices,is_reasoning, type="3D")
            else:
                sample["Question_Type"] = "OPEN"
                prompt = get_open_ended_prompt(question, is_reasoning, lang=lang)
                
        nii_info = {"image_path":image_path,"nii_num_slices":nii_num_slices, "nii_axis":nii_axis}
        messages = {"prompt":prompt,"images_3d":nii_info}
        sample["messages"] = messages
        sample["language"] = lang
        
        return sample

        
    def cal_metrics(self,out_samples):
        
        if self.task == "Report":
            for sample_id, out_sample in enumerate(out_samples):
                findings = out_sample["findings"]
                impression = out_sample["impression"].replace("IMPRESSION:","")
                out_samples[sample_id]["report"] = f"Findings: {findings} Impression: {impression}."
            
            metrics = self.eval_report(out_samples)
        else:
            results_text, metrics, wrong_answers = _evaluate_core(
                                                            out_samples=out_samples,
                                                            desc="Evaluating CT-RATE predictions",
                                                            category_key='type',
                                                            category_name_mapping=None,
                                                            calculate_overall=True
                                                        )
        return metrics,out_samples