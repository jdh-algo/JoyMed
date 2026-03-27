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

from ..question_formats import get_report_generation_prompt

import re


import socket
ip_adress = socket.gethostbyname(socket.gethostname())
from ..eval_report import Evalreport

class CheXpert_Plus(BaseDataset):
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

        for idx,sample in tqdm(enumerate(dataset.items())):
            if idx % self.num_chunks == self.chunk_idx:
                sample = sample[1]
                # print(">>>Chao[CheXpert_Plus], sample")
                # pprint(sample)
                if pd.isna(sample["section_findings"]):
                    sample["section_findings"] = ""
                if pd.isna(sample["section_impression"]):
                    sample["section_impression"] = ""
                if sample["section_findings"].strip() == "" and sample["section_impression"].strip() == "":
                    continue
                sample = self.construct_messages(sample)
                sample["sample_id"] = idx
                self.samples.append(sample)
        print("total samples number:", len(self.samples))
        return self.samples

    def construct_messages(self,sample):
        image_root = os.path.join(self.dataset_path,"images")
        image = sample["key_image_path"]
        image = Image.open(os.path.join(image_root,image))
        findings = sample["section_findings"]
        impression = sample["section_impression"]

        findings = "None" if findings.strip() == "" else findings
        impression = "None" if impression.strip() == "" else impression
        
        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        question_version = os.environ.get("REASONING_QUESTION_VERSION", 'Lingshu')


        if os.environ.get("TEST_LANGUAGE") == "zh":
            lang = "zh"
        else:
            lang = "en"
        prompt = get_report_generation_prompt(is_reasoning, question_version=question_version, lang=lang)
        
        messages = {"prompt":prompt,"image":image}
        sample["messages"] = messages
        sample["language"] = lang
        sample["Question_Type"] = 'REPORT'
        return sample


    def cal_metrics(self,out_samples):
        import pandas as pd

        predictions_data = []
        ground_truth_data = []

        for i,sample in enumerate(out_samples):
            response = sample["response"]
            findings = sample["section_findings"]
            impression = sample["section_impression"]
            golden = f"Findings: {findings} Impression: {impression}."

            # # 从 response 中提取 <answer></answer>之间的内容
            if "</answer>" in response:
                match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if match:
                    print(">>>[match] before match: ", response)
                    response = match.group(1).strip()
                    print(">>>[match] after match: ", response)
                else:
                    print(">>>[match] no match: ")

            # 生成唯一的study_id
            study_id = f"study_{i+1}"
            
            # 添加预测数据
            predictions_data.append({
                'study_id': study_id,
                'report': response
            })

            # 添加真实标签数据
            ground_truth_data.append({
                'study_id': study_id,
                'report': golden
            })


        # 创建DataFrame
        predictions_df = pd.DataFrame(predictions_data)
        ground_truth_df = pd.DataFrame(ground_truth_data)

        prediction_path = os.path.join(self.output_path,'predictions.csv')
        ground_truth_path = os.path.join(self.output_path,'ground_truth.csv')
        # 保存为CSV文件
        predictions_df.to_csv(prediction_path, index=False)
        ground_truth_df.to_csv(ground_truth_path, index=False)

        metrics = self.eval_report(out_samples)
        return metrics,out_samples
                