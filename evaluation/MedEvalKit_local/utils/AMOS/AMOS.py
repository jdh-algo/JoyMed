import torch
import os
import json
import gc
import csv
import numpy as np
from PIL import Image
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
from ..question_formats import get_report_generation_prompt, get_multiple_choice_prompt

from ..utils import save_json, extract
from ..base_dataset import BaseDataset
from ..eval_3d import _evaluate_core
from ..eval_report import Evalreport


class AMOS(BaseDataset):
    def __init__(self, model, dataset_path, output_path, task, split = "MM"):
        super().__init__()
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path 
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx", 0))
        self.num_chunks = int(os.environ.get("num_chunks", 1))
        self.eval_report = Evalreport()
        self.split = split
        self.task = task


    def load_data(self):
        dataset_path = self.dataset_path
        if self.task == "Report":
            # json_path = os.path.join(dataset_path,self.split,"valid_report.json") 
            json_path = os.path.join(dataset_path,self.split,"amos_val_mrg_sft.json")
        else:
            json_path = os.path.join(dataset_path,self.split,"valid_vqa.json") 
        
        
        print(f"Loading AMOS data from: {json_path}")
        with open(json_path,"r") as f:
            dataset = json.load(f)
            
        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:            
                sample = self.construct_messages(sample)
                sample["sample_id"] = idx
                self.samples.append(sample)

        # self.samples = [sample for sample in self.samples if sample["type"]=="multiple_choice"][:100]
        print("total samples number:", len(self.samples))
        return self.samples


    def construct_messages(self, sample):
        try:
            question = sample["conversations"][0]["value"]
            answer = sample["conversations"][1]["value"]
            question_type = sample.get("Question_Type", "OPEN")
            main_type = sample.get("type", "Unknown")
            sub_type = sample.get("sub-type", "Unknown")
        except (KeyError, IndexError) as e:
            print(f"Warning: Invalid sample format. Error: {e}")
            return None
        
        video_dir = os.path.join(self.dataset_path,self.split, sample["video"][0])
        
        
        # ## debug 加载nii文件，测试输入nii和image，模型结果是否一致
        # axis_map = {'2':'axial', '1':'coronal', '0':'sagittal'}
        # nii_num_slices = 32
        # nii_axis = 2
        # image_path = os.path.join(self.dataset_path.replace("3D-RAD","3D-RAD_ori"), 
        #                           "valid_fixed",
        #                          '_'.join(os.path.basename(sample["video"][0]).split('_')[:2]),
        #                          '_'.join(os.path.basename(sample["video"][0]).split('_')[:3]),
        #                          os.path.basename(sample["video"][0])+".nii.gz"
        #                         )
        # video_dir = {"image_path":image_path,"nii_num_slices":nii_num_slices,"nii_axis":nii_axis}
        
        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        question_version = 'Lingshu'
        lang = "en"
        prompt = get_report_generation_prompt(is_reasoning, question_version=question_version, lang=lang)
        messages = {
            "prompt": prompt,
            "images_3d": video_dir
        }
        sample["messages"] = messages
        sample["report"] = answer
        sample["question"] = prompt
        if self.task == "Report":
            sample["Question_Type"] = "REPORT"
        else:
            sample["Question_Type"] = "OPEN"
            
        
        return sample
        
    def cal_metrics(self,out_samples):
        # print(f"==== cal_metrics: {self.task}")
        
        if self.task == "Report":
            new_out_samples = []
            for sample in out_samples:
                sample["response"] = sample["response"].replace('Findings:','').replace('Impression:','')
                new_out_samples.append(sample)
            # print(f"===len: {len(new_out_samples)}")
            metrics = self.eval_report(new_out_samples)
        else:
            # print(f"== in score")
            results_text, metrics, wrong_answers = _evaluate_core(
                                                            out_samples=out_samples,
                                                            desc="Evaluating AMOS predictions",
                                                            category_key='type',
                                                            category_name_mapping=None,
                                                            calculate_overall=True
                                                        )
        return metrics,out_samples