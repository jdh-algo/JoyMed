import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset,load_from_disk
from collections import defaultdict
from tqdm import tqdm

from ..utils import save_json,judge_multi_choice
from ..base_dataset import BaseDataset

from ..question_formats import get_multiple_choice_prompt
from .image_mcq import GMAIMMBenchDataset

class GMAIMMBench(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        super().__init__()
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
        if "GMAI-MMBench-test" in os.path.basename(dataset_path):
            self.dataset = GMAIMMBenchDataset(dataset='GMAI-MMBench_TEST', dataset_path=os.path.join(os.path.dirname(self.dataset_path), "GMAI-MMBench"))
        else:
            self.dataset = GMAIMMBenchDataset(dataset='GMAI-MMBench_VAL', dataset_path=os.path.join(os.path.dirname(self.dataset_path), "GMAI-MMBench"))

    def load_data(self):
        for idx,sample in tqdm(enumerate(self.dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                sample["sample_id"] = idx
                self.samples.append(sample)
        return self.samples

    def construct_messages(self,sample):
        question = sample['question']
        
        msgs, options = self.dataset.build_prompt(sample)
        if "GMAI-MMBench-test" in os.path.basename(self.dataset_path):
            category = sample['category'].replace('.','')
            answer = dict(zip(options.values(), options.keys()))[category]
        else:
            answer = sample["answer"]
        
        images = []
        for msg in msgs:
            if msg['type'] == 'image':
                images.append(Image.open(msg['value']).copy())
            elif msg['type'] == 'text':
                prompt = msg['value']
        
        messages = {"prompt":prompt,"images":images}
        sample = {"question":question, "answer":answer}
        sample["messages"] = messages
        sample["choices"] = options
        sample["Question_Type"] = 'CHOICE'
        # sample["answer"] = answer
        
        return sample


    def cal_metrics(self,out_samples):
        total = 0
        right = 0
        total_modality = defaultdict(int)
        right_modality = defaultdict(int)
        for i,sample in enumerate(out_samples):
            response = sample["response"]
            choices = sample["choices"]
            answer = sample["answer"]

            correct = judge_multi_choice(choices,answer,response)
            out_samples[i]["correct"] = correct
            if correct:
                right += 1
            total += 1

        metrics = {"total metrics":{"total":total,"right":right,"acc":right/total}}
        return metrics,out_samples



                