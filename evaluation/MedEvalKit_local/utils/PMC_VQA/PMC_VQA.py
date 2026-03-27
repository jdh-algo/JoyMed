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

class PMC_VQA(BaseDataset):
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
        dataset = []
        csv_path = os.path.join(dataset_path,"test_2.csv")
        reader = csv.reader(open(csv_path, 'r', encoding='utf-8'))
        next(reader)
        for i,row in enumerate(reader):

            index,figure_path,caption,question,choiceA,choiceB,choiceC,choiceD,answer,split = row
            choices = [choiceA,choiceB,choiceC,choiceD]
            is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
            prompt = get_multiple_choice_prompt(question,choices,is_reasoning)
            image_path = os.path.join(dataset_path,"figures",figure_path)
            sample = {"prompt":prompt,"answer":answer,"image":image_path,"choices":choices,"image_path":image_path}
            dataset.append(sample)

        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                sample["sample_id"] = idx
                self.samples.append(sample)
        # self.samples = self.samples[:10]
        return self.samples

    def construct_messages(self,sample):
        prompt = sample["prompt"]
        image = sample["image"]
        messages = {"prompt":prompt,"image":image}
        sample["messages"] = messages
        sample["Question_Type"] = "CHOICE"
        del sample["image"]
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


                