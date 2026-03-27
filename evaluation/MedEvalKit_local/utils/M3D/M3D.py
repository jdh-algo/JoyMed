import torch
import os
import json
import gc
import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import re
# from ..utils import save_json, extract, judge_multi_choice, judger, get_compare_messages, judge_open_end_vqa, judge_judgement
from ..base_dataset import BaseDataset
# from ..question_formats import get_judgement_prompt, get_open_ended_prompt
from ..eval_3d import _evaluate_core
# from ..mm_utils import load_images, process_images, load_video, process_video, tokenizer_multimodal_token, get_model_name_from_path, KeywordsStoppingCriteria
class M3D(BaseDataset):
    def __init__(self, model, dataset_path, output_path, mode):
        super().__init__()
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "./M3D_test.json"
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx", 0))
        self.num_chunks = int(os.environ.get("num_chunks", 1))
        self.mode = mode

    def load_data(self):
        json_path = os.path.join(self.dataset_path,"M3D_test.json")
        
        print(f"Loading M3D data from: {json_path}")
        
        with open(json_path,"r") as f:
            dataset = json.load(f)

        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                sample["sample_id"] = idx
                self.samples.append(sample)
                
        # self.samples = [sample for sample in self.samples if len(os.listdir(os.path.join(self.dataset_path, sample["video"][0])))>220]
        # self.samples = sorted(self.samples,key=lambda item: -len(os.listdir(os.path.join(self.dataset_path, item["video"][0]))))[:16]
        # self.samples = self.samples[1020:]
        print("total samples number:", len(self.samples))
        return self.samples

    def construct_messages(self, sample):
        try:
            question = sample["conversations"][0]["value"]
            answer = sample["conversations"][1]["value"]
            Qtype = sample.get("type", "Default")
            if "Question_Type" in sample and "CLOSE" in sample["Question_Type"].upper():
                sample["Question_Type"] = "CHOICE"
                sample["choices"] = question.split('\n')[1:-1]
            else:
                sample["Question_Type"] = "OPEN"
        except (KeyError, IndexError) as e:
            print(f"Warning: Invalid sample format. Error: {e}")
            return None

        if self.mode == "Slice":
            # 抽取32帧
            video_dir = os.path.join(self.dataset_path, sample["video"][0])

        else:
            #使用原始3d数据
            json_path = os.path.join(self.dataset_path.replace("M3D","M3D_ori"),"nii_info.json")
            nii_info = json.load(open(json_path,"r"))
            nii_num_slices = nii_info[sample["video"][0]]["nii_num_slices"]
            # nii_axis = nii_info[sample["video"][0]]["nii_axis"]
            nii_axis = 0
            image_path = os.path.join(self.dataset_path.replace("M3D","M3D_ori"), "M3D_Cap_nii",sample["video"][0]+".nii.gz")
            video_dir = {"image_path":image_path,"nii_num_slices":nii_num_slices,"nii_axis":nii_axis}
        
        
        prompt = question.replace('<image>\n', '').replace('<video>\n', '').strip()
        
        messages = {
            "prompt": prompt,
            "images_3d": video_dir
        }
        sample["messages"] = messages
        
        return sample

    def extract_choice_letter(text):
    
        text = str(text).lower().strip()
        
        match = re.match(r'^([a-d])[.\):]?\s*', text)
        if match:
            return match.group(1)
        
        if text in ['a', 'b', 'c', 'd']:
            return text
        
        return text

    def cal_metrics(self, out_samples):
        
        print("\n" + "="*80)
        print("Starting M3D Evaluation")
        print("="*80)
        QUESTION_TYPE_MAPPING = {
            1: "Plane", 2: "Phase", 3: "Organ", 4: "Abnormality", 5: "Location"
}
        results_text, metrics, wrong_answers = _evaluate_core(
                                                        out_samples=out_samples,
                                                        desc="Evaluating M3D predictions",
                                                        category_key='type',
                                                        category_name_mapping=QUESTION_TYPE_MAPPING,
                                                        calculate_overall=True
                                                    )
        print(results_text)
        
        wrong_answers_readable = {
            QUESTION_TYPE_MAPPING.get(k, f"Type_{k}"): v 
            for k, v in wrong_answers.items()
        }
        
        os.makedirs(self.output_path, exist_ok=True)
        wrong_answers_path = os.path.join(self.output_path, "wrong_answers.json")
        
        with open(wrong_answers_path, 'w', encoding='utf-8') as f:
            json.dump(wrong_answers_readable, f, indent=4, ensure_ascii=False)
        
        print(f"\n✅ Wrong answers saved to: {wrong_answers_path}")
        print("="*80 + "\n")
        
        return metrics, out_samples