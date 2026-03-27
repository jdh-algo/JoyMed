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

class RAD_3D(BaseDataset):
    def __init__(self, model, dataset_path, output_path):
        super().__init__()
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "./Rad3D_test.json"
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx", 0))
        self.num_chunks = int(os.environ.get("num_chunks", 1))

    def load_data(self):
        json_path = os.path.join(self.dataset_path,"Rad3D_test.json")
        
        print(f"Loading 3D-RAD data from: {json_path}")
        
        
        with open(json_path,"r") as f:
            dataset = json.load(f)

        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                sample["sample_id"] = idx
                self.samples.append(sample)

        # self.samples = [sample for sample in self.samples if sample["type"]=="Static_Temporal_Diagnosis"]
        print("total samples number:", len(self.samples))
        return self.samples

    def construct_messages(self, sample):
        try:
            question = sample["conversations"][0]["value"]
            answer = sample["conversations"][1]["value"]
            main_type = sample.get("type", "Unknown")
            sub_type = sample.get("sub-type", "Unknown")
            if "Question_Type" in sample and "CLOSE" in sample["Question_Type"].upper():
                sample["Question_Type"] = "CHOICE"
            else:
                sample["Question_Type"] = "OPEN"
                
        except (KeyError, IndexError) as e:
            print(f"Warning: Invalid sample format. Error: {e}")
            return None
        
        video_dir = os.path.join(self.dataset_path, sample["video"][0])
        
        
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
        
        
        question_clean = question.replace('<image>\n', '').replace('<video>\n', '').strip()
        
        # if question_type.lower() in ['close', 'closed']:
        #     sample["Question_Type"] = 'CLOSED'
        #     prompt = question_clean + "\nAnswer with the option's letter from the given choices directly."
        # else:
        #     sample["Question_Type"] = 'OPEN'
        #     prompt = question_clean + "\nPlease answer the question concisely."
        
        prompt = question_clean 
        
        messages = {
            "prompt": prompt,
            "images_3d": video_dir
        }
        sample["messages"] = messages
        
            
        return sample

    def extract_choice_letter(self, text):
        text = str(text).lower().strip()
        
        match = re.match(r'^([a-d])[.\):]?\s*', text)
        if match:
            return match.group(1)
        
        if text in ['a', 'b', 'c', 'd']:
            return text
        
        return text

    def cal_metrics(self, out_samples):
        print("\n" + "="*80)
        print("Starting 3D-RAD Evaluation")
        print("="*80)
        
        TYPE_MAPPING = {
            "Medical_Computation": "Medical Computation",
            "Spatial_Relationship": "Spatial Relationship",
            "Abnormality_Detection": "Abnormality Detection",
            "Organ_Identification": "Organ Identification",
            "Image_Quality": "Image Quality"
        }
        
        SUB_TYPE_MAPPING = {
            "Thickness": "Thickness",
            "Distance": "Distance",
            "Volume": "Volume",
            "Density": "Density",
            "Location": "Location",
            "Relative_Position": "Relative Position",
            "Presence": "Presence",
            "Severity": "Severity",
            "Organ_Name": "Organ Name",
            "Organ_Count": "Organ Count",
            "Contrast": "Contrast",
            "Noise": "Noise"
        }
        
        results_text, metrics, wrong_answers = _evaluate_core(
                                                            out_samples=out_samples,
                                                            desc="Evaluating 3D-RAD predictions",
                                                            category_key='type',
                                                            category_name_mapping=TYPE_MAPPING,
                                                            calculate_overall=True
                                                        )
        
        print(results_text)
        
        wrong_answers_readable = {}
        for key, value in wrong_answers.items():
            readable_key = TYPE_MAPPING.get(key, key)
            wrong_answers_readable[readable_key] = value
        
        os.makedirs(self.output_path, exist_ok=True)
        wrong_answers_path = os.path.join(self.output_path, "wrong_answers.json")
        
        with open(wrong_answers_path, 'w', encoding='utf-8') as f:
            json.dump(wrong_answers_readable, f, indent=4, ensure_ascii=False)
        
        print(f"\n✅ Wrong answers saved to: {wrong_answers_path}")
        print("="*80 + "\n")
        
        return metrics, out_samples
