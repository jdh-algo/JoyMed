import random
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

import json
import pandas as pd

import monai.transforms as mtf
from monai.data import load_decathlon_datalist
from monai.data import set_track_meta
from monai.transforms import Resize

from ..utils.utils import mask2box
from .dataset_info import dataset_info
from .prompt_templates import Caption_templates, PosREC_templates, PosREG_templates, Seg_templates
from .term_dictionary import term_dict

import nibabel as nib
from scipy import sparse
import ast
from typing import Dict, List


CLOSE_ENDED_RESPONSE_PROMPT = "You are answering a multiple-choice question. Choose exactly one answer from the listed options."
SHORT_ANSWER_RESPONSE_PROMPT = "You are answering a short clinical question. Respond with one concise phrase or sentence, staying focused and specific."
LONG_ANSWER_RESPONSE_PROMPT = "You are answering a detailed clinical question. Provide a concise, clinically focused answer in 2 to 4 sentences."
REPORT_GENERATION_PROMPT = "You are generating a radiology report from the provided 3D image. Summarize key findings and an impression in clear, professional language."
# Retain the legacy name for compatibility with existing callers that expect an open-ended prompt.
OPEN_ENDED_RESPONSE_PROMPT = LONG_ANSWER_RESPONSE_PROMPT
QUESTION_TYPE_TO_PROMPT = {
    "multiple_choice": CLOSE_ENDED_RESPONSE_PROMPT,
    "short_answer": SHORT_ANSWER_RESPONSE_PROMPT,
    "long_answer": LONG_ANSWER_RESPONSE_PROMPT,
    "report_generation": REPORT_GENERATION_PROMPT,
}
MULTIPLE_CHOICE_LABEL_PATTERN = re.compile(r'\(\s*([A-Da-d])\s*\)')

# ---- CT-RATE organ grouping (pre-merged masks) ----
# 0 = background. 1-34 correspond to the grouped organ ids produced by
# fvlm/fvlm-main/data/resize.py (see merged_organ_id mapping).
GROUP_ID_TO_NAME: Dict[int, str] = {
    0: "background",
    1: "lung",
    2: "heart",
    3: "esophagus",
    4: "aorta",
    5: "brain",
    6: "adrenal_gland",
    7: "kidney",
    8: "stomach",
    9: "liver",
    10: "gallbladder",
    11: "pancreas",
    12: "spleen",
    13: "colon",
    14: "small_bowel",
    15: "urinary_bladder",
    16: "trachea",
    17: "inferior_vena_cava",
    18: "portal_vein_and_splenic_vein",
    19: "pulmonary_vein",
    20: "iliac_artery",
    21: "iliac_vena",
    22: "lumbar_vertebrae",
    23: "thoracic_vertebrae",
    24: "cervical_vertebrae",
    25: "rib",
    26: "humerus",
    27: "scapula",
    28: "clavicula",
    29: "femur",
    30: "hip",
    31: "sacrum",
    32: "gluteus",
    33: "iliopsoas",
    34: "autochthon",
}

# Absolute paths provided by the user for grouped masks and organ text.
CT_RATE_MASK_ROOT = "/mnt/workspace/offline/xiehuidong.6688/fvlm/fvlm-main/data/merged_train_masks"
CT_RATE_CONC_INFO = "/mnt/workspace/offline/xiehuidong.6688/fvlm/des/conc_info.json"
CT_RATE_DESC_INFO = "/mnt/workspace/offline/xiehuidong.6688/fvlm/des/desc_info.json"

# class CT_RATE_CAP(Dataset):

class CTRATEDataset_w_Seg_fvlm(Dataset):
    def __init__(self, args, model_args, tokenizer, mode="train"):
        self.args = args
        self.model_args = model_args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.ct_rate_mask_root = getattr(args, "ct_rate_mask_root", CT_RATE_MASK_ROOT)
        self.id_to_group_name = GROUP_ID_TO_NAME

        # Load organ-level textual descriptions once (Conclusion + Findings)
        conc_path = getattr(args, "ct_rate_conc_info", CT_RATE_CONC_INFO)
        desc_path = getattr(args, "ct_rate_desc_info", CT_RATE_DESC_INFO)
        with open(conc_path, "r") as f:
            self.conc_info = json.load(f)
        with open(desc_path, "r") as f:
            self.desc_info = json.load(f)

        json_file = []
        with open(args.cap_data_path_CT_rate, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                json_file.append(obj)
        self.data_list_ct_rate = json_file

        # apply the same transform to both images and segmentations
        train_transform = mtf.Compose(
            [
                # spatial transforms applied to both A and B
                mtf.RandRotate90d(keys=["A", "B"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["A", "B"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["A", "B"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["A", "B"], prob=0.10, spatial_axis=2),
        
                # intensity transforms: usually only on the “image” (e.g. A)
                mtf.RandScaleIntensityd(keys=["A"], factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys=["A"], offsets=0.1, prob=0.5),
        
                mtf.ToTensord(keys=["A"], dtype=torch.float32),
                mtf.ToTensord(keys=["B"], dtype=torch.int8),
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensord(keys=["A"], dtype=torch.float32),
                mtf.ToTensord(keys=["B"], dtype=torch.int8),
            ]
        )
        set_track_meta(False)
        
        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
            self.data_list_ct_rate = self.data_list_ct_rate[:512]
            
        elif 'test' in mode:
            self.transform = val_transform
            
        self.len_ct_rate = len(self.data_list_ct_rate) if args.use_CT_rate_data else 0
        
    def __len__(self):
        return self.len_ct_rate

    def _get_ct_rate_item(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list_ct_rate[idx]
                image_abs_path = data["image_3d"]
                # Map the CT-RATE image path to the pre-grouped mask root.
                seg_rel = os.path.relpath(
                    image_abs_path,
                    "/mnt/workspace/offline/shared_data/Medical_Data_3D/CT-RATE/dataset/train_fixed",
                )
                seg_abs_path = os.path.join(self.ct_rate_mask_root, seg_rel)

                #image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                image = nib.load(image_abs_path).get_fdata()
                seg = nib.load(seg_abs_path).get_fdata()

                if np.max(image) == 0:
                    raise ValueError("Empty image (all zeros)")

                image = (image - (-1000)) / (2000)
                image[image>1.0] = 1.0
                image[image<0.0] = 0.0
                image = np.expand_dims(image, 0)
                seg = np.expand_dims(seg, 0)
                image = np.transpose(image,[0, 3, 1, 2])
                seg = np.transpose(seg,[0, 3, 1, 2])
                
                resizer = Resize(spatial_size=self.model_args.img_size)  
                seg_resizer = Resize(
                    spatial_size=self.model_args.img_size,
                    mode="nearest"
                )

                image = resizer(image).detach().cpu().numpy()  # still numpy
                seg = seg_resizer(seg).detach().cpu().numpy()  # still numpy

                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                sample = {
                    "A": image,   # numpy array or torch tensor
                    "B": seg,   # same shape / spatial size as A
                }
                out = self.transform(sample)
                image = out["A"]
                seg = out["B"]

                # ---- organ-level text (new CLIP pipeline) ----
                seg_unique = torch.unique(seg)
                organ_input_ids: List[torch.Tensor] = []
                organ_attention_mask: List[torch.Tensor] = []
                organ_ids: List[int] = []
                organ_texts: List[str] = []
                organ_normal_flags: List[bool] = []

                study_id = self._derive_study_id(image_abs_path)
                for oid in seg_unique.tolist():
                    if oid <= 0:
                        continue
                    organ_id_int = int(oid)
                    organ_ids.append(organ_id_int)
                    organ_name = self._organ_name_from_id(organ_id_int)
                    organ_text, is_normal = self._build_organ_text(study_id, organ_name)
                    organ_texts.append(organ_text)
                    organ_normal_flags.append(is_normal)

                    organ_text_tensor = self.tokenizer(
                        organ_text,
                        max_length=self.args.max_length,
                        truncation=True,
                        padding="longest",
                        return_tensors="pt",
                    )
                    organ_input_ids.append(organ_text_tensor["input_ids"][0])
                    organ_attention_mask.append(organ_text_tensor["attention_mask"][0])

                if organ_input_ids:
                    organ_input_ids_tensor = torch.stack(organ_input_ids, dim=0)
                    organ_attention_mask_tensor = torch.stack(organ_attention_mask, dim=0)
                    organ_ids_tensor = torch.tensor(organ_ids, dtype=torch.long)
                    organ_normal_flags_tensor = torch.tensor(organ_normal_flags, dtype=torch.bool)
                else:
                    # rare fallback; keep shapes consistent
                    organ_input_ids_tensor = torch.zeros(
                        (0, self.args.max_length), dtype=torch.long
                    )
                    organ_attention_mask_tensor = torch.zeros_like(organ_input_ids_tensor)
                    organ_ids_tensor = torch.zeros((0,), dtype=torch.long)
                    organ_normal_flags_tensor = torch.zeros((0,), dtype=torch.bool)

                # Use aggregated organ text (not the full findings) for caption-level CLIP
                if organ_texts:
                    text = " ".join(organ_texts)
                else:
                    text = "No significant abnormalities."

                text_tensor = self.tokenizer(
                    text, max_length=self.args.max_length, truncation=True, padding="longest", return_tensors="pt"
                )
                
                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                ret = {
                    'image': image,
                    'seg': seg,
                    'text': text,
                    'input_id': input_id,
                    'attention_mask': attention_mask,
                    'organ_input_ids': organ_input_ids_tensor,
                    'organ_attention_mask': organ_attention_mask_tensor,
                    'organ_ids': organ_ids_tensor,
                    'organ_normal_flags': organ_normal_flags_tensor,
                    'question_type': "Image_text_retrieval",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list_ct_rate) - 1)

    def __getitem__(self, idx):
        return self._get_ct_rate_item(idx)

    @staticmethod
    def _derive_study_id(image_path: str) -> str:
        """train_1_a_1.nii.gz -> train_1_a (drop last numeric suffix)."""
        base = os.path.basename(image_path)
        stem = os.path.splitext(os.path.splitext(base)[0])[0]
        parts = stem.split("_")
        if parts and parts[-1].isdigit():
            parts = parts[:-1]
        return "_".join(parts)

    def _organ_name_from_id(self, organ_id: int) -> str:
        return self.id_to_group_name.get(organ_id, f"organ_{organ_id}")

    def _strip_neutral_template(self, organ_name: str, text: str) -> str:
        """Remove leading neutral template to mirror original fVLM handling."""
        template = f"{organ_name} shows no significant abnormalities."
        if text.startswith(template) and text != template:
            stripped = text[len(template):].strip()
            # Clean leading punctuation/whitespace if template was concatenated.
            return stripped.lstrip(",.;: ").strip()
        return text

    def _build_organ_text(self, study_id: str, organ_name: str) -> str:
        """Combine conclusion + findings organ descriptions; fallback to neutral text."""
        conc_entry = self.conc_info.get(study_id, {})
        desc_entry = self.desc_info.get(study_id, {})
        text_parts: List[str] = []

        conc_text = conc_entry.get(organ_name, "")
        desc_text = desc_entry.get(organ_name, "")

        if conc_text:
            text_parts.append(conc_text.strip())
        if desc_text:
            text_parts.append(desc_text.strip())

        if not text_parts:
            # If organ-specific text is missing, provide a neutral statement.
            neutral = f"{organ_name} shows no significant abnormalities."
            text_parts.append(neutral)
            merged_text = " ".join(text_parts)
            merged_text = self._strip_neutral_template(organ_name, merged_text)
            return merged_text, True

        merged_text = " ".join(text_parts)
        merged_text = self._strip_neutral_template(organ_name, merged_text)
        is_normal = "no significant abnormalities" in merged_text.lower()
        return merged_text, is_normal
        

class CTRATEDataset_w_Seg(Dataset):
    def __init__(self, args, model_args, tokenizer, mode="train"):
        self.args = args
        self.model_args = model_args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        json_file = []
        with open(args.cap_data_path_CT_rate, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                json_file.append(obj)
        self.data_list_ct_rate = json_file

        # apply the same transform to both images and segmentations
        train_transform = mtf.Compose(
            [
                # spatial transforms applied to both A and B
                mtf.RandRotate90d(keys=["A", "B"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["A", "B"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["A", "B"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["A", "B"], prob=0.10, spatial_axis=2),
        
                # intensity transforms: usually only on the “image” (e.g. A)
                mtf.RandScaleIntensityd(keys=["A"], factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys=["A"], offsets=0.1, prob=0.5),
        
                mtf.ToTensord(keys=["A"], dtype=torch.float32),
                mtf.ToTensord(keys=["B"], dtype=torch.int8),
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensord(keys=["A"], dtype=torch.float32),
                mtf.ToTensord(keys=["B"], dtype=torch.int8),
            ]
        )
        set_track_meta(False)
        
        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
            self.data_list_ct_rate = self.data_list_ct_rate[:512]
            
        elif 'test' in mode:
            self.transform = val_transform
            
        self.len_ct_rate = len(self.data_list_ct_rate) if args.use_CT_rate_data else 0
        
    def __len__(self):
        return self.len_ct_rate

    def _get_ct_rate_item(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list_ct_rate[idx]
                image_abs_path = data["image_3d"]
                seg_abs_path = image_abs_path.replace("/CT-RATE/dataset/","/CT-RATE/dataset/ts_seg/ts_total/",1)

                #image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                image = nib.load(image_abs_path).get_fdata()
                seg = nib.load(seg_abs_path).get_fdata()

                if np.max(image) == 0:
                    raise ValueError("Empty image (all zeros)")

                image = (image - (-1000)) / (2000)
                image[image>1.0] = 1.0
                image[image<0.0] = 0.0
                image = np.expand_dims(image, 0)
                seg = np.expand_dims(seg, 0)
                image = np.transpose(image,[0, 3, 1, 2])
                seg = np.transpose(seg,[0, 3, 1, 2])
                
                resizer = Resize(spatial_size=self.model_args.img_size)  
                seg_resizer = Resize(
                    spatial_size=self.model_args.img_size,
                    mode="nearest"
                )

                image = resizer(image).detach().cpu().numpy()  # still numpy
                seg = seg_resizer(seg).detach().cpu().numpy()  # still numpy

                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                sample = {
                    "A": image,   # numpy array or torch tensor
                    "B": seg,   # same shape / spatial size as A
                }
                out = self.transform(sample)
                image = out["A"]
                seg = out["B"]

                text = data['conversations'][1]['value']

                text_tensor = self.tokenizer(
                    text, max_length=self.args.max_length, truncation=True, padding="longest", return_tensors="pt"
                )
                
                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                ret = {
                    'image': image,
                    'seg': seg,
                    'text': text,
                    'input_id': input_id,
                    'attention_mask': attention_mask,
                    'question_type': "Image_text_retrieval",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list_ct_rate) - 1)

    def __getitem__(self, idx):
        return self._get_ct_rate_item(idx)
        

class ITRDataset(Dataset):
    def __init__(self, args, model_args, tokenizer, mode="train"):
        self.args = args
        self.model_args = model_args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        if args.use_M3D_data:
            json_file = []
            with open(args.cap_data_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    json_file.append(obj)
            self.data_list = json_file

        if args.use_CT_rate_data:
            json_file = []
            with open(args.cap_data_path_CT_rate, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    json_file.append(obj)
            self.data_list_ct_rate = json_file
            

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        set_track_meta(False)
        
        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
            self.data_list = self.data_list_ct_rate[:512]
            
        elif 'test' in mode:
            self.transform = val_transform
            
        self.len_m3d = len(self.data_list) if args.use_M3D_data else 0
        self.len_ct_rate = len(self.data_list_ct_rate) if args.use_CT_rate_data else 0
        
    def __len__(self):
        return self.len_ct_rate
        # use_m3d = self.args.use_M3D_data
        # use_ct = self.args.use_CT_rate_data

        # if use_m3d and use_ct:
        #     return self.len_m3d + self.len_ct_rate
        # elif use_m3d:
        #     return self.len_m3d
        # elif use_ct:
        #     return self.len_ct_rate
        # else:
        #     return 0
            

    def truncate_text(self, input_text, max_tokens):
        def count_tokens(text):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        if count_tokens(input_text) <= max_tokens:
            return input_text

        sentences = input_text.split('.')

        selected_sentences = []
        current_tokens = 0

        if sentences:
            selected_sentences.append(sentences.pop(0))

        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        truncated_text = '.'.join(selected_sentences)
        return truncated_text

    def _get_ct_rate_item(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list_ct_rate[idx]
                image_abs_path = data["image_3d"]

                #image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                image = nib.load(image_abs_path).get_fdata()

                image = (image - (-1000)) / (2000)
                image[image>1.0] = 1.0
                image[image<0.0] = 0.0
                image = np.expand_dims(image, 0)
                image = np.transpose(image,[0, 3, 1, 2])

                if np.max(image) == 0:
                    raise ValueError("Empty image (all zeros)")

                resizer = Resize(spatial_size=self.model_args.img_size)  
                image = resizer(image).detach().cpu().numpy()  # still numpy

                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image)

                # affine = np.eye(4)
                # nifti_image = nib.Nifti1Image(np.squeeze(image.detach().cpu().numpy()), affine)
                # nib.save(nifti_image, "./image" + str(random.randint(1, 10000)) + ".nii.gz")
                # exit(0)

                #text = self.truncate_text(data['conversations'][1]['value'], self.args.max_length)
                text = data['conversations'][1]['value']

                text_tensor = self.tokenizer(
                    text, max_length=self.args.max_length, truncation=True, padding="longest", return_tensors="pt"
                )
                
                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                ret = {
                    'image': image,
                    'text': text,
                    'input_id': input_id,
                    'attention_mask': attention_mask,
                    'question_type': "Image_text_retrieval",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list_ct_rate) - 1)
    
    def _get_m3d_item(self,idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_abs_path = data["image_3d"]

                #image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                image = nib.load(image_abs_path).get_fdata(dtype=np.float32)

                if np.max(image) == 0:
                    raise ValueError("Empty image (all zeros)")
    
                image = image / np.max(image)
                image[image>1.0] = 1.0
                image[image<0.0] = 0.0
                image = np.expand_dims(image, 0)

                resizer = Resize(spatial_size=self.model_args.img_size)  
                image = resizer(image).detach().cpu().numpy()  # still numpy

                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image)

                # affine = np.eye(4)
                # nifti_image = nib.Nifti1Image(np.squeeze(image.detach().cpu().numpy()), affine)
                # nib.save(nifti_image, "./image" + str(random.randint(1, 10000)) + ".nii.gz")
                # exit(0)
                #text = self.truncate_text(data['conversations'][1]['value'], self.args.max_length)
                text = data['conversations'][1]['value']

                text_tensor = self.tokenizer(
                    text, max_length=self.args.max_length, truncation=True, padding="longest", return_tensors="pt"
                )
                
                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                ret = {
                    'image': image,
                    'text': text,
                    'input_id': input_id,
                    'attention_mask': attention_mask,
                    'question_type': "Image_text_retrieval",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)
                

    def __getitem__(self, idx):
        # use_m3d = self.args.use_M3D_data
        # use_ct = self.args.use_CT_rate_data

        # if use_m3d and use_ct:
        #     # flip a coin: <0.5 -> M3D, >=0.5 -> CT-RATE
        #     if random.random() < 0.5:
        #         # randomly pick one M3D sample
        #         m3d_idx = random.randint(0, self.len_m3d - 1)
        #         return self._get_m3d_item(m3d_idx)
        #     else:
        #         # randomly pick one CT-RATE sample
        #         ct_idx = random.randint(0, self.len_ct_rate - 1)
        #         return self._get_ct_rate_item(ct_idx)
                    
        # if use_m3d:
        #     # only m3d
        #     return self._get_m3d_item(idx)
        # if use_ct:
        #     # only ct-rate
        ## i use CT rate no matter what, M3D data quality is too bad. if you really want to use M3D, uncomment the code above.
        return self._get_ct_rate_item(idx)
            
        raise RuntimeError("Both use_M3D_data and use_CT_rate_data are False.")


class CapDataset(Dataset):
    def __init__(self, args, model_args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if args.use_CT_rate_data:
            json_file = []
            with open(args.cap_data_path_CT_rate, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    json_file.append(obj)
            self.data_list_ct_rate = json_file

        self.caption_prompts = Caption_templates

        # Build train_transform: exclude RandRotate90 for fvlm vision_tower to avoid shape inconsistencies
        transform_list = []
        if not (model_args.vision_tower and "fvlm" in model_args.vision_tower.lower()):
            transform_list.append(mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)))
        transform_list.extend([
            mtf.RandFlip(prob=0.10, spatial_axis=0),
            mtf.RandFlip(prob=0.10, spatial_axis=1),
            mtf.RandFlip(prob=0.10, spatial_axis=2),
            mtf.RandScaleIntensity(factors=0.1, prob=0.5),
            mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
            mtf.ToTensor(dtype=torch.float),
        ])
        train_transform = mtf.Compose(transform_list)

        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list_ct_rate)

    def _get_ct_rate_item(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list_ct_rate[idx]
                image_abs_path = data["image_3d"]

                #image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                image = nib.load(image_abs_path).get_fdata()

                image = (image - (-1000)) / (2000)
                image[image>1.0] = 1.0
                image[image<0.0] = 0.0
                image = np.expand_dims(image, 0)
                image = np.transpose(image,[0, 3, 1, 2])

                if np.max(image) == 0:
                    raise ValueError("Empty image (all zeros)")

                resizer = Resize(spatial_size=self.model_args.img_size)  
                image = resizer(image).detach().cpu().numpy()  # still numpy

                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image)

                #text = self.truncate_text(data['conversations'][1]['value'], self.args.max_length)
                answer = data['conversations'][1]['value']

                prompt_question = random.choice(self.caption_prompts)

                question = self.image_tokens + prompt_question

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="longest", return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = int(torch.sum(attention_mask).item())
                eos_pos = None
                if len(input_id) < self.args.max_length:
                    # append EOS when there is room
                    eos_pos = len(input_id)
                    input_id = torch.cat(
                        [
                            input_id,
                            torch.tensor([self.tokenizer.eos_token_id], dtype=input_id.dtype, device=input_id.device),
                        ],
                        dim=0,
                    )
                    attention_mask = torch.cat(
                        [attention_mask, torch.tensor([1], dtype=attention_mask.dtype, device=attention_mask.device)],
                        dim=0,
                    )
                else:
                    # no room; force EOS at last slot
                    eos_pos = len(input_id) - 1
                    input_id[eos_pos] = self.tokenizer.eos_token_id
                    attention_mask[eos_pos] = 1

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="longest", return_tensors="pt"
                )

                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                else:
                    label[label == self.tokenizer.pad_token_id] = -100
                if eos_pos is not None:
                    label[eos_pos] = self.tokenizer.eos_token_id

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "Caption",
                }
                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


    def __getitem__(self, idx):
        return self._get_ct_rate_item(idx)


class VQADataset(Dataset):
    def __init__(self, args, model_args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.model_args = model_args

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "train":
            if args.use_CT_rate_data:
                json_file = []
                with open(args.vqa_data_train_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        json_file.append(obj)
                self.data_list_ct_rate = json_file
                
        elif mode == "validation":
            if args.use_CT_rate_data:
                json_file = []
                with open(args.vqa_data_val_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        json_file.append(obj)
                self.data_list_ct_rate = json_file[:100]
        elif "test" in mode:
            if args.use_CT_rate_data:
                json_file = []
                with open(args.vqa_data_val_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        json_file.append(obj)
                self.data_list_ct_rate = json_file[:100]
        else:
            print("The mode is not desired ! ")

        # Build train_transform: exclude RandRotate90 for fvlm vision_tower to avoid shape inconsistencies
        transform_list = []
        if not (model_args.vision_tower and "fvlm" in model_args.vision_tower.lower()):
            transform_list.append(mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)))
        transform_list.extend([
            mtf.RandFlip(prob=0.10, spatial_axis=0),
            mtf.RandFlip(prob=0.10, spatial_axis=1),
            mtf.RandFlip(prob=0.10, spatial_axis=2),
            mtf.RandScaleIntensity(factors=0.1, prob=0.5),
            mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
            mtf.ToTensor(dtype=torch.float),
        ])
        train_transform = mtf.Compose(transform_list)

        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list_ct_rate)

    def _normalize_question_type(self, data_entry) -> str:
        """
        Infer question type using only the sample id field (no question text or type field).
        """
        entry_id = str(data_entry.get("id", "")).lower()
        if "multiple_choice" in entry_id:
            return "multiple_choice"
        if "short_answer" in entry_id:
            return "short_answer"
        if "report" in entry_id:
            return "report_generation"
        return "long_answer"

    def __getitem__(self, idx):
        max_attempts = 10000
        for _ in range(max_attempts):
            try:
                data = self.data_list_ct_rate[idx]
                image_abs_path = data["image_3d"]

                #image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                image = nib.load(image_abs_path).get_fdata()

                image = (image - (-1000)) / (2000)
                image[image>1.0] = 1.0
                image[image<0.0] = 0.0
                image = np.expand_dims(image, 0)
                image = np.transpose(image,[0, 3, 1, 2])

                if np.max(image) == 0:
                    raise ValueError("Empty image (all zeros)")

                resizer = Resize(spatial_size=self.model_args.img_size)  
                image = resizer(image).detach().cpu().numpy()  # still numpy

                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image)

                question_entry = data["conversations"][0]
                answer_entry = data["conversations"][1]

                question_raw = question_entry["value"]
                answer = answer_entry["value"]

                question_type = self._normalize_question_type(data)

                is_multiple_choice = question_type == "multiple_choice"
                if is_multiple_choice:
                    answer_labels = MULTIPLE_CHOICE_LABEL_PATTERN.findall(answer or "")
                    unique_labels = {label.upper() for label in answer_labels}
                    # Drop samples with no answer label or multiple/ambiguous labels.
                    if len(unique_labels) != 1:
                        idx = random.randint(0, len(self.data_list_ct_rate) - 1)
                        continue
                        
                prompt_suffix = QUESTION_TYPE_TO_PROMPT.get(
                    question_type, LONG_ANSWER_RESPONSE_PROMPT
                )
                question = self.image_tokens + ' ' + question_raw
                question = f"{question.strip()}\n{prompt_suffix}"
                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="longest", return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = int(torch.sum(attention_mask).item())
                eos_pos = None
                if len(input_id) < self.args.max_length:
                    # append EOS when there is room
                    eos_pos = len(input_id)
                    input_id = torch.cat(
                        [
                            input_id,
                            torch.tensor([self.tokenizer.eos_token_id], dtype=input_id.dtype, device=input_id.device),
                        ],
                        dim=0,
                    )
                    attention_mask = torch.cat(
                        [attention_mask, torch.tensor([1], dtype=attention_mask.dtype, device=attention_mask.device)],
                        dim=0,
                    )
                else:
                    # no room; force EOS at last slot
                    eos_pos = len(input_id) - 1
                    input_id[eos_pos] = self.tokenizer.eos_token_id
                    attention_mask[eos_pos] = 1

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="longest", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                else:
                    label[label == self.tokenizer.pad_token_id] = -100
                if eos_pos is not None:
                    label[eos_pos] = self.tokenizer.eos_token_id

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer
                }

                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                # print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list_ct_rate) - 1)


class VQADataset_test(Dataset):
    def __init__(self, args, tokenizer, question_subset="auto", mode="test"):
        self.args = args
        self.tokenizer = tokenizer
        self.question_subset = question_subset
        self.mode = mode
        self.image_tokens = "<im_patch>" * args.proj_out_num
        self.image_root = getattr(args, "vqa_image_root", args.data_root)
        self.img_size = getattr(args, "img_size", (32, 256, 256))
        self.image_norm_min = getattr(args, "image_norm_min", -1000.0)
        self.image_norm_max = getattr(args, "image_norm_max", 1000.0)
        if self.image_norm_max <= self.image_norm_min:
            raise ValueError(
                f"image_norm_max ({self.image_norm_max}) must be greater than image_norm_min ({self.image_norm_min})."
            )

        set_track_meta(False)
        self.resizer = Resize(spatial_size=self.img_size)
        self.transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),
            ]
        )

        self.samples = []
        self._load_open_vqa(args.vqa_data_test_path)
        if self.question_subset == "open":
            self.samples = [s for s in self.samples if s["question_type"] != "multiple_choice"]
        elif self.question_subset == "close":
            self.samples = [s for s in self.samples if s["question_type"] == "multiple_choice"]

    def _infer_question_type(self, entry_id):
        # Infer question type using only the sample id field.
        entry_id_lower = str(entry_id).lower()
        if "multiple_choice" in entry_id_lower:
            return "multiple_choice"
        if "short_answer" in entry_id_lower:
            return "short_answer"
        if "report" in entry_id_lower:
            return "report_generation"
        return "long_answer"

    def _prepare_question(self, raw_question, question_type):
        question = raw_question.replace("<image>", self.image_tokens)
        question = question.replace("<multiple_choice>", "")
        question = question.replace("<long_answer>", "")
        question = question.replace("<short_answer>", "")
        question = question.replace("<report_generation>", "")

        question = question.replace("\n", " ").strip()
        if not question.startswith(self.image_tokens):
            question = f"{self.image_tokens} {question}"
            
        answer_choices = self._extract_answer_choices(question) if question_type == "multiple_choice" else []
        prompt_suffix = QUESTION_TYPE_TO_PROMPT.get(
            question_type, LONG_ANSWER_RESPONSE_PROMPT
        )
        question = f"{question.strip()}\n{prompt_suffix}"
        return question, answer_choices

    def _extract_answer_choices(self, question_text):
        matches = list(re.finditer(r'\(([a-jA-J])\)', question_text))
        if not matches:
            return []
        choices = []
        for idx, match in enumerate(matches):
            label = match.group(1).upper()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(question_text)
            option_text = question_text[start:end].strip()
            if option_text:
                choices.append({"label": label, "text": option_text})
        return choices

    def _load_open_vqa(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        for entry in data:
            image_name = entry["image"]
            image_path = self._resolve_image_path(image_name)
            conversations = entry.get("conversations", [])
            for i in range(0, len(conversations) - 1, 2):
                question_entry = conversations[i]
                answer_entry = conversations[i + 1]
                if question_entry.get("from") != "human":
                    continue
                if answer_entry.get("from") not in ("gpt", "assistant"):
                    continue
                question_type = self._infer_question_type(entry.get("id", ""))
                answer_text = answer_entry.get("value", "").strip()
                if question_type == "multiple_choice":
                    answer_labels = MULTIPLE_CHOICE_LABEL_PATTERN.findall(answer_text or "")
                    unique_labels = {label.upper() for label in answer_labels}
                    if len(unique_labels) != 1:
                        # Skip ambiguous or missing labels for MC.
                        continue
                question_text, answer_choices = self._prepare_question(question_entry.get("value", ""), question_type)
                self.samples.append(
                    {
                        "image_path": image_path,
                        "question": question_text,
                        "answer": answer_text,
                        "question_type": question_type,
                        "answer_choice": json.dumps(answer_choices, ensure_ascii=False),
                        "question_id": entry.get("id", ""),
                    }
                )

    def _resolve_image_path(self, image_name):
        if os.path.isabs(image_name) and os.path.exists(image_name):
            return image_name

        direct_path = os.path.join(self.image_root, image_name)
        if os.path.exists(direct_path):
            return direct_path

        stem = image_name.replace(".nii.gz", "")
        parts = stem.split("_")
        if len(parts) >= 3:
            level1 = "_".join(parts[:2])
            level2 = "_".join(parts[:3])
            candidate = os.path.join(self.image_root, level1, level2, image_name)
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(f"Unable to resolve image path for {image_name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]

        image = nib.load(image_path).get_fdata()
        denom = self.image_norm_max - self.image_norm_min
        if denom == 0:
            denom = 1e-8
        image = (image - self.image_norm_min) / denom
        image = np.clip(image, 0.0, 1.0)
        image = np.expand_dims(image, 0)
        image = np.transpose(image, [0, 3, 1, 2])

        image = self.resizer(image).detach().cpu().numpy()
        image = self.transform(image)

        question = sample["question"]
        answer = sample["answer"].strip()

        return {
            "image": image,
            "question": question,
            "answer": answer,
            "question_type": sample.get("question_type", "long_answer"),
            "answer_choice": sample.get("answer_choice", "[]"),
            "question_id": sample.get("question_id", ""),
        }


class VQAYNDataset(Dataset):
    def __init__(self, args, model_args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.model_args = model_args

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "train":
            self.data_list = pd.read_csv(args.vqa_yn_data_train_path)
        elif mode == "validation":
            self.data_list = pd.read_csv(args.vqa_yn_data_val_path, nrows=2048)
        elif "test" in mode:
            self.data_list = pd.read_csv(args.vqa_yn_data_test_path)
        else:
            print("The mode is not desired ! ")

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_abs_path = os.path.join(self.args.data_root, data["Image Path"])

                # remove "M3D_Cap_npy" from path
                parts = image_abs_path.split(os.sep)
                new_parts = [p for p in parts if p != "M3D_Cap_npy"]
                image_abs_path = os.sep.join(new_parts)

                # replace .npy with .nii.gz
                image_abs_path = image_abs_path.replace(".npy", ".nii.gz")

                #image = np.load(image_abs_path)  # nomalized, 0-1, C,D,H,W
                # image = np.load(img_path)[np.newaxis, ...]  # nomalized
                image = nib.load(image_abs_path).get_fdata()

                if np.max(image) == 0:
                    raise ValueError("Empty image (all zeros)")
                    
                image = image / np.max(image)
                image = np.expand_dims(image, 0)
                
                resizer = Resize(spatial_size=self.model_args.img_size)  
                image = resizer(image)  # still numpy

                image = self.transform(image)

                question = data["Question"]
                answer = str(data["Answer"])

                question = self.image_tokens + ' ' + question
                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="longest", return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="longest", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'answer_choice': data["Answer Choice"],
                    'question_type': data["Question Type"],
                }
                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)



class PosRECDataset(Dataset):
    def __init__(self, args, model_args, tokenizer, tag="0000", description=True, mode='train'):
        self.args = args
        self.tokenizer = tokenizer
        self.model_args = model_args

        self.tag = tag
        self.mode = mode
        self.description = description

        self.dataset_info = dataset_info

        self.image_tokens = "<im_patch>" * args.proj_out_num
        self.box_tokens = ["<bx_start>", "<bx_end>"]

        root_path = args.seg_data_path
        if mode == "train":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="train",
            )
        elif mode == "validation":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )
        elif mode == "test":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif mode == 'test':
            self.transform = val_transform

        self.cls_questions = PosREC_templates["cls_questions"]
        self.des_qustions = PosREC_templates["des_questions"]
        self.cls_answers = PosREC_templates["cls_answers"]
        self.des_answers = PosREC_templates["des_answers"]
        self.cls_no_answers = PosREC_templates["cls_no_answers"]
        self.des_no_answers = PosREC_templates["des_no_answers"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_path = data['image']
            seg_path = data['label']

            image_array = np.load(image_path) #1*32*256*256, normalized

            seg_array = sparse.load_npz(seg_path)
            gt_shape = ast.literal_eval(seg_path.split('.')[-2].split('_')[-1])
            seg_array = seg_array.toarray().reshape(gt_shape)

            cls_id = np.random.choice(seg_array.shape[0])
            seg_array = seg_array[cls_id:cls_id+1,:,:,:]

            seg_array = np.transpose(seg_array,[0, 3, 1, 2])
            image_array = np.transpose(image_array, [0, 3, 1, 2])
            
            #print("before", seg_array.shape, image_array.shape)
            
            resizer = Resize(spatial_size=self.model_args.img_size)  
            image_array = resizer(image_array)  # still numpy
            seg_array = resizer(seg_array)  # still numpy

            #print("after", seg_array.shape, image_array.shape)

            # affine = np.eye(4) 
            # nifti_image = nib.Nifti1Image(np.sum(seg_array.astype(np.float32),axis=0), affine)
            # nib.save(nifti_image, "/mnt/workspace/offline/xiehuidong.6688/p.nii.gz")
            
            # seg_array = np.load(seg_path)['data'].reshape(np.squeeze(image_array).shape)
            #cls_id = int(os.path.basename(seg_path).split('_')[1].split('.')[0])
            
            try:
                item = {
                    'image': image_array,
                    'seg': seg_array,
                }

                it = self.transform(item)

                image = it['image']
                seg = it['seg']  # 1*D*H*W

                cls_list = self.dataset_info[self.tag]

                # print(cls_id)
                # print(cls_list[cls_id])
                # print(term_dict[cls_list[cls_id]])
                      
                vld_cls = torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()

                if vld_cls:
                    box = mask2box(seg[0])
                    if not self.description:
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]
                        answer = random.choice(self.cls_answers).format(box_text)
                    else:
                        question_temple = random.choice(self.des_qustions)
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]
                        answer = random.choice(self.des_answers).format(cls_list[cls_id], box_text)
                else:
                    if not self.description:
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.cls_no_answers).format(cls_list[cls_id])
                    else:
                        question_temple = random.choice(self.des_qustions)
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.des_no_answers).format(cls_list[cls_id])

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="longest",
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="longest", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "REC",
                }

                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class PosREGDataset(Dataset):
    def __init__(self, args, model_args, tokenizer, tag="0000", description=True, mode='train'):
        self.args = args
        self.tokenizer = tokenizer
        self.model_args = model_args

        self.tag = tag
        self.mode = mode
        self.description = description

        self.dataset_info = dataset_info

        self.image_tokens = "<im_patch>" * args.proj_out_num
        self.box_tokens = ["<bx_start>", "<bx_end>"]

        root_path = args.seg_data_path
        if mode == "train":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="train",
            )
        elif mode == "validation":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )
        elif mode == "test":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif mode == 'test':
            self.transform = val_transform

        self.cls_questions = PosREG_templates["cls_questions"]
        self.des_questions = PosREG_templates["des_questions"]
        self.cls_answers = PosREG_templates["cls_answers"]
        self.des_answers = PosREG_templates["des_answers"]

        self.cls_no_questions = PosREC_templates["cls_questions"]
        self.des_no_questions = PosREC_templates["des_questions"]

        self.cls_no_answers = PosREG_templates["cls_no_answers"]
        self.des_no_answers = PosREG_templates["des_no_answers"]


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_path = data['image']
            seg_path = data['label']

            image_array = np.load(image_path) #1*32*256*256, normalized

            seg_array = sparse.load_npz(seg_path)
            gt_shape = ast.literal_eval(seg_path.split('.')[-2].split('_')[-1])
            seg_array = seg_array.toarray().reshape(gt_shape)

            cls_id = np.random.choice(seg_array.shape[0])
            seg_array = seg_array[cls_id:cls_id+1,:,:,:]

            seg_array = np.transpose(seg_array,[0, 3, 1, 2])
            image_array = np.transpose(image_array, [0, 3, 1, 2])

            resizer = Resize(spatial_size=self.model_args.img_size)  
            image_array = resizer(image_array)  # still numpy
            seg_array = resizer(seg_array)  # still numpy

            # seg_array = np.load(seg_path)
            # cls_id = int(os.path.basename(seg_path).split('_')[1].split('.')[0][1])
            
            try:
                item = {
                    'image': image_array,
                    'seg': seg_array,
                }

                it = self.transform(item)
                image = it['image']
                seg = it['seg']  # 1*D*H*W

                cls_list = self.dataset_info[self.tag]
                vld_cls = torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()

                if vld_cls:
                    box = mask2box(seg[0])
                    if not self.description:
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(box_text)
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.cls_answers).format(cls_list[cls_id])
                    else:
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]
                        question_temple = random.choice(self.des_questions)
                        question = question_temple.format(box_text)
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.des_answers).format(cls_list[cls_id], random.choice(term_dict[cls_list[cls_id]]))
                else:
                    if not self.description:
                        question_temple = random.choice(self.cls_no_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.cls_no_answers).format(cls_list[cls_id])
                    else:
                        question_temple = random.choice(self.des_no_questions)
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.des_no_answers).format(cls_list[cls_id])

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="longest",
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="longest", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "REG",
                }

                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)



class SegDataset(Dataset):
    def __init__(self, args, model_args, tokenizer, tag="0000", description=False, mode='train'):
        self.args = args
        self.tokenizer = tokenizer
        self.model_args = model_args

        self.tag = tag
        self.description = description
        self.mode = mode
        self.dataset_info = dataset_info

        self.image_tokens = "<im_patch>" * args.proj_out_num

        root_path = args.seg_data_path
        if mode == "train":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="train",
            )
        elif mode == "validation":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )
        elif mode == "test":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif mode == 'test':
            self.transform = val_transform

        self.cls_questions = Seg_templates["cls_questions"]
        self.des_questions = Seg_templates["des_questions"]
        self.cls_answers = Seg_templates["cls_answers"]
        self.des_answers = Seg_templates["des_answers"]
        self.cls_no_answers = Seg_templates["cls_no_answers"]
        self.des_no_answers = Seg_templates["des_no_answers"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_path = data['image']
            seg_path = data['label']
            
            image_array = np.load(image_path) #1*32*256*256, normalized
            
            seg_array = sparse.load_npz(seg_path)
            gt_shape = ast.literal_eval(seg_path.split('.')[-2].split('_')[-1])
            seg_array = seg_array.toarray().reshape(gt_shape)

            cls_id = np.random.choice(seg_array.shape[0])
            seg_array = seg_array[cls_id:cls_id+1,:,:,:]

            seg_array = np.transpose(seg_array,[0, 3, 1, 2])
            image_array = np.transpose(image_array, [0, 3, 1, 2])

            resizer = Resize(spatial_size=self.model_args.img_size)  
            image_array = resizer(image_array)  # still numpy
            seg_array = resizer(seg_array)  # still numpy

            # print("after", seg_array.shape, image_array.shape)

            # seg_array = np.load(seg_path)
            # cls_id = int(os.path.basename(seg_path).split('_')[1].split('.')[0][1])
            
            try:
                item = {
                    'image': image_array,
                    'seg': seg_array,
                }

                it = self.transform(item)

                image = it['image']
                seg = it['seg']  # 1*D*H*W

                cls_list = self.dataset_info[self.tag]
                vld_cls = torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()
                if vld_cls:
                    if not self.description:
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.cls_answers)
                    else:
                        question_temple = random.choice(self.des_questions)
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.des_answers).format(cls_list[cls_id])
                else:
                    if not self.description:
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.cls_no_answers).format(cls_list[cls_id])
                    else:
                        question_temple = random.choice(self.des_questions)
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.des_no_answers).format(cls_list[cls_id])

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="longest",
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="longest", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'seg': seg,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "seg",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)



class RefSegDataset(Dataset):
    def __init__(self, args, model_args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.model_args = model_args

        self.image_tokens = "<im_patch>" * args.proj_out_num

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.data_list = pd.read_csv(args.refseg_data_train_path, engine='python')
            self.transform = train_transform
        elif mode == 'validation':
            self.data_list = pd.read_csv(args.refseg_data_test_path, engine='python')
            self.transform = val_transform
        elif mode == 'test':
            self.data_list = pd.read_csv(args.refseg_data_test_path, engine='python')
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_path = os.path.join(self.args.refseg_data_train_path[:-14], data["Image"])

                #nib.load(image_abs_path).get_fdata()
                #image_array = np.load(image_path)  # 1*32*256*256, normalized
                image_array = nib.load(image_path).get_fdata()

                seg_path = os.path.join(self.args.refseg_data_train_path[:-14], data["Mask"])
                
                #seg_array = np.load(seg_path)
                seg_array = nib.load(seg_path).get_fdata()
                seg_array = (seg_array == data["Mask_ID"]).astype(np.int8)

                image_array = np.expand_dims(np.transpose(image_array,[2,0,1]), 0)
                seg_array = np.expand_dims(np.transpose(seg_array,[2,0,1]), 0)

                resizer = Resize(spatial_size=self.model_args.img_size)  
                image_array = resizer(image_array)  # still numpy
                seg_array = resizer(seg_array)  # still numpy
                
                item = {
                    "image": image_array,
                    "seg": seg_array,
                }

                it = self.transform(item)

                image = it['image']
                seg = it['seg']  # C*D*H*W

                question = data["Question"]
                question = self.image_tokens + ' ' + question

                answer = data["Answer"]

                self.tokenizer.padding_side = "right"
                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="longest", return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="longest", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'seg': seg,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "refseg",
                }

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class MultiSegDataset(Dataset):
    def __init__(self, args, model_args, tokenizer, mode='train'):
        super(MultiSegDataset, self).__init__()
        self.tokenizer = tokenizer

        self.dataset_info = dataset_info

        self.ds_list = []
        #self.ds_list.append(RefSegDataset(args, model_args, tokenizer, mode=mode))
        for dataset_code in self.dataset_info.keys():
           self.ds_list.append(SegDataset(args, model_args, tokenizer, tag=dataset_code, description=False, mode=mode))
           self.ds_list.append(SegDataset(args, model_args, tokenizer, tag=dataset_code, description=True, mode=mode))
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class MultiPosDataset(Dataset):
    def __init__(self, args, model_args, tokenizer, mode='train'):
        super(MultiPosDataset, self).__init__()
        self.tokenizer = tokenizer

        self.dataset_info = dataset_info

        self.ds_list = []
        for dataset_code in self.dataset_info.keys():
            self.ds_list.append(PosRECDataset(args, model_args, tokenizer, tag=dataset_code, description=False, mode=mode))
            self.ds_list.append(PosRECDataset(args, model_args, tokenizer, tag=dataset_code, description=True, mode=mode))
            self.ds_list.append(PosREGDataset(args, model_args, tokenizer, tag=dataset_code, description=False, mode=mode))
            self.ds_list.append(PosREGDataset(args, model_args, tokenizer, tag=dataset_code, description=True, mode=mode))
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]



class PosSegDatasets(Dataset):
    def __init__(self, args, model_args, tokenizer, mode='train'):
        super(PosSegDatasets, self).__init__()
        self.ds_list = [
            MultiPosDataset(args, model_args, tokenizer, mode),
            MultiSegDataset(args, model_args, tokenizer, mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class TextDatasets(Dataset):
    def __init__(self, args, model_args, tokenizer, mode='train'):
        super(TextDatasets, self).__init__()
        self.ds_list = [
            CapDataset(args, model_args, tokenizer, mode),
            VQADataset(args, model_args, tokenizer, mode=mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)
        self.model_args = model_args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class UniDatasets(Dataset):
    def __init__(self, args, model_args, tokenizer, mode='train'):
        super(UniDatasets, self).__init__()
        self.ds_list = [
            CapDataset(args, model_args, tokenizer, mode),
            VQADataset(args, model_args, tokenizer, mode=mode),
            #VQAYNDataset(args, model_args, tokenizer, mode=mode),
            #MultiPosDataset(args, model_args, tokenizer, mode),
            #MultiSegDataset(args, model_args, tokenizer, mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)
        self.model_args = model_args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]



