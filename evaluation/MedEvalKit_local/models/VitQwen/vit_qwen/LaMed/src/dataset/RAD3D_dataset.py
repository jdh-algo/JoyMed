import os
import re
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import monai.transforms as mtf
from monai.data import set_track_meta
from monai.transforms import Resize
import nibabel as nib

# Import prompt constants from multi_dataset
from .multi_dataset import (
    CLOSE_ENDED_RESPONSE_PROMPT,
    SHORT_ANSWER_RESPONSE_PROMPT,
    LONG_ANSWER_RESPONSE_PROMPT,
    REPORT_GENERATION_PROMPT,
    OPEN_ENDED_RESPONSE_PROMPT,
    QUESTION_TYPE_TO_PROMPT,
    MULTIPLE_CHOICE_LABEL_PATTERN,
)


class VQADataset_test_3D_RAD(Dataset):
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
        self._load_3d_rad_csv(args.vqa_data_test_path)
        if self.question_subset == "open":
            self.samples = [s for s in self.samples if s["question_type"] != "multiple_choice"]
        elif self.question_subset == "close":
            self.samples = [s for s in self.samples if s["question_type"] == "multiple_choice"]

    def _map_question_type(self, question_type_str):
        """Map CSV QuestionType to internal question type."""
        question_type_lower = str(question_type_str).lower().strip()
        if question_type_lower == "close" or "close" in question_type_lower:
            return "multiple_choice"
        elif question_type_lower == "open" or "open" in question_type_lower:
            return "short_answer"  # Use SHORT_ANSWER_RESPONSE_PROMPT for open questions
        else:
            return "short_answer"  # default to short_answer

    def _prepare_question(self, raw_question, question_type, answer_choices=None):
        question = raw_question.replace("<image>", self.image_tokens)
        question = question.replace("<multiple_choice>", "")
        question = question.replace("<long_answer>", "")
        question = question.replace("<short_answer>", "")
        question = question.replace("<report_generation>", "")

        question = question.replace("\n", " ").strip()
        if not question.startswith(self.image_tokens):
            question = f"{self.image_tokens} {question}"
        prompt_suffix = QUESTION_TYPE_TO_PROMPT.get(
            question_type, LONG_ANSWER_RESPONSE_PROMPT
        )

        if question_type == "multiple_choice":
            choices = answer_choices or []
            if not choices:
                choices = self._extract_answer_choices(question)
            if choices:
                first_choice_match = re.search(r'\([a-jA-J]\)', question)
                if first_choice_match:
                    question = question[:first_choice_match.start()].rstrip()
                choices_text = " ".join([f"({c['label']}) {c['text']}" for c in choices])
                question = f"{question.strip()} {choices_text}\n{prompt_suffix}"
            else:
                question = f"{question.strip()}\n{prompt_suffix}"
            return question, choices

        question = f"{question.strip()}\n{prompt_suffix}"
        return question, []

    def _extract_answer_choices(self, question_text):
        """Extract answer choices from question text in format (A) option1 (B) option2 etc."""
        matches = list(re.finditer(r'\(([a-jA-J])\)', question_text))
        if not matches:
            return []
        choices = []
        for idx, match in enumerate(matches):
            label = match.group(1).lower()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(question_text)
            option_text = question_text[start:end].strip()
            if option_text:
                choices.append({"label": label, "text": option_text})
        return choices

    def _build_answer_choices_from_csv(self, row):
        """Build answer choices list from CSV row (Choice A, Choice B, etc.)."""
        choices = []
        choice_labels = ['A', 'B', 'C', 'D']
        for label in choice_labels:
            choice_col = f"Choice {label}"
            if choice_col in row and pd.notna(row[choice_col]):
                choice_text = str(row[choice_col]).strip()
                if choice_text and choice_text != "-" and choice_text.lower() != "nan":
                    choices.append({"label": label.lower(), "text": choice_text})
        return choices

    def _resolve_image_path(self, volume_name):
        """
        Convert VolumeName like 'test_1016_d_2.nii.gz' to full path:
        '/mnt/workspace/offline/shared_data/Medical_Data_3D/CT-RATE/dataset/valid/valid_1016/valid_1016_d/valid_1016_d_2.nii.gz'
        """
        # Remove .nii.gz extension
        stem = volume_name.replace(".nii.gz", "")
        parts = stem.split("_")
        
        # Pattern: test_<num>_<letter>_<num> -> valid_<num>/valid_<num>_<letter>/valid_<num>_<letter>_<num>.nii.gz
        if len(parts) >= 4 and parts[0] == "test":
            # Extract number, letter, and final number
            num = parts[1]
            letter = parts[2]
            final_num = parts[3]
            
            # Build path: valid_<num>/valid_<num>_<letter>/valid_<num>_<letter>_<num>.nii.gz
            image_filename = f"valid_{num}_{letter}_{final_num}.nii.gz"
            image_path = os.path.join(
                self.image_root,
                f"valid_{num}",
                f"valid_{num}_{letter}",
                image_filename
            )
            
            if os.path.exists(image_path):
                return image_path
        
        # Fallback: try direct path
        direct_path = os.path.join(self.image_root, volume_name)
        if os.path.exists(direct_path):
            return direct_path
        
        # Fallback: try alternative path structure
        if len(parts) >= 3:
            level1 = "_".join(parts[:2])
            level2 = "_".join(parts[:3])
            candidate = os.path.join(self.image_root, level1, level2, volume_name)
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(f"Unable to resolve image path for {volume_name}")

    def _load_3d_rad_csv(self, csv_path):
        """Load samples from 3D-RAD CSV file."""
        df = pd.read_csv(csv_path)
        
        for idx, row in df.iterrows():
            volume_name = row["VolumeName"]
            image_path = self._resolve_image_path(volume_name)
            
            question_text = str(row["Question"]).strip()
            answer_text = str(row["Answer"]).strip()
            question_type_str = str(row.get("QuestionType", "Open")).strip()
            question_type = self._map_question_type(question_type_str)
            
            csv_choices = []
            # For multiple choice questions, collect answer choices if available in CSV.
            if question_type == "multiple_choice":
                csv_choices = self._build_answer_choices_from_csv(row)
                # Validate answer has a label for multiple choice.
                answer_labels = MULTIPLE_CHOICE_LABEL_PATTERN.findall(answer_text or "")
                unique_labels = {label.upper() for label in answer_labels}

                raw_answer = str(answer_text).strip() if answer_text is not None else ""
                if re.fullmatch(r"[A-Da-d]", raw_answer):
                    unique_labels.add(raw_answer.upper())

                answer_choice = str(row.get("AnswerChoice", "")).strip()
                if re.fullmatch(r"[A-Da-d]", answer_choice):
                    unique_labels.add(answer_choice.upper())
                elif re.fullmatch(r"\([A-Da-d]\)", answer_choice):
                    unique_labels.add(answer_choice.strip("()").upper())

                if not unique_labels and csv_choices:
                    normalized_answer = raw_answer.lower()
                    for choice in csv_choices:
                        choice_text = str(choice.get("text", "")).strip().lower()
                        choice_label = str(choice.get("label", "")).strip().upper()
                        if normalized_answer and choice_text == normalized_answer and choice_label:
                            unique_labels.add(choice_label)

                if len(unique_labels) != 1:
                    # Skip ambiguous or missing labels for MC.
                    continue

                label = next(iter(unique_labels))
                label_lower = label.lower()
                choice_text = None
                for choice in csv_choices:
                    choice_label = str(choice.get("label", "")).strip().upper()
                    if choice_label == label:
                        choice_text = str(choice.get("text", "")).strip()
                        break

                if choice_text:
                    answer_text = f"({label_lower}) {choice_text}"
                elif raw_answer:
                    if MULTIPLE_CHOICE_LABEL_PATTERN.search(raw_answer):
                        answer_text = raw_answer
                    else:
                        answer_text = f"({label_lower}) {raw_answer}"
                else:
                    answer_text = f"({label_lower})"
            
            # Prepare question with prompt suffix
            question_text, answer_choices = self._prepare_question(
                question_text, question_type, answer_choices=csv_choices
            )
            # print(question_text, answer_text)
            # exit(0)
            
            # Generate question_id from row index
            question_id = f"3d_rad_{idx}"
            
            self.samples.append(
                {
                    "image_path": image_path,
                    "question": question_text,
                    "answer": answer_text,
                    "question_type": question_type,
                    "answer_choice": json.dumps(answer_choices, ensure_ascii=False),
                    "question_id": question_id,
                }
            )

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
            "question_type": sample.get("question_type", "short_answer"),
            "answer_choice": sample.get("answer_choice", "[]"),
            "question_id": sample.get("question_id", ""),
        }


class VQADataset_train_3D_RAD(Dataset):
    """
    Training dataset for 3D-RAD VQA.
    Loads all CSV files from all task subdirectories under a given root directory.
    Includes data augmentation transforms for training.
    """
    def __init__(self, args, model_args, tokenizer, mode="train"):
        self.args = args
        self.model_args = model_args
        self.tokenizer = tokenizer
        self.mode = mode
        self.image_tokens = "<im_patch>" * args.proj_out_num
        
        # Image root for training data: train_fixed directory
        self.image_root = getattr(
            args, "rad3d_train_image_root",
            "/mnt/workspace/offline/shared_data/Medical_Data_3D/CT-RATE/dataset/train_fixed"
        )
        self.img_size = model_args.img_size
        self.image_norm_min = getattr(args, "image_norm_min", -1000.0)
        self.image_norm_max = getattr(args, "image_norm_max", 1000.0)
        if self.image_norm_max <= self.image_norm_min:
            raise ValueError(
                f"image_norm_max ({self.image_norm_max}) must be greater than image_norm_min ({self.image_norm_min})."
            )

        set_track_meta(False)
        self.resizer = Resize(spatial_size=self.img_size)

        # Build transforms: include augmentation for training
        transform_list = []
        if mode == 'train':
            # Exclude RandRotate90 for fvlm vision_tower to avoid shape inconsistencies
            if not (model_args.vision_tower and "fvlm" in model_args.vision_tower.lower()):
                transform_list.append(mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)))
            transform_list.extend([
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
            ])
        transform_list.append(mtf.ToTensor(dtype=torch.float))
        self.transform = mtf.Compose(transform_list)

        self.samples = []
        # Load all CSV files from the training directory
        train_root = getattr(
            args, "rad3d_train_data_path",
            "/mnt/workspace/offline/shared_data/Medical_Data_3D/3D-RAD/train"
        )
        self._load_all_csv_files(train_root)

    def _map_question_type(self, question_type_str):
        """Map CSV QuestionType to internal question type."""
        question_type_lower = str(question_type_str).lower().strip()
        if question_type_lower == "close" or "close" in question_type_lower:
            return "multiple_choice"
        elif question_type_lower == "open" or "open" in question_type_lower:
            return "short_answer"
        else:
            return "short_answer"

    def _prepare_question(self, raw_question, question_type, answer_choices=None):
        question = raw_question.replace("<image>", self.image_tokens)
        question = question.replace("<multiple_choice>", "")
        question = question.replace("<long_answer>", "")
        question = question.replace("<short_answer>", "")
        question = question.replace("<report_generation>", "")

        question = question.replace("\n", " ").strip()
        if not question.startswith(self.image_tokens):
            question = f"{self.image_tokens} {question}"
        prompt_suffix = QUESTION_TYPE_TO_PROMPT.get(
            question_type, LONG_ANSWER_RESPONSE_PROMPT
        )

        if question_type == "multiple_choice":
            choices = answer_choices or []
            if not choices:
                choices = self._extract_answer_choices(question)
            if choices:
                first_choice_match = re.search(r'\([a-jA-J]\)', question)
                if first_choice_match:
                    question = question[:first_choice_match.start()].rstrip()
                choices_text = " ".join([f"({c['label']}) {c['text']}" for c in choices])
                question = f"{question.strip()} {choices_text}\n{prompt_suffix}"
            else:
                question = f"{question.strip()}\n{prompt_suffix}"
            return question, choices

        question = f"{question.strip()}\n{prompt_suffix}"
        return question, []

    def _extract_answer_choices(self, question_text):
        """Extract answer choices from question text in format (A) option1 (B) option2 etc."""
        matches = list(re.finditer(r'\(([a-jA-J])\)', question_text))
        if not matches:
            return []
        choices = []
        for idx, match in enumerate(matches):
            label = match.group(1).lower()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(question_text)
            option_text = question_text[start:end].strip()
            if option_text:
                choices.append({"label": label, "text": option_text})
        return choices

    def _build_answer_choices_from_csv(self, row):
        """Build answer choices list from CSV row (Choice A, Choice B, etc.)."""
        choices = []
        choice_labels = ['A', 'B', 'C', 'D']
        for label in choice_labels:
            choice_col = f"Choice {label}"
            if choice_col in row and pd.notna(row[choice_col]):
                choice_text = str(row[choice_col]).strip()
                if choice_text and choice_text != "-" and choice_text.lower() != "nan":
                    choices.append({"label": label.lower(), "text": choice_text})
        return choices

    def _resolve_image_path(self, volume_name):
        """
        Convert VolumeName like 'train_10025_a_1.nii.gz' to full path:
        '/mnt/workspace/offline/shared_data/Medical_Data_3D/CT-RATE/dataset/train_fixed/train_10025/train_10025_a/train_10025_a_1.nii.gz'
        """
        # Remove .nii.gz extension
        stem = volume_name.replace(".nii.gz", "")
        parts = stem.split("_")
        
        # Pattern: train_<num>_<letter>_<num> -> train_<num>/train_<num>_<letter>/train_<num>_<letter>_<num>.nii.gz
        if len(parts) >= 4 and parts[0] == "train":
            num = parts[1]
            letter = parts[2]
            final_num = parts[3]
            
            # Build path: train_<num>/train_<num>_<letter>/train_<num>_<letter>_<num>.nii.gz
            image_filename = f"train_{num}_{letter}_{final_num}.nii.gz"
            image_path = os.path.join(
                self.image_root,
                f"train_{num}",
                f"train_{num}_{letter}",
                image_filename
            )
            
            if os.path.exists(image_path):
                return image_path
        
        # Fallback: try direct path
        direct_path = os.path.join(self.image_root, volume_name)
        if os.path.exists(direct_path):
            return direct_path
        
        # Fallback: try alternative path structure
        if len(parts) >= 3:
            level1 = "_".join(parts[:2])
            level2 = "_".join(parts[:3])
            candidate = os.path.join(self.image_root, level1, level2, volume_name)
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(f"Unable to resolve image path for {volume_name}")

    def _load_all_csv_files(self, train_root):
        """
        Load all CSV files from all task subdirectories under train_root.
        Discovers Task1_*, Task2_*, etc. directories and loads all .csv files within them.
        """
        if not os.path.isdir(train_root):
            raise ValueError(f"Training root directory does not exist: {train_root}")
        
        csv_files = []
        # Iterate through all task directories
        for task_dir in sorted(os.listdir(train_root)):
            task_path = os.path.join(train_root, task_dir)
            if os.path.isdir(task_path) and task_dir.startswith("Task"):
                # Find all CSV files in this task directory
                for csv_file in os.listdir(task_path):
                    if csv_file.endswith(".csv"):
                        csv_files.append(os.path.join(task_path, csv_file))
        
        print(f"[VQADataset_train_3D_RAD] Found {len(csv_files)} CSV files in {train_root}")
        
        # Load samples from each CSV file
        for csv_path in csv_files:
            self._load_3d_rad_csv(csv_path)
        
        print(f"[VQADataset_train_3D_RAD] Loaded {len(self.samples)} total samples")

    def _load_3d_rad_csv(self, csv_path):
        """Load samples from a single 3D-RAD CSV file."""
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[Warning] Failed to load CSV {csv_path}: {e}")
            return
        
        for idx, row in df.iterrows():
            try:
                volume_name = row["VolumeName"]
                image_path = self._resolve_image_path(volume_name)
                
                question_text = str(row["Question"]).strip()
                answer_text = str(row["Answer"]).strip()
                question_type_str = str(row.get("QuestionType", "Open")).strip()
                question_type = self._map_question_type(question_type_str)
                
                csv_choices = []
                # For multiple choice questions, collect answer choices if available in CSV.
                if question_type == "multiple_choice":
                    csv_choices = self._build_answer_choices_from_csv(row)
                    # Validate answer has a label for multiple choice.
                    answer_labels = MULTIPLE_CHOICE_LABEL_PATTERN.findall(answer_text or "")
                    unique_labels = {label.upper() for label in answer_labels}

                    raw_answer = str(answer_text).strip() if answer_text is not None else ""
                    if re.fullmatch(r"[A-Da-d]", raw_answer):
                        unique_labels.add(raw_answer.upper())

                    answer_choice = str(row.get("AnswerChoice", "")).strip()
                    if re.fullmatch(r"[A-Da-d]", answer_choice):
                        unique_labels.add(answer_choice.upper())
                    elif re.fullmatch(r"\([A-Da-d]\)", answer_choice):
                        unique_labels.add(answer_choice.strip("()").upper())

                    if not unique_labels and csv_choices:
                        normalized_answer = raw_answer.lower()
                        for choice in csv_choices:
                            choice_text = str(choice.get("text", "")).strip().lower()
                            choice_label = str(choice.get("label", "")).strip().upper()
                            if normalized_answer and choice_text == normalized_answer and choice_label:
                                unique_labels.add(choice_label)

                    if len(unique_labels) != 1:
                        # Skip ambiguous or missing labels for MC.
                        continue

                    label = next(iter(unique_labels))
                    label_lower = label.lower()
                    choice_text = None
                    for choice in csv_choices:
                        choice_label = str(choice.get("label", "")).strip().upper()
                        if choice_label == label:
                            choice_text = str(choice.get("text", "")).strip()
                            break

                    if choice_text:
                        answer_text = f"({label_lower}) {choice_text}"
                    elif raw_answer:
                        if MULTIPLE_CHOICE_LABEL_PATTERN.search(raw_answer):
                            answer_text = raw_answer
                        else:
                            answer_text = f"({label_lower}) {raw_answer}"
                    else:
                        answer_text = f"({label_lower})"
                
                # Prepare question with prompt suffix
                question_text, answer_choices = self._prepare_question(
                    question_text, question_type, answer_choices=csv_choices
                )
                
                # Generate question_id from CSV file and row index
                csv_basename = os.path.basename(csv_path).replace(".csv", "")
                question_id = f"3d_rad_train_{csv_basename}_{idx}"
                
                self.samples.append(
                    {
                        "image_path": image_path,
                        "question": question_text,
                        "answer": answer_text,
                        "question_type": question_type,
                        "answer_choice": json.dumps(answer_choices, ensure_ascii=False),
                        "question_id": question_id,
                    }
                )
            except FileNotFoundError as e:
                # Skip samples with missing images
                continue
            except Exception as e:
                # Skip problematic rows
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
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

                # Tokenize question + answer
                text_tensor = self.tokenizer(
                    question + ' ' + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                # Append EOS token
                eos_pos = None
                if len(input_id) < self.args.max_length:
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
                    eos_pos = len(input_id) - 1
                    input_id[eos_pos] = self.tokenizer.eos_token_id
                    attention_mask[eos_pos] = 1

                # Tokenize question only to get question length
                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                # Create label: mask question tokens with -100
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
                    'question_type': sample.get("question_type", "short_answer"),
                }

                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                # On error, try a random different sample
                idx = random.randint(0, len(self.samples) - 1)
        
        # If all attempts fail, raise error
        raise RuntimeError(f"Failed to load sample after {max_attempts} attempts")
