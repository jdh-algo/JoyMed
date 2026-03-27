"""
3D-RAD 原始格式评测：使用 shared_data/3D-RAD_ori/valid_vqa.json 作为测试样本集。
数据格式为 nii.gz 路径 + question/answer，与 RAD_3D (Rad3D_test.json) 的 slice 目录格式不同。
"""

import os
import json
from tqdm import tqdm
from ..base_dataset import BaseDataset
from ..eval_3d import _evaluate_core


class RAD_3D_ori(BaseDataset):
    """使用 valid_vqa.json 的 3D-RAD 评测，图像为 nii 文件路径。"""

    def __init__(self, model, dataset_path, output_path):
        super().__init__()
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else ""
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx", 0))
        self.num_chunks = int(os.environ.get("num_chunks", 1))

    def load_data(self):
        json_path = os.path.join(self.dataset_path, "valid_vqa.json")
        if not os.path.isfile(json_path):
            # 评测数据集名为 3D-RAD-Ori，实际数据目录多为 3D-RAD_ori，尝试该路径
            base_dir = os.path.dirname(self.dataset_path.rstrip(os.sep))
            last_name = os.path.basename(self.dataset_path.rstrip(os.sep))
            if last_name == "3D-RAD-Ori":
                alt_data_path = os.path.join(base_dir, "3D-RAD_ori")
                alt_json = os.path.join(alt_data_path, "valid_vqa.json")
                if os.path.isfile(alt_json):
                    json_path = alt_json
                    self.dataset_path = alt_data_path
            if not os.path.isfile(json_path):
                raise FileNotFoundError(f"valid_vqa.json not found at: {json_path}")

        print(f"Loading 3D-RAD (ori) data from: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        for idx, raw in enumerate(tqdm(dataset)):
            if idx % self.num_chunks != self.chunk_idx:
                continue
            sample = self.construct_messages(raw)
            if sample is None:
                continue
            sample["sample_id"] = idx
            self.samples.append(sample)

        print("total samples number:", len(self.samples))
        return self.samples

    def construct_messages(self, raw):
        try:
            question = raw.get("question", "").strip()
            answer = raw.get("answer", "").strip()
            image_3d_rel = raw.get("image_3d", "")
            num_slices = raw.get("num_slices", "32")
            axis = raw.get("axis", "2")
            task = raw.get("Task", "Unknown")
            sub_task = raw.get("SubTask", "Unknown")
            question_type = raw.get("QuestionType", "Open")
        except (KeyError, TypeError) as e:
            print(f"Warning: Invalid sample format. Error: {e}")
            return None

        if not question or not image_3d_rel:
            return None

        # 闭合题需有有效选项字母；answer_idx 为 "-" 或空时按开放题处理
        answer_idx_raw = str(raw.get("answer_idx", "")).strip()
        has_valid_choice = answer_idx_raw not in ("", "-") and answer_idx_raw.upper() in ("A", "B", "C", "D")
        is_choice = (
            str(question_type).lower() in ("choice", "closed", "close")
            and has_valid_choice
            and raw.get("options")
        )

        # ===== 按 Rad3D_test.json 的风格组织问句 =====
        full_question = question
        if is_choice and raw.get("options"):
            option_lines = []
            for k, v in raw["options"].items():
                if v is None or str(v).strip() in ("", "-"):
                    continue
                option_lines.append(f"{k}. {v}")
            if option_lines:
                full_question = question + "\n" + "\n".join(option_lines)

        human_value = "<video>\n" + full_question

        image_path = os.path.join(self.dataset_path, image_3d_rel)
        try:
            nii_num_slices = int(num_slices)
        except (ValueError, TypeError):
            nii_num_slices = 32
        try:
            nii_axis = int(axis)
        except (ValueError, TypeError):
            nii_axis = 2

        # 与 Deeptumor 一致：images_3d 传 nii 信息字典，供模型从 nii 加载
        nii_info = {
            "image_path": image_path,
            "nii_num_slices": nii_num_slices,
            "nii_axis": nii_axis,
        }
        messages = {
            # prompt 不带 <video> 前缀，仅保留问题+选项，和 RAD_3D 中的处理保持一致
            "prompt": full_question,
            "images_3d": nii_info,
        }

        gt_value = answer_idx_raw if is_choice else answer
        # 兼容 _evaluate_core：需要 type/sub-type、Question_Type、conversations、question（错题记录用）
        sample = {
            "type": task,
            "sub-type": sub_task,
            "Question_Type": "CHOICE" if is_choice else "OPEN",
            "question": human_value,
            "conversations": [
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": gt_value},
            ],
            "messages": messages,
        }
        if raw.get("options"):
            sample["options"] = raw["options"]
        if is_choice:
            sample["answer_idx"] = answer_idx_raw
        return sample

    def cal_metrics(self, out_samples):
        print("\n" + "=" * 80)
        print("Starting 3D-RAD (ori, valid_vqa) Evaluation")
        print("=" * 80)

        category_name_mapping = {
            "Task1_Image_Observation": "Image Observation",
            "Task2_Anomaly_Detection": "Anomaly Detection",
            "Task3_Medical_Computation": "Medical Computation",
            "Task4_Existence_Detection": "Existence Detection",
            "Task5_Static_Temporal_Diagnosis": "Static Temporal Diagnosis",
            "Task6_Longitudinal_Temporal_Diagnosis": "Longitudinal Temporal Diagnosis",
        }

        results_text, metrics, wrong_answers = _evaluate_core(
            out_samples=out_samples,
            desc="Evaluating 3D-RAD (ori) predictions",
            category_key="type",
            category_name_mapping=category_name_mapping,
            calculate_overall=True,
        )

        print(results_text)

        wrong_answers_readable = {}
        for key, value in wrong_answers.items():
            readable_key = category_name_mapping.get(key, key)
            wrong_answers_readable[readable_key] = value

        os.makedirs(self.output_path, exist_ok=True)
        wrong_answers_path = os.path.join(self.output_path, "wrong_answers.json")
        with open(wrong_answers_path, "w", encoding="utf-8") as f:
            json.dump(wrong_answers_readable, f, indent=4, ensure_ascii=False)

        print(f"\n✅ Wrong answers saved to: {wrong_answers_path}")
        print("=" * 80 + "\n")

        return metrics, out_samples
