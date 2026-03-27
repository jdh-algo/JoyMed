"""
AMOS 原始格式评测：使用 shared_data/AMOS/ORI/valid_report.json 作为测试样本集。
数据格式为 nii.gz 路径 + report，与 AMOS (slice 目录格式) 不同。
"""

import os
import json
from tqdm import tqdm
from ..base_dataset import BaseDataset
from ..eval_report import Evalreport
from ..question_formats import get_report_generation_prompt


class AMOS_ori(BaseDataset):
    """使用 valid_report.json 的 AMOS 评测，图像为 nii 文件路径。"""

    def __init__(self, model, dataset_path, output_path):
        super().__init__()
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else ""
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx", 0))
        self.num_chunks = int(os.environ.get("num_chunks", 1))
        self.eval_report = Evalreport()

    def load_data(self):
        json_path = os.path.join(self.dataset_path, "valid_report.json")
        if not os.path.isfile(json_path):
            base_dir = os.path.dirname(self.dataset_path.rstrip(os.sep))
            last_name = os.path.basename(self.dataset_path.rstrip(os.sep))
            if last_name == "AMOS-Ori-Report":
                alt_data_path = os.path.join(base_dir, "AMOS", "ORI")
                alt_json = os.path.join(alt_data_path, "valid_report.json")
                if os.path.isfile(alt_json):
                    json_path = alt_json
                    self.dataset_path = alt_data_path
            if not os.path.isfile(json_path):
                raise FileNotFoundError(f"valid_report.json not found at: {json_path}")

        print(f"Loading AMOS (ori) data from: {json_path}")

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
            image_3d_rel = raw.get("image_3d", "")
            num_slices = raw.get("num_slices", 32)
            axis = raw.get("axis", "2")
            report = raw.get("report", "").strip()
        except (KeyError, TypeError) as e:
            print(f"Warning: Invalid sample format. Error: {e}")
            return None

        if not image_3d_rel or not report:
            return None

        image_path = os.path.join(self.dataset_path, image_3d_rel)
        try:
            nii_num_slices = int(num_slices)
        except (ValueError, TypeError):
            nii_num_slices = 32
        try:
            nii_axis = int(axis)
        except (ValueError, TypeError):
            nii_axis = 2

        is_reasoning = True if os.environ.get("REASONING", "False") == "True" else False
        prompt = get_report_generation_prompt(is_reasoning, question_version='Lingshu', lang="en")

        nii_info = {
            "image_path": image_path,
            "nii_num_slices": nii_num_slices,
            "nii_axis": nii_axis,
        }
        messages = {
            "prompt": prompt,
            "images_3d": nii_info,
        }

        sample = {
            "Question_Type": "REPORT",
            "report": report,
            "question": prompt,
            "conversations": [
                {"from": "human", "value": "<video>\n" + prompt},
                {"from": "gpt", "value": report},
            ],
            "messages": messages,
        }
        return sample

    def cal_metrics(self, out_samples):
        print("\n" + "=" * 80)
        print("Starting AMOS (ori, valid_report) Evaluation")
        print("=" * 80)

        new_out_samples = []
        for sample in out_samples:
            sample["response"] = sample["response"].replace('Findings:', '').replace('Impression:', '')
            new_out_samples.append(sample)

        metrics = self.eval_report(new_out_samples)

        print(f"\nMetrics: {metrics}")
        print("=" * 80 + "\n")

        return metrics, out_samples
