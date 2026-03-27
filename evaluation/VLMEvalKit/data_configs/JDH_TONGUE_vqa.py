import os
import re
import json
import numpy as np
import sys
from pathlib import Path
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.image_vqa import CustomVQADataset
from vlmeval.smp import load, dump, d2df
from vlmeval.dataset.utils import build_judge, DEBUG_MESSAGE
from vlmeval.utils import track_progress_rich

# from mm_pipeline.data_juicer.projects.tongue.prompts import sft_short_prompt, long_prompt
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
prompts_path = project_root / "data_juicer" / "projects"
sys.path.insert(0, str(prompts_path))
from tongue.prompts import tongue_main


def build_judge_prompt(answer_text, prediction_text):
    """Build evaluation prompt for AI judge"""

    prompt = f"""你是一名中医舌诊专家，请根据以下标准对模型的回答进行评分。

**评估任务**：
比较模型预测答案与标准答案在各个维度的一致性，分几个分数段对其进行评分。

**标准答案**：
{answer_text}

**模型预测**：
{prediction_text}

**评分标准**：
1. 完全正确：1分
2. 非常接近：0.75分
3. 基本一致：0.5分
4. 较多不一致：0.25分
5. 非常不一致：0分

**请仅输出最终评分，不要输出任何其他内容**：
"""

    return prompt


def tongue_auxeval(model, line):
    """Auxiliary evaluation function for tongue VQA"""
    try:
        # Extract answer and prediction
        answer_text = str(line["answer"])
        prediction_text = str(line["prediction"])

        # Build judge prompt
        judge_prompt = build_judge_prompt(answer_text, prediction_text)
        if not judge_prompt:
            return {"score": 0.0, "log": f"Failed to build judge prompt", "raw_score": 0.0}

        # Get judge response
        msgs = [{"type": "text", "value": judge_prompt}]
        response = model.generate(msgs)
        score = float(response.strip().split("：")[-1].split("：")[-1].split("分")[0].strip())
        return {"score": score, "log": f"模型预测：{prediction_text}\n标准答案：{answer_text}", "raw_score": score}
    except Exception as e:
        return {"score": 0.0, "log": f"Evaluation error: {str(e)}", "raw_score": 0.0}


class TongueVQADataset:
    def load_data(self, dataset):
        data_path = os.path.join("/mnt/workspace/offline/shared_benchmarks", f"{dataset}.tsv")
        return load(data_path)

    def build_prompt(self, line):
        msgs = ImageBaseDataset.build_prompt(self, line)
        msgs[-1]["value"] = tongue_main
        return msgs

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """Evaluate medical chat VQA dataset"""
        import os.path as osp

        # Setup file paths
        suffix = eval_file.split(".")[-1]
        model_name = judge_kwargs.get("model", "Qwen3-32B")
        storage = eval_file.replace(f".{suffix}", f"_{model_name}.xlsx")
        tmp_file = eval_file.replace(f".{suffix}", f"_{model_name}.pkl")
        nproc = judge_kwargs.pop("nproc", 4)

        if not osp.exists(storage):
            # Load data
            data = load(eval_file)
            assert "answer" in data and "prediction" in data

            # Convert to string
            data["prediction"] = [str(x) for x in data["prediction"]]
            data["answer"] = [str(x) for x in data["answer"]]

            # Build judge model
            model = build_judge(**judge_kwargs)
            assert model.working(), "Tongue VQA evaluation requires a working judge model\n" + DEBUG_MESSAGE

            # Prepare evaluation data
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [i for i in range(lt)]

            # Load existing results if available
            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)

            # Filter out already processed items
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            # Process remaining items
            if len(indices):
                new_results = track_progress_rich(
                    tongue_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    # Verify consistency of cached vs newly computed results
                    assert ans[k]["log"] == v["log"] and ans[k]["score"] == v["score"]

            # Collect results
            data["score"] = [ans[i]["score"] for i in range(lt)]
            data["log"] = [ans[i]["log"] for i in range(lt)]
            data["raw_score"] = [ans[i].get("raw_score", 0.0) for i in range(lt)]

            # Save detailed results
            dump(data, storage)

        # Load results and calculate statistics
        data = load(storage)

        # Calculate overall score
        overall_score = np.mean(data["score"])

        # Calculate scores by image type
        results = {"Overall": overall_score}
        # Convert to DataFrame and save
        ret = d2df(results).round(2)
        score_pth = storage.replace(".xlsx", "_score.csv")
        dump(ret, score_pth)

        return ret


class JDH_TONGUE_vqa_test(TongueVQADataset):
    pass


# Override CustomVQADataset methods
CustomVQADataset.load_data = TongueVQADataset.load_data
CustomVQADataset.build_prompt = TongueVQADataset.build_prompt
CustomVQADataset.evaluate = TongueVQADataset.evaluate
# Create specific dataset classes for VLMEvalKit compatibility


# from vlmeval.dataset.utils.vqa_eval import anls_compute, process_line
# def fallback_dimension_score(pred_value, gt_value, dimension_type="text"):
#     """
#     Fallback scoring function using ANLS when AI judge fails
#     This is kept as a backup option but not used by default
#     """
#     if not pred_value or not gt_value:
#         return 0.0

#     if dimension_type == "exact":
#         # For dimensions that require exact match (like image type)
#         return 1.0 if str(pred_value).strip().lower() == str(gt_value).strip().lower() else 0.0
#     elif dimension_type == "text":
#         # For text dimensions, use ANLS with medical-friendly thresholds
#         anls_score = 1 - anls_compute(str(gt_value), str(pred_value))
#         if anls_score >= 0.8:
#             return 1.0  # Full credit for high similarity
#         elif anls_score >= 0.5:
#             return 0.5  # Partial credit for moderate similarity
#         else:
#             return 0.0  # No credit for low similarity
#     else:
#         return 0.0
