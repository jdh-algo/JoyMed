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

# from mm_pipeline.data_juicer.projects.med_chat.prompts import sft_short_prompt, long_prompt
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
prompts_path = project_root / "data_juicer" / "projects"
sys.path.insert(0, str(prompts_path))
from med_chat.prompts import sft_short_prompt, long_prompt

# Define evaluation criteria for each image type
eval_criteria = {
    "皮损图片": {
        "dimensions": ["图片类型", "是否存在异常", "皮损部位", "皮损类型", "皮损表现"],
        "scoring": {
            "图片类型": {"exact": 1},
            "是否存在异常": {"exact": 1},
            "皮损部位": {"exact": 1},
            "皮损类型": {"exact": 1, "partial": 0.5},
            "皮损表现": {"exact": 1, "partial": 0.5},
        },
        "bonus_checks": [],
        "penalty_checks": ["hallucination"],
    },
    "药盒": {
        "dimensions": ["图片类型", "药品名称", "药品信息"],
        "scoring": {"图片类型": {"exact": 1}, "药品名称": {"exact": 1}, "药品信息": {"exact": 1}},
        "bonus_checks": [],
        "penalty_checks": ["hallucination"],
    },
    "处方": {
        "dimensions": ["图片类型", "疾病诊断", "药品名称", "药品信息"],
        "scoring": {
            "图片类型": {"exact": 1},
            "疾病诊断": {"exact": 1, "partial": 0.5},
            "药品名称": {"exact": 1, "partial": 0.5},
            "药品信息": {"exact": 1},
        },
        "bonus_checks": [],
        "penalty_checks": ["hallucination"],
    },
    "血液检验报告": {
        "dimensions": ["图片类型", "是否存在异常", "异常结果"],
        "scoring": {"图片类型": {"exact": 1}, "是否存在异常": {"exact": 1}, "异常结果": {"exact": 1, "partial": 0.5}},
        "bonus_checks": [],
        "penalty_checks": ["hallucination"],
    },
    "影像检查报告": {
        "dimensions": ["图片类型", "是否存在异常", "异常结果"],
        "scoring": {"图片类型": {"exact": 1}, "是否存在异常": {"exact": 1}, "异常结果": {"exact": 1, "partial": 0.5}},
        "bonus_checks": [],
        "penalty_checks": ["hallucination"],
    },
    "影像片子": {
        "dimensions": ["图片类型", "检查部位", "检查类型", "是否存在异常", "影像所见", "临床意义及建议"],
        "scoring": {
            "图片类型": {"exact": 1},
            "检查部位": {"exact": 1},
            "检查类型": {"exact": 1},
            "是否存在异常": {"exact": 1},
            "影像所见": {"exact": 1, "partial": 0.5},
            "临床意义及建议": {"exact": 1, "partial": 0.5},
        },
        "bonus_checks": [],
        "penalty_checks": ["hallucination"],
    },
    "超声检查报告": {
        "dimensions": ["图片类型", "是否存在异常", "异常结果"],
        "scoring": {"图片类型": {"exact": 1}, "是否存在异常": {"exact": 1}, "异常结果": {"exact": 1, "partial": 0.5}},
        "bonus_checks": [],
        "penalty_checks": ["hallucination"],
    },
    "体检报告": {
        "dimensions": ["图片类型", "是否存在异常", "异常结果"],
        "scoring": {"图片类型": {"exact": 1}, "是否存在异常": {"exact": 1}, "异常结果": {"exact": 1, "partial": 0.5}},
        "bonus_checks": [],
        "penalty_checks": ["hallucination"],
    },
    "血液照片": {
        "dimensions": ["图片类型", "是否存在异常", "异常描述"],
        "scoring": {"图片类型": {"exact": 1}, "是否存在异常": {"exact": 1}, "异常描述": {"exact": 1, "partial": 0.5}},
        "bonus_checks": [],
        "penalty_checks": ["hallucination"],
    },
    "病历": {
        "dimensions": [
            "图片类型",
            "就诊日期/入出院日期",
            "疾病诊断/临床诊断/入出院诊断",
            "诊疗意见或处置措施",
            "其他病历信息",
        ],
        "scoring": {
            "图片类型": {"exact": 1},
            "就诊日期/入出院日期": {"exact": 1},
            "疾病诊断/临床诊断/入出院诊断": {"exact": 1, "partial": 0.5},
            "诊疗意见或处置措施": {"exact": 1, "partial": 0.5},
            "其他病历信息": {"exact": 1},
        },
        "bonus_checks": [],
        "penalty_checks": ["hallucination"],
    },
    "舌部图": {
        "dimensions": ["图片类型", "是否存在异常", "异常描述"],
        "scoring": {"图片类型": {"exact": 1}, "是否存在异常": {"exact": 1}, "异常描述": {"exact": 1, "partial": 0.5}},
        "bonus_checks": [],
        "penalty_checks": ["hallucination"],
    },
    "口腔图": {
        "dimensions": ["图片类型", "是否存在异常", "异常描述"],
        "scoring": {"图片类型": {"exact": 1}, "是否存在异常": {"exact": 1}, "异常描述": {"exact": 1, "partial": 0.5}},
        "bonus_checks": [],
        "penalty_checks": ["hallucination"],
    },
    "眼部图": {
        "dimensions": ["图片类型", "是否存在异常", "异常描述"],
        "scoring": {"图片类型": {"exact": 1}, "是否存在异常": {"exact": 1}, "异常描述": {"exact": 1, "partial": 0.5}},
        "bonus_checks": [],
        "penalty_checks": ["hallucination"],
    },
    "大便图": {
        "dimensions": ["图片类型", "是否存在异常", "异常描述"],
        "scoring": {"图片类型": {"exact": 1}, "是否存在异常": {"exact": 1}, "异常描述": {"exact": 1, "partial": 0.5}},
        "bonus_checks": [],
        "penalty_checks": ["hallucination"],
    },
    "其他检验检查报告": {
        "dimensions": ["图片类型", "是否存在异常", "异常描述"],
        "scoring": {"图片类型": {"exact": 1}, "是否存在异常": {"exact": 1}, "异常描述": {"exact": 1, "partial": 0.5}},
        "bonus_checks": [],
        "penalty_checks": ["hallucination"],
    },
    "其他": {
        "dimensions": ["图片类型", "是否存在异常", "异常描述"],
        "scoring": {"图片类型": {"exact": 1}, "是否存在异常": {"exact": 1}, "异常描述": {"exact": 1, "partial": 0.5}},
        "bonus_checks": [],
        "penalty_checks": ["hallucination"],
    },
}


def get_max_score(image_type, add_bonus=False):
    """Calculate maximum possible score for an image type"""
    if image_type not in eval_criteria:
        return 0

    criteria = eval_criteria[image_type]
    max_score = sum([max(scoring.values()) for scoring in criteria["scoring"].values()])
    if add_bonus:
        max_score += len(criteria["bonus_checks"])
    return max_score


def build_judge_prompt(image_type, answer_text, prediction_text):
    """Build evaluation prompt for AI judge"""
    if image_type not in eval_criteria:
        return None

    criteria = eval_criteria[image_type]
    dimensions = criteria["dimensions"]
    scoring = criteria["scoring"]

    prompt = f"""你是一个专业的医学图像分析评估专家。请根据以下标准对模型的回答进行评分。

**评估任务**：
比较模型预测答案与标准答案在各个维度的一致性，并进行一些附加检查。

**图片类型**：{image_type}

**标准答案**：
{answer_text}

**模型预测**：
{prediction_text}

**评分标准**：
"""

    for dim in dimensions:
        if dim in scoring:
            score_info = scoring[dim]
            prompt += f"\n{dim}：\n"
            if "exact" in score_info and "partial" in score_info:
                prompt += f"  - 完全一致：{score_info['exact']}分\n"
                prompt += f"  - 基本一致（语义相似）：{score_info['partial']}分\n"
                prompt += f"  - 不一致：0分\n"
            else:
                prompt += f"  - 一致：{score_info['exact']}分\n"
                prompt += f"  - 不一致：0分\n"

    # Add penalty checks
    if criteria.get("penalty_checks"):
        prompt += f"\n附加检查：\n"
        for penalty_check in criteria["penalty_checks"]:
            if penalty_check == "hallucination":
                prompt += (
                    f"  - 幻觉检查：如果模型回答中包含图片中一定无法获取的信息（如患者主观感受、时间长度等），扣1分\n"
                )
            # Can add more penalty checks here in the future

    # Add bonus checks (for future extensibility)
    if criteria.get("bonus_checks"):
        for bonus_check in criteria["bonus_checks"]:
            # Can add bonus check logic here in the future
            pass

    prompt += f"""
**请按以下JSON格式返回评分结果**：
{{
  "dimension_scores": {{
"""

    for i, dim in enumerate(dimensions):
        prompt += f'    "{dim}": 分数'
        if i < len(dimensions) - 1:
            prompt += ","
        prompt += "\n"

    prompt += f"""  }},"""

    # Add penalty checks to output format
    if criteria.get("penalty_checks"):
        for i, penalty_check in enumerate(criteria["penalty_checks"]):
            if penalty_check == "hallucination":
                prompt += f'\n  "hallucination_penalty": 0或-1'
                if i < len(criteria["penalty_checks"]) - 1:
                    prompt += ","
                prompt += "\n"
            # Can add more penalty output formats here in the future

    # Add bonus checks to output format (for future extensibility)
    if criteria.get("bonus_checks"):
        for bonus_check in criteria["bonus_checks"]:
            # Can add bonus output formats here in the future
            pass

    prompt += f"""
}}

注意： 
1. 请仔细比较每个维度模型预测与标准答案的内容一致性
2. 模型预测可能有多种格式，请忽略格式，只关注其内容
3. 从模型预测文本中提取各维度的相关信息进行比较，对于提取不到的视为null
"""

    return prompt


def extract_json_from_response(response_text):
    """
    Extract JSON from AI judge response that may contain additional text or formatting
    Handles various formats like ```json, ```, {}, or plain JSON
    """
    import re

    if not response_text or not isinstance(response_text, str):
        return None

    response_text = response_text.strip()

    # Strategy 1: Try direct JSON parsing first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract JSON from markdown code blocks
    # Pattern for ```json ... ``` or ``` ... ```
    code_block_patterns = [r"```json\s*\n(.*?)\n```", r"```\s*\n(.*?)\n```", r"```json(.*?)```", r"```(.*?)```"]

    for pattern in code_block_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    # Strategy 3: Find JSON object by looking for { ... } patterns
    # Find the first complete JSON object
    json_patterns = [
        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",  # Simple nested JSON
        r"\{.*?\}(?=\s*(?:\n|$))",  # JSON ending at line break
        r"\{.*\}",  # Any JSON-like structure
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    # Strategy 4: Try to extract JSON from mixed content
    # Look for lines that might contain JSON
    lines = response_text.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    # Strategy 5: Try to find and extract the largest JSON-like substring
    start_idx = response_text.find("{")
    if start_idx != -1:
        # Find the matching closing brace
        brace_count = 0
        for i, char in enumerate(response_text[start_idx:], start_idx):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    try:
                        return json.loads(response_text[start_idx : i + 1])
                    except json.JSONDecodeError:
                        break

    return None


def med_chat_auxeval(model, line):
    """Auxiliary evaluation function for medical chat VQA"""
    try:
        # Extract answer and prediction
        answer_text = str(line["answer"])
        prediction_text = str(line["prediction"])

        # Determine image type from answer json string
        answer_dict = json.loads(answer_text)
        image_type = answer_dict.get("图片类型", "")

        if not image_type or image_type not in eval_criteria:
            return {"score": 0.0, "log": f"Unknown image type: {image_type}", "raw_score": 0.0}

        # Build judge prompt
        judge_prompt = build_judge_prompt(image_type, answer_text, prediction_text)
        if not judge_prompt:
            return {"score": 0.0, "log": f"Failed to build judge prompt for {image_type}", "raw_score": 0.0}

        # Get judge response
        msgs = [{"type": "text", "value": judge_prompt}]
        response = model.generate(msgs)

        # Parse judge response using robust JSON extraction
        judge_result = extract_json_from_response(response)
        if judge_result:
            dimension_scores = judge_result.get("dimension_scores", {})

            # Calculate penalty scores based on criteria configuration
            penalty_score = 0
            criteria = eval_criteria[image_type]
            if criteria.get("penalty_checks"):
                for penalty_check in criteria["penalty_checks"]:
                    if penalty_check == "hallucination":
                        penalty_score += judge_result.get("hallucination_penalty", 0)
                    # Can add more penalty types here in the future

            # Calculate bonus scores (for future extensibility)
            bonus_score = 0
            if criteria.get("bonus_checks"):
                for bonus_check in criteria["bonus_checks"]:
                    # Can add bonus scoring logic here in the future
                    pass

            # Calculate total raw score
            raw_score = sum(dimension_scores.values()) + penalty_score + bonus_score

            # Normalize to 100 points using dynamic max score calculation
            max_possible_score = get_max_score(image_type)
            if max_possible_score > 0:
                normalized_score = max(0, (raw_score / max_possible_score) * 100)
            else:
                normalized_score = 0.0

            return {
                "score": normalized_score,
                "log": f"模型预测：{prediction_text}\n标准答案：{answer_text}",
                "raw_score": raw_score,
                "dimension_scores": dimension_scores,
                "penalty_score": penalty_score,
                "bonus_score": bonus_score,
                "hallucination_penalty": judge_result.get(
                    "hallucination_penalty", 0
                ),  # Keep for backward compatibility
            }
        else:
            return {"score": 0.0, "log": f"Failed to parse judge response: {response}", "raw_score": 0.0}

    except Exception as e:
        return {"score": 0.0, "log": f"Evaluation error: {str(e)}", "raw_score": 0.0}


class MedChatVQADataset:
    def load_data(self, dataset):
        data_path = os.path.join("/mnt/workspace/offline/shared_benchmarks", f"{dataset}.tsv")
        return load(data_path)

    def build_prompt(self, line):
        msgs = ImageBaseDataset.build_prompt(self, line)
        msgs[-1]["value"] = long_prompt  # long_prompt, sft_short_prompt
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
            assert model.working(), "Medical Chat VQA evaluation requires a working judge model\n" + DEBUG_MESSAGE

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
                    med_chat_auxeval,
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

        # Group by image type if available
        if "answer" in data:
            image_type_scores = {}
            for i in range(len(data)):
                try:
                    answer_text = str(data["answer"].iloc[i])
                    answer_dict = json.loads(answer_text)
                    image_type = answer_dict.get("图片类型", "")
                    if image_type not in [
                        "皮损图片",
                        "药盒",
                        "处方",
                        "血液检验报告",
                        "影像检查报告",
                        "影像片子",
                        "超声检查报告",
                        "体检报告",
                        "血液照片",
                        "病历",
                    ]:
                        image_type = "其他"
                    if image_type and image_type not in image_type_scores:
                        image_type_scores[image_type] = []
                    if image_type:
                        image_type_scores[image_type].append(data["score"].iloc[i])
                except:
                    continue

            # Calculate average score for each image type
            for img_type, scores in image_type_scores.items():
                if len(scores) > 0:
                    results[img_type] = np.mean(scores)

        # Convert to DataFrame and save
        ret = d2df(results).round(2)
        score_pth = storage.replace(".xlsx", "_score.csv")
        dump(ret, score_pth)

        return ret


class JDH_MEDCHAT_vqa_val_gpt4_1(MedChatVQADataset):
    pass


class JDH_MEDCHAT_vqa_test_gpt4_1(MedChatVQADataset):
    pass


class JDH_MEDCHAT_vqa_val_doubao(MedChatVQADataset):
    pass


class JDH_MEDCHAT_vqa_test_doubao(MedChatVQADataset):
    pass


# Override CustomVQADataset methods
CustomVQADataset.load_data = MedChatVQADataset.load_data
CustomVQADataset.build_prompt = MedChatVQADataset.build_prompt
CustomVQADataset.evaluate = MedChatVQADataset.evaluate
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
