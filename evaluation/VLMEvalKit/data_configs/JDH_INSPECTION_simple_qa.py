#!/usr/bin/env python3
"""
VLMEvalKit evaluation script for JDH_INSPECTION_simple_qa dataset.
Evaluates model's ability to extract specific entry values from inspection reports.
"""

import os
import re
import json
import numpy as np
from pathlib import Path
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.image_vqa import CustomVQADataset
from vlmeval.smp import load, dump, d2df, LMUDataRoot
from vlmeval.dataset.utils import build_judge, DEBUG_MESSAGE
from vlmeval.utils import track_progress_rich


def normalize_value(value):
    """Normalize value for comparison."""
    if not value or not isinstance(value, str):
        return ""

    # Convert to string and strip whitespace
    result = str(value).strip()

    # Remove extra whitespace
    result = re.sub(r"\s+", " ", result)

    # Normalize units
    # Handle different unit representations
    result = re.sub(r"μmol/L", "umol/L", result)
    result = re.sub(r"×10\^(\d+)", r"*10^\1", result)
    result = re.sub(r"×10(\d+)", r"*10^\1", result)
    result = re.sub(r"\*10\^(\d+)", r"E\1", result)  # Scientific notation

    # Normalize decimal points
    result = re.sub(r"\.0+$", "", result)  # Remove trailing zeros after decimal
    result = re.sub(r"\.0+\s", " ", result)  # Remove trailing zeros before space

    # Normalize reference range separators
    result = re.sub(r"(\d+)\s*[-~～–—]\s*(\d+)", r"\1-\2", result)
    result = re.sub(r"(\d+)--(\d+)", r"\1-\2", result)  # Double dash

    # Handle Chinese characters for reference ranges
    result = re.sub(r"参考范围[：:]", "", result)
    result = re.sub(r"参考区间[：:]", "", result)
    result = re.sub(r"正常值范围[：:]", "", result)

    # Remove common punctuation at the end
    result = re.sub(r"[。，、；：]$", "", result)

    return result.lower()


def extract_numerical_value_and_unit(text):
    """Extract numerical value and unit from text."""
    if not text or not isinstance(text, str):
        return None, None

    text = text.strip()

    # Pattern to match number with optional unit
    patterns = [
        r"^([\d\.]+)\s*([a-zA-Z/%μ]+.*)$",  # Number followed by unit
        r"^([\d\.]+)\s*\*\s*10\^([\d]+)\s*(.*)$",  # Scientific notation
        r"^([\d\.]+)E([\d]+)\s*(.*)$",  # E notation
        r"^([\d\.]+)$",  # Just number
    ]

    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            if len(match.groups()) == 1:
                return float(match.group(1)), None
            elif len(match.groups()) == 2:
                try:
                    return float(match.group(1)), match.group(2).strip()
                except:
                    return match.group(1), match.group(2).strip()
            elif len(match.groups()) == 3:  # Scientific notation
                try:
                    base = float(match.group(1))
                    exp = int(match.group(2))
                    value = base * (10**exp)
                    unit = match.group(3).strip() if match.group(3) else None
                    return value, unit
                except:
                    return text, None

    return text, None


def is_reference_range(text):
    """Check if text represents a reference range."""
    if not text or not isinstance(text, str):
        return False

    # Patterns for reference ranges
    patterns = [
        r"\d+\.?\d*\s*[-~～–—]\s*\d+\.?\d*",  # Number range
        r"[<>≤≥]\s*\d+\.?\d*",  # Less than/greater than
        r"阴性|阳性",  # Qualitative results
    ]

    for pattern in patterns:
        if re.search(pattern, text):
            return True

    return False


def compare_values(pred_value, label_value, field_type="result"):
    """Compare predicted and labeled values."""
    # Normalize both values
    pred_norm = normalize_value(pred_value)
    label_norm = normalize_value(label_value)

    # Exact match after normalization
    if pred_norm == label_norm:
        return 1.0

    # For empty predictions
    if not pred_norm:
        return 0.0

    # For reference ranges, check if the range values match
    if field_type == "reference" and is_reference_range(label_value):
        # Extract range values
        pred_range = re.findall(r"[\d\.]+", pred_norm)
        label_range = re.findall(r"[\d\.]+", label_norm)

        if pred_range and label_range:
            if len(pred_range) == len(label_range):
                try:
                    pred_nums = [float(x) for x in pred_range]
                    label_nums = [float(x) for x in label_range]

                    # Check if all numbers match (with small tolerance)
                    if all(abs(p - l) < 0.01 for p, l in zip(pred_nums, label_nums)):
                        return 1.0
                except:
                    pass

    # For result values, try numerical comparison
    if field_type == "result":
        pred_num, pred_unit = extract_numerical_value_and_unit(pred_norm)
        label_num, label_unit = extract_numerical_value_and_unit(label_norm)

        # If both are numbers
        if isinstance(pred_num, (int, float)) and isinstance(label_num, (int, float)):
            # Check if numbers are close enough (relative tolerance)
            rel_diff = abs(pred_num - label_num) / max(abs(label_num), 1e-10)
            if rel_diff < 0.01:  # 1% tolerance
                # Check unit match
                if pred_unit == label_unit or (not pred_unit and not label_unit):
                    return 1.0
                # Partial score if number matches but unit doesn't
                return 0.5

    # Fuzzy string matching for partial credit
    # Calculate character-level similarity
    if len(pred_norm) > 0 and len(label_norm) > 0:
        # Simple character overlap ratio
        common_chars = set(pred_norm) & set(label_norm)
        similarity = len(common_chars) / max(len(set(pred_norm)), len(set(label_norm)))

        # If very similar (>80% character overlap), give partial credit
        if similarity > 0.8:
            return 0.5

    return 0.0


def get_llm_judgment(pred_value, label_value, field_type, question, model):
    """Use LLM to judge if two values are equivalent."""
    try:
        prompt = f"""请判断以下两个答案是否等价或基本正确：

问题：{question}
预测答案：{pred_value}
标准答案：{label_value}

请考虑：
1. 数值的等价性（如0.5和0.50是等价的）
2. 单位的不同表示方式（如umol/L和μmol/L是等价的）
3. 范围表示的不同方式（如10-20和10~20是等价的）
4. 科学记数法的不同表示

如果预测答案基本正确（数值正确但格式略有差异），请回答"是"。
如果预测答案错误或偏差较大，请回答"否"。

请只回答"是"或"否"。"""

        response = model.generate(prompt)
        response = response.strip().lower()

        return "是" in response or "yes" in response or "对" in response or "正确" in response
    except Exception:
        return False


def simple_qa_auxeval(model, line):
    """Auxiliary evaluation function for entry QA."""
    try:
        # Extract fields
        question = str(line.get("question", ""))
        answer = str(line.get("answer", ""))
        prediction = str(line.get("prediction", ""))
        field_type = str(line.get("field_type", "result"))
        entryname = str(line.get("entryname", ""))

        # Clean prediction (remove common prefixes/suffixes)
        prediction_clean = prediction.strip()

        # Remove common response patterns
        patterns_to_remove = [
            r"^答案[是：:]\s*",
            r"^回答[是：:]\s*",
            r"^该项?的?测量结果[是：:]\s*",
            r"^该项?的?参考区间[是：:]\s*",
            r"^根据.*?[，,]\s*",
            r".*?[的地]测量结果[是为：:]\s*",
            r".*?[的地]参考区间[是为：:]\s*",
        ]

        for pattern in patterns_to_remove:
            prediction_clean = re.sub(pattern, "", prediction_clean, flags=re.IGNORECASE)

        # Basic comparison
        score = compare_values(prediction_clean, answer, field_type)

        # If not perfect match and model available, try LLM judgment
        if score < 1.0 and model and model.working():
            llm_match = get_llm_judgment(prediction_clean, answer, field_type, question, model)
            if llm_match:
                score = 1.0

        # Create detailed log
        log_message = f"问题：{question}\n预测：{prediction}\n标准答案：{answer}\n得分：{score}"

        return {
            "score": score * 100,  # Convert to percentage
            "log": log_message,
            "raw_score": score,
            "field_type": field_type,
            "entryname": entryname,
            "exact_match": score == 1.0,
            "prediction_clean": prediction_clean,
            "answer_clean": answer,
        }

    except Exception as e:
        return {
            "score": 0.0,
            "log": f"Evaluation error: {str(e)}",
            "raw_score": 0.0,
            "field_type": field_type if "field_type" in locals() else "unknown",
            "entryname": entryname if "entryname" in locals() else "unknown",
            "exact_match": False,
            "prediction_clean": "",
            "answer_clean": "",
        }


class JDH_INSPECTION_simple_qa:
    def load_data(self, dataset):
        """Load custom dataset."""
        data_path = os.path.join(LMUDataRoot(), f"{dataset}.tsv")
        return load(data_path)

    def build_prompt(self, line):
        """Build prompt for the model."""
        msgs = ImageBaseDataset.build_prompt(self, line)

        # Add system prompt at the beginning
        # Use custom system prompt if provided in the line, otherwise use default
        if isinstance(line, int):
            line_data = self.data.iloc[line]
        else:
            line_data = line

        if "system_prompt" in line_data and line_data["system_prompt"] and str(line_data["system_prompt"]).strip():
            system_prompt = str(line_data["system_prompt"]).strip()
        else:
            system_prompt = "你是医学文档解析专家"

        msgs.insert(0, dict(type="text", value=system_prompt))
        
        is_reasoning = os.environ.get("REASONING", "False") == "True"
        if is_reasoning:
            msgs[-1]["value"] += "/think"

        return msgs

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """Evaluate entry QA dataset"""
        import os.path as osp

        # Setup file paths
        suffix = eval_file.split(".")[-1]
        model_name = judge_kwargs.get("model", "gpt-4o")
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

            # Build judge model (optional for this task)
            model = None
            if judge_kwargs:
                try:
                    model = build_judge(**judge_kwargs)
                    if not model.working():
                        model = None
                        print("Judge model not working, using rule-based evaluation only")
                except:
                    model = None
                    print("Could not build judge model, using rule-based evaluation only")

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
                    simple_qa_auxeval,
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
            data["field_type"] = [ans[i].get("field_type", "unknown") for i in range(lt)]
            data["entryname"] = [ans[i].get("entryname", "") for i in range(lt)]
            data["exact_match"] = [ans[i].get("exact_match", False) for i in range(lt)]
            data["prediction_clean"] = [ans[i].get("prediction_clean", "") for i in range(lt)]
            data["answer_clean"] = [ans[i].get("answer_clean", "") for i in range(lt)]

            # Save detailed results
            dump(data, storage)

        # Load results and calculate statistics
        data = load(storage)

        # Overall statistics
        total_samples = len(data)
        overall_score = data["score"].mean() if total_samples > 0 else 0.0
        exact_match_rate = (
            sum(data.get("exact_match", [False] * total_samples)) / total_samples if total_samples > 0 else 0.0
        )

        # Field type statistics
        field_types = data.get("field_type", ["unknown"] * total_samples)
        field_type_scores = {}
        field_type_counts = {}
        field_type_exact_match = {}

        for field_type in set(field_types):
            mask = [ft == field_type for ft in field_types]
            field_samples = sum(mask)
            if field_samples > 0:
                field_scores = [score for score, m in zip(data["score"], mask) if m]
                field_exact = [em for em, m in zip(data.get("exact_match", [False] * total_samples), mask) if m]

                field_type_counts[field_type] = field_samples
                field_type_scores[field_type] = np.mean(field_scores)
                field_type_exact_match[field_type] = sum(field_exact) / field_samples

        # Entry name statistics (top errors)
        entry_scores = {}
        for i in range(total_samples):
            entry = data.iloc[i].get("entryname", "unknown")
            score = data.iloc[i]["score"]

            if entry not in entry_scores:
                entry_scores[entry] = []
            entry_scores[entry].append(score)

        # Calculate average score per entry
        entry_avg_scores = {entry: np.mean(scores) for entry, scores in entry_scores.items()}

        # Find entries with lowest scores (most difficult)
        difficult_entries = sorted(entry_avg_scores.items(), key=lambda x: x[1])[:10]

        # Format results
        results = {
            "Overall": round(overall_score, 2),
            "Exact_Match_Rate": round(exact_match_rate * 100, 2),
            "Total_Samples": total_samples,
        }

        # Add field type results
        for field_type in sorted(field_type_counts.keys()):
            results[f"{field_type}_score"] = round(field_type_scores[field_type], 2)
            results[f"{field_type}_exact_match"] = round(field_type_exact_match[field_type] * 100, 2)
            results[f"{field_type}_count"] = field_type_counts[field_type]

        # Add difficult entries information
        results["Most_Difficult_Entries"] = ", ".join([f"{entry}({score:.1f})" for entry, score in difficult_entries])

        # Calculate score distribution
        score_bins = [0, 50, 80, 100]
        score_distribution = np.histogram(data["score"], bins=score_bins)[0]
        results["Score_0_50"] = int(score_distribution[0])
        results["Score_50_80"] = int(score_distribution[1])
        results["Score_80_100"] = int(score_distribution[2])

        # Convert to DataFrame and save
        ret = d2df(results).round(2)
        score_pth = storage.replace(".xlsx", "_score.csv")
        dump(ret, score_pth)

        # Also save a detailed error analysis
        error_analysis_path = storage.replace(".xlsx", "_errors.txt")
        with open(error_analysis_path, "w", encoding="utf-8") as f:
            f.write("=== Error Analysis ===\n\n")

            # Find samples with score < 50
            errors = [(i, row) for i, row in data.iterrows() if row["score"] < 50]

            f.write(f"Total Errors (score < 50): {len(errors)}\n\n")

            for idx, (i, row) in enumerate(errors[:20]):  # Show first 20 errors
                f.write(f"Error {idx+1}:\n")
                f.write(f"  Entry: {row.get('entryname', 'unknown')}\n")
                f.write(f"  Field Type: {row.get('field_type', 'unknown')}\n")
                f.write(f"  Question: {row.get('question', '')}\n")
                f.write(f"  Answer: {row.get('answer_clean', row.get('answer', ''))}\n")
                f.write(f"  Prediction: {row.get('prediction_clean', row.get('prediction', ''))}\n")
                f.write(f"  Score: {row['score']}\n")
                f.write("-" * 50 + "\n")

        return ret


class LTR_simpleQA(JDH_INSPECTION_simple_qa):
    pass


# Override the default dataset class methods
CustomVQADataset.load_data = JDH_INSPECTION_simple_qa.load_data
CustomVQADataset.build_prompt = JDH_INSPECTION_simple_qa.build_prompt
CustomVQADataset.evaluate = JDH_INSPECTION_simple_qa.evaluate
