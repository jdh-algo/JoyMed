#!/usr/bin/env python3
"""
VLMEvalKit evaluation script for JDH_ALLDOC_qa dataset.
Evaluates model's ability to extract specific entry values from all medicaldocuments.
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


def get_llm_judgment(pred_value, label_value, question, model):
    """Use LLM to judge if two values are equivalent."""
    try:
        prompt = f"""请判断以下两个答案是否在主要内容上一致：

问题：{question}
预测答案：{pred_value}
标准答案：{label_value}

如果答案在主要内容上完全一致，请输出1.0。
如果答案在主要内容上不一致，请输出0.0。
如果答案在主要内容上部分一致，请输出0-1之间的分数。

你可以推理思考，请将最终评分输出在<score>标签中。
"""
        response = model.generate(prompt)
        
        # First try to extract score from <score> tags
        score_pattern = r"<score>\s*(\d+(?:\.\d+)?)\s*</score>"
        score_matches = re.findall(score_pattern, response, re.IGNORECASE)
        
        if score_matches:
            score = float(score_matches[0])
            # Ensure score is in 0-1 range, then convert to 0-100
            score = max(0.0, min(1.0, score)) * 100
            return score, response
        
        # If no <score> tags found, look for numeric values in the response
        # Try to find the last numeric value (likely the final score)
        numeric_matches = re.findall(r"\d+(?:\.\d+)?", response)
        
        if numeric_matches:
            # Get the last numeric value as it's likely the final score
            last_score = float(numeric_matches[-1])
            
            # If the score is already in 0-100 range, use it as is
            if 0 <= last_score <= 100:
                return last_score, response
            # If the score is in 0-1 range, convert to 0-100
            elif 0 <= last_score <= 1:
                return last_score * 100, response
            # If score is outside expected ranges, default to 0
            else:
                return 0.0, response
        
        # If no numeric values found, return 0
        return 0.0, response
        
    except Exception as e:
        # Return 0 score with error information in response
        error_response = f"Error in score extraction: {str(e)}"
        return 0.0, error_response


def alldoc_qa_auxeval(model, line):
    """Auxiliary evaluation function for entry QA."""
    try:
        # Extract fields
        question = str(line.get("question", ""))
        answer = str(line.get("answer", ""))
        prediction = str(line.get("prediction", ""))

        assert model and model.working()

        score, log = get_llm_judgment(prediction, answer, question, model)

        # Create detailed log
        log_message = f"问题：{question}\n预测：{prediction}\n标准答案：{answer}\n评判结果：{log}\n得分：{score}"

        return {
            "score": score,
            "log": log_message,
        }

    except Exception as e:
        return {
            "score": 0.0,
            "log": f"Evaluation error: {str(e)}",
        }


class JDH_ALLDOC_qa:
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
                    alldoc_qa_auxeval,
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

            # Add class_name and department fields from original data
            data["class_name"] = [data.iloc[i].get("doc_type", "unknown") for i in range(lt)]
            data["department"] = [data.iloc[i].get("dept_name", "unknown") for i in range(lt)]

            # Save detailed results
            dump(data, storage)

        # Load results and calculate statistics
        data = load(storage)

        # Overall statistics
        total_samples = len(data)
        overall_score = data["score"].mean() if total_samples > 0 else 0.0

        # Class name statistics
        class_name_scores = {}
        class_name_counts = {}

        for class_name in set(data.get("class_name", ["unknown"] * total_samples)):
            mask = [cn == class_name for cn in data.get("class_name", ["unknown"] * total_samples)]
            class_samples = sum(mask)
            if class_samples > 0:
                class_scores = [score for score, m in zip(data["score"], mask) if m]
                class_name_counts[class_name] = class_samples
                class_name_scores[class_name] = np.mean(class_scores)

        # Department statistics
        department_scores = {}
        department_counts = {}

        for department in set(data.get("department", ["unknown"] * total_samples)):
            mask = [dept == department for dept in data.get("department", ["unknown"] * total_samples)]
            dept_samples = sum(mask)
            if dept_samples > 0:
                dept_scores = [score for score, m in zip(data["score"], mask) if m]
                department_counts[department] = dept_samples
                department_scores[department] = np.mean(dept_scores)

        # Find samples with lowest scores (most difficult)
        low_score_samples = [(i, data.iloc[i]["score"]) for i in range(total_samples)]
        difficult_samples = sorted(low_score_samples, key=lambda x: x[1])[:10]

        # Format results
        results = {
            "Overall": round(overall_score, 2),
            "Total_Samples": total_samples,
        }

        # Add class name results
        for class_name in sorted(class_name_counts.keys()):
            results[f"{class_name}_score"] = round(class_name_scores[class_name], 2)
            results[f"{class_name}_count"] = class_name_counts[class_name]

        # Add department results
        for department in sorted(department_counts.keys()):
            results[f"{department}_score"] = round(department_scores[department], 2)
            results[f"{department}_count"] = department_counts[department]

        # Add difficult samples information
        results["Most_Difficult_Samples"] = ", ".join([f"Sample_{i}({score:.1f})" for i, score in difficult_samples])

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
                f.write(f"  Class: {row.get('class_name', 'unknown')}\n")
                f.write(f"  Department: {row.get('department', 'unknown')}\n")
                f.write(f"  Question: {row.get('question', '')}\n")
                f.write(f"  Answer: {row.get('answer', '')}\n")
                f.write(f"  Prediction: {row.get('prediction', '')}\n")
                f.write(f"  Score: {row['score']}\n")
                f.write("-" * 50 + "\n")

        return ret


class GMD_simpleQA(JDH_ALLDOC_qa):
    pass


class GMD_complexQA(JDH_ALLDOC_qa):
    pass


# Override the default dataset class methods
CustomVQADataset.load_data = JDH_ALLDOC_qa.load_data
CustomVQADataset.build_prompt = JDH_ALLDOC_qa.build_prompt
CustomVQADataset.evaluate = JDH_ALLDOC_qa.evaluate
