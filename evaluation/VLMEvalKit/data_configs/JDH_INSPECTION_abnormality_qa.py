import os
import re
import json
import numpy as np
import sys
from pathlib import Path
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.image_vqa import CustomVQADataset
from vlmeval.smp import load, dump, d2df, LMUDataRoot
from vlmeval.dataset.utils import build_judge, DEBUG_MESSAGE
from vlmeval.utils import track_progress_rich
import Levenshtein


def extract_json_from_text(text):
    """Extract JSON from text, handling various formats including natural language."""
    if not text or not isinstance(text, str):
        return [], False  # Return (items, parsing_failed)

    text = text.strip()

    # Remove <think> tags and their content
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()

    # If text is empty after cleaning, it's successful parsing with no results
    if not text:
        return [], False

    # First try to find JSON array in the text
    json_patterns = [
        r"\[.*\]",  # Simple array pattern
        r"\{.*\}",  # Simple object pattern
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    # Normalize field names in the parsed data
                    normalized_list = []
                    for item in parsed:
                        if isinstance(item, dict):
                            normalized_item = normalize_field_names(item)
                            normalized_list.append(normalized_item)
                    return normalized_list, False  # Success
                elif isinstance(parsed, dict):
                    normalized_item = normalize_field_names(parsed)
                    return [normalized_item], False  # Success
            except json.JSONDecodeError:
                continue

    # Try to find multiple JSON objects in the text
    # Pattern for multiple separate JSON objects
    multiple_json_pattern = r"\{[^}]*\}"
    json_matches = re.findall(multiple_json_pattern, text, re.DOTALL)
    if json_matches:
        items = []
        for match in json_matches:
            try:
                # Handle single quotes in JSON (replace with double quotes)
                match = match.replace("'", '"')
                # Handle unquoted keys by adding quotes around them
                match = re.sub(r"(\s*)(\w+)(\s*:)", r'\1"\2"\3', match)
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    normalized_item = normalize_field_names(parsed)
                    items.append(normalized_item)
            except json.JSONDecodeError:
                continue
        if items:
            return items, False  # Success

    # If no JSON found, try to parse the entire text as JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            # Normalize field names in the parsed data
            normalized_list = []
            for item in parsed:
                if isinstance(item, dict):
                    normalized_item = normalize_field_names(item)
                    normalized_list.append(normalized_item)
            return normalized_list, False  # Success
        elif isinstance(parsed, dict):
            normalized_item = normalize_field_names(parsed)
            return [normalized_item], False  # Success
    except json.JSONDecodeError:
        pass

    # If no JSON found, try to parse natural language responses
    nl_result = parse_natural_language_response(text)
    # Natural language parsing is also considered successful if it returns results
    return nl_result, len(nl_result) == 0  # Failed only if no results from NL parsing


def parse_natural_language_response(text):
    """Parse natural language responses and convert to JSON format."""
    if not text or not isinstance(text, str):
        return []

    text = text.strip()

    # Skip if text indicates no abnormalities
    if any(phrase in text for phrase in ["无异常", "正常", "无异", "该检验单无异常项", "检验结果正常"]):
        return []

    # Remove lines that end with colons (introductory lines)
    lines = text.split("\n")
    filtered_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.endswith(":") and not line.endswith("："):
            filtered_lines.append(line)

    # Reconstruct text without introductory lines
    text = "\n".join(filtered_lines)

    items = []
    processed_entries = set()  # To avoid duplicates

    # Pattern 1: Handle special cases like "阳性(R)" format first
    # Example: "梅毒甲苯胺红不加热血清试验(滴度 1:4) 阳性(R)"
    positive_pattern = r"^([^\n]+?)\s+阳性\(R\)$"
    matches = re.findall(positive_pattern, text, re.MULTILINE)
    for match in matches:
        entryname = match.strip()
        if entryname and entryname not in processed_entries:
            items.append({"entryname": entryname, "result": "阳性", "reference": "", "status": "阳性"})
            processed_entries.add(entryname)

    # Pattern 2: Format with units and reference ranges
    # Example: "平均血红蛋白浓度 32.3 g/L 参考范围：320-360 偏低"
    unit_pattern = (
        r"^([^\n]+?)\s+([\d\.]+(?:\s*[^\s]+)?)\s+(参考范围[：:]\s*[^\s]+(?:\s*[^\s]+)*?)\s+(偏高|偏低|阳性|阴性|异常)$"
    )
    matches = re.findall(unit_pattern, text, re.MULTILINE)
    for match in matches:
        entryname, result, reference, status = match
        entryname = entryname.strip()
        if entryname and entryname not in processed_entries:
            items.append(
                {
                    "entryname": entryname,
                    "result": result.strip(),
                    "reference": reference.strip(),
                    "status": status.strip(),
                }
            )
            processed_entries.add(entryname)

    # Pattern 3: Multi-line format with 【】brackets
    # Example: "【互】谷丙转氨酶 54 偏高"
    bracket_pattern = r"^【[^】]*】\s*([^\n]+?)\s+([\d\.]+(?:\s*[^\s]+)?)\s+(偏高|偏低|阳性|阴性|异常)$"
    matches = re.findall(bracket_pattern, text, re.MULTILINE)
    for match in matches:
        entryname, result, status = match
        entryname = entryname.strip()
        if entryname and entryname not in processed_entries:
            items.append({"entryname": entryname, "result": result.strip(), "reference": "", "status": status.strip()})
            processed_entries.add(entryname)

    # Pattern 3.5: Handle negative results with parentheses
    # Example: "MP-β2M  阴性(-)  偏高"
    negative_pattern = r"^([^\n]+?)\s+阴性\([^)]*\)\s+(偏高|偏低|阳性|阴性|异常)$"
    matches = re.findall(negative_pattern, text, re.MULTILINE)
    for match in matches:
        entryname, status = match
        entryname = entryname.strip()
        if entryname and entryname not in processed_entries:
            items.append({"entryname": entryname, "result": "阴性", "reference": "", "status": status.strip()})
            processed_entries.add(entryname)

    # Pattern 4: Simple format with reference range (more specific)
    # Example: "肌酐 52.8 umol/L 41--73 偏高"
    simple_ref_pattern = (
        r"^([^\n]+?)\s+([\d\.]+(?:\s*[^\s]+)?)\s+([\d\.]+(?:--|-|~)[\d\.]+(?:\s*[^\s]+)?)\s+(偏高|偏低|阳性|阴性|异常)$"
    )
    matches = re.findall(simple_ref_pattern, text, re.MULTILINE)
    for match in matches:
        entryname, result, reference, status = match
        entryname = entryname.strip()
        if entryname and entryname not in processed_entries:
            items.append(
                {
                    "entryname": entryname,
                    "result": result.strip(),
                    "reference": reference.strip(),
                    "status": status.strip(),
                }
            )
            processed_entries.add(entryname)

    # Pattern 5: Simple abnormality indicators (most general, should be last)
    # Example: "项目名 结果 偏高/偏低"
    # But exclude lines that start with 【】brackets (already handled by Pattern 3)
    simple_abnormality_pattern = r"^(?!【[^】]*】)([^\n]+?)\s+([\d\.]+(?:\s*[^\s]+)?)\s+(偏高|偏低|阳性|阴性|异常)$"
    matches = re.findall(simple_abnormality_pattern, text, re.MULTILINE)
    for match in matches:
        entryname, result, status = match
        entryname = entryname.strip()
        if entryname and entryname not in processed_entries:
            items.append({"entryname": entryname, "result": result.strip(), "reference": "", "status": status.strip()})
            processed_entries.add(entryname)

    # Pattern 6: Handle formats with arrows and symbols
    # Example: "红细胞分布宽度 35.60 ↓"
    arrow_pattern = r"^([^\n]+?)\s+([\d\.]+(?:\s*[^\s]+)?)\s*([↑↓])$"
    matches = re.findall(arrow_pattern, text, re.MULTILINE)
    for match in matches:
        entryname, result, arrow = match
        entryname = entryname.strip()
        if entryname and entryname not in processed_entries:
            status = "偏高" if arrow == "↑" else "偏低"
            items.append({"entryname": entryname, "result": result.strip(), "reference": "", "status": status})
            processed_entries.add(entryname)

    # Pattern 7: Handle formats with comparison symbols
    # Example: "血清游离甲状腺素11.46＜下限"
    comparison_pattern = r"^([^\n]+?)([\d\.]+(?:\s*[^\s]+)?)\s*([＜＞])\s*(下限|上限)$"
    matches = re.findall(comparison_pattern, text, re.MULTILINE)
    for match in matches:
        entryname, result, symbol, limit = match
        entryname = entryname.strip()
        if entryname and entryname not in processed_entries:
            status = "偏低" if symbol == "＜" else "偏高"
            items.append({"entryname": entryname, "result": result.strip(), "reference": f"{limit}", "status": status})
            processed_entries.add(entryname)

    return items


def normalize_field_names(item):
    """Normalize field names to standard English names."""
    if not isinstance(item, dict):
        return item

    # Field name mapping from Chinese to English
    field_mapping = {
        # Entry name variations
        "检验项目名称": "entryname",
        "检验项目": "entryname",
        "项目名称": "entryname",
        "名称": "entryname",
        "项目": "entryname",
        # Result variations
        "结果": "result",
        "检测结果": "result",
        "值": "result",
        "检测值": "result",
        # Reference variations
        "参考范围": "reference",
        "参考值": "reference",
        "正常值范围": "reference",
        "参考": "reference",
        "正常范围": "reference",
        "参考区间": "reference",
        # Status variations
        "异常状态": "status",
        "状态": "status",
        "异常": "status",
        "异常类型": "status",
    }

    normalized_item = {}
    for key, value in item.items():
        # Normalize the key
        normalized_key = field_mapping.get(key, key)
        normalized_item[normalized_key] = value

    return normalized_item


def normalize_text(text):
    """Normalize text for comparison."""
    if not text or not isinstance(text, str):
        return ""

    # Remove extra whitespace and normalize
    text = re.sub(r"\s+", " ", text.strip())
    # Remove common punctuation that doesn't affect meaning
    text = re.sub(r"[，。、；：！？]", "", text)
    return text.lower()


def is_meaningful_field(text):
    """Check if a field value is meaningful."""
    if not text or not isinstance(text, str):
        return False

    text = text.strip()
    if not text:
        return False

    # Check for meaningful content
    has_content = bool(re.search(r"[\w\u4e00-\u9fff]", text))
    return has_content


def standardize_value(value):
    """Standardize value for comparison."""
    if not isinstance(value, str):
        return str(value) if value is not None else ""

    # Remove extra whitespace and normalize
    result = re.sub(r"\s+", " ", value.strip())
    # Remove common punctuation that doesn't affect meaning
    result = re.sub(r"[，。、；：！？]", "", result)

    # Handle common variations in medical test results
    # Remove parentheses content like "(R)", "(-)", "(+)" etc.
    result = re.sub(r"\([^)]*\)", "", result)

    # Normalize units and spacing
    result = re.sub(r"\s*([μμ]mol/L|mmol/L|g/L|mg/L|U/L|%|umol/L)\s*", r"\1", result)

    # Remove extra spaces around numbers
    result = re.sub(r"(\d+)\s*([~-])\s*(\d+)", r"\1\2\3", result)

    # Handle double dashes in reference ranges
    result = re.sub(r"(\d+)--(\d+)", r"\1-\2", result)

    # Normalize common medical terms
    result = re.sub(r"正常值范围[：:]", "参考范围：", result)
    result = re.sub(r"参考区间[：:]", "参考范围：", result)

    result = re.sub(r"[↑↓*]", "", result)

    # Final whitespace cleanup
    result = re.sub(r"\s+", "", result)

    return result.lower()


def get_llm_judgment(pred_value, label_value, field_name, model):
    """Use LLM to judge if two values are equivalent."""
    try:
        prompt = f"""请判断以下两个{field_name}是否等价，如果是数值类型的，请忽略单位：

预测值：{pred_value}
标准值：{label_value}

请只回答"是"或"否"。"""

        response = model.generate(prompt)
        response = response.strip().lower()

        return "是" in response or "yes" in response or "true" in response
    except Exception:
        return False


def find_best_matches_levenshtein(pred_lookup, label_lookup):
    """Find optimal one-to-one matches between predicted and labeled items using Levenshtein distance.

    This function computes all pairwise similarity scores and finds the optimal matching
    that maximizes the total similarity score. Each predicted item is matched to at most
    one labeled item and vice versa.

    Args:
        pred_lookup: Dictionary mapping standardized entrynames to predicted items
        label_lookup: Dictionary mapping standardized entrynames to labeled items

    Returns:
        Dictionary mapping predicted entrynames to their best matching labeled entrynames
    """
    if not pred_lookup or not label_lookup:
        return {}

    pred_keys = list(pred_lookup.keys())
    label_keys = list(label_lookup.keys())

    # Compute all pairwise similarity scores
    # scores[i][j] = similarity between pred_keys[i] and label_keys[j]
    scores = []
    for pred_key in pred_keys:
        row = []
        for label_key in label_keys:
            # Check for exact match first (for efficiency)
            if pred_key == label_key:
                score = 1.0
            else:
                # Calculate Levenshtein ratio (0-1, where 1 is perfect match)
                score = Levenshtein.ratio(pred_key, label_key)
            row.append(score)
        scores.append(row)

    # Find optimal matching using greedy approach
    # (For a more optimal solution, we could use the Hungarian algorithm, but greedy works well for this use case)
    matches = {}
    used_labels = set()

    # Create a list of all (score, pred_idx, label_idx) tuples
    all_pairs = []
    for i, pred_key in enumerate(pred_keys):
        for j, label_key in enumerate(label_keys):
            all_pairs.append((scores[i][j], i, j))

    # Sort by score in descending order (highest similarity first)
    all_pairs.sort(reverse=True, key=lambda x: x[0])

    # Greedily assign matches
    used_preds = set()
    for score, pred_idx, label_idx in all_pairs:
        if pred_idx not in used_preds and label_idx not in used_labels:
            pred_key = pred_keys[pred_idx]
            label_key = label_keys[label_idx]
            matches[pred_key] = label_key
            used_preds.add(pred_idx)
            used_labels.add(label_idx)

            # Stop when we've matched all possible items
            if len(matches) == min(len(pred_keys), len(label_keys)):
                break

    return matches


def compare_abnormality_items(pred_items, label_items, model):
    """Compare predicted and labeled abnormality items and return detailed metrics."""
    if not pred_items or not label_items:
        return {
            "entryname": {"matches": [], "pred_count": 0, "label_count": 0},
            "result": {"matches": [], "pred_count": 0, "label_count": 0},
            "reference": {"matches": [], "pred_count": 0, "label_count": 0},
            "status": {"matches": [], "pred_count": 0, "label_count": 0},
        }

    # Ensure both are lists
    if isinstance(pred_items, dict):
        pred_items = [pred_items]
    if isinstance(label_items, dict):
        label_items = [label_items]

    # Create lookup dictionaries for efficient matching
    pred_lookup = {}
    for item in pred_items:
        entryname = standardize_value(item.get("entryname", ""))
        if entryname and is_meaningful_field(entryname):
            pred_lookup[entryname] = item

    label_lookup = {}
    for item in label_items:
        entryname = standardize_value(item.get("entryname", ""))
        if entryname and is_meaningful_field(entryname):
            label_lookup[entryname] = item

    # Initialize results structure
    results = {
        "entryname": {"matches": [], "pred_count": len(pred_lookup), "label_count": len(label_lookup)},
        "result": {"matches": [], "pred_count": len(pred_lookup), "label_count": len(label_lookup)},
        "reference": {"matches": [], "pred_count": len(pred_lookup), "label_count": len(label_lookup)},
        "status": {"matches": [], "pred_count": len(pred_lookup), "label_count": len(label_lookup)},
    }

    # Find best matches using Levenshtein distance
    entryname_matches = find_best_matches_levenshtein(pred_lookup, label_lookup)

    # Process matched pairs
    for pred_entryname, label_entryname in entryname_matches.items():
        pred_item = pred_lookup[pred_entryname]
        label_item = label_lookup[label_entryname]

        # Compare each field
        for field in ["entryname", "result", "reference", "status"]:
            pred_val = standardize_value(pred_item.get(field, ""))
            label_val = standardize_value(label_item.get(field, ""))

            # Check if both values are meaningful
            pred_meaningful = is_meaningful_field(pred_val)
            label_meaningful = is_meaningful_field(label_val)

            if pred_meaningful != label_meaningful:
                is_match = False
            elif not pred_meaningful and not label_meaningful:
                is_match = True
            else:
                # Exact match after standardization
                is_match = pred_val == label_val

                # If not match and we have a model, try LLM judgment
                if not is_match and model:
                    original_pred_val = pred_item.get(field, "")
                    original_label_val = label_item.get(field, "")
                    is_match = get_llm_judgment(original_pred_val, original_label_val, field, model)

            results[field]["matches"].append(is_match)

    return results


def abnormality_auxeval(model, line, enable_debug_columns=True):
    """Auxiliary evaluation function for abnormality detection."""
    # line is already a pandas Series from data.iloc[i], no need for tuple unpacking

    try:
        # Extract answer and prediction
        answer_text = str(line["answer"])
        prediction_text = str(line["prediction"])

        # Parse prediction and answer
        pred_items, pred_failed = extract_json_from_text(prediction_text)
        label_items, label_failed = extract_json_from_text(answer_text)

        # Check if both prediction and answer are empty (perfect match case)
        pred_empty = (
            (len(pred_items) == 0)
            or (all(not entry.get("status") for entry in pred_items))
            or (all(entry.get("status", "").strip().startswith("正常") for entry in pred_items))
        )
        label_empty = (
            (len(label_items) == 0)
            or (all(not entry.get("status") for entry in label_items))
            or (all(entry.get("status", "").strip().startswith("正常") for entry in label_items))
        )
        both_empty = pred_empty and label_empty

        # Check if parsing failed for either prediction or answer
        parsing_failed = pred_failed or label_failed

        # Generate debug columns
        if enable_debug_columns:
            pred_json_debug = json.dumps(pred_items, ensure_ascii=False, separators=(",", ":")) if pred_items else "[]"
            label_json_debug = (
                json.dumps(label_items, ensure_ascii=False, separators=(",", ":")) if label_items else "[]"
            )
        else:
            pred_json_debug = ""
            label_json_debug = ""

        # If both are empty, return perfect score
        if both_empty:
            return {
                "score": 100.0,  # Perfect score
                "log": f"模型预测：{prediction_text}\n标准答案：{answer_text}\n[Both empty - perfect match]",
                "raw_score": 1.0,
                "field_metrics": {
                    "entryname": {
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                        "true_positives": 0,
                        "predicted_count": 0,
                        "labeled_count": 0,
                    },
                    "result": {
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                        "true_positives": 0,
                        "predicted_count": 0,
                        "labeled_count": 0,
                    },
                    "reference": {
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                        "true_positives": 0,
                        "predicted_count": 0,
                        "labeled_count": 0,
                    },
                    "status": {
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                        "true_positives": 0,
                        "predicted_count": 0,
                        "labeled_count": 0,
                    },
                },
                "individual_field_results": {"entryname": [], "result": [], "reference": [], "status": []},
                "parsing_failed": False,
                "predicted_items": 0,
                "labeled_items": 0,
                "pred_json_debug": pred_json_debug,
                "label_json_debug": label_json_debug,
            }

        # Compare items
        field_results = compare_abnormality_items(pred_items, label_items, model)

        # Calculate metrics for this sample
        field_metrics = {}
        individual_field_results = {}

        for field in ["entryname", "result", "reference", "status"]:
            field_data = field_results.get(field, {"matches": [], "pred_count": 0, "label_count": 0})
            matches = field_data["matches"]
            pred_count = field_data["pred_count"]
            label_count = field_data["label_count"]

            individual_field_results[field] = matches

            if matches:
                # Calculate true positives (correct matches)
                true_positives = sum(matches)

                # Calculate precision: TP / (TP + FP) = TP / predicted_count
                precision = true_positives / pred_count if pred_count > 0 else 0.0

                # Calculate recall: TP / (TP + FN) = TP / label_count
                recall = true_positives / label_count if label_count > 0 else 0.0

                # Calculate F1 score
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                field_metrics[field] = {
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "true_positives": true_positives,
                    "predicted_count": pred_count,
                    "labeled_count": label_count,
                }
            else:
                field_metrics[field] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "true_positives": 0,
                    "predicted_count": pred_count,
                    "labeled_count": label_count,
                }

        # Calculate overall score (average F1 across fields)
        overall_f1 = np.mean([m["f1"] for m in field_metrics.values()])

        return {
            "score": overall_f1 * 100,  # Convert to percentage
            "log": f"模型预测：{prediction_text}\n标准答案：{answer_text}",
            "raw_score": overall_f1,
            "field_metrics": field_metrics,
            "individual_field_results": individual_field_results,
            "parsing_failed": parsing_failed,
            "predicted_items": len(pred_items),
            "labeled_items": len(label_items),
            "pred_json_debug": pred_json_debug,
            "label_json_debug": label_json_debug,
        }

    except Exception as e:
        return {
            "score": 0.0,
            "log": f"Evaluation error: {str(e)}",
            "raw_score": 0.0,
            "field_metrics": {},
            "individual_field_results": {},
            "parsing_failed": True,
            "predicted_items": 0,
            "labeled_items": 0,
            "pred_json_debug": "[]",
            "label_json_debug": f"Error: {str(e)}",
        }


class JDH_INSPECTION_abnormality_qa:
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
        """Evaluate abnormality detection VQA dataset"""
        import os.path as osp

        # Setup file paths
        suffix = eval_file.split(".")[-1]
        model_name = judge_kwargs.get("model", "gpt-4o")
        storage = eval_file.replace(f".{suffix}", f"_{model_name}.xlsx")
        tmp_file = eval_file.replace(f".{suffix}", f"_{model_name}.pkl")
        nproc = judge_kwargs.pop("nproc", 4)

        # Debug columns configuration
        enable_debug_columns = judge_kwargs.pop("enable_debug_columns", True)

        if not osp.exists(storage):
            # Load data
            data = load(eval_file)
            assert "answer" in data and "prediction" in data

            # Convert to string
            data["prediction"] = [str(x) for x in data["prediction"]]
            data["answer"] = [str(x) for x in data["answer"]]

            # Build judge model
            model = build_judge(**judge_kwargs)
            assert model.working(), "Abnormality VQA evaluation requires a working judge model\n" + DEBUG_MESSAGE

            # Prepare evaluation data
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line, enable_debug_columns) for line in lines]
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
                    abnormality_auxeval,
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
            data["field_metrics"] = [ans[i].get("field_metrics", {}) for i in range(lt)]
            data["individual_field_results"] = [ans[i].get("individual_field_results", {}) for i in range(lt)]
            data["parsing_failed"] = [ans[i].get("parsing_failed", True) for i in range(lt)]
            data["predicted_items"] = [ans[i].get("predicted_items", 0) for i in range(lt)]
            data["labeled_items"] = [ans[i].get("labeled_items", 0) for i in range(lt)]
            data["pred_json_debug"] = [ans[i].get("pred_json_debug", "[]") for i in range(lt)]
            data["label_json_debug"] = [ans[i].get("label_json_debug", "") for i in range(lt)]

            # Save detailed results
            dump(data, storage)

        # Load results and calculate statistics
        data = load(storage)

        # Calculate parsing failure statistics
        total_samples = len(data)
        parsing_failed_count = sum(data.get("parsing_failed", [True] * total_samples))
        parsing_success_rate = (total_samples - parsing_failed_count) / total_samples if total_samples > 0 else 0.0
        total_predicted_items = sum(data.get("predicted_items", [0] * total_samples))
        total_labeled_items = sum(data.get("labeled_items", [0] * total_samples))
        avg_predicted_items = total_predicted_items / total_samples if total_samples > 0 else 0.0
        avg_labeled_items = total_labeled_items / total_samples if total_samples > 0 else 0.0

        # Calculate field-wise metrics (MACRO AVERAGE)
        field_metrics_macro = {
            "entryname": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "result": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "reference": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "status": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        }

        # Calculate field-wise metrics (MICRO AVERAGE)
        field_metrics_micro = {
            "entryname": {"true_positives": 0, "predicted_count": 0, "labeled_count": 0},
            "result": {"true_positives": 0, "predicted_count": 0, "labeled_count": 0},
            "reference": {"true_positives": 0, "predicted_count": 0, "labeled_count": 0},
            "status": {"true_positives": 0, "predicted_count": 0, "labeled_count": 0},
        }

        # Aggregate field metrics from all samples
        valid_samples = 0
        for i in range(total_samples):
            sample_metrics = data.iloc[i].get("field_metrics", {})
            individual_results = data.iloc[i].get("individual_field_results", {})

            # Handle case where field_metrics might be a string (from Excel serialization)
            if isinstance(sample_metrics, str):
                try:
                    import ast

                    sample_metrics = ast.literal_eval(sample_metrics)
                except:
                    sample_metrics = {}

            # Handle case where individual_field_results might be a string
            if isinstance(individual_results, str):
                try:
                    import ast

                    individual_results = ast.literal_eval(individual_results)
                except:
                    individual_results = {}

            # Handle case where field_metrics might be a string or other type
            if isinstance(sample_metrics, dict) and sample_metrics:  # Only count samples with valid field metrics
                valid_samples += 1
                for field in field_metrics_macro:
                    if field in sample_metrics and isinstance(sample_metrics[field], dict):
                        field_metrics_macro[field]["precision"] += sample_metrics[field].get("precision", 0.0)
                        field_metrics_macro[field]["recall"] += sample_metrics[field].get("recall", 0.0)
                        field_metrics_macro[field]["f1"] += sample_metrics[field].get("f1", 0.0)

                    # Aggregate individual results for micro averaging
                    if field in individual_results and isinstance(individual_results[field], list):
                        field_results = individual_results[field]
                        field_metrics_micro[field]["true_positives"] += sum(field_results)

                        # For micro averaging, we need the total counts from the sample
                        # Use predicted_items and labeled_items from the sample
                        predicted_items = data.iloc[i].get("predicted_items", 0)
                        labeled_items = data.iloc[i].get("labeled_items", 0)

                        # Distribute the counts equally across all 4 fields
                        # This is a reasonable approximation for micro averaging
                        field_metrics_micro[field]["predicted_count"] += predicted_items
                        field_metrics_micro[field]["labeled_count"] += labeled_items

        # Calculate macro averages (average of per-sample metrics)
        if valid_samples > 0:
            for field in field_metrics_macro:
                field_metrics_macro[field]["precision"] /= valid_samples
                field_metrics_macro[field]["recall"] /= valid_samples
                field_metrics_macro[field]["f1"] /= valid_samples

        # Calculate micro averages (metrics over all individual entries)
        for field in field_metrics_micro:
            true_positives = field_metrics_micro[field]["true_positives"]
            predicted_count = field_metrics_micro[field]["predicted_count"]
            labeled_count = field_metrics_micro[field]["labeled_count"]

            # Calculate precision: TP / (TP + FP) = TP / predicted_count
            precision = true_positives / predicted_count if predicted_count > 0 else 0.0

            # Calculate recall: TP / (TP + FN) = TP / labeled_count
            recall = true_positives / labeled_count if labeled_count > 0 else 0.0

            # Calculate F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            field_metrics_micro[field]["precision"] = round(precision, 4)
            field_metrics_micro[field]["recall"] = round(recall, 4)
            field_metrics_micro[field]["f1"] = round(f1, 4)

        # Calculate overall macro averages
        overall_precision_macro = np.mean([m["precision"] for m in field_metrics_macro.values()])
        overall_recall_macro = np.mean([m["recall"] for m in field_metrics_macro.values()])
        overall_f1_macro = np.mean([m["f1"] for m in field_metrics_macro.values()])

        # Calculate overall micro averages
        total_true_positives_all_fields = sum(
            field_metrics_micro[field]["true_positives"] for field in field_metrics_micro
        )
        total_predicted_all_fields = sum(field_metrics_micro[field]["predicted_count"] for field in field_metrics_micro)
        total_labeled_all_fields = sum(field_metrics_micro[field]["labeled_count"] for field in field_metrics_micro)

        overall_precision_micro = (
            total_true_positives_all_fields / total_predicted_all_fields if total_predicted_all_fields > 0 else 0.0
        )
        overall_recall_micro = (
            total_true_positives_all_fields / total_labeled_all_fields if total_labeled_all_fields > 0 else 0.0
        )
        overall_f1_micro = (
            2 * (overall_precision_micro * overall_recall_micro) / (overall_precision_micro + overall_recall_micro)
            if (overall_precision_micro + overall_recall_micro) > 0
            else 0.0
        )

        # Format results
        results = {
            "Overall": round(0.5 * overall_f1_micro + 0.5 * overall_f1_macro, 4),  # Use micro F1 as overall metric
            "Parsing_Failed_Count": parsing_failed_count,
            "Parsing_Success_Rate": round(parsing_success_rate, 4),
            "Total_Predicted_Items": total_predicted_items,
            "Total_Labeled_Items": total_labeled_items,
            "Avg_Predicted_Items": round(avg_predicted_items, 2),
            "Avg_Labeled_Items": round(avg_labeled_items, 2),
        }

        # Add field-wise macro results
        for field, metrics in field_metrics_macro.items():
            results[f"{field}_precision_macro"] = round(metrics["precision"], 4)
            results[f"{field}_recall_macro"] = round(metrics["recall"], 4)
            results[f"{field}_f1_macro"] = round(metrics["f1"], 4)

        # Add field-wise micro results
        for field, metrics in field_metrics_micro.items():
            results[f"{field}_precision_micro"] = metrics["precision"]
            results[f"{field}_recall_micro"] = metrics["recall"]
            results[f"{field}_f1_micro"] = metrics["f1"]
            results[f"{field}_true_positives"] = metrics["true_positives"]
            results[f"{field}_predicted_count"] = metrics["predicted_count"]
            results[f"{field}_labeled_count"] = metrics["labeled_count"]

        # Add overall field metrics
        results["Overall_precision_macro"] = round(overall_precision_macro, 4)
        results["Overall_recall_macro"] = round(overall_recall_macro, 4)
        results["Overall_f1_macro"] = round(overall_f1_macro, 4)
        results["Overall_precision_micro"] = round(overall_precision_micro, 4)
        results["Overall_recall_micro"] = round(overall_recall_micro, 4)
        results["Overall_f1_micro"] = round(overall_f1_micro, 4)
        results["Total_Labeled_Items"] = total_labeled_all_fields
        results["Total_Predicted_Items"] = total_predicted_all_fields
        results["Total_Correct_Items"] = total_true_positives_all_fields

        # Convert to DataFrame and save
        ret = d2df(results).round(4)
        score_pth = storage.replace(".xlsx", "_score.csv")
        dump(ret, score_pth)

        return ret


class LTR_abnormalityQA(JDH_INSPECTION_abnormality_qa):
    pass


# Keep the following code to override the default dataset class
CustomVQADataset.load_data = JDH_INSPECTION_abnormality_qa.load_data
CustomVQADataset.build_prompt = JDH_INSPECTION_abnormality_qa.build_prompt
CustomVQADataset.evaluate = JDH_INSPECTION_abnormality_qa.evaluate
