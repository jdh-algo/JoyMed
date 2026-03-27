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

# Import the new utility functions
from eval_utils.markdown_json_converter import markdown_to_json, json_to_markdown, match_field_names, MatchingResult


def ignore_precision(row):
    for k, v in row.items():
        if k in ["notes", "diagnose", "isNeedParse", "abnormalMarkType"]:
            # 不处理
            continue
        if k == "reference":
            # 忽略reference中的数字精度
            try:
                row[k]["min"] = str(float(v["min"])).rstrip("0").rstrip(".")
                row[k]["max"] = str(float(v["max"])).rstrip("0").rstrip(".")
            except Exception:
                pass
            try:
                intervals = re.findall(r"\d+\.?\d*\-\d+\.?\d*", row[k]["ref"])
                if len(intervals) == 1:
                    endpts = intervals[0].split("-")
                    row[k]["ref"] = row[k]["ref"].replace(
                        intervals[0],
                        str(float(endpts[0])).rstrip("0").rstrip(".")
                        + "-"
                        + str(float(endpts[1])).rstrip("0").rstrip("."),
                    )
            except Exception:
                pass
        elif k == "normalRangeValue":
            try:
                intervals = re.findall(r"\d+\.?\d*\-\d+\.?\d*", row[k])
                if len(intervals) == 1:
                    endpts = intervals[0].split("-")
                    row[k] = row[k].replace(
                        intervals[0],
                        str(float(endpts[0])).rstrip("0").rstrip(".")
                        + "-"
                        + str(float(endpts[1])).rstrip("0").rstrip("."),
                    )
            except Exception:
                pass
        else:
            # 忽略其他字段数字精度
            try:
                row[k] = str(float(v)).rstrip("0").rstrip(".")
            except Exception:
                pass
    return row


def is_meaningful_field(text):
    """Check if a field value is meaningful (contains numbers, letters, Chinese chars, or +/- signs)."""
    if not text or not isinstance(text, str):
        return False

    text = text.strip()
    if not text:
        return False

    # Check for meaningless patterns first
    meaningless_patterns = [
        r"^[-\s]{2,}$",  # Two or more dashes and spaces only
        r"^[~\s]+$",  # Only tildes and spaces
        r"^[.\s]+$",  # Only dots and spaces
        r"^[()\s]+$",  # Only parentheses and spaces
        r"^[\[\]\s]+$",  # Only brackets and spaces
        r"^[^\w\u4e00-\u9fff+\-]+$",  # Only non-meaningful characters
    ]

    for pattern in meaningless_patterns:
        if re.match(pattern, text):
            return False

    # Check for meaningful content: numbers, English letters, Chinese characters, or +/- signs
    has_numbers = bool(re.search(r"\d", text))
    has_english = bool(re.search(r"[a-zA-Z]", text))
    has_chinese = bool(re.search(r"[\u4e00-\u9fff]", text))
    has_signs = bool(re.search(r"[+\-]", text))

    return has_numbers or has_english or has_chinese or has_signs


def normalize_text(text):
    """Normalize text for comparison."""
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower().strip()

    # Normalize Greek letters
    text = text.replace("μ", "u").replace("μl", "ul").replace("μmol", "umol")
    text = text.replace("μl", "ul").replace("μmol/l", "umol/l")

    # Normalize scientific notation
    text = re.sub(r"10\^(\d+)", r"10\1", text)
    text = re.sub(r"10\s+(\d+)", r"10\1", text)

    # Normalize range notations
    text = re.sub(r"[-~]\s*[-~]", "-", text)  # "--" or "~~" to "-"
    text = re.sub(r"[-~]", "-", text)  # "~" to "-"
    text = re.sub(r" ~ ", "-", text)  # " ~ " to "-"
    text = re.sub(r"\(-\)", "-", text)  # "(-)" to "-"
    text = re.sub(r"\(\+\)", "+", text)  # "(+)" to "+"

    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_json_from_markdown(markdown_text):
    """Extract JSON from markdown response with improved parsing."""
    if not markdown_text:
        return []

    # Clean the text first
    text = str(markdown_text).strip()

    # CRITICAL FIX: Handle the specific format from ground truth
    # Replace literal \\n with actual newlines BEFORE any other processing
    text = text.replace("\\n", "\n")
    text = text.replace("\\t", "\t")
    text = text.replace("\\r", "")

    # Handle escaped quotes and backslashes
    if text.startswith('"') and text.endswith('"'):
        # Remove outer quotes and unescape
        text = text[1:-1]
        text = text.replace('""', '"')  # Unescape doubled quotes
        text = text.replace('\\"', '"')  # Unescape escaped quotes
        text = text.replace("\\\\", "\\")  # Unescape escaped backslashes

    # First try to parse as direct JSON array
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find and extract JSON array pattern
    # More robust pattern that handles nested objects
    json_array_pattern = r"\[\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}(?:\s*,\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})*\s*\]"
    matches = re.findall(json_array_pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue

    # Try to extract JSON from code blocks with various formats
    json_patterns = [
        r"```json\s*(.*?)\s*```",
        r"```JSON\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict):
                    return [parsed]
            except (json.JSONDecodeError, ValueError):
                continue

    # Look for markdown tables and convert them
    # This is the most likely case based on the ground truth format
    if "|" in text:
        try:
            result = markdown_to_json(text)
            if result:
                return result
        except Exception:
            pass

    # Try to extract structured data that looks like a table
    # Look for patterns like "项目名称：xxx 结果：xxx"
    structured_pattern = (
        r"(项目名称|检验项目|项目|名称)[：:]\s*([^\s，,；;]+)\s*[，,；;]?\s*(结果|数值|值)[：:]\s*([^\s，,；;]+)"
    )
    matches = re.findall(structured_pattern, text, re.IGNORECASE)
    if matches:
        result = []
        for match in matches:
            item = {}
            # Extract field names and values
            if len(match) >= 4:
                item["entryname"] = match[1].strip()
                item["result"] = match[3].strip()
                result.append(item)
        if result:
            return result

    return []


def compare_field_values(pred_value, label_value, field_name):
    """Compare field values with optimized normalization rules."""
    pred_str = str(pred_value)
    label_str = str(label_value)

    # Apply all standardization in one pass
    pred_standardized = standardize_value(pred_str)
    label_standardized = standardize_value(label_str)

    # Check meaningfulness of standardized values
    pred_meaningful = is_meaningful_field(pred_standardized)
    label_meaningful = is_meaningful_field(label_standardized)

    # If one is meaningful and the other is not, they don't match
    if pred_meaningful != label_meaningful:
        return False

    # If both are meaningless, consider them equal
    if not pred_meaningful and not label_meaningful:
        return True

    # Exact match after all standardization
    if pred_standardized == label_standardized:
        return True

    # For reference fields, try removing units and comparing
    if field_name == "reference":
        # Remove common units and compare
        # Remove units from both values
        pred_no_unit = re.sub(r"\s*[A-Z/]+$", "", pred_standardized)
        label_no_unit = re.sub(r"\s*[A-Z/]+$", "", label_standardized)

        if pred_no_unit == label_no_unit:
            return True

        # Try removing units from the beginning of the string too
        pred_no_unit_start = re.sub(r"^[A-Z/]+\s*", "", pred_standardized)
        label_no_unit_start = re.sub(r"^[A-Z/]+\s*", "", label_standardized)

        if pred_no_unit_start == label_no_unit_start:
            return True

    # Fallback to original normalization for special cases
    pred_norm = normalize_text(pred_str)
    label_norm = normalize_text(label_str)

    # Exact match after normalization
    if pred_norm == label_norm:
        return True

    # Special cases for result and reference fields
    if field_name in ["result", "reference"]:
        # Remove unit part if present
        pred_clean = re.sub(r"\s*[a-zA-Z/%]+$", "", pred_norm)
        label_clean = re.sub(r"\s*[a-zA-Z/%]+$", "", label_norm)

        if pred_clean == label_clean:
            return True

    return False


def get_llm_judgment(pred_value, label_value, field_name, model):
    """Get LLM judgment for field comparison."""
    # Check meaningfulness first to avoid unnecessary LLM calls
    pred_meaningful = is_meaningful_field(str(pred_value))
    label_meaningful = is_meaningful_field(str(label_value))

    # If one is meaningful and the other is not, they don't match
    if pred_meaningful != label_meaningful:
        return False

    # If both are meaningless, they match
    if not pred_meaningful and not label_meaningful:
        return True

    prompt = f"""请判断以下两个{field_name}字段的值是否相等。考虑以下规则：
1. 希腊字母与英文字母等价（如μL和uL相同）
2. 科学计数法等价（如10^6和10 6相同）
3. 忽略大小写差异
4. 范围表示法等价（如"-"、"~"等）
5. 对于结果和参考值，有无单位部分都算正确
6. 只要主要信息相等就算正确

预测值: {pred_value}
真实值: {label_value}

请回答"相等"或"不相等"。"""

    try:
        msgs = [{"type": "text", "value": prompt}]
        response = model.generate(msgs)
        return "相等" in response or "相同" in response or "正确" in response
    except:
        return False


def inspection_auxeval(model, line, enable_debug_columns=True):
    """Auxiliary evaluation function for inspection VQA"""
    # Handle case where line is a pandas Series (from DataFrame iteration)
    if hasattr(line, "to_dict"):
        # Convert pandas Series to dict
        line = line.to_dict()

    try:
        # Extract answer and prediction
        answer_text = str(line.get("answer", ""))
        prediction_text = str(line.get("prediction", ""))

        # Parse prediction with improved extraction
        pred_json = extract_json_from_markdown(prediction_text)

        # Parse answer - CRITICAL FIX for the specific ground truth format
        label_json = None

        # The answer field contains escaped markdown tables
        # First, clean up the answer text
        cleaned_answer = answer_text

        # Handle if it's wrapped in quotes (from TSV/CSV)
        if cleaned_answer.startswith('"') and cleaned_answer.endswith('"'):
            cleaned_answer = cleaned_answer[1:-1]
            cleaned_answer = cleaned_answer.replace('""', '"')  # Unescape doubled quotes

        # Replace escaped newlines with actual newlines
        cleaned_answer = cleaned_answer.replace("\\n", "\n")
        cleaned_answer = cleaned_answer.replace("\\t", "\t")
        cleaned_answer = cleaned_answer.replace("\\r", "")

        # Now try to parse the cleaned answer text
        try:
            # First attempt: direct JSON parsing (unlikely for markdown tables)
            parsed = json.loads(cleaned_answer)
            if isinstance(parsed, list):
                label_json = parsed
            elif isinstance(parsed, dict):
                label_json = [parsed]
        except (json.JSONDecodeError, ValueError):
            # Most likely case: it's a markdown table
            if "|" in cleaned_answer:
                try:
                    # Parse the markdown table directly
                    label_json = markdown_to_json(cleaned_answer)
                except Exception as e:
                    print(f"Failed to parse markdown table: {e}")
                    # Fallback: try extract_json_from_markdown
                    label_json = extract_json_from_markdown(cleaned_answer)
            else:
                # Try other extraction methods
                label_json = extract_json_from_markdown(cleaned_answer)

        # Check if parsing failed
        parsing_failed = len(pred_json) == 0

        # Generate debug columns
        if enable_debug_columns:
            # 1. Model prediction markdown to JSON conversion result
            pred_json_debug = json.dumps(pred_json, ensure_ascii=False, separators=(",", ":")) if pred_json else "[]"

            # 2. Ground truth format conversion
            label_markdown_debug = ""
            label_json_debug = ""

            if label_json is not None and len(label_json) > 0:
                label_json_debug = json.dumps(label_json, ensure_ascii=False, separators=(",", ":"))
                # Only generate markdown if it's a reasonable size
                if len(label_json) <= 50:  # Increased limit
                    try:
                        label_markdown_debug = json_to_markdown(label_json)
                    except Exception as e:
                        label_markdown_debug = f"[Markdown conversion error: {str(e)}]"
                else:
                    label_markdown_debug = f"[Large table with {len(label_json)} rows]"
            else:
                label_json_debug = (
                    f"Failed to parse answer to JSON. Raw answer (first 500 chars): {cleaned_answer[:500]}"
                )
                label_markdown_debug = cleaned_answer[:500] if len(cleaned_answer) > 500 else cleaned_answer.strip()
        else:
            pred_json_debug = ""
            label_markdown_debug = ""
            label_json_debug = ""

        # Evaluate single sample
        if label_json is not None and pred_json:
            sample_results = evaluate_single_sample(pred_json, label_json, model)
        else:
            sample_results = {
                "entryname": {"matches": [], "pred_count": 0, "label_count": 0},
                "result": {"matches": [], "pred_count": 0, "label_count": 0},
                "unit": {"matches": [], "pred_count": 0, "label_count": 0},
                "reference": {"matches": [], "pred_count": 0, "label_count": 0},
            }

        # Calculate metrics
        field_metrics = {}
        individual_field_results = {}

        for field in ["entryname", "result", "unit", "reference"]:
            field_data = sample_results.get(field, {"matches": [], "pred_count": 0, "label_count": 0})
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

        # Calculate overall score
        if any(m["predicted_count"] > 0 or m["labeled_count"] > 0 for m in field_metrics.values()):
            overall_f1 = np.mean([m["f1"] for m in field_metrics.values()])
        else:
            overall_f1 = 0.0

        return {
            "score": overall_f1 * 100,
            "log": f"Pred items: {len(pred_json)}, Label items: {len(label_json) if label_json else 0}",
            "raw_score": overall_f1,
            "field_metrics": field_metrics,
            "individual_field_results": individual_field_results,
            "sample_results": sample_results,
            "parsing_failed": parsing_failed,
            "parsed_items": len(pred_json),
            "pred_json_debug": pred_json_debug,
            "label_markdown_debug": label_markdown_debug,
            "label_json_debug": label_json_debug,
        }

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        print(f"ERROR in inspection_auxeval: {error_trace}")
        return {
            "score": 0.0,
            "log": f"Evaluation error: {str(e)}",
            "raw_score": 0.0,
            "field_metrics": {},
            "individual_field_results": {},
            "sample_results": {},
            "parsing_failed": True,
            "parsed_items": 0,
            "pred_json_debug": "[]",
            "label_markdown_debug": f"Error: {str(e)}",
            "label_json_debug": f"Error: {error_trace}",
        }


def standardize_value(value):
    """Apply all standardization steps in one pass for efficiency."""
    if not isinstance(value, str):
        value = str(value) if value is not None else ""

    # Apply all standardization in one pass
    result = value

    # Remove all whitespace
    result = "".join(result.split())

    # Convert special characters
    special_mapping = {
        "μ": "u",
        "：": ":",
        "，": ",",
        "。": ".",
        "？": "?",
        "！": "!",
        "；": ";",
        "'": "'",
        "'": "'",
        """: '"',
        """: '"',
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "《": "<",
        "》": ">",
        "、": ",",
        "—": "-",
        "～": "~",
    }
    for special_char, normal_char in special_mapping.items():
        result = result.replace(special_char, normal_char)

    # Convert full-width to half-width
    full_width_chars = []
    for char in result:
        code = ord(char)
        if code == 0x3000:  # Full-width space
            full_width_chars.append(chr(0x0020))
        elif 0xFF01 <= code <= 0xFF5E:
            full_width_chars.append(chr(code - 0xFEE0))
        else:
            full_width_chars.append(char)
    result = "".join(full_width_chars)

    # Convert to uppercase
    result = result.upper()

    # Handle special cases
    if result == "阴性(-)":
        result = "阴性-"
    if result == "阳性(+)":
        result = "阳性+"

    # Normalize range separators
    result = result.replace("~", "-")

    # Normalize precision for numeric values
    if result:
        # Handle range values (e.g., "11.0-45.0" -> "11-45")
        try:
            pattern = r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)"
            matches = re.findall(pattern, result)

            for start, end in matches:
                # Normalize each number in the range
                start_norm = str(float(start)).rstrip("0").rstrip(".")
                end_norm = str(float(end)).rstrip("0").rstrip(".")

                # Replace the original range
                original_range = f"{start}-{end}"
                normalized_range = f"{start_norm}-{end_norm}"
                result = result.replace(original_range, normalized_range)
        except:
            pass

        # Handle single numeric values
        try:
            # Only convert if the entire string is numeric
            if re.match(r"^[\d\.]+$", result):
                result = str(float(result)).rstrip("0").rstrip(".")
        except:
            pass

    return result


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


def evaluate_single_sample(pred_data, label_data, model):
    """Evaluate a single sample using field matching and optimized standardization."""
    if not pred_data or not label_data:
        return {
            "entryname": {"matches": [], "pred_count": 0, "label_count": 0},
            "result": {"matches": [], "pred_count": 0, "label_count": 0},
            "unit": {"matches": [], "pred_count": 0, "label_count": 0},
            "reference": {"matches": [], "pred_count": 0, "label_count": 0},
        }

    # Ensure both are lists
    if isinstance(pred_data, dict):
        pred_data = [pred_data]
    if isinstance(label_data, dict):
        label_data = [label_data]

    # Define keyword groups for field matching
    keyword_groups = [
        ["检验项目", "检查项目", "项目", "项目名称", "entryname", "test_name", "代号", "no", "序号", "name", "名称", "检测项目"],
        ["结果", "检测结果", "result", "value", "test_result", "结果/单位", "值", "数值", "检验结果"],
        ["单位", "unit", "measurement_unit", "units"],
        ["参考范围", "参考值", "reference", "normal_range", "reference_range", "异常提示", "范围", "正常值", "参考区间"],
        ["备注", "notes", "comment", "remark", "状态", "status"],
        ["缩写", "abbreviation", "abbr", "简称"],
        ["检测方法", "test_method", "method", "方法"],
        ["样本类型", "sample_type", "specimen", "标本"],
        ["报告时间", "report_time", "time", "时间"],
        ["医生", "doctor", "physician", "医师"],
        ["医院", "hospital", "institution", "机构"],
    ]

    # Get standard field mapping
    standard_fields = ["entryname", "result", "unit", "reference"]

    # Create standardized lookup for predictions
    pred_lookup = {}
    for item in pred_data:
        if not isinstance(item, dict):
            continue
        
        # Find the entryname field in the item
        entryname_value = None
        for key, value in item.items():
            key_lower = key.lower()
            # Check if this key matches entryname keywords
            for keyword in keyword_groups[0]:  # First group is for entryname
                if keyword.lower() in key_lower or key_lower in keyword.lower():
                    entryname_value = standardize_value(str(value))
                    break
            if entryname_value:
                break
        
        if entryname_value and is_meaningful_field(entryname_value):
            # Store the full item indexed by entryname
            pred_lookup[entryname_value] = item

    # Create standardized lookup for labels
    label_lookup = {}
    for item in label_data:
        if not isinstance(item, dict):
            continue
        
        # Find the entryname field in the item
        entryname_value = None
        for key, value in item.items():
            key_lower = key.lower()
            # Check if this key matches entryname keywords
            for keyword in keyword_groups[0]:  # First group is for entryname
                if keyword.lower() in key_lower or key_lower in keyword.lower():
                    entryname_value = standardize_value(str(value))
                    break
            if entryname_value:
                break
        
        if entryname_value and is_meaningful_field(entryname_value):
            # Store the full item indexed by entryname
            label_lookup[entryname_value] = item

    # Initialize results structure with counts based on lookups
    results = {
        "entryname": {"matches": [], "pred_count": len(pred_lookup), "label_count": len(label_lookup)},
        "result": {"matches": [], "pred_count": len(pred_lookup), "label_count": len(label_lookup)},
        "unit": {"matches": [], "pred_count": len(pred_lookup), "label_count": len(label_lookup)},
        "reference": {"matches": [], "pred_count": len(pred_lookup), "label_count": len(label_lookup)},
    }

    # Find best matches using Levenshtein distance
    entryname_matches = find_best_matches_levenshtein(pred_lookup, label_lookup)

    # Track matched items
    matched_pred_entrynames = set(entryname_matches.keys())
    matched_label_entrynames = set(entryname_matches.values())

    # Process matched pairs
    for pred_entryname, label_entryname in entryname_matches.items():
        pred_item = pred_lookup[pred_entryname]
        label_item = label_lookup[label_entryname]
        
        # Now compare each field
        for field_idx, field in enumerate(standard_fields):
            # Find the corresponding keys in both pred and label
            pred_value = None
            label_value = None
            
            # Get label value
            for key, value in label_item.items():
                key_lower = key.lower()
                for keyword in keyword_groups[field_idx]:
                    if keyword.lower() in key_lower or key_lower in keyword.lower():
                        label_value = standardize_value(str(value))
                        break
                if label_value is not None:
                    break
            
            # Get pred value
            for key, value in pred_item.items():
                key_lower = key.lower()
                for keyword in keyword_groups[field_idx]:
                    if keyword.lower() in key_lower or key_lower in keyword.lower():
                        pred_value = standardize_value(str(value))
                        break
                if pred_value is not None:
                    break
            
            # Compare values
            if pred_value is None:
                pred_value = ""
            if label_value is None:
                label_value = ""
            
            pred_meaningful = is_meaningful_field(pred_value)
            label_meaningful = is_meaningful_field(label_value)
            
            if pred_meaningful != label_meaningful:
                is_match = False
            elif not pred_meaningful and not label_meaningful:
                is_match = True
            else:
                is_match = (pred_value == label_value)
                
                # Try LLM judgment if not matching
                if not is_match and model and field != "entryname":
                    # Get original values for LLM judgment
                    orig_pred = None
                    orig_label = None
                    
                    for key, value in pred_item.items():
                        key_lower = key.lower()
                        for keyword in keyword_groups[field_idx]:
                            if keyword.lower() in key_lower:
                                orig_pred = str(value)
                                break
                    
                    for key, value in label_item.items():
                        key_lower = key.lower()
                        for keyword in keyword_groups[field_idx]:
                            if keyword.lower() in key_lower:
                                orig_label = str(value)
                                break
                    
                    if orig_pred and orig_label:
                        is_match = get_llm_judgment(orig_pred, orig_label, field, model)
            
            results[field]["matches"].append(is_match)

    # Add false entries for unmatched predictions (false positives)
    for pred_entryname in pred_lookup:
        if pred_entryname not in matched_pred_entrynames:
            # This predicted item was not matched to any label item
            for field in standard_fields:
                results[field]["matches"].append(False)

    # Add false entries for unmatched labels (false negatives)
    for label_entryname in label_lookup:
        if label_entryname not in matched_label_entrynames:
            # This label item was not matched to any prediction
            for field in standard_fields:
                results[field]["matches"].append(False)

    return results


def calculate_metrics(results_list):
    """Calculate precision, recall, and F1 for each field."""
    field_metrics = {}

    for field in ["entryname", "result", "unit", "reference"]:
        all_results = []
        for result in results_list:
            all_results.extend(result.get(field, []))

        if not all_results:
            field_metrics[field] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            continue

        correct = sum(all_results)
        total = len(all_results)

        # Note: With the current implementation after fixing unmatched items,
        # precision and recall are still the same because we're adding the same
        # number of False entries for both unmatched predictions and unmatched labels.
        # For more accurate metrics, we would need to track pred_count and label_count separately.
        precision = correct / total if total > 0 else 0.0
        recall = precision  # In this case, precision = recall since we're comparing against ground truth
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        field_metrics[field] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    # Calculate overall metrics
    overall_precision = np.mean([m["precision"] for m in field_metrics.values()])
    overall_recall = np.mean([m["recall"] for m in field_metrics.values()])
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    field_metrics["Overall"] = {
        "precision": round(overall_precision, 4),
        "recall": round(overall_recall, 4),
        "f1": round(overall_f1, 4),
    }

    return field_metrics


class JDH_INSPECTION_full_parsing:
    def load_data(self, dataset):
        """Load custom dataset."""
        data_path = os.path.join(LMUDataRoot(), f"{dataset}.tsv")
        return load(data_path)

    def build_prompt(self, line):
        """Build prompt for the model."""
        msgs = ImageBaseDataset.build_prompt(self, line)

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
        """Evaluate inspection VQA dataset"""
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
            assert model.working(), "Inspection VQA evaluation requires a working judge model\n" + DEBUG_MESSAGE

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
                    inspection_auxeval,
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
            data["parsed_items"] = [ans[i].get("parsed_items", 0) for i in range(lt)]
            data["pred_json_debug"] = [ans[i].get("pred_json_debug", "[]") for i in range(lt)]
            data["label_markdown_debug"] = [ans[i].get("label_markdown_debug", "") for i in range(lt)]
            data["label_json_debug"] = [ans[i].get("label_json_debug", "") for i in range(lt)]

            # Save detailed results
            dump(data, storage)

        # Load results and calculate statistics
        data = load(storage)

        # Calculate parsing failure statistics
        total_samples = len(data)
        parsing_failed_count = sum(data.get("parsing_failed", [True] * total_samples))
        parsing_success_rate = (total_samples - parsing_failed_count) / total_samples if total_samples > 0 else 0.0
        total_parsed_items = sum(data.get("parsed_items", [0] * total_samples))
        avg_parsed_items = total_parsed_items / total_samples if total_samples > 0 else 0.0

        # Calculate field-wise metrics (MACRO AVERAGE)
        field_metrics_macro = {
            "entryname": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "result": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "unit": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "reference": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        }

        # Calculate field-wise metrics (MICRO AVERAGE)
        field_metrics_micro = {
            "entryname": {"true_positives": 0, "predicted_count": 0, "labeled_count": 0},
            "result": {"true_positives": 0, "predicted_count": 0, "labeled_count": 0},
            "unit": {"true_positives": 0, "predicted_count": 0, "labeled_count": 0},
            "reference": {"true_positives": 0, "predicted_count": 0, "labeled_count": 0},
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

            # Only count samples with valid field metrics
            if isinstance(sample_metrics, dict) and sample_metrics:
                valid_samples += 1
                for field in field_metrics_macro:
                    if field in sample_metrics and isinstance(sample_metrics[field], dict):
                        field_metrics_macro[field]["precision"] += sample_metrics[field].get("precision", 0.0)
                        field_metrics_macro[field]["recall"] += sample_metrics[field].get("recall", 0.0)
                        field_metrics_macro[field]["f1"] += sample_metrics[field].get("f1", 0.0)

                    # Aggregate counts for micro averaging
                    if field in sample_metrics and isinstance(sample_metrics[field], dict):
                        field_metrics_micro[field]["true_positives"] += sample_metrics[field].get("true_positives", 0)
                        field_metrics_micro[field]["predicted_count"] += sample_metrics[field].get("predicted_count", 0)
                        field_metrics_micro[field]["labeled_count"] += sample_metrics[field].get("labeled_count", 0)

        # Calculate macro averages
        if valid_samples > 0:
            for field in field_metrics_macro:
                field_metrics_macro[field]["precision"] /= valid_samples
                field_metrics_macro[field]["recall"] /= valid_samples
                field_metrics_macro[field]["f1"] /= valid_samples

        # Calculate micro averages
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

        # Calculate overall averages
        overall_precision_macro = np.mean([m["precision"] for m in field_metrics_macro.values()])
        overall_recall_macro = np.mean([m["recall"] for m in field_metrics_macro.values()])
        overall_f1_macro = np.mean([m["f1"] for m in field_metrics_macro.values()])

        total_true_positives_all_fields = sum(field_metrics_micro[field]["true_positives"] for field in field_metrics_micro)
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
            "Overall": round(0.5 * overall_f1_micro + 0.5 * overall_f1_macro, 4),
            "Parsing_Failed_Count": parsing_failed_count,
            "Parsing_Success_Rate": round(parsing_success_rate, 4),
            "Total_Parsed_Items": total_parsed_items,
            "Avg_Parsed_Items": round(avg_parsed_items, 2),
        }

        # Add field-wise results
        for field, metrics in field_metrics_macro.items():
            results[f"{field}_precision_macro"] = round(metrics["precision"], 4)
            results[f"{field}_recall_macro"] = round(metrics["recall"], 4)
            results[f"{field}_f1_macro"] = round(metrics["f1"], 4)

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


class LTR_fullparsing(JDH_INSPECTION_full_parsing):
    pass


# Keep the following code to override the default dataset class
CustomVQADataset.load_data = JDH_INSPECTION_full_parsing.load_data
CustomVQADataset.build_prompt = JDH_INSPECTION_full_parsing.build_prompt
CustomVQADataset.evaluate = JDH_INSPECTION_full_parsing.evaluate
