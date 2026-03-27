#!/usr/bin/env python3
"""
Script to clean full parsing dataset and generate simple QA and abnormality QA training data.
Supports both JSON and TSV input/output formats.
Outputs three separate files: cleaned full parsing, simple QA, and abnormality QA.
"""

import os
import json
import random
import re
import sys
import csv
from pathlib import Path
from tqdm import tqdm
import argparse

# Add parent directory to sys.path to import from eval_utils
sys.path.append(str(Path(__file__).parent.parent))
from eval_utils.markdown_json_converter import markdown_to_json


def clean_table_content(content):
    """
    Clean table content based on specified rules:
    - Replace values containing keywords like "提供", "提及", "明确", "给出", "补充" with "/"
    """
    if not content or not isinstance(content, str):
        return content

    # Keywords that trigger replacement
    keywords = ["提供", "提及", "明确", "给出", "补充"]

    # Check if content contains any keyword
    for keyword in keywords:
        if keyword in content:
            return "/"

    return content


def clean_markdown_table(markdown_content):
    """
    Clean markdown table by applying cleaning rules to each cell value.
    """
    if not markdown_content:
        return markdown_content

    lines = markdown_content.split("\n")
    cleaned_lines = []

    for line in lines:
        if "|" in line:
            # Split by pipe and clean each cell
            cells = line.split("|")
            cleaned_cells = []

            for cell in cells:
                # Skip empty cells at the beginning and end (markdown table format)
                if cell == "" and (len(cleaned_cells) == 0 or cells.index(cell) == len(cells) - 1):
                    cleaned_cells.append(cell)
                else:
                    cleaned_cell = clean_table_content(cell.strip())
                    # Preserve spacing for alignment
                    if cleaned_cell == "/":
                        # Center the "/" in the cell space
                        padding = max(0, len(cell) - 1)
                        left_pad = padding // 2
                        right_pad = padding - left_pad
                        cleaned_cells.append(" " * left_pad + "/" + " " * right_pad)
                    else:
                        cleaned_cells.append(" " + cleaned_cell + " " if cell.startswith(" ") else cleaned_cell)

            cleaned_lines.append("|".join(cleaned_cells))
        else:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def clean_full_parsing_data(data):
    """
    Clean the full parsing dataset by applying cleaning rules to assistant responses.
    """
    cleaned_data = []

    for sample in tqdm(data, desc="Cleaning full parsing data"):
        cleaned_sample = sample.copy()

        if "messages" in cleaned_sample:
            cleaned_messages = []
            for message in cleaned_sample["messages"]:
                cleaned_message = message.copy()

                # Clean assistant responses containing markdown tables
                if message.get("role") == "assistant" and message.get("content"):
                    cleaned_content = clean_markdown_table(message["content"])
                    cleaned_message["content"] = cleaned_content

                cleaned_messages.append(cleaned_message)

            cleaned_sample["messages"] = cleaned_messages

        # If there's an original_row (for TSV format), clean relevant fields
        if "original_row" in cleaned_sample:
            original_row = cleaned_sample["original_row"].copy()

            # Clean the answer field if it exists
            if "answer" in original_row:
                original_row["answer"] = clean_markdown_table(original_row["answer"])

            # Clean full_parse_result fields if they exist
            for field in ["full_parse_result", "full_parse_result_doubao"]:
                if field in original_row:
                    original_row[field] = clean_markdown_table(original_row[field])

            cleaned_sample["original_row"] = original_row

        cleaned_data.append(cleaned_sample)

    return cleaned_data


def load_input_data(input_file):
    """
    Load input data from either JSON or TSV file.
    Returns data in unified format (list of dicts with 'messages' and 'images' keys).
    """
    file_ext = Path(input_file).suffix.lower()

    if file_ext == ".json":
        with open(input_file, "r", encoding="utf-8") as f:
            return json.load(f)

    elif file_ext == ".tsv":
        data = []
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in tqdm(reader, desc="Loading TSV data"):
                # Convert TSV row to JSON-like format
                messages = [
                    {"content": row.get("system_prompt", "你是医学文档解析专家"), "role": "system"},
                    {"content": f"<image>{row.get('question', '')}", "role": "user"},
                    {"content": row.get("answer", ""), "role": "assistant"},
                ]

                # Handle image path
                image_path = row.get("image_path", "")
                if not image_path and "image_url" in row:
                    # If image_path is not provided, use image_url
                    image_path = row["image_url"]

                images = [image_path] if image_path else []

                data.append(
                    {
                        "messages": messages,
                        "images": images,
                        # Keep additional fields for TSV output
                        "original_row": row,
                    }
                )
        return data

    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Only .json and .tsv are supported.")


def save_output_data(output_file, samples, original_format="json"):
    """
    Save output data in the same format as input (JSON or TSV).
    """
    file_ext = Path(output_file).suffix.lower()

    if file_ext == ".json":
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

    elif file_ext == ".tsv":
        if not samples:
            print(f"Warning: No samples to save to {output_file}")
            return

        # Extract headers from the first sample if it has original_row
        if "original_row" in samples[0]:
            # Use original headers
            headers = list(samples[0]["original_row"].keys())
        else:
            # Create standard headers for TSV output
            headers = [
                "index",
                "diag_id",
                "image_url",
                "classify_res",
                "angle",
                "full_parse_result_doubao",
                "question",
                "answer",
                "image_path",
                "system_prompt",
            ]

        with open(output_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
            writer.writeheader()

            for idx, sample in enumerate(samples):
                row = {}

                # If we have original row data, use it as base
                if "original_row" in sample:
                    row = sample["original_row"].copy()

                # Update with new question and answer
                messages = sample["messages"]
                row["system_prompt"] = messages[0]["content"] if len(messages) > 0 else ""
                row["question"] = messages[1]["content"].replace("<image>", "") if len(messages) > 1 else ""
                row["answer"] = messages[2]["content"] if len(messages) > 2 else ""

                # Update image path
                if sample.get("images"):
                    row["image_path"] = sample["images"][0]

                # Fill in missing fields with defaults if needed
                for header in headers:
                    if header not in row:
                        if header == "index":
                            row[header] = str(idx)
                        else:
                            row[header] = ""

                writer.writerow(row)

    else:
        raise ValueError(f"Unsupported output format: {file_ext}. Only .json and .tsv are supported.")


def parse_markdown_table(markdown_content):
    """Parse markdown table to extract inspection data using robust converter."""
    # Use the robust markdown parser from eval_utils
    parsed_data = markdown_to_json(markdown_content)

    # Normalize field names to standard format
    normalized_data = []
    for item in parsed_data:
        normalized_item = {}
        for key, value in item.items():
            # Normalize common field name variations
            key_lower = key.lower()
            if any(term in key_lower for term in ["项目", "entryname", "test", "名称", "检验", "检查"]):
                normalized_item["entryname"] = value
            elif any(term in key_lower for term in ["结果", "result", "value", "值"]):
                normalized_item["result"] = value
            elif any(term in key_lower for term in ["单位", "unit", "measurement"]):
                normalized_item["unit"] = value
            elif any(term in key_lower for term in ["参考", "reference", "normal", "范围", "区间"]):
                normalized_item["reference"] = value
            else:
                # Keep original key if not recognized
                normalized_item[key] = value

        # Ensure all required fields exist
        if "entryname" not in normalized_item:
            normalized_item["entryname"] = ""
        if "result" not in normalized_item:
            normalized_item["result"] = ""
        if "unit" not in normalized_item:
            normalized_item["unit"] = ""
        if "reference" not in normalized_item:
            normalized_item["reference"] = ""

        # Only add if entryname and result are present
        if normalized_item["entryname"] and normalized_item["result"]:
            normalized_data.append(normalized_item)

    return normalized_data


def extract_numerical_value(value_str):
    """Extract numerical value from string."""
    if not value_str or not isinstance(value_str, str):
        return None
    # Find the first number in the string
    match = re.search(r"(-?\d+\.?\d*)", value_str.strip())
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def extract_numerical_range(reference_str):
    """Extract min and max values from reference range string."""
    if not reference_str or not isinstance(reference_str, str):
        return None, None

    ref_str = reference_str.strip()

    # Remove whitespaces around separators (e.g., "12 - 88" -> "12-88")
    ref_str = re.sub(r"\s*[-~～\-—–]{1,3}\s*", lambda m: m.group().strip(), ref_str)
    ref_str = re.sub(r"\s*到\s*", "到", ref_str)

    # Handle various reference range formats
    patterns = [
        r"(-?\d+\.?\d*)\s*[-~～\-—–]{1,3}\s*(-?\d+\.?\d*)",  # 10-20, 10--20, 10---20, 10~20, 10～20, 10—20
        r"(-?\d+\.?\d*)\s*到\s*(-?\d+\.?\d*)",  # 10到20
        r".*?[：:]\s*(-?\d+\.?\d*)\s*[-~～\-—–]{1,3}\s*(-?\d+\.?\d*)",  # Any text: 10-20, 10--20, 10---20
    ]

    # Check for multiple ranges first
    range_count = 0
    for pattern in patterns:
        matches = re.findall(pattern, ref_str)
        range_count += len(matches)

    # Check for multiple single value comparisons
    single_patterns = [
        (r"[<≤＜]\s*(-?\d+\.?\d*)", "max"),  # <5 or ≤5 or ＜5
        (r"[>≥＞]\s*(-?\d+\.?\d*)", "min"),  # >5 or ≥5 or ＞5
        (r"<=\s*(-?\d+\.?\d*)", "max"),  # <=5
        (r">=\s*(-?\d+\.?\d*)", "min"),  # >=5
    ]

    single_count = 0
    for pattern, val_type in single_patterns:
        matches = re.findall(pattern, ref_str)
        single_count += len(matches)

    # If multiple ranges or inequalities found, return None
    if range_count > 1 or single_count > 1:
        return "UNKNOWN", "UNKNOWN"

    # Process patterns for single range
    for pattern in patterns:
        match = re.search(pattern, ref_str)
        if match:
            try:
                min_val = float(match.group(1))
                max_val = float(match.group(2))
                return min_val, max_val
            except ValueError:
                continue

    # Handle single value comparisons
    for pattern, val_type in single_patterns:
        match = re.search(pattern, ref_str)
        if match:
            try:
                val = float(match.group(1))
                if val_type == "max":
                    return None, val
                else:
                    return val, None
            except ValueError:
                continue

    return None, None


def roman_to_int(roman):
    """Convert Roman numeral to integer."""
    roman_values = {
        "I": 1,
        "Ⅰ": 1,
        "II": 2,
        "Ⅱ": 2,
        "III": 3,
        "Ⅲ": 3,
        "IV": 4,
        "Ⅳ": 4,
        "V": 5,
        "Ⅴ": 5,
        "VI": 6,
        "Ⅵ": 6,
        "VII": 7,
        "Ⅶ": 7,
        "VIII": 8,
        "Ⅷ": 8,
        "IX": 9,
        "Ⅸ": 9,
        "X": 10,
        "Ⅹ": 10,
    }
    return roman_values.get(roman.upper(), None)


def count_plus_signs(text):
    """Count the number of consecutive + signs in a string."""
    if not text:
        return 0
    # Extract consecutive + signs
    import re

    plus_match = re.search(r"\++", text)
    if plus_match:
        return len(plus_match.group())
    return 0


def extract_plus_range(reference):
    """Extract min and max plus count from ranges like '++~+++'."""
    if not reference:
        return None, None

    # Remove whitespaces around separators (e.g., "++ - +++" -> "++-+++")
    reference = re.sub(r"\s*[-~～\-—–]{1,3}\s*", lambda m: m.group().strip(), reference)
    reference = re.sub(r"\s*到\s*", "到", reference)

    # Handle various range separators similar to extract_numerical_range
    # First check for multiple distinct plus ranges (not just multiple pattern matches)
    # Look for patterns that indicate "or" conditions or multiple ranges
    if any(separator in reference for separator in ["或", ",", "，", ";", "；"]):
        # Check if we have multiple plus ranges separated by these
        plus_range_pattern = r"(\++)\s*[-~～\-—–]{1,3}\s*(\++)"
        matches = re.findall(plus_range_pattern, reference)
        if len(matches) > 1:
            return None, None

    # Match patterns like ++~+++, ++--+++, ++到+++, etc.
    patterns = [
        r"(\++)\s*[-~～\-—–]{1,3}\s*(\++)",  # ++-+++, ++--+++, ++---+++, ++~+++, ++～+++
        r"(\++)\s*到\s*(\++)",  # ++到+++
        r".*?[：:]\s*(\++)\s*[-~～\-—–]{1,3}\s*(\++)",  # Any text: ++-+++
    ]

    # Process patterns for single range
    for pattern in patterns:
        match = re.search(pattern, reference)
        if match:
            min_plus = len(match.group(1))
            max_plus = len(match.group(2))
            return min_plus, max_plus

    return None, None


def extract_roman_range(reference):
    """Extract min and max Roman numeral values from ranges like 'Ⅰ~Ⅱ'."""
    if not reference:
        return None, None
    # Remove unit "度" in reference, e.g. "I度-II度"
    reference = re.sub(r"度", "", reference)
    # Remove whitespaces around separators (e.g., "Ⅰ - Ⅱ" -> "Ⅰ-Ⅱ")
    reference = re.sub(r"\s*[-~～\-—–]{1,3}\s*", lambda m: m.group().strip(), reference)
    reference = re.sub(r"\s*到\s*", "到", reference)

    # Handle various range separators similar to extract_numerical_range
    # Create pattern for Roman numerals (both ASCII and Unicode versions)
    roman_pattern = r"([IVXⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩivx]+)"

    # First check for multiple distinct Roman ranges (not just multiple pattern matches)
    # Look for patterns that indicate "or" conditions or multiple ranges
    if any(separator in reference for separator in ["或", ",", "，", ";", "；"]):
        # Check if we have multiple Roman ranges separated by these
        roman_range_pattern = roman_pattern + r"\s*[-~～\-—–]{1,3}\s*" + roman_pattern
        matches = re.findall(roman_range_pattern, reference, re.IGNORECASE)
        if len(matches) > 1:
            return None, None

    patterns = [
        roman_pattern + r"\s*[-~～\-—–]{1,3}\s*" + roman_pattern,  # Ⅰ-Ⅱ, Ⅰ--Ⅱ, Ⅰ---Ⅱ, Ⅰ~Ⅱ, Ⅰ～Ⅱ
        roman_pattern + r"\s*到\s*" + roman_pattern,  # Ⅰ到Ⅱ
        r".*?[：:]\s*" + roman_pattern + r"\s*[-~～\-—–]{1,3}\s*" + roman_pattern,  # Any text: Ⅰ-Ⅱ
    ]

    # Process patterns for single range
    for pattern in patterns:
        match = re.search(pattern, reference, re.IGNORECASE)
        if match:
            min_val = roman_to_int(match.group(1).strip())
            max_val = roman_to_int(match.group(2).strip())
            if min_val is not None and max_val is not None:
                return min_val, max_val

    return None, None


def is_chinese_range_normal(result, reference):
    """Check if result is normal for Chinese ranges like '0-偶见'."""
    if not reference or not result:
        return None

    # Remove whitespaces around separators (e.g., "0 - 偶见" -> "0-偶见")
    reference = re.sub(r"\s*[-~～\-—–]{1,3}\s*", lambda m: m.group().strip(), reference)
    reference = re.sub(r"\s*到\s*", "到", reference)

    # Only consider ranges with separators (ignore single values like "少许")
    if not any(sep in reference for sep in ["-", "--", "---", "~", "～", "到"]):
        return None

    # Check for multiple ranges - count how many separators exist
    separator_pattern = r"[-~～]{1,3}|到"
    separators = re.findall(separator_pattern, reference)

    # If multiple separators found (indicating multiple ranges), return None
    if len(separators) > 1:
        return None

    # Extract values before and after separators
    # Split by various range separators
    parts = re.split(separator_pattern, reference)

    if len(parts) == 2:
        # Get the first and second parts (values before and after separator)
        value1 = parts[0].strip()
        value2 = parts[1].strip()

        # Check if either part contains 1-3 continuous Chinese characters
        chinese_char_pattern = r"[\u4e00-\u9fff]{1,3}"
        has_chinese_chars = bool(re.search(chinese_char_pattern, value1) or re.search(chinese_char_pattern, value2))

        # Only proceed if at least one part contains Chinese characters
        if not has_chinese_chars:
            return None

        # If result equals either of the two values, it's normal
        if result == value1 or result == value2:
            return True  # Normal
        else:
            return False  # Abnormal

    return None


def determine_abnormality_status(result, reference):
    """
    Determine abnormality status based on result and reference.
    Returns: "偏高", "偏低", "阳性", "阴性", "弱阳性", "异常", or None (normal)

    测试图片
    - 2871_656923910968020996.png
    - 5724_660909523425054720.png
    - 7855_661008311390711817.png
    - 互医产品侧W9检验检查单数据汇总_IMG_3192.png
    - 可用检查单_2.png
    - 西京医院标注_尿常规4.png

    测试数据：
    res_ref_pairs = [
        ("弱阳性(±)", "阴性", "弱阳性"),
        ("阳性(+)", "阴性", "阳性"),
        ("阴性", "阴性", None),
        ("++(1.5)", "-", "阳性"),
        ("-", "-", None),
        ("正常", "-", None),
        ("++", "++~+++", None),
        ("++++", "++~+++", "阳性"),
        ("-", "++~+++", "阴性"),
        ("Ⅲ", "Ⅰ~Ⅱ", "偏高"),
        ("0", "0-偶见", None),
        ("少许", "0-偶见", "异常"),
        ("0", "少见", None),
        ("++", "-", "阳性"),
        ("黄色", "", None),
        ("黄色", "黄褐色", "异常"),
        ("2+", "-", "阳性"),
        ("normal", "normal", None),
        ("清亮", "", None),
        ("9", "5--6", "偏高"),
        ("2.0", "1.0-1.5", "偏高"),
        ("2.0", "1.0~1.5", "偏高"),
        ("+-", "-", "弱阳性"),
        ("1+", "+-", "阳性"),
    ]
    """
    if not result or not isinstance(result, str):
        return None

    # Check for normal results
    if any(keyword in result.lower() for keyword in ("无", "正常", "未见", "未查见", "normal")):
        return None

    # Check for arrows indicating abnormality
    if "↑" in result:
        return "偏高"
    elif "↓" in result:
        return "偏低"
    # elif "←" in result:
    #     return "左偏"
    # elif "→" in result:
    #     return "右偏"

    # Handle numerical values
    result_val = extract_numerical_value(result)
    if result_val is not None:
        min_ref, max_ref = extract_numerical_range(reference)
        if min_ref == "UNKNOWN" and max_ref == "UNKNOWN":
            return None
        if min_ref is not None and max_ref is not None:
            if result_val > max_ref:
                return "偏高"
            elif result_val < min_ref:
                return "偏低"
            else:
                return None  # Normal, within range
        elif max_ref is not None:  # Only max reference (e.g., <5)
            if result_val > max_ref:
                return "偏高"
            else:
                return None
        elif min_ref is not None:  # Only min reference (e.g., >5)
            if result_val < min_ref:
                return "偏低"
            else:
                return None

    # Handle qualitative results (弱阳性/阳性/阴性)
    if any(keyword in result for keyword in ("弱阳性", "±", "+-", "+/-", "弱+")):
        if not reference or (not any(keyword in reference for keyword in ("弱阳性", "±", "+-", "+/-", "弱+"))):
            return "弱阳性"
        else:
            return None
    elif any(keyword in result for keyword in ("阳性", "+")):
        if (
            not reference  # 阳性默认异常
            or (not any(keyword in reference for keyword in ("阳性", "+")))
            or any(keyword in reference for keyword in ("弱阳性", "±", "+-", "+/-", "弱+"))
        ):
            return "阳性"
        elif result.count("+") > reference.count("+"):
            return "阳性"
        else:
            return None
    elif any(keyword in result for keyword in ("阴性", "-", "—", "–")):
        if reference and any(keyword in reference for keyword in ("阳性", "+", "弱阳性", "±", "+-", "+/-", "弱+")):
            return "阴性"
        else:  # 阴性默认正常
            return None

    # Handle plus sign ranges (e.g., "++~+++")
    min_plus_ref, max_plus_ref = extract_plus_range(reference)
    if min_plus_ref is not None and max_plus_ref is not None:
        result_plus = count_plus_signs(result)
        if result_plus > 0:
            if result_plus < min_plus_ref:
                return "偏低"
            elif result_plus > max_plus_ref:
                return "阳性"
            else:
                return None  # Normal, within range

    # Handle Roman numeral ranges (e.g., "Ⅰ~Ⅱ")
    min_roman_ref, max_roman_ref = extract_roman_range(reference)
    if min_roman_ref is not None and max_roman_ref is not None:
        result_roman = roman_to_int(result)
        if result_roman is not None:
            if result_roman < min_roman_ref:
                return "偏低"
            elif result_roman > max_roman_ref:
                return "偏高"
            else:
                return None  # Normal, within range

    # Handle special Chinese ranges (e.g., "0-偶见")
    chinese_check = is_chinese_range_normal(result, reference)
    if chinese_check is not None:
        return None if chinese_check else "异常"

    # result = 0, reference 是复杂内容，算正常
    try:
        if int(result) == 0:
            return None
    except ValueError:
        pass

    # 其他异常情况
    if reference and result != reference and len(reference.strip()) < 10:
        return "异常"
    if any(keyword in result for keyword in ("?", "？")):
        return "异常"

    # 未发现异常
    return None


def select_entries_from_table(entries, num_entries_per_image):
    """Select random entries from table data, same logic as generate_ins_simpleqa_dataset.py."""
    if not entries:
        return []

    valid_entries = []

    for item in entries:
        entryname = item.get("entryname", "").strip()
        result = item.get("result", "").strip()

        # Skip items without meaningful entryname or result
        if not entryname or not result:
            continue

        # Skip placeholder or empty values
        if result.lower() in ["nan", "none", "null", "--", "---", ""]:
            continue

        valid_entries.append(item)

    # If we have fewer valid entries than requested, return all
    if len(valid_entries) <= num_entries_per_image:
        return valid_entries

    # Otherwise, randomly sample the requested number
    return random.sample(valid_entries, num_entries_per_image)


def create_simple_qa_from_entry(entry, image_path):
    """Create simple QA sample from a single entry, matching generate_ins_simpleqa_dataset.py format."""
    entryname = entry.get("entryname", "").strip()
    result = entry.get("result", "").strip()
    unit = entry.get("unit", "").strip()
    reference = entry.get("reference", "").strip()

    if not entryname or not result:
        return None

    # Skip placeholder values
    if result.lower() in ["", "nan", "none", "null", "--", "---"]:
        return None

    # Format result with unit (same logic as original)
    if unit and unit.strip() and unit.strip().lower() not in ["", "nan", "none", "null", result.lower()]:
        formatted_result = f"{result} {unit}"
    else:
        formatted_result = result

    # Create question and answer for result - exact same format as generate_ins_simpleqa_dataset.py
    question = f"图中显示{entryname}的测量结果是多少？"
    if unit and unit.strip() and unit.strip().lower() not in ["", "nan", "none", "null"]:
        question += "如果有单位请同时包含单位"

    answer = formatted_result

    return {"question": question, "answer": answer, "image_path": image_path, "field_type": "result"}


def create_reference_qa_from_entry(entry, image_path):
    """Create reference QA sample from a single entry."""
    entryname = entry.get("entryname", "").strip()
    reference = entry.get("reference", "").strip()

    if not entryname or not reference:
        return None

    # Skip empty references
    if reference.lower() in ["", "nan", "none", "null", "--", "---"]:
        return None

    # Create question and answer for reference
    question = f"图中显示{entryname}的参考区间是多少？"
    answer = reference.replace("\n", " ").replace("\r", " ")

    return {"question": question, "answer": answer, "image_path": image_path, "field_type": "reference"}


def create_abnormality_detection_data(entries):
    """Create abnormality detection data from entries."""
    abnormal_items = []

    for entry in entries:
        entryname = entry.get("entryname", "").strip()
        result = entry.get("result", "").strip()
        unit = entry.get("unit", "").strip()
        reference = entry.get("reference", "").strip()

        # Skip items without meaningful entryname or result
        if not entryname or not result:
            continue

        # Determine abnormality status
        status = determine_abnormality_status(result, reference)

        # Only include abnormal items
        if status:
            # Format result with unit
            if unit and unit.strip() and unit.strip().lower() not in ["", "nan", "none", "null", result.lower()]:
                formatted_result = f"{result} {unit}"
            else:
                formatted_result = result

            abnormal_item = {
                "entryname": entryname,
                "result": formatted_result,
                "reference": reference,
                "status": status,
            }
            abnormal_items.append(abnormal_item)

    return json.dumps(abnormal_items, ensure_ascii=False)

def clean_entries(entries):
    """Clean entries."""
    cleaned_entries = []
    for entry in entries:
        # empty values
        entryname = entry.get("entryname", "").strip()
        unit = entry.get("unit", "").strip()
        reference = entry.get("reference", "").strip()
        if entryname == "无":
            entryname = ""
        if reference == "无":
            reference = ""
        if unit == "无":
            unit = ""
        for null_char in ["/", "-", "—", "–"]:
            if entryname == null_char:
                entryname = ""
            if unit == null_char:
                unit = ""
            if reference == null_char:
                reference = ""

        if not entryname:
            continue

        result = entry.get("result", "").strip("/").strip()
        
        # remove unit from reference and result
        if unit and reference and unit in reference:
            reference = reference.replace(f"({unit})", "").replace(f"（{unit}）", "").replace(unit, "").strip()
        if unit and result and unit in result:
            result = result.replace(f"({unit})", "").replace(f"（{unit}）", "").replace(unit, "").strip()

        cleaned_entries.append({"entryname": entryname, "result": result, "reference": reference, "unit": unit})

    return cleaned_entries


def generate_training_data(
    input_file, output_cleaned_file, output_simple_qa_file, output_abnormality_file, num_entries_per_image=1, seed=42
):
    """
    Clean full parsing data and generate training data.

    Args:
        input_file: Path to ins_full_parsing_0813_22534.json or TSV file
        output_cleaned_file: Path to output cleaned full parsing data file (JSON or TSV)
        output_simple_qa_file: Path to output simple QA training data file (JSON or TSV)
        output_abnormality_file: Path to output abnormality QA training data file (JSON or TSV)
        num_entries_per_image: Number of entries to sample per image for EACH type (result and reference) of simple QA (default: 1)
        seed: Random seed for reproducibility

    Note:
        Abnormality QA samples are always generated. When no abnormalities are detected,
        the answer will be an empty list [] to provide training data for normal cases.
    """

    # Set random seed
    random.seed(seed)

    # Load input data
    print(f"Loading data from {input_file}...")
    data = load_input_data(input_file)
    print(f"Loaded {len(data)} samples")

    # Step 1: Clean the full parsing data
    print("\nCleaning full parsing data...")
    cleaned_data = clean_full_parsing_data(data)

    # Save cleaned data
    input_format = "tsv" if Path(input_file).suffix.lower() == ".tsv" else "json"
    print(f"Saving cleaned data to {output_cleaned_file}...")
    save_output_data(output_cleaned_file, cleaned_data, input_format)
    print(f"Saved {len(cleaned_data)} cleaned samples")

    # Step 2: Generate QA datasets from cleaned data
    print("\nGenerating QA datasets from cleaned data...")

    # System prompt
    system_prompt = "你是医学文档解析专家"

    # Use exact same question from JDH_INSPECTION_abnormality_qa.py
    abnormality_qa_question = "请识别检验单中的异常项目，输出JSON格式，包含检验项目名称、结果、参考范围和异常状态。异常状态包括：偏高、偏低、阳性、阴性、弱阳性、异常。只输出异常项目，正常项目不要输出。"

    # Process cleaned data for QA generation
    simple_qa_samples = []
    abnormality_qa_samples = []

    for sample in tqdm(cleaned_data, desc="Processing samples for QA"):
        messages = sample.get("messages", [])
        images = sample.get("images", [])

        if len(messages) < 3 or not images:
            continue

        # Get the markdown table from assistant response
        assistant_response = messages[2]["content"]
        image_path = images[0]

        # Parse markdown table
        entries = parse_markdown_table(assistant_response)

        entries = clean_entries(entries)

        if not entries:
            continue

        # Keep original row data if it exists (for TSV output)
        original_row = sample.get("original_row", {})

        # 1. Create abnormality detection sample (always generate, even for normal cases)
        abnormality_answer = create_abnormality_detection_data(entries)
        abnormality_sample = {
            "messages": [
                {"content": system_prompt, "role": "system"},
                {"content": f"<image>{abnormality_qa_question}", "role": "user"},
                {"content": abnormality_answer, "role": "assistant"},
            ],
            "images": images,
        }
        if original_row:
            abnormality_sample["original_row"] = original_row.copy()
        abnormality_qa_samples.append(abnormality_sample)

        # 2. Create simple QA samples with balanced distribution
        # Separate entries by what type of QA they can generate
        result_capable_entries = []
        reference_capable_entries = []
        
        for item in entries:
            entryname = item.get("entryname", "").strip()
            result = item.get("result", "").strip()
            reference = item.get("reference", "").strip()

            # Skip items without meaningful entryname
            if not entryname:
                continue

            # Check if this entry can generate each type of QA
            can_generate_result_qa = result and result.lower() not in ["", "nan", "none", "null", "--", "---"]
            can_generate_reference_qa = reference and reference.lower() not in ["", "nan", "none", "null", "--", "---"]

            # Add to appropriate lists (an entry can be in both lists)
            if can_generate_result_qa:
                result_capable_entries.append(item)
            if can_generate_reference_qa:
                reference_capable_entries.append(item)

        # Shuffle both lists for random selection
        random.shuffle(result_capable_entries)
        random.shuffle(reference_capable_entries)

        # Generate result QA samples
        result_qa_generated = 0
        used_for_result = set()
        
        for entry in result_capable_entries:
            if result_qa_generated >= num_entries_per_image:
                break
                
            entry_key = f"{entry.get('entryname', '')}_{entry.get('result', '')}"
            if entry_key in used_for_result:
                continue

            result_qa = create_simple_qa_from_entry(entry, image_path)
            if result_qa:
                used_for_result.add(entry_key)
                simple_qa_sample = {
                    "messages": [
                        {"content": system_prompt, "role": "system"},
                        {"content": f"<image>{result_qa['question']}", "role": "user"},
                        {"content": result_qa["answer"], "role": "assistant"},
                    ],
                    "images": images,
                }
                if original_row:
                    simple_qa_sample["original_row"] = original_row.copy()
                simple_qa_samples.append(simple_qa_sample)
                result_qa_generated += 1

        # Generate reference QA samples
        reference_qa_generated = 0
        used_for_reference = set()
        
        for entry in reference_capable_entries:
            if reference_qa_generated >= num_entries_per_image:
                break
                
            entry_key = f"{entry.get('entryname', '')}_{entry.get('reference', '')}"
            if entry_key in used_for_reference:
                continue

            reference_qa = create_reference_qa_from_entry(entry, image_path)
            if reference_qa:
                used_for_reference.add(entry_key)
                reference_qa_sample = {
                    "messages": [
                        {"content": system_prompt, "role": "system"},
                        {"content": f"<image>{reference_qa['question']}", "role": "user"},
                        {"content": reference_qa["answer"], "role": "assistant"},
                    ],
                    "images": images,
                }
                if original_row:
                    reference_qa_sample["original_row"] = original_row.copy()
                simple_qa_samples.append(reference_qa_sample)
                reference_qa_generated += 1

    # Shuffle the samples
    random.shuffle(simple_qa_samples)
    random.shuffle(abnormality_qa_samples)

    # Save outputs to separate files
    print(f"\nSaving {len(simple_qa_samples)} simple QA samples to {output_simple_qa_file}...")
    save_output_data(output_simple_qa_file, simple_qa_samples, input_format)

    print(f"Saving {len(abnormality_qa_samples)} abnormality QA samples to {output_abnormality_file}...")
    save_output_data(output_abnormality_file, abnormality_qa_samples, input_format)

    # Print statistics
    print(f"\nDataset statistics:")
    print(f"Cleaned full parsing samples: {len(cleaned_data)}")
    print(f"Simple QA samples: {len(simple_qa_samples)}")
    print(f"Abnormality QA samples: {len(abnormality_qa_samples)}")
    print(f"Total samples: {len(cleaned_data)} + {len(simple_qa_samples)} + {len(abnormality_qa_samples)} = {len(cleaned_data) + len(simple_qa_samples) + len(abnormality_qa_samples)}")

    # Field type distribution for simple QA
    if simple_qa_samples:
        result_count = sum(1 for s in simple_qa_samples if "测量结果" in s["messages"][1]["content"])
        reference_count = sum(1 for s in simple_qa_samples if "参考区间" in s["messages"][1]["content"])
        print(f"\nSimple QA field type distribution:")
        print(f"  Result questions: {result_count} ({result_count/len(simple_qa_samples)*100:.1f}%)")
        print(f"  Reference questions: {reference_count} ({reference_count/len(simple_qa_samples)*100:.1f}%)")
        print(f"  Target per image: {num_entries_per_image} result + {num_entries_per_image} reference = {num_entries_per_image * 2} total")

    return cleaned_data, simple_qa_samples, abnormality_qa_samples


def main():
    parser = argparse.ArgumentParser(
        description="Clean and generate training data from full parsing label file (JSON or TSV)"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="/mnt/afs/sunqintian3/mm_pipeline/llamafactory/data/ins_full_parsing_0813_22534.json",
        help="Path to input full parsing label file (JSON or TSV)",
    )
    parser.add_argument(
        "--output-cleaned",
        type=str,
        default="ins_full_parsing_cleaned.json",
        help="Path to output cleaned full parsing data file (JSON or TSV)",
    )
    parser.add_argument(
        "--output-simple-qa",
        type=str,
        default="ins_training_simple_qa.json",
        help="Path to output simple QA training data file (JSON or TSV)",
    )
    parser.add_argument(
        "--output-abnormality",
        type=str,
        default="ins_training_abnormality_qa.json",
        help="Path to output abnormality QA training data file (JSON or TSV)",
    )
    parser.add_argument(
        "--num-entries", type=int, default=1, help="Number of entries to select from each image for EACH type (result and reference) of simple QA (default: 1)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--auto-format", action="store_true", help="Automatically match output format to input format")

    args = parser.parse_args()

    # Convert to absolute paths
    input_file = Path(args.input_file).absolute()

    # Determine output format
    if args.auto_format:
        # Match output format to input format
        input_ext = input_file.suffix.lower()
        if input_ext == ".tsv":
            output_cleaned_file = Path(args.output_cleaned).with_suffix(".tsv").absolute()
            output_simple_qa_file = Path(args.output_simple_qa).with_suffix(".tsv").absolute()
            output_abnormality_file = Path(args.output_abnormality).with_suffix(".tsv").absolute()
        else:
            output_cleaned_file = Path(args.output_cleaned).with_suffix(".json").absolute()
            output_simple_qa_file = Path(args.output_simple_qa).with_suffix(".json").absolute()
            output_abnormality_file = Path(args.output_abnormality).with_suffix(".json").absolute()
    else:
        output_cleaned_file = Path(args.output_cleaned).absolute()
        output_simple_qa_file = Path(args.output_simple_qa).absolute()
        output_abnormality_file = Path(args.output_abnormality).absolute()

    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist!")
        return

    # Check file formats
    supported_formats = [".json", ".tsv"]
    if input_file.suffix.lower() not in supported_formats:
        print(f"Error: Unsupported input format {input_file.suffix}. Supported formats: {supported_formats}")
        return

    # Generate training data
    generate_training_data(
        input_file=str(input_file),
        output_cleaned_file=str(output_cleaned_file),
        output_simple_qa_file=str(output_simple_qa_file),
        output_abnormality_file=str(output_abnormality_file),
        num_entries_per_image=args.num_entries,
        seed=args.seed,
    )

    print(f"\nAll data successfully generated:")
    print(f"  Cleaned full parsing: {output_cleaned_file}")
    print(f"  Simple QA: {output_simple_qa_file}")
    print(f"  Abnormality QA: {output_abnormality_file}")


if __name__ == "__main__":
    main()
