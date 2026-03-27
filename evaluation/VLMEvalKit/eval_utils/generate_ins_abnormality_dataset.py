#!/usr/bin/env python3
"""
Script to generate JDH_ABNORMALITY_test.tsv dataset from inspection_label_res.json
Downloads images and creates the dataset focused on abnormality detection.
"""

import os
import json
import requests
import pandas as pd
import re
from pathlib import Path
from urllib.parse import urlparse
import time
from tqdm import tqdm


def download_image(url, save_path, timeout=30):
    """Download image from URL to local path."""
    if os.path.exists(save_path):
        return True
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def extract_filename_from_url(url):
    """Extract filename from URL."""
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)
    if not filename:
        filename = f"image_{hash(url) % 1000000}.jpg"
    return filename


def is_numerical_value(value_str):
    """Check if a value is numerical."""
    if not value_str or not isinstance(value_str, str):
        return False
    # Remove common non-numeric characters and check if it's a number
    cleaned = re.sub(r"[^\d.-]", "", value_str.strip())
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def extract_numerical_value(value_str):
    """Extract numerical value from string."""
    if not value_str or not isinstance(value_str, str):
        return None
    # Find the first number in the string
    match = re.search(r"(\d+\.?\d*)", value_str.strip())
    if match:
        return float(match.group(1))
    return None


def extract_reference_range(reference_str):
    """Extract min and max values from reference range string."""
    if not reference_str or not isinstance(reference_str, str):
        return None, None

    ref_str = reference_str.strip()

    # Handle various reference range formats
    patterns = [
        r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",  # 10-20
        r"(\d+\.?\d*)\s*~\s*(\d+\.?\d*)",  # 10~20
        r"(\d+\.?\d*)\s*到\s*(\d+\.?\d*)",  # 10到20
        r"(\d+\.?\d*)--(\d+\.?\d*)",  # 10--20 (double dash)
        r"参考范围[：:]\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",  # 参考范围：10-20
        r"参考区间[：:]\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",  # 参考区间：10-20
        r"正常值范围[：:]\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",  # 正常值范围：10-20
        r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*[μμ]mol/L",  # 10-20 μmol/L
        r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*mmol/L",  # 10-20 mmol/L
        r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*g/L",  # 10-20 g/L
        r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*U/L",  # 10-20 U/L
    ]

    for pattern in patterns:
        match = re.search(pattern, ref_str)
        if match:
            try:
                min_val = float(match.group(1))
                max_val = float(match.group(2))
                return min_val, max_val
            except ValueError:
                continue

    return None, None


def determine_abnormality_status(result, reference, notes):
    """
    Determine abnormality status based on result, reference, and notes.
    Returns: "偏高", "偏低", "阳性", "阴性", "异常", or None (normal)
    """
    if not result or not isinstance(result, str):
        return None

    result = result.strip()
    reference = reference.strip() if reference else ""

    # Handle qualitative results (阳性/阴性)
    if result in ["阳性", "阴性"]:
        if reference and "阴性" in reference:
            if result == "阳性":
                return "阳性"  # Abnormal: should be negative but is positive
            else:
                return None  # Normal: negative as expected
        elif reference and "阳性" in reference:
            if result == "阴性":
                return "阴性"  # Abnormal: should be positive but is negative
            else:
                return None  # Normal: positive as expected
        else:
            # No clear reference, use notes or mark as abnormal
            if notes in ["1", "2"]:
                return "异常"
            # For positive results without clear reference, mark as abnormal
            if result == "阳性":
                return "阳性"
            return None

    # Handle numerical values
    if is_numerical_value(result):
        result_val = extract_numerical_value(result)
        if result_val is None:
            return None

        min_ref, max_ref = extract_reference_range(reference)

        # If no reference range, check if notes indicate abnormality
        if min_ref is None or max_ref is None:
            if notes in ["1"]:
                return "偏低"
            elif notes in ["2"]:
                return "偏高"
            elif notes in ["3", "4", "5"]:  # Other abnormality indicators
                return "异常"
            # If no reference range and no notes, consider normal (don't assume abnormal)
            return None

        # Check if result is outside reference range
        if result_val > max_ref:
            return "偏高"
        elif result_val < min_ref:
            return "偏低"
        else:
            return None

    # For other cases, use notes if available
    if notes in ["1"]:
        return "偏低"
    elif notes in ["2"]:
        return "偏高"
    elif notes in ["3", "4", "5"]:  # Other abnormality indicators
        return "异常"
    else:
        # For non-numerical results without clear reference, only mark as abnormal if we have notes
        # This prevents marking normal items as abnormal when reference is empty
        if notes:  # Only mark as abnormal if there are notes indicating abnormality
            return "异常"
        return None  # If no reference and no notes, consider normal


def format_result_with_unit(result, unit):
    """Format result with unit if available."""
    if not result:
        return ""

    if unit and unit.strip():
        return f"{result} {unit.strip()}"
    else:
        return result


def convert_table_to_abnormality_json(table_data):
    """Convert table data to JSON format for abnormality detection."""
    if not table_data or "data" not in table_data:
        return "[]"

    abnormal_items = []

    for item in table_data["data"]:
        entryname = item.get("entryname", "").strip()
        result = item.get("result", "").strip()
        unit = item.get("unit", "").strip()
        reference = item.get("reference", {}).get("ref", "").strip()
        notes = item.get("notes", "").strip()

        # Skip items without meaningful entryname or result
        if not entryname or not result:
            continue

        # Determine abnormality status
        status = determine_abnormality_status(result, reference, notes)

        # Only include abnormal items
        if status:
            formatted_result = format_result_with_unit(result, unit)

            abnormal_item = {
                "entryname": entryname,
                "result": formatted_result,
                "reference": reference,
                "status": status,
            }
            abnormal_items.append(abnormal_item)

    return json.dumps(abnormal_items, ensure_ascii=False)


def generate_dataset():
    """Generate the JDH_INSPECTION_abnormality_qa.tsv dataset."""

    # Paths
    current_dir = Path(__file__).parent
    label_file = current_dir / "inspection_label_res.json"
    output_dir = Path("/mnt/workspace/offline/shared_benchmarks/images/JDH_INSPECTION_abnormality_qa")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load label data
    print("Loading label data...")
    with open(label_file, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    # Prepare dataset
    dataset_rows = []
    question = "请识别检验单中的异常项目，输出JSON格式，包含检验项目名称、结果、参考范围和异常状态。异常状态包括：偏高、偏低、阳性、阴性、弱阳性、异常。只输出异常项目，正常项目不要输出。"

    print("Processing data and downloading images...")
    for idx, (image_id, data) in enumerate(tqdm(label_data.items())):
        try:
            # Get image URL and fix the domain
            image_url = data.get("url", "")
            if not image_url:
                print(f"Skipping {image_id}: no URL")
                continue

            # Replace s3-internal with s3 to make URLs accessible
            # image_url = image_url.replace("s3-internal", "s3")

            # Download image
            filename = extract_filename_from_url(image_url)
            image_path = output_dir / filename

            if not image_path.exists():
                success = download_image(image_url, image_path)
                if not success:
                    print(f"Failed to download image for {image_id}")
                    continue
                time.sleep(0.1)  # Small delay to be respectful

            # Convert table data to abnormality JSON
            table_data = data.get("table", {})
            answer = convert_table_to_abnormality_json(table_data)

            # Create row
            row = {
                "index": idx,
                "question": question,
                "answer": answer,
                "image_url": image_url,
                "image_path": str(image_path.absolute()),
            }
            dataset_rows.append(row)

        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            continue

    # Create DataFrame and save
    print("Creating dataset...")
    df = pd.DataFrame(dataset_rows)

    # Save to TSV
    output_file = current_dir / "JDH_INSPECTION_abnormality_qa.tsv"
    df.to_csv(output_file, sep="\t", index=False, encoding="utf-8")

    print(f"Dataset created successfully!")
    print(f"Total samples: {len(dataset_rows)}")
    print(f"Output file: {output_file}")
    print(f"Images saved to: {output_dir}")


if __name__ == "__main__":
    generate_dataset()
