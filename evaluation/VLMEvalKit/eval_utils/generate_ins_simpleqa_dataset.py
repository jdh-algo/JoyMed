#!/usr/bin/env python3
"""
Script to generate JDH_INSPECTION_simple_qa.tsv dataset from inspection_label_res.json
Downloads images and creates a QA dataset for specific entry name queries.
"""

import os
import json
import requests
import pandas as pd
import re
import random
from pathlib import Path
from urllib.parse import urlparse
import time
from tqdm import tqdm
import argparse


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


def format_value_with_unit(value, unit):
    """Format value with unit if available."""
    if not value:
        return ""

    value = str(value).strip()
    if unit and unit.strip() and unit.strip() != "nan":
        return f"{value} {unit.strip()}"
    else:
        return value


def create_question(entryname, field_type, has_unit):
    """Create a question for a specific entry name and field type."""
    field_name_map = {"result": "测量结果", "reference": "参考区间"}

    field_chinese = field_name_map[field_type]

    # Base question format
    base_question = f"图中显示{entryname}的{field_chinese}是多少？"

    # Add unit instruction if applicable
    if has_unit and field_type == "result":
        base_question += "如果有单位请同时包含单位"

    return base_question


def extract_answer(item, field_type):
    """Extract answer from item based on field type."""
    if field_type == "result":
        result = item.get("result", "").strip()
        unit = item.get("unit", "").strip()

        # Check if unit is meaningful (not empty, not 'nan', etc.)
        if unit and unit.lower() not in ["", "nan", "none", "null"]:
            has_unit = True
            answer = format_value_with_unit(result, unit)
        else:
            has_unit = False
            answer = result

        return answer, has_unit

    elif field_type == "reference":
        reference_data = item.get("reference", {})
        if isinstance(reference_data, dict):
            ref_value = reference_data.get("ref", "").strip()
        else:
            ref_value = str(reference_data).strip()

        # Replace newlines with spaces to maintain TSV format integrity
        ref_value = ref_value.replace("\n", " ").replace("\r", " ")

        # Reference values typically don't need separate unit handling
        # as they often include units in the string itself
        return ref_value, False

    return "", False


def select_entries_from_table(table_data, num_entries_per_image):
    """Select random entries from table data."""
    if not table_data or "data" not in table_data:
        return []

    valid_entries = []

    for item in table_data["data"]:
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


def generate_dataset(num_entries_per_image=5, seed=42):
    """Generate the JDH_INSPECTION_simple_qa.tsv dataset.

    Args:
        num_entries_per_image: Number of entry names to select from each image
        seed: Random seed for reproducibility
    """

    # Set random seed for reproducibility
    random.seed(seed)

    # Paths
    current_dir = Path(__file__).parent
    label_file = current_dir / "inspection_label_res.json"
    output_dir = Path("/mnt/workspace/offline/shared_benchmarks/images/JDH_INSPECTION_simple_qa")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load label data
    print("Loading label data...")
    with open(label_file, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    # Prepare dataset
    dataset_rows = []

    print(f"Processing data with {num_entries_per_image} entries per image...")
    print("Downloading images if needed...")

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

            # Select random entries from this image
            table_data = data.get("table", {})
            selected_entries = select_entries_from_table(table_data, num_entries_per_image)

            if not selected_entries:
                print(f"No valid entries found for {image_id}")
                continue

            # Create questions for each selected entry
            for entry in selected_entries:
                entryname = entry.get("entryname", "").strip()

                # Create question for result field
                result_answer, has_unit_result = extract_answer(entry, "result")
                if result_answer:  # Only create question if there's a valid answer
                    result_question = create_question(entryname, "result", has_unit_result)
                    row = {
                        "index": len(dataset_rows),
                        "question": result_question,
                        "answer": result_answer,
                        "image_url": image_url,
                        "image_path": str(image_path.absolute()),
                        "entryname": entryname,
                        "field_type": "result",
                    }
                    dataset_rows.append(row)

                # Create question for reference field
                reference_answer, _ = extract_answer(entry, "reference")
                if reference_answer:  # Only create question if there's a valid answer
                    reference_question = create_question(entryname, "reference", False)
                    row = {
                        "index": len(dataset_rows),
                        "question": reference_question,
                        "answer": reference_answer,
                        "image_url": image_url,
                        "image_path": str(image_path.absolute()),
                        "entryname": entryname,
                        "field_type": "reference",
                    }
                    dataset_rows.append(row)

        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            continue

    # Shuffle the dataset to mix questions from different images
    random.shuffle(dataset_rows)

    # Re-index after shuffling
    for i, row in enumerate(dataset_rows):
        row["index"] = i

    # Create DataFrame and save
    print("Creating dataset...")
    df = pd.DataFrame(dataset_rows)

    # Save to TSV
    output_file = current_dir / "JDH_INSPECTION_simple_qa.tsv"
    df.to_csv(output_file, sep="\t", index=False, encoding="utf-8")

    # Print statistics
    print(f"\nDataset created successfully!")
    print(f"Total samples: {len(dataset_rows)}")
    print(f"Total images processed: {len(set(df['image_path'].tolist()))}")
    print(f"Average questions per image: {len(dataset_rows) / len(set(df['image_path'].tolist())):.2f}")

    # Field type distribution
    field_counts = df["field_type"].value_counts()
    print(f"\nField type distribution:")
    for field, count in field_counts.items():
        print(f"  {field}: {count} ({count/len(df)*100:.1f}%)")

    print(f"\nOutput file: {output_file}")
    print(f"Images saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate inspection entry QA dataset")
    parser.add_argument(
        "--num-entries", type=int, default=5, help="Number of entry names to select from each image (default: 5)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    generate_dataset(num_entries_per_image=args.num_entries, seed=args.seed)


if __name__ == "__main__":
    main()
