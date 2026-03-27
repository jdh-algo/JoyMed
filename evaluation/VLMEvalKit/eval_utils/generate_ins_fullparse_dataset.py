#!/usr/bin/env python3
"""
Script to generate JDH_INSPECTION_full_parsing.tsv dataset from inspection_label_res.json
Downloads images and creates the dataset in the required format.
"""

import os
import json
import requests
import pandas as pd
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


def convert_table_to_json(table_data):
    """Convert table data to JSON format for answer field."""
    if not table_data or "data" not in table_data:
        return "{}"

    # Extract only the required fields: entryname, result, unit, reference
    formatted_data = []
    for item in table_data["data"]:
        formatted_item = {
            "entryname": item.get("entryname", ""),
            "result": item.get("result", ""),
            "unit": item.get("unit", ""),
            "reference": item.get("reference", {}).get("ref", ""),
        }
        formatted_data.append(formatted_item)

    return json.dumps(formatted_data, ensure_ascii=False)


def generate_dataset():
    """Generate the JDH_INSPECTION_full_parsing.tsv dataset."""

    # Paths
    current_dir = Path(__file__).parent
    label_file = current_dir / "inspection_label_res.json"
    output_dir = Path("/mnt/workspace/offline/shared_benchmarks/images/JDH_INSPECTION_full_parsing")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load label data
    print("Loading label data...")
    with open(label_file, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    # Prepare dataset
    dataset_rows = []
    question = "请将图中检验单表格解析为Markdown格式，表头包括检验项目、结果、单位、参考范围"

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

            # Convert table data to JSON
            table_data = data.get("table", {})
            answer = convert_table_to_json(table_data)

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
    output_file = current_dir / "JDH_INSPECTION_full_parsing.tsv"
    df.to_csv(output_file, sep="\t", index=False, encoding="utf-8")

    print(f"Dataset created successfully!")
    print(f"Total samples: {len(dataset_rows)}")
    print(f"Output file: {output_file}")
    print(f"Images saved to: {output_dir}")


if __name__ == "__main__":
    generate_dataset()
