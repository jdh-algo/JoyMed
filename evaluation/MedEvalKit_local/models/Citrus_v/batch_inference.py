import json
import os
import csv
import argparse
import citrus_v_3
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

def main(json_path: str, output_csv: str):
    # ----------------------------
    # 配置
    # ----------------------------
    model_path = "/mnt/workspace/offline/caoxuyang5/code/citrus_v3/checkpoints/checkpoint-6000"
    root_dir = "/mnt/workspace/offline/shared_data/M3D"  # base dir for video folders

    # ----------------------------
    # 加载模型和 processor
    # ----------------------------
    print("Loading model and processor...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("✅ Model loaded.")

    # ----------------------------
    # 读取 JSON 数据
    # ----------------------------
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {json_path}")

    # ----------------------------
    # 准备 CSV 写入
    # ----------------------------
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ground_truth", "prediction"])  # header
        csvfile.flush()  # 立即写入磁盘

        for idx, item in enumerate(data):
            print(f"\n[{idx+1}/{len(data)}] Processing...")

            # --- 1. 构造 image_3d 绝对路径 ---
            rel_video = item["video"][0]  # e.g., "./ct_quizze/005322/..."
            abs_image_folder = os.path.join(root_dir, rel_video.lstrip("./"))

            if not os.path.isdir(abs_image_folder):
                print(f"⚠️ Folder not found: {abs_image_folder}")
                gt = next(msg["value"] for msg in item["conversations"] if msg["from"] == "gpt")
                writer.writerow([gt, "[ERROR] Image folder not found"])
                csvfile.flush()  # 👈 立即刷新
                continue

            # --- 2. 提取问题和真实答案 ---
            human_msg = next(msg for msg in item["conversations"] if msg["from"] == "human")
            gpt_msg = next(msg for msg in item["conversations"] if msg["from"] == "gpt")
            question = human_msg["value"]
            ground_truth = gpt_msg["value"]

            # --- 3. 构造 messages（与 inference.py 一致）---
            # 注意：这里直接使用原始 question，不额外加 "Options:" 等（因为 CLOSED 问题已含选项）
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_3d", "image_3d": abs_image_folder},
                        {"type": "text", "text": f"Question:\n{question}"}
                    ]
                }
            ]

            # --- 4. Tokenize & Inference ---
            try:
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(model.device)

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=128)

                # Trim input tokens
                input_len = inputs.input_ids.shape[1]
                output_ids = generated_ids[0][input_len:]
                prediction = processor.decode(output_ids, skip_special_tokens=True).strip()

                print(f"✅ Prediction: {prediction}, \n✅ GroundTruth: \n{ground_truth}")
                writer.writerow([ground_truth, prediction])
                csvfile.flush()  # 👈 立即刷新

            except Exception as e:
                print(f"❌ Error: {e}")
                writer.writerow([ground_truth, f"[ERROR] {str(e)}"])
                csvfile.flush()  # 👈 立即刷新

    print(f"\n🎉 Batch inference complete! Results saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference on M3D test JSON using Citrus-V.")
    parser.add_argument(
        "--json",
        type=str,
        default="/mnt/workspace/offline/shared_data/M3D/M3D_test.json",
        help="Path to input JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="m3d_predictions.csv",
        help="Output CSV file path"
    )
    args = parser.parse_args()
    main(args.json, args.output)