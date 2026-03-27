import json
import os
import pandas as pd
import logging
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import List, Dict, Any, Set
import requests
import re
from api_token import api_map

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_name = "gpt-5"
api_url = api_map[model_name]["API_URL"]
api_token = api_map[model_name]["TOKEN"]
def call_api(prompt,retry_num=5):
    """调用API处理文本"""
    # 构建 prompt
    retry_count = 0
    while True:
        try:
            response = requests.post(
            url = api_url,
            headers = {
                "Content-Type": "application/json", 
                "Authorization": f"Bearer {api_token}", 
                "Accept": "application/json"
            },
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens":16392,
                "temperature": 1
            }
            )
            
            if response:
                gpt_return_list = response.json().get('choices')
                # print(response.json())
                item = gpt_return_list[0]
                gpt = item.get('message').get('content')
                return gpt
        except Exception as e:
            logging.error(f"API 调用错误: {e}")
            
        retry_count += 1
        if retry_count>=retry_num:
            return None

def process_single_record(record):
    """
    处理单条记录
    返回: (volumename, Examined_Area, Examined_Type, processed_findings_list) 或 None
    """
    try:
        # 提取 report 
        if "reports" in record:
            report = record["reports"].replace("\n", " ").lower() if isinstance(record["reports"],str) else record["reports"][0].replace("\n", " ").lower()
        elif "report" in record:
            report = record["report"].replace("\n", " ").lower() if isinstance(record["report"],str) else record["report"][0].replace("\n", " ").lower()
        else:
            findings = record["findings"] if record.get("findings") else record["section_findings"]
            impression = record["impression"] if record.get("impression") else record["section_impression"]
            report = f"Findings: {findings} Impression: {impression}.".replace("\n", " ").lower()
            
        #提取findings
        match = re.search(rf"findings[:：]\s*(.*?)(?=impression[:：]|$)", report.lower(), flags=re.S)
        report = match.group(1).strip() if match else report
        if report is None or report == "" or (len(report)<10 and "none" in report):
            logging.warning(f"findings内容为空: {record.get('name', '')}")
            return None
            
        volumename = record.get('name', [])
        Examined_Area = record.get('Examined_Area', [])
        Examined_Type = record.get('Examined_Type', [])
        
        # 调用 API 进行文本分割
        # logging.info(f"正在处理 volumename: {volumename}，进行文本分割")
        prompt = """Task: Rewrite a medical imaging report into atomic sentences under strict constraints.
You MUST use only the Findings section as input. Do not read, reference, or derive anything from Impression or any other section.
Definitions
- “Sentence” = one atomic finding statement.
- Allowed sentence forms ONLY:
  1) <Anatomical location> + <normal finding>
  2) <Anatomical location> + <one abnormal finding>
- “Paired organ” includes left/right sides (e.g., lungs, kidneys, breasts, ovaries, etc.).
- If the report contains multiple MRI sequences, each sentence must belong to EXACTLY ONE sequence.
-If any foreign bodies are identified on imaging—such as tubes/catheters or other medical devices—these should also be considered as an ABNORMAL finding.

Rules (apply in order)
R1. Atomicity of findings
- Each sentence must contain exactly ONE finding.
- A sentence cannot mix normal + abnormal.
- A sentence cannot contain multiple abnormal findings.
=> If violated, split into multiple sentences until every sentence has exactly one finding.

R2. Left/right separation for paired organs
- Do NOT mention both sides in one sentence.
- If a sentence refers to a paired organ without specifying side (e.g., “lungs”), rewrite into TWO sentences:
  - one for the left side
  - one for the right side
- If a sentence explicitly mentions both sides, split into two side-specific sentences.

R3. MRI sequence isolation (only if sequences exist)
- Each sentence must contain information from exactly ONE sequence.
=> If a sentence mixes sequences, split by sequence.

R4. Removal rules
Remove a sentence if:
- it is not in the allowed forms (see Definitions), OR
- it is comparison-to-prior wording (e.g., “compared with prior”, “previous study”, “interval change”), OR
- it contains non-finding content (history, indication, technique, recommendation, impression headings, measurements not tied to a finding, etc.).

Procedure
1) Split the report into candidate sentences.
2) For each candidate, apply R1–R3 to split as needed.
3) Apply R4 to delete invalid sentences.
4) Output the remaining sentences in the original report order.

Output (STRICT)
Return ONLY valid JSON with exactly one key:
{"rewritten_report":"<sentence1>. <sentence2>. <sentence3>. ..."}
- Each sentence MUST end with a period "." (no exceptions).
- Separate sentences using exactly ONE space after each period.
- Use English.
- Do NOT add any extra keys, commentary, markdown, or explanations.""" + f''' The report is{report}'''
        divided_findings_str = call_api(prompt)
        if not divided_findings_str:
            logging.warning(f"拆句API 调用失败: {volumename}")
            return None
        
        # 清理返回内容，提取 JSON 部分
        # 移除可能的 markdown 代码块标记
        cleaned_str = divided_findings_str.strip()
        if cleaned_str.startswith('```'):
            # 移除开头的 ```json 或 ```
            lines = cleaned_str.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            # 移除结尾的 ```
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            cleaned_str = '\n'.join(lines)
        
        # 解析 JSON 字符串
        try:
            json_res = json.loads(cleaned_str)
        except json.JSONDecodeError as e:
            logging.error(f"JSON 解析失败 {volumename}: {e}")
            logging.error(f"返回内容: {divided_findings_str}")
            return None
        
        # if not isinstance(json_list, list):
        #     logging.warning(f"解析结果不是列表: {volumename}")
        #     return None

        res_list = []
        # for sentence in json_list:
        rewritten_report = json_res.get('rewritten_report')
        return volumename, Examined_Area, Examined_Type, rewritten_report
        
    except Exception as e:
        logging.error(f"处理记录时出错: {e}")
        return None


def load_existing_names(output_file):
    """
    如果输出文件存在，读取其中所有的 name 字段
    返回: name 的集合
    """
    existing_names = set()
    try:
        if os.path.exists(output_file):
            print(f"检测到输出文件已存在: {output_file}，正在读取已有的 name 字段...")
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        name = data.get('name')
                        rewritten_report = data.get('rewritten_report')
                        if rewritten_report != "":
                            existing_names.add(name)
                    except json.JSONDecodeError:
                        continue
            print(f"已读取 {len(existing_names)} 个已存在的 name")
        else:
            print(f"输出文件不存在，将创建新文件: {output_file}")
    except Exception as e:
        logging.warning(f"读取已存在文件时出错: {e}，将继续处理所有记录")
    
    return existing_names

def generate_seg_file(samples, output_file):
    """主函数"""
    
    # 读取所有report
    if isinstance(samples, str):
        print(f"正在读取文件: {samples}")
        records = []
        try:
            with open(samples, 'r', encoding='utf-8') as in_f:
                for line_num, line in enumerate(in_f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        records.append((line_num, record))
                    except json.JSONDecodeError as e:
                        logging.error(f"第 {line_num} 行 JSON 解析失败: {e}")
        except FileNotFoundError:
            print(f"错误: 输入文件不存在: {input_file}")
            return
        except Exception as e:
            print(f"读取文件时出错: {e}")
            return
    else:
        for sample in samples:
            sample['name'] = sample['sample_id']
        records = [(id, sample) for id, sample in enumerate(samples) if sample['Question_Type'] == 'REPORT']
            

    # 加载已存在的 name 集合
    existing_names = load_existing_names(output_file)
    # 如果 output_file 存在，过滤掉 name 已存在的记录
    if existing_names:
        records = [record for record in records if record[1].get('name') not in existing_names]

    
    total_records = len(records)
    if total_records == 0:
        print("没有需要处理的记录，所有记录都已存在。")
        return
    print(f"开始并发处理 {total_records} 条数据（并发数: 15，每 5 个结果保存一次）...")
    
    # 用于线程安全的文件写入和计数
    file_lock = Lock()
    success_count = 0
    failed_count = 0
    save_batch_size = 1
    
    # 用于按顺序保存结果
    results_dict = {}  # key: 索引, value: (line_num, result) 或 None（表示失败）
    next_write_index = 0  # 下一个应该写入的索引
    
    # 打开输出文件（追加模式，保留已存在的内容）
    file_mode = 'a' if existing_names else 'w'
    with open(output_file, file_mode, encoding='utf-8') as out_f:
        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=30) as executor:
            # 提交所有任务，使用索引作为key
            future_to_index = {}
            for idx, (line_num, record) in enumerate(records):
                future = executor.submit(process_single_record, record)
                future_to_index[future] = (idx, line_num, record)
            
            # 使用 tqdm 显示进度
            with tqdm(total=total_records, desc="gt处理进度") as pbar:
                # 处理完成的任务
                for future in as_completed(future_to_index):
                    idx, line_num, record = future_to_index[future]
                    
                    try:
                        result = future.result()
                        results_dict[idx] = (line_num, result)
                        if result is not None:
                            success_count += 1
                        else:
                            failed_count += 1
                    except Exception as e:
                        logging.error(f"第 {line_num} 行处理失败: {e}")
                        results_dict[idx] = (line_num, None)
                        failed_count += 1
                    
                    # 检查是否有连续的结果可以按顺序保存
                    while next_write_index in results_dict:
                        line_num, result = results_dict[next_write_index]
                        
                        if result is not None:
                            volumename, Examined_Area, Examined_Type, json_sentence = result
                            if json_sentence != "":
                                output_record = {
                                    'name': volumename,
                                    'Examined_Area': Examined_Area,
                                    'Examined_Type': Examined_Type,
                                    'rewritten_report': json_sentence
                                }
                                json_line = json.dumps(output_record, ensure_ascii=False)
                                with file_lock:
                                    out_f.write(json_line + '\n')
                                    out_f.flush()  # 立即刷新到磁盘
                        
                        # 删除已写入的结果
                        del results_dict[next_write_index]
                        next_write_index += 1
                        
                        # 每写入 5 个结果，强制刷新一次
                        if (next_write_index - 1) % save_batch_size == 0:
                            with file_lock:
                                out_f.flush()
                    
                    pbar.update(1)
                
                # 保存剩余的结果（按顺序）
                while next_write_index < total_records:
                    if next_write_index in results_dict:
                        line_num, result = results_dict[next_write_index]
                        if result is not None:
                            volumename, Examined_Area, Examined_Type, processed_findings = result
                            if processed_findings != "":
                                output_record = {
                                    'name': volumename,
                                    'Examined_Area': Examined_Area,
                                    'Examined_Type': Examined_Type,
                                    'rewritten_report': processed_findings
                                }
                                json_line = json.dumps(output_record, ensure_ascii=False)
                                with file_lock:
                                    out_f.write(json_line + '\n')
                        del results_dict[next_write_index]
                    next_write_index += 1
                
                # 最终刷新
                with file_lock:
                    out_f.flush()
    
    print(f"\n处理完成!")
    print(f"总数据: {total_records}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"输出文件: {output_file}")


if __name__ == "__main__":
    # 文件路径
    dataset_path = "/mnt/workspace/shared_data/"
    input_file = "groundtruth.jsonl"
    output_file = os.path.join(dataset_path, "groundtruth_step1_res.jsonl")
    
    generate_seg_file(input_file, output_file)