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
import numpy as np
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
    # prompt = prompt + response

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
                "model": "gpt-5",
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

def clean_api_response(response_text):
    """
    清理API返回的内容，准备JSON解析
    1. 移除markdown代码块标记
    2. 移除空行
    3. 处理控制字符（转义JSON字符串值中的换行符等）
    """
    if not response_text:
        return None
    
    import re
    
    # 移除首尾空白
    cleaned = response_text.strip()
    
    # 移除markdown代码块标记（```json 或 ```）
    if cleaned.startswith('```'):
        lines = cleaned.split('\n')
        # 移除第一行的```json或```
        if lines[0].startswith('```'):
            lines = lines[1:]
        # 移除最后一行的```
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        cleaned = '\n'.join(lines)
    
    # 移除空行
    lines = [line for line in cleaned.split('\n') if line.strip()]
    cleaned = '\n'.join(lines)
    # print(cleaned)
    
    # 处理JSON字符串值中的控制字符
    # 方法：逐字符解析，跟踪是否在字符串值内，转义控制字符
    result = []
    in_string = False
    escape_next = False
    i = 0
    
    while i < len(cleaned):
        char = cleaned[i]
        
        if escape_next:
            # 转义字符后的字符，直接添加
            result.append(char)
            escape_next = False
        elif char == '\\':
            # 转义字符
            result.append(char)
            escape_next = True
        elif char == '"':
            # 引号，切换字符串状态
            result.append(char)
            in_string = not in_string
        elif in_string and ord(char) < 32:
            # 在字符串值内的控制字符，需要转义
            if char == '\n':
                result.append('\\n')
            elif char == '\r':
                result.append('\\r')
            elif char == '\t':
                result.append('\\t')
            elif char == '\b':
                result.append('\\b')
            elif char == '\f':
                result.append('\\f')
            else:
                # 其他控制字符，转义为Unicode格式或跳过
                result.append(f'\\u{ord(char):04x}')
        else:
            # 普通字符，直接添加
            result.append(char)
        
        i += 1
    
    cleaned = ''.join(result)
    
    return cleaned.strip()

def process_step1_rewrite(record):
    """
    第一步：处理 gen_file 的记录，生成 rewritten_report
    返回: (volumename, Examined_Area, Examined_Type, rewritten_report) 或 None
    """
    try:
        # 提取 Findings_EN 字段
        response = record["response"].replace("\n", " ").lower()
        response_match = re.search(r'<answer>(.*?)</answer>', response)
        response = response_match.group(1).strip() if response_match else response.strip()
        #提取findings
        
        match = re.search(rf"findings[:：]\s*(.*?)(?=impression[:：]|$)", response.lower(), flags=re.S)
        findings_en = match.group(1).strip() if match else response
        if findings_en is None or findings_en == "" or (len(findings_en)<10 and "none" in findings_en):
            logging.warning(f"findings内容为空: {record.get('name', '')}")
            return None
            
        volumename = record.get('name', '')
        Examined_Area = record.get('Examined_Area', '')
        Examined_Type = record.get('Examined_Type', '')
        
        # 调用第一个 API 进行文本分割
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
- Do NOT add any extra keys, commentary, markdown, or explanations.""" + f''' The report is{findings_en}'''
        
        divided_findings_str = call_api(prompt)
        if not divided_findings_str:
            logging.warning(f"拆句API 调用失败: {volumename}")
            return None
        
        cleaned_str = clean_api_response(divided_findings_str)
        if not cleaned_str:
            logging.warning(f"清理后内容为空: {volumename}")
            return None
        
        # 解析 JSON 字符串
        try:
            json_res = json.loads(cleaned_str)
        except json.JSONDecodeError as e:
            logging.error(f"JSON 解析失败 {volumename}: {e}")
            logging.error(f"返回内容: {divided_findings_str}")
            return None
        
        rewritten_report = json_res.get('rewritten_report')
        if not rewritten_report:
            logging.warning(f"未找到 rewritten_report: {volumename}")
            return None
            
        return volumename, Examined_Area, Examined_Type, rewritten_report
        
    except Exception as e:
        logging.error(f"处理记录时出错: {e}")
        return None


def process_step2_tag(gen_record, gt_record):
    """
    第二步：处理 pred_seg_path 和 gt_seg_report_file 的记录，生成标签
    返回: (volumename, Examined_Area, Examined_Type, tag_result) 或 None
    """
    try:
        volumename = gen_record.get('name', '')
        Examined_Area = gen_record.get('Examined_Area', '')
        Examined_Type = gen_record.get('Examined_Type', '')
        
        gen_rewritten_report = gen_record.get('rewritten_report', '')
        gt_rewritten_report = gt_record.get('rewritten_report', '')
        
        if not gen_rewritten_report or not gt_rewritten_report:
            logging.warning(f"缺少 rewritten_report: {volumename} \ngen_record:{gen_record}\ngt_record{gt_record}")
            return None
        
        # 调用第二个 API 进行标签生成
        logging.info(f"正在处理 volumename: {volumename}，进行标签生成")
        prompt = f'''
        You are given two radiology reports:

- Ref: {gt_rewritten_report}
- Gen: {gen_rewritten_report}

Task:
1) Identify sentence pairs (Ref ↔ Gen) that have substantially the same clinical meaning. When matching two sentences, you should consider whether the entity/object being described and the situation being described are consistent—or at least stand in an inclusion (containment) relationship. Many-to-many alignment is allowed: a sentence may appear in multiple pairs.
   - Pair sentences when the core clinical statement is the same, even if the granularity differs (e.g., broader vs more specific anatomy/abnormality). Use the relationship labels to capture the difference.

2) Also output sentences from either report that do NOT have any meaningfully similar match in the other report ("unmatched" sentences).

IMPORTANT RULE: Anti-contradiction constraint
Ensure both statements from Gen and Ref could be true at the same time.
If one sentence affirms a finding while the other negates it (e.g., “present/seen” vs “no/absent/not seen”), or they state opposite directions (e.g., increased vs decreased; patent vs obstructed; normal vs abnormal), they are contradictions and MUST NOT be paired; place them in unmatched.

For each paired Ref–Gen sentence, assign these labels:

- normality: "normal" if the paired sentences describe only normal findings; otherwise "abnormal".
- anatomical_relationship: "equivalent" if the anatomical sites are the same; "part-whole" if one site is contained within the other.
- asserted_abnormality_relationship: compare affirmed (present) abnormality concepts; use "equivalent" or "part-whole". If no affirmed abnormality is stated in either sentence, output null.
- negated_abnormality_relationship: compare negated (absent) abnormality concepts; use "equivalent" or "part-whole". If no negated abnormality is stated in either sentence, output null.
"details_of_abnormality":
- Abnormality details are any descriptors that refine an affirmed abnormal finding, such as:
  measurements (size/number), severity (mild/moderate/severe), morphology/appearance (shape/margins/density/signal/enhancement), temporal descriptors (acute/chronic), and diagnostic impression/inference (e.g., "consistent with", "suggesting").
- Exclude:
  (1) anatomical localization words (laterality, organ/region/segment names), and
  (2) the core abnormality concept word(s) themselves (e.g., "nodule", "fracture", "hemorrhage").
- If uncertainty/hedging is present (possible/probable/likely/suspected/cannot exclude/etc.), count the uncertainty cue(s) as part of the details.
- If a sentence contains no such details, treat it as having "no details".

Details-of-abnormality comparison rules:
- Compare abnormality details between Ref and Gen by meaning (not exact string match).
- Output "equivalent" if the meaning coverage is essentially the same.
- Output "partial" if there is meaningful overlap but at least one meaningful detail is missing, extra, or contradictory.
- Output "none" if there is no meaningful overlap, or if one side has no details while the other has details.
- If normality = "normal", set details_of_abnormality_relationship = null.

Output:
Return ONLY one valid JSON object with exactly these two top-level fields: "pairs" and "unmatched_sentences".

Schema:

{{
  "pairs": [
    {{
      "ref_sentence": "<string>",
      "gen_sentence": "<string>",
      "normality": "normal" | "abnormal",
      "anatomical_relationship": "equivalent" | "part-whole",
      "asserted_abnormality_relationship": "equivalent" | "part-whole" ,
      "negated_abnormality_relationship": "equivalent" | "part-whole",
      "details_of_abnormality_relationship": "none" | "partial" | "equivalent"
    }}
  ],
  "unmatched_sentences": [
    {{
      "sentence_is_from": "Ref" | "Gen",
      "sentence": "<string>",
      "normality": "normal" | "abnormal"
    }}
  ]
}}

Return ONLY valid JSON. Do not include any extra text.
'''
        # print(prompt)
        processed_sentence_str = call_api(prompt)
        if not processed_sentence_str:
            logging.warning(f"标签API 调用失败: {volumename}")
            return None
        
        # 清理返回内容，提取 JSON 部分
        cleaned_str = processed_sentence_str.strip()
        if cleaned_str.startswith('```'):
            lines = cleaned_str.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            cleaned_str = '\n'.join(lines)
        
        # 解析 JSON 字符串
        try:
            json_sentence = json.loads(cleaned_str)
        except json.JSONDecodeError as e:
            logging.error(f"JSON 解析失败 {volumename}: {e}")
            logging.error(f"返回内容: {processed_sentence_str[:200]}...")
            return None
            
        return volumename, Examined_Area, Examined_Type, json_sentence
        
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

def calculate_score(tag_result):
    """
    根据tag_result计算RadSEM score
    
    Args:
        tag_result: 包含pairs和unmatched_sentences的字典
    
    Returns:
        score: RadSEM score (0-1之间)
    """
    if not tag_result or not isinstance(tag_result, dict):
        return 0.0
    
    pairs = tag_result.get('pairs', [])
    unmatched_sentences = tag_result.get('unmatched_sentences', [])
    
    # 统计匹配上的异常句和正常句数量（加权后）
    matched_abnormal_count = 0.0
    matched_normal_count = 0.0
    
    # 统计未匹配的异常句和正常句数量
    unmatched_ref_abnormal = 0
    unmatched_ref_normal = 0
    unmatched_gen_abnormal = 0
    unmatched_gen_normal = 0
    
    # 处理pairs
    for pair in pairs:
        normality = pair.get('normality', '')
        if not normality:
            continue
        
        # 计算part-whole的数量
        part_whole_count = 0
        if pair.get('anatomical_relationship') == 'part-whole':
            part_whole_count += 1
        if pair.get('asserted_abnormality_relationship') == 'part-whole':
            part_whole_count += 1
        if pair.get('negated_abnormality_relationship') == 'part-whole':
            part_whole_count += 1
        
        # part-whole系数：每出现一个part-whole，系数多乘1/3
        part_whole_coeff = (1.0 / 3.0) ** part_whole_count
        
        # 计算details_of_abnormality_relationship的系数
        details_coeff = 1.0
        details_rel = pair.get('details_of_abnormality_relationship')
        if details_rel == 'equivalent':
            details_coeff = 1.0
        elif details_rel == 'partial':
            details_coeff = 0.75
        elif details_rel == 'none':
            details_coeff = 0.5
        elif details_rel is None:
            # 如果为None（正常句的情况），系数为1.0
            details_coeff = 1.0
        
        # 计算该句子的加权数量
        sentence_weight = part_whole_coeff * details_coeff
        
        # 根据normality分类
        if normality == 'abnormal':
            matched_abnormal_count += sentence_weight
        elif normality == 'normal':
            matched_normal_count += sentence_weight
    
    # 处理unmatched_sentences
    for unmatched in unmatched_sentences:
        sentence_from = unmatched.get('sentence_is_from', '')
        normality = unmatched.get('normality', '')
        
        if normality == 'abnormal':
            if sentence_from == 'Ref':
                unmatched_ref_abnormal += 1
            elif sentence_from == 'Gen':
                unmatched_gen_abnormal += 1
        elif normality == 'normal':
            if sentence_from == 'Ref':
                unmatched_ref_normal += 1
            elif sentence_from == 'Gen':
                unmatched_gen_normal += 1
    
    # 计算F1值
    # 异常句F1 = 2*匹配上的异常句数量 / (2*匹配上的异常句数量 + 两个报告的未匹配的异常句数量之和)
    abnormal_denominator = 2 * matched_abnormal_count + unmatched_ref_abnormal + unmatched_gen_abnormal
    abnormal_f1 = (2 * matched_abnormal_count / abnormal_denominator) if abnormal_denominator > 0 else 0.0
    
    # 正常句F1 = 2*匹配上的正常句数量 / (2*匹配上的正常句数量 + 两个报告的未匹配的正常句数量之和)
    normal_denominator = 2 * matched_normal_count + unmatched_ref_normal + unmatched_gen_normal
    normal_f1 = (2 * matched_normal_count / normal_denominator) if normal_denominator > 0 else 0.0

    if unmatched_ref_abnormal + unmatched_gen_abnormal == 0:
        abnormal_f1 = 1
    if unmatched_ref_normal + unmatched_gen_normal == 0:
        normal_f1 = 1
    
    # 最终得分：RadSEM score = 0.9*异常句F1 + 0.1*正常句F1
    score = 0.9 * abnormal_f1 + 0.1 * normal_f1
    
    return score

def eval_report(gt_seg_report_file, out_samples, pred_seg_path, eval_tag_path, output_score_path):
    """主函数"""
    # ========== 第一步：处理 gen_file，生成 pred_seg_path ==========
    print("="*60)
    print("第一步：处理 response，生成 seg_report")
    print("="*60)

    
    # 读取所有report
    if isinstance(out_samples, str):
        print(f"正在读取文件: {out_samples}")
        gen_records = []
        try:
            with open(out_samples, 'r', encoding='utf-8') as in_f:
                for line_num, line in enumerate(in_f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        gen_records.append((line_num, record))
                    except json.JSONDecodeError as e:
                        logging.error(f"第 {line_num} 行 JSON 解析失败: {e}")
        except FileNotFoundError:
            print(f"错误: 输入文件不存在: {input_file}")
            return
        except Exception as e:
            print(f"读取文件时出错: {e}")
            return
    else:
        for sample in out_samples:
            sample['name'] = sample['sample_id']
        gen_records = [(id, sample) for id, sample in enumerate(out_samples) if sample['Question_Type'] == 'REPORT']
            

    # 加载已存在的 name 集合
    existing_names_step1 = load_existing_names(pred_seg_path)
    # 如果 pred_seg_path 存在，过滤掉 name 已存在的记录
    if existing_names_step1:
        gen_records = [record for record in gen_records if record[1].get('name') not in existing_names_step1]

    

    
    total_records_step1 = len(gen_records)
    if total_records_step1 > 0:
        print(f"开始并发处理 {total_records_step1} 条数据（第一步）...")
        
        file_lock = Lock()
        success_count_step1 = 0
        failed_count_step1 = 0
        save_batch_size = 5
        results_dict = {}
        next_write_index = 0
        
        file_mode = 'a' if existing_names_step1 else 'w'
        with open(pred_seg_path, file_mode, encoding='utf-8') as out_f:
            with ThreadPoolExecutor(max_workers=50) as executor:
                future_to_index = {}
                for idx, (line_num, record) in enumerate(gen_records):
                    future = executor.submit(process_step1_rewrite, record)
                    future_to_index[future] = (idx, line_num, record)
                
                with tqdm(total=total_records_step1, desc="第一步处理进度") as pbar:
                    for future in as_completed(future_to_index):
                        idx, line_num, record = future_to_index[future]
                        try:
                            result = future.result()
                            results_dict[idx] = (line_num, result)
                            if result is not None:
                                success_count_step1 += 1
                            else:
                                failed_count_step1 += 1
                        except Exception as e:
                            logging.error(f"第 {line_num} 行处理失败: {e}")
                            results_dict[idx] = (line_num, None)
                            failed_count_step1 += 1
                        
                        while next_write_index in results_dict:
                            line_num, result = results_dict[next_write_index]
                            if result is not None:
                                volumename, Examined_Area, Examined_Type, rewritten_report = result
                                if rewritten_report != "":
                                    output_record = {
                                        'name': volumename,
                                        'Examined_Area': Examined_Area,
                                        'Examined_Type': Examined_Type,
                                        'rewritten_report': rewritten_report
                                    }
                                    json_line = json.dumps(output_record, ensure_ascii=False)
    
                                    with file_lock:
                                        out_f.write(json_line + '\n')
                                        out_f.flush()
                            del results_dict[next_write_index]
                            next_write_index += 1
                            if (next_write_index - 1) % save_batch_size == 0:
                                with file_lock:
                                    out_f.flush()
                        pbar.update(1)
                    
                    while next_write_index < total_records_step1:
                        if next_write_index in results_dict:
                            line_num, result = results_dict[next_write_index]
                            if result is not None:
                                volumename, Examined_Area, Examined_Type, rewritten_report = result
                                if rewritten_report != "":
                                    output_record = {
                                        'name': volumename,
                                        'Examined_Area': Examined_Area,
                                        'Examined_Type': Examined_Type,
                                        'rewritten_report': rewritten_report
                                    }
                                    json_line = json.dumps(output_record, ensure_ascii=False)
                                    with file_lock:
                                        out_f.write(json_line + '\n')
                            del results_dict[next_write_index]
                        next_write_index += 1
                    
                    with file_lock:
                        out_f.flush()
        
        print(f"\n第一步处理完成!")
        print(f"总数据: {total_records_step1}")
        print(f"成功: {success_count_step1}")
        print(f"失败: {failed_count_step1}")
        print(f"输出文件: {pred_seg_path}")
    else:
        print("第一步：没有需要处理的记录，所有记录都已存在。")
    
    # ========== 第二步：处理 pred_seg_path 和 gt_seg_report_file，生成 eval_tag_path ==========
    print("\n" + "="*60)
    print("第二步：处理 pred_seg_path 和 gt_seg_report_file，生成标签")
    print("="*60)
    
    # 加载已存在的 name 集合（用于 eval_tag_path）
    existing_names_step2 = load_existing_names(eval_tag_path)
    
    # 读取 pred_seg_path 和 gt_seg_report_file
    print(f"正在读取文件: {pred_seg_path}")
    gen_rewritten_data = {}
    try:
        with open(pred_seg_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    name = data.get('name')
                    gen_rewritten_data[name] = data
    except FileNotFoundError:
        print(f"错误: 文件不存在: {pred_seg_path}")
        return
    
    print(f"正在读取文件: {gt_seg_report_file}")
    gt_data = {}
    try:
        with open(gt_seg_report_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    name = data.get('name')
                    gt_data[name] = data
    except FileNotFoundError:
        print(f"错误: 文件不存在: {gt_seg_report_file}")
        return
    
    # 找出匹配的 name
    matched_names = set(gen_rewritten_data.keys()) & set(gt_data.keys())
    print(f"找到 {len(matched_names)} 个匹配的name")
    
    # 过滤掉已存在的记录
    records_step2 = []
    for name in sorted(matched_names):
        if existing_names_step2 and name in existing_names_step2:
            logging.info(f"name={name} 已存在，跳过")
            continue
        records_step2.append((name, gen_rewritten_data[name], gt_data[name]))
    
    total_records_step2 = len(records_step2)
    if total_records_step2 > 0:
        print(f"开始并发处理 {total_records_step2} 条数据（第二步）...")
        
        file_lock = Lock()
        success_count_step2 = 0
        failed_count_step2 = 0
        save_batch_size = 5
        results_dict = {}
        next_write_index = 0
        
        file_mode = 'a' if existing_names_step2 else 'w'
        with open(eval_tag_path, file_mode, encoding='utf-8') as out_f:
            with ThreadPoolExecutor(max_workers=50) as executor:
                future_to_index = {}
                for idx, (name, gen_record, gt_record) in enumerate(records_step2):
                    future = executor.submit(process_step2_tag, gen_record, gt_record)
                    future_to_index[future] = (idx, name, gen_record, gt_record)
                
                with tqdm(total=total_records_step2, desc="第二步处理进度") as pbar:
                    for future in as_completed(future_to_index):
                        idx, name, gen_record, gt_record = future_to_index[future]
                        try:
                            result = future.result()
                            results_dict[idx] = (name, result)
                            if result is not None:
                                success_count_step2 += 1
                            else:
                                failed_count_step2 += 1
                        except Exception as e:
                            logging.error(f"name={name} 处理失败: {e}")
                            results_dict[idx] = (name, None)
                            failed_count_step2 += 1
                        
                        while next_write_index in results_dict:
                            name, result = results_dict[next_write_index]
                            if result is not None:
                                volumename, Examined_Area, Examined_Type, tag_result = result
                                output_record = {
                                    'name': volumename,
                                    'Examined_Area': Examined_Area,
                                    'Examined_Type': Examined_Type,
                                    'findings': tag_result
                                }
                                json_line = json.dumps(output_record, ensure_ascii=False)
                                with file_lock:
                                    out_f.write(json_line + '\n')
                                    out_f.flush()
                            del results_dict[next_write_index]
                            next_write_index += 1
                            if (next_write_index - 1) % save_batch_size == 0:
                                with file_lock:
                                    out_f.flush()
                        pbar.update(1)
                    
                    while next_write_index < total_records_step2:
                        if next_write_index in results_dict:
                            name, result = results_dict[next_write_index]
                            if result is not None:
                                volumename, Examined_Area, Examined_Type, tag_result = result
                                output_record = {
                                    'name': volumename,
                                    'Examined_Area': Examined_Area,
                                    'Examined_Type': Examined_Type,
                                    'findings': tag_result
                                }
                                json_line = json.dumps(output_record, ensure_ascii=False)
                                with file_lock:
                                    out_f.write(json_line + '\n')
                            del results_dict[next_write_index]
                        next_write_index += 1
                    
                    with file_lock:
                        out_f.flush()
        
        print(f"\n第二步处理完成!")
        print(f"总数据: {total_records_step2}")
        print(f"成功: {success_count_step2}")
        print(f"失败: {failed_count_step2}")
        print(f"输出文件: {eval_tag_path}")
    else:
        print("第二步：没有需要处理的记录，所有记录都已存在。")
    
    # ========== 第三步：计算score并保存 ==========
    print("\n" + "="*60)
    print("第三步：计算score并保存")
    print("="*60)
    
    # 加载已存在的 name 集合（用于 output_score_path）
    existing_names_step3 = load_existing_names(output_score_path)
    
    # 读取eval_tag_path
    print(f"正在读取文件: {eval_tag_path}")
    tag_records = []
    try:
        with open(eval_tag_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # 如果 output_score_path 存在，过滤掉 name 已存在的记录
                    if existing_names_step3:
                        record_name = record.get('name')
                        if record_name in existing_names_step3:
                            logging.info(f"第 {line_num} 行 name={record_name} 已存在，跳过")
                            continue
                    tag_records.append((line_num, record))
                except json.JSONDecodeError as e:
                    logging.error(f"第 {line_num} 行 JSON 解析失败: {e}")
    except FileNotFoundError:
        print(f"错误: 输入文件不存在: {eval_tag_path}")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    total_records_step3 = len(tag_records)
    if total_records_step3 > 0:
        print(f"开始计算 {total_records_step3} 条数据的score...")
        
        file_mode = 'a' if existing_names_step3 else 'w'
        with open(output_score_path, file_mode, encoding='utf-8') as out_f:
            score_list = []
            for line_num, record in tqdm(tag_records, desc="计算score"):
                name = record.get('name', '')
                Examined_Area = record.get('Examined_Area', '')
                Examined_Type = record.get('Examined_Type', '')
                findings = record.get('findings', {})
                
                # 计算score
                score = calculate_score(findings)
                score_list.append(score)
                
                # 保存结果
                output_record = {
                    'name': name,
                    'Examined_Area': Examined_Area,
                    'Examined_Type': Examined_Type,
                    'score': score
                }
                json_line = json.dumps(output_record, ensure_ascii=False)
                out_f.write(json_line + '\n')
                out_f.flush()
        # score = np.array(score_list)
        print(f"mean:{np.mean(score_list)}")
        print(f"\n第三步处理完成!")
        print(f"总数据: {total_records_step3}")
        print(f"输出文件: {output_score_path}")
    else:
        print("第三步：没有需要处理的记录，所有记录都已存在。")
    
    print(f"处理完成!")
    ##计算平均分
    metric = {"total":0,"score":0}
    if os.path.exists(output_score_path):
        with open(output_score_path, 'r', encoding='utf-8') as out_f:
            for line in out_f:
                metric["total"] += 1
                line = line.strip()
                data = json.loads(line)
                score = data.get('score')
                metric["score"] += score
        metric["score"] /= metric["total"]
    return metric

if __name__ == "__main__":
    main()