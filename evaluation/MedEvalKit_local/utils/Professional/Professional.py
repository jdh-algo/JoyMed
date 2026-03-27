import torch
import os
import json
import gc
import csv
import pandas as pd

from PIL import Image
from datasets import load_dataset,load_from_disk
from collections import defaultdict
from tqdm import tqdm

from ..utils import save_json,extract,judge_multi_choice
from ..base_dataset import BaseDataset

from ..question_formats import get_multiple_choice_prompt
from .parse_think_anwser import parser_answer


Professional_subject = {
    "主治305":"P1-主治305呼吸内科学考题题库整理-医考帮-9761",
    "正高001":"P1-正高001心血管内科考题题库3384道-阿虎医考（有图）",
    "正高002":"P0-正高002呼吸内科考题题库1111道-阿虎医考（有图）",
    "正高003":"P1-正高003消化内科考题题库2521道-阿虎医考（有图）",
    "正高015":"P1-正高015泌尿外科考题题库1841道-阿虎医考（有图）",
    "正高019":"P1-正高019妇产科考题题库2523道-阿虎医考（有图）",
    "正高020":"P1-正高020小儿内科考题题库3219道-阿虎医考（无图）",
    "正高028":"P0-正高028皮肤与性病考题题库1323道-阿虎医考（无图）",
    "正高029":"P0-正高029肿瘤内科考题题库1661道-阿虎医考（有图）",
    "正高030":"P1-正高030肿瘤外科考题题库1086道-阿虎医考（有图）",
    "正高044":"P0-正高044临床营养考题题库978道-阿虎医考（无图）",
    "正高053":"P0-正高053放射医学技术考题题库2098道-阿虎医考（有图）",
    "正高068":"P0-正高068精神病考题题库1030道-阿虎医考（无图）",
    "正高111":"P1-正高111心电图技术考题题库334道-阿虎医考（有图）",
    "副高002":"P0-副高002呼吸内科考题题库7243道-医考帮（有图）",
    "副高015":"P1-副高015泌尿外科考题题库5495道-医考帮（有图）",
    "副高028":"P0-副高028皮肤与性病考题题库4022道-医考帮（有图）",
    "副高029":"P0-副高029肿瘤内科考题题库5109道-医考帮（有图）",
    "副高030":"P1-副高030肿瘤外科考题题库2413道-医考帮（有图）",
    "副高044":"P0-副高044临床营养考题题库3284道-医考帮（无图）",
    "副高053":"P0-副高053放射医学技术考题题库6124道-阿虎医考（有图）",
    "副高068":"P0-副高068精神病考题题库3950道-医考帮（有图）",
    "副高111":"P1-副高111心电图技术考题题库1837道-阿虎医考（有图）",

    "新正高001": "新001心血管内科正高题库3384道",
    "新正高002": "新002呼吸内科正高题库1111道",
    "新正高003": "新003消化内科正高题库2521道",
    "新正高028": "新028皮肤与性病正高题库1323道",
    "新正高029": "新029肿瘤内科正高题库1661道",
    "新正高030": "新030肿瘤外科正高题库1086道",
    "新正高044": "新044临床营养正高题库978道",
    "新正高053": "新053放射医学技术正高题库2098道",
    "新正高068": "新068精神病正高题库1030道",
    "新正高111": "新111心电图技术正高题库334道",

    "新副高001": "新P1-副高001心血管内科考题题库8056道-医考帮（有图）",
    "新副高002": "新P0-副高002呼吸内科考题题库7243道-医考帮（有图）",
    "新副高003": "新P1-副高003消化内科考题题库8153道-医考帮（有图）",
    "新副高028": "新P0-副高028皮肤与性病考题题库4022道-医考帮（有图）",
    "新副高029": "新P0-副高029肿瘤内科考题题库5109道-医考帮（有图）",
    "新副高030": "新P1-副高030肿瘤外科考题题库2413道-医考帮（有图）",
    "新副高044": "新P0-副高044临床营养考题题库3284道-医考帮（无图）",
    "新副高053": "新P0-副高053放射医学技术考题题库6124道-阿虎医考（有图）",
    "新副高068": "新P0-副高068精神病考题题库3950道-医考帮（有图）",
    "新副高111": "新P1-副高111心电图技术考题题库1837道-阿虎医考（有图）",
}


prompt_map = {
    'X': '请详细阅读上述题目，选择多个正确答案，给出对应答案选项，不要输出其它任何信息。请按照下面格式给出：\n【答案】:[选项]',
    'A1': '请详细阅读上述题目，选择一个最佳答案，给出对应答案选项，不要给其他任何信息。请按照下面格式给出：\n【答案】:[选项]',
    'A2': '请详细阅读上述题目，选择一个最佳答案，给出对应答案选项，不要给其他任何信息。请按照下面格式给出：\n【答案】:[选项]',
    'A3': '请详细阅读上述题目，选择一个最佳答案，给出对应答案选项，不要给其他任何信息。请按照下面格式给出：\n【答案】:[选项]',
    'B': '请详细阅读上述题目，选择一个最佳答案，给出对应答案选项，不要给其他任何信息。请按照下面格式给出：\n【答案】:[选项]',
    'other': '请详细阅读上述题目，选择一个或多个正确答案，给出对应答案选项，不要输出其它任何信息。请按照下面格式给出：\n【答案】:[选项]',
    'std': '请详细阅读上述题目，给出对应答案选项，不要给其他任何信息。请按照下面格式给出：\n【答案】:[选项]'
}


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    计算两个字符串之间的 Levenshtein 编辑距离
    编辑操作包括：插入、删除、替换一个字符，每种操作代价为 1
    
    参数:
        s1: 第一个字符串
        s2: 第二个字符串
    
    返回:
        两个字符串的编辑距离（整数）
    """
    # 获取两个字符串的长度
    m, n = len(s1), len(s2)
    
    # 创建 (m+1) x (n+1) 的二维矩阵，初始化第一行和第一列
    # dp[i][j] 表示 s1[0..i-1] 转换为 s2[0..j-1] 的最小编辑距离
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 初始化第一列：s2 为空时，需要删除 s1 的前 i 个字符，代价为 i
    for i in range(m + 1):
        dp[i][0] = i
    
    # 初始化第一行：s1 为空时，需要插入 s2 的前 j 个字符，代价为 j
    for j in range(n + 1):
        dp[0][j] = j
    
    # 填充动态规划矩阵
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # 如果当前字符相等，无需操作，代价等于左上角的值
            if s1[i-1] == s2[j-1]:
                cost = 0
            else:
                # 字符不相等，替换操作的代价为 1
                cost = 1
            
            # 取删除、插入、替换三种操作的最小代价
            # dp[i-1][j] + 1: 删除 s1[i-1]
            # dp[i][j-1] + 1: 插入 s2[j-1] 到 s1
            # dp[i-1][j-1] + cost: 替换 s1[i-1] 为 s2[j-1]
            dp[i][j] = min(
                dp[i-1][j] + 1,        # 删除
                dp[i][j-1] + 1,        # 插入
                dp[i-1][j-1] + cost    # 替换
            )
    
    # 矩阵右下角的值就是最终的编辑距离
    return dp[m][n]


class Professional(BaseDataset):
    def __init__(self,model,dataset_path,output_path,task="副高002"):
        super().__init__()
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "haonan-li/cmmlu"
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
        self.task = task
    
    def load_data(self):
        # dataset = load_dataset(self.dataset_path)['test']
        # dataset = load_dataset(self.dataset_path)
        # print(dataset)
        file_path = os.path.join(self.dataset_path, f"data/{Professional_subject[self.task]}.csv")
        df = pd.read_csv(file_path)
        dataset = df.to_dict(orient='records')

        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                print(f"idx: {idx}")
                sample = self.construct_messages(sample)
                sample["sample_id"] = idx
                self.samples.append(sample)
        return self.samples

    def construct_messages(self,sample):
        
        question_type = None
        for tmp_key in ["题目类型", "题型"]:
            if tmp_key in sample.keys():
                question_type = sample[tmp_key]
                break
        print(f"question_type: {question_type}")

        type = 'other'
        if question_type:
            for key_word in prompt_map.keys():
                if key_word in question_type:
                    type = key_word
                    break
        
        question = sample["题干"]
        Option = sample["选项"]
        answer = sample["正确答案"]
        include_img = sample["是否含图片"]
        imgs_name = sample["图片名称"]

        choices = Option.split('\n')
        
        #处理图片
        imgs_data = []
        if include_img == '是':
            split_chars = [';', '；', '\n']
            img_name_list = []
            for split_char in split_chars:
                if split_char in imgs_name:
                    img_name_list = imgs_name.split(split_char)
                    break
            if len(img_name_list) == 0:
                img_name_list = [imgs_name]

            for img_name in img_name_list:
                img_name = img_name.replace(' ', '').strip()
                img_path = os.path.join(self.dataset_path, f"img/{Professional_subject[self.task]}/{img_name}.jpg")
                if os.path.exists(img_path):
                    # img_data = read_byte(img_path)
                    img_data = Image.open(img_path)
                    imgs_data.append(img_data)
                else:
                    print(f'Warning: image not found ! {img_path}')
        else:
            sample["图片名称"] = ""

        is_reasoning = True if self.REASONING else True if os.environ.get("REASONING","False") == "True" else False
        # print(is_reasoning)
        prompt = get_multiple_choice_prompt(question,choices,is_reasoning,lang = "zh")
        
        prompt = f"{prompt}\n{prompt_map[type]}/think" if is_reasoning else f"{prompt}\n{prompt_map[type]}"
        
        if imgs_data == []:
            messages = {"prompt":prompt}
        else:
            messages = {"prompt":prompt,"images":imgs_data}
            
        sample["prompt"] = prompt
        sample["messages"] = messages
        sample["answer"] = answer
        sample["choices"] = choices
        sample["language"] = "zh"
        sample["Question_Type"] = 'CHOICE'
        # import pdb;pdb.set_trace()
        return sample

    def cal_metrics(self,out_samples):
        total = 0
        right = 0
        edit_distance = 0

        metrics = {}
        for i,sample in enumerate(out_samples):
            response = sample["response"]
            response = extract(response,"answer")
            response = parser_answer(response)
            out_samples[i]["response"] = response
            
            answer = sample["answer"]

            correct = answer == response
            out_samples[i]["correct"] = correct
            if correct:
                right += 1
            total += 1

            tmp_ed = 1 - (levenshtein_distance(answer, response) / (max(len(answer), len(response))))
            edit_distance += tmp_ed

            question_type = ''
            if "题目类型" in sample.keys():
                question_type = sample["题目类型"].strip('[]【】')
            elif "题型" in sample.keys():
                question_type = sample["题型"].strip('[]【】')

            if question_type not in metrics:
                metrics[question_type] = {"total":1, "right": int(correct), "edit_distance": tmp_ed}
            else:
                metrics[question_type]['total'] += 1
                metrics[question_type]['right'] += int(correct)
                metrics[question_type]['edit_distance'] += tmp_ed


        for metric_key, metric_info in metrics.items():
            metrics[metric_key]['acc'] = metric_info['right'] / metric_info['total']
            metrics[metric_key]['avg_edit_distance'] = metric_info['edit_distance'] / metric_info['total']
            del metrics[metric_key]['edit_distance']

        if total > 0:
            metrics['total metrics'] = {"total": total, "right": right, "acc": right / total, "avg_edit_distance": edit_distance / total}
        else:
            metrics['total metrics'] = {"total": 0, "right": 0, "acc": 0.0, "avg_edit_distance": 0.0}
        return metrics, out_samples

