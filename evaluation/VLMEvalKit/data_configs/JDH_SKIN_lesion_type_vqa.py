import os
import numpy as np
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.image_vqa import CustomVQADataset
from vlmeval.smp import load, dump, d2df
from tqdm import tqdm
import time
from functools import partial

# from test_chatgpt import llm_model_api, prepare_messages, remote_model_client

# remote_llm_model_api = partial(llm_model_api, model_name="gpt-4o-0806", client=remote_model_client)


class LesionTypeVQADataset:
    def load_data(self, dataset):
        data_path = os.path.join("/mnt/workspace/offline/shared_benchmarks", f"{dataset}.tsv")
        return load(data_path)

    def build_prompt(self, line):
        msgs = ImageBaseDataset.build_prompt(self, line)
        msgs[-1]["value"] += (
            "\n仅可从下方可能的类型名称中选取最接近的作答，如果有多种类型，请用英文逗号分隔。\n"
            "水疱,丘疹,斑疹,斑块,脓肿,脓疱,大疱,斑片,结节,溃疡,结痂,糜烂,抓痕,萎缩,渗出,紫癜/淤点,"
            "皲裂,硬结,皮肤干燥,毛细血管扩张,鳞屑,瘢痕,易破溃,硬化,根细头大,息肉样,疣状/乳头状,圆顶状,"
            "平顶,棕色,半透明,色素减退,紫色,黄色,黑色,红斑,粉刺,苔藓样变,蓝色,脐状,斑丘疹萎缩,三文鱼色,"
            "风团,尖锐,皮下螨道,灰色,色素沉着,囊肿,浸渍,斑丘疹,丘疱疹,干燥角化,角质增厚,表皮粗糙,脱屑,"
            "角化,领圈状脱屑,纹理加深,浸润,脓栓,甲板增厚,灰黄混浊,白色混浊,角化脱屑,甲板破坏,毛囊性丘疹,"
            "圆锥状角栓,淡褐色角栓,毛发稀疏,头发脱落,发际线后移"
        )
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert "answer" in data and "prediction" in data

        data["prediction"] = [str(x) for x in data["prediction"]]
        data["answer"] = [str(x) for x in data["answer"]]

        import multiprocessing as mp
        from concurrent.futures import ThreadPoolExecutor

        def process_line(line):
            """
            Process a single line and compute score based on the multi-choice scoring logic:
            - If model output exactly equals all possible correct options: 1.0 point
            - If partially correct, not including any wrong answer: 0.5 point
            - If partially correct but including some wrong answer: 0.25 point
            - If includes no correct answer: 0.0 point
            """
            pred = line["prediction"].strip()
            ans = line["answer"].strip()

            answer_set = set([a.strip() for a in ans.split(",") if a.strip()])
            for possible_sep in ["：", ":"]:
                if possible_sep in pred:
                    pred = pred.split(possible_sep)[-1]
                    break
            predicted_set1 = set([p.strip() for p in pred.split(",") if p.strip()])
            predicted_set2 = set([p.strip() for p in pred.split("，") if p.strip()])
            predicted_set = predicted_set1 if len(predicted_set1) > len(predicted_set2) else predicted_set2

            # Scoring logic
            correct_count = len(predicted_set & answer_set)  # intersection
            wrong_count = len(predicted_set - answer_set)  # predicted but not correct
            total_correct = len(answer_set)

            if correct_count == total_correct and wrong_count == 0:
                return 1.0
            elif correct_count > 0 and wrong_count == 0:
                return 0.5
            elif correct_count > 0 and wrong_count > 0:
                return 0.25
            else:
                return 0.0

        # 将数据转换为列表
        lines = [
            data.iloc[i] if hasattr(data, "iloc") else data.loc[i] if hasattr(data, "loc") else data[i]
            for i in range(len(data))
        ]

        # 使用16线程并行处理
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(process_line, lines), total=len(lines)))

        # 将结果添加到数据中
        data["result"] = results
        # print(data)

        # ========根据需要计算评测指标=========
        # 精确匹配
        result = np.mean(data["result"])
        ret = {"Overall": result}
        ret = d2df(ret).round(2)
        # 保存结果
        suffix = eval_file.split(".")[-1]
        result_file = eval_file.replace(f".{suffix}", "_acc.csv")
        dump(ret, result_file)
        return ret


CustomVQADataset.load_data = LesionTypeVQADataset.load_data
CustomVQADataset.build_prompt = LesionTypeVQADataset.build_prompt
CustomVQADataset.evaluate = LesionTypeVQADataset.evaluate
