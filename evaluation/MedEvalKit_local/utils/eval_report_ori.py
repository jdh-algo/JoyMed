from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import sentence_bleu
from .eval_metrics.rouge_report import Rouge


from cidereval import cider, ciderD
from bert_score import BERTScorer
from RaTEScore import RaTEScore
# from green_score import GREEN
from tqdm import tqdm
import json

import os
import nltk
import re
import numpy as np
import sys

recursionlimit = 1000
sys.setrecursionlimit(recursionlimit)
# green_scorer = GREEN("GREEN-radllama2-7b", output_dir=".")
os.environ['PYTHONIOENCODING'] = 'utf-8'

import socket
ip_adress = socket.gethostbyname(socket.gethostname())

def prep_reports(reports):
    """Preprocesses reports"""
    return [list(filter(
        lambda val: val !=  "", str(elem)\
            .lower().replace(".", " .").split(" "))) for elem in reports]
    

class Evalreport:
    def __init__(self):
        self.rouge_scorer = Rouge()
        self.ratescore = RaTEScore(bert_model = "/mnt/workspace/offline/shared_model/report_eval/RaTE-NER-Deberta", eval_model='/mnt/workspace/offline/shared_model/report_eval/BioLORD-2023-C')
        self.dataset_path = os.environ.get("dataset_path",None)
    def __call__(self, datas):
        global recursionlimit

        reports = []
        reports2 = []
        preds = []
    
        total_bleu1 = 0
        total_bleu2 = 0
        total_bleu3 = 0
        total_bleu4 = 0
    
        total_rouge1 = 0
        total_rouge2 = 0
        total_rougel = 0
    
        total_precision = 0
        total_recall = 0
        total_f1 = 0
    
        total_meteor_scores = 0
    
        messages_list = []
        for sample in tqdm(datas):
            response = sample["response"].replace("\n", " ").lower()
            if response == "":
                continue
                
            response_match = re.search(r'<answer>(.*?)</answer>', response)
            response = response_match.group(1).strip() if response_match else response.strip()

            if "reports" in sample:
                golden = sample["reports"].replace("\n", " ").lower() if isinstance(sample["reports"],str) else sample["reports"][0].replace("\n", " ").lower()
            elif "report" in sample:
                golden = sample["report"].replace("\n", " ").lower() if isinstance(sample["report"],str) else sample["report"][0].replace("\n", " ").lower()
            else:
                findings = sample["findings"] if sample.get("findings") else sample["section_findings"]
                impression = sample["impression"] if sample.get("impression") else sample["section_impression"]
                golden = f"Findings: {findings} Impression: {impression}.".replace("\n", " ").lower()
    
            tokenized_response = prep_reports([response.lower()])[0]
            tokenized_golden = prep_reports([golden.lower()])[0] 
    
            reports.append(golden)
            reports2.append([golden.lower()])
    
            preds.append(response)
            # print(f">>>response: \n{response}")
            # print(f">>>golden: \n{golden}")
    
            bleu1 = sentence_bleu([tokenized_golden], tokenized_response, weights=[1])
            bleu2 = sentence_bleu([tokenized_golden], tokenized_response, weights=[0.5,0.5])
            bleu3 = sentence_bleu([tokenized_golden], tokenized_response, weights=[1/3,1/3,1/3])
            bleu4 = sentence_bleu([tokenized_golden], tokenized_response, weights=[0.25,0.25,0.25,0.25])
            total_bleu1 += bleu1
            total_bleu2 += bleu2
            total_bleu3 += bleu3
            total_bleu4 += bleu4

            suc = False
            while not suc:
                try:
                    rouge_scores = self.rouge_scorer.get_scores(response.lower()[:2048], golden.lower())
                    suc = True
                except:
                    recursionlimit *= 2
                    sys.setrecursionlimit(recursionlimit)
                        
            total_rouge1 += rouge_scores[0]["rouge-1"]["f"]
            total_rouge2 += rouge_scores[0]["rouge-2"]["f"]
            total_rougel += rouge_scores[0]["rouge-l"]["f"]
    
            meteor_scores = single_meteor_score(hypothesis = tokenized_response,reference  = tokenized_golden)
            total_meteor_scores += meteor_scores
    
    
        print("begin to compute cider metrics...")
        cider_score = cider(predictions=[pred.lower() for pred in preds], references= reports2)
        cider_score = cider_score["avg_score"]
    
        print("begin to compute RaTE score...")
        # if ip_adress =="11.48.109.25":
        #     import ipdb; ipdb.set_trace()
        rate_scores = self.ratescore.compute_score(preds, reports)
        
        rate_score = sum(rate_scores)/len(rate_scores)
    
        # print("begin to compute Green score...")
        # green_mean, green_std, green_score_list, summary, result_df = green_scorer(reports, preds)
    
    
        print(f"Meteor Score: {total_meteor_scores/len(datas)}")
        print(f"RaTE Score: {rate_score}")
        # print(f"Green Score: {green_mean} ± {green_std}")
        print(f"Cider Score: {cider_score}")
    
        print("Metrics computed successfully!")
        print(f"Bleu-1: {total_bleu1/len(datas)}")
        print(f"Bleu-2: {total_bleu2/len(datas)}")
        print(f"Bleu-3: {total_bleu3/len(datas)}")
        print(f"Bleu-4: {total_bleu4/len(datas)}")
        print(f"Rouge-1: {total_rouge1/len(datas)}")
        print(f"Rouge-2: {total_rouge2/len(datas)}")
        print(f"Rouge-L: {total_rougel/len(datas)}")
    
        metrics = {
                "cider": float(cider_score), # lingshu
                "meteor": float(total_meteor_scores/len(datas)),
                "bleu1": float(total_bleu1/len(datas)),
                "bleu2": float(total_bleu2/len(datas)),
                "bleu3": float(total_bleu3/len(datas)),
                "bleu4": float(total_bleu4/len(datas)),
                "rouge1": float(total_rouge1/len(datas)),
                "rouge2": float(total_rouge2/len(datas)),
                "rougeL": float(total_rougel/len(datas)), # lingshu
                "rate": float(rate_score), # lingshu
                # "green_mean": float(green_mean),
                # "green_std": float(green_std),
                # "green_score" : f"{green_mean} ± {green_std}",
        }
    
    
        return metrics
