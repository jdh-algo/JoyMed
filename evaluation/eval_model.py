import os
import time
import json
import sys
import gc
from argparse import ArgumentParser
from utils_main import cal_gpu

supported_datasets = ["PMC_VQA","SLAKE","VQA_RAD","MedXpertQA-MM","PATH_VQA","MMMU-Medical-val","OmniMedVQA",
                      "PubMedQA","MedMCQA","MedQA_USMLE","MedXpertQA-Text","CMMLU","MedQA_MCMLE","Medbullets_op4","Medbullets_op5","SuperGPQA","CMB","CMExam","MedFrameQA",
                      "CheXpert_Plus", "IU_XRAY",
                      "HealthBench",
                      "DiagnosisArena", "GMAI-MMBench-val", "GMAI-MMBench-test","Know2Do",
                      "CT-RATE-VQA","CT-RATE-Report","M3D-Ori","M3D-Slice","3D-RAD","3D-RAD-Ori","3D-MIR","INSPECT","AMOS-MM-Report","AMOS-Ori-Report","Deeptumor-VQA","Deeptumor-Report",
                      "OCR_LTR_fullparsing", "OCR_LTR_simpleQA", "OCR_LTR_abnormalityQA", "OCR_GMD_simpleQA", "OCR_GMD_complexQA",
                       ] + \
                    [f"Professional-{task}" for task in ["新正高001","新正高002","新正高003","新正高028","新正高029","新正高030","新正高044","新正高053","新正高068","新正高111","新副高001","新副高002","新副高003","新副高028","新副高029","新副高030","新副高044","新副高053","新副高068","新副高111","副高002","正高029","正高001","副高028","正高044","正高003","副高029","正高053","正高015","副高044","正高068","正高019","副高053","主治305","正高020","副高068","副高015","正高030","正高002","副高030","正高111","正高028","副高111"]]
                  
supported_models = ["JoyMed", "Qwen3-VL", "Qwen2.5-VL", "Lingshu", "HealthGPT", "HuatuoGPT", "MedGemma", "MedGemma_1_5","Hulu","Baichuan"] #注意需要输入对应的model_size

def main():
    parser = ArgumentParser()
    parser.add_argument('--eval_datasets', type=str, default='Medbullets_op4',
                    help='name of eval dataset')
    parser.add_argument('--datasets_path', type=str, default="/mnt/workspace/offline/shared_data",
                    help='path of eval dataset')
    parser.add_argument('--output_path', type=str, default=os.path.join(os.path.dirname(__file__),"outputs"),
                        help='name of saved json')
    parser.add_argument('--model_name', type=str, default="Citrus_v",
                        help='name of model')
    parser.add_argument('--model_path', type=str, default="/path/of/model/Citrus-V-8B-v1.0")
    parser.add_argument('--prompt_version', type=str, default="v1",
                        help='version of prompt')
    parser.add_argument('--model_size', type=int, default=8,
                        help='size of model')
    parser.add_argument('--model_id', type=int, default=0)

    args = parser.parse_args()
    
    
    id = args.model_id
    model_name = args.model_name
    model_path = args.model_path
    result_save_name = str(id)+'_'+os.path.basename(model_path)
    # eval_datasets = ['CMMLU']
    # eval_datasets = supported_datasets
    eval_datasets = args.eval_datasets.split(',')
    datasets_path = args.datasets_path
    prompt_version = args.prompt_version
    model_size = args.model_size
    output_path = args.output_path

    vqa_datasets = ','.join([data for data in eval_datasets if not data.startswith('OCR_')])
    ocr_datasets = ','.join([data for data in eval_datasets if data.startswith('OCR_')])

    # ### 评测 vqa_datasets
    if vqa_datasets != "":
        suc, gpuDeviceStr, chunks = cal_gpu(model_size, vqa_datasets)
        
        if model_path.startswith("online_api") and (model_name.lower().startswith("gpt") or model_name.lower().startswith('doubao')):
            cmd = f"cd MedEvalKit_local && ./eval_server_multi_api.sh {model_name} {model_path} {result_save_name} \
                        {vqa_datasets} {prompt_version} {datasets_path} {output_path}"
        else:
            cmd = f"cd MedEvalKit_local && ./eval_server_multi.sh {model_name} {model_path} {result_save_name} \
                            {vqa_datasets} {prompt_version} {gpuDeviceStr} {chunks} {datasets_path} {output_path}"
        print(cmd)
        os.system(cmd)
            
    ### 评测 ocr_datasets
    if ocr_datasets != "":
        chunks = 8
        cmd = f"cd VLMEvalKit && ./eval.sh {model_name} {model_path} {result_save_name} {ocr_datasets} {prompt_version} {model_size} {chunks} {datasets_path} {output_path}"
        
        print(cmd)
        os.system(cmd)

if __name__ == "__main__":
    main()