# from evalscope.backend.vlm_eval_kit import VLMEvalKitBackendManager
# print(f'** All models from VLMEvalKit backend: {VLMEvalKitBackendManager.list_supported_datasets()}')
import json
from argparse import ArgumentParser
import yaml
import importlib.util
import sys
import os
import torch
import random
import numpy as np
from evalscope.run import run_task
from evalscope.summarizer import Summarizer

from data_configs.data_configs import DATASET_CONFIGS
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from api_token import api_map


def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def default_config_local(model_name, model_path, dataset, nproc, outputs_path):
    config = {"work_dir": outputs_path,
              "eval_backend": "VLMEvalKit",
              "eval_config": {"model":[{"name":model_name, "model_path":model_path}],
                              "data":[datasets_map[dataset]],
                              "mode": "all",
                              "reuse": True,
                              "nproc": nproc,
                              "OPENAI_API_KEY": api_map["gpt-4o"]["TOKEN"],
                              "OPENAI_API_BASE": api_map["gpt-4o"]["API_URL"],
                              "LOCAL_LLM": "gpt-4o",
                              },
              "analysis_report": True,
              "use_cache": f"{outputs_path}",
             }
    return config

def default_config_api(model_name, dataset, nproc, outputs_path):
    
    api_base = api_model[model_name]["API_URL"]
    key = api_model[model_name]["TOKEN"]
    
    if "gpt-5" in model_name.lower():
        temperature = 1.0
        reuse = False
    elif "gpt-4.1" in model_name.lower() or model_name.lower() == "Doubao-1.5-vision-pro-32k":
        temperature = 0
        reuse = True
    elif model_name.lower() == "doubao-seed-1.6-250615":
        temperature = 0
        reuse = False
    else:
        raise ValueError(f"模型{model_name}不被支持")
    
    config = {"work_dir": outputs_path,
              "eval_backend": "VLMEvalKit",
              "eval_config": {"model":[{"type":model_name, 
                                        "name":"CustomAPIModel", 
                                        "api_base":api_base,
                                        "key":key,
                                        "temperature":temperature,
                                        "img_size":-1,
                                        "max_tokens":8192,
                                       }
                                      ],
                              "data":[datasets_map[dataset]],
                              "mode": "all",
                              "reuse": reuse,
                              "nproc": nproc,
                              "OPENAI_API_KEY": api_map["gpt-4o"]["TOKEN"],
                              "OPENAI_API_BASE": api_map["gpt-4o"]["API_URL"],
                              },
              "analysis_report": True,
              "use_cache": f"{outputs_path}",
             }
    return config
    
def import_custom_dataset(dataset_name):
    """Dynamically import a custom dataset module for the given dataset."""
    if dataset_name in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset_name]
        if config["custom_script"] and config["custom_eval_class"]:
            print(f"Importing custom dataset module for {dataset_name}")

            # Get the script path
            script_path = config["custom_script"]

            # Handle relative paths - if path starts with mm_pipeline/eval/, make it relative to current directory
            if script_path.startswith("mm_pipeline/eval/"):
                # Remove the mm_pipeline/eval/ prefix since we're already in that directory
                script_path = script_path.replace("mm_pipeline/eval/", "", 1)
            elif not os.path.isabs(script_path):
                # If it's a relative path not starting with mm_pipeline/eval/,
                # assume it's relative to the project root (two levels up)
                script_path = script_path[script_path.index("data_configs/"):]

            # Extract module name from script path
            module_name = f"custom_dataset_{dataset_name}"

            try:
                # Check if module is already loaded
                if module_name in sys.modules:
                    # Remove it to force reload
                    del sys.modules[module_name]

                # Load the module dynamically
                spec = importlib.util.spec_from_file_location(module_name, script_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)

                    # The module should automatically patch CustomVQADataset when loaded
                    print(f"Successfully imported custom dataset module for {dataset_name}")
                    return True
                else:
                    print(f"Failed to create module spec for {dataset_name}")
                    return False
            except Exception as e:
                print(f"Failed to import custom dataset module for {dataset_name}: {str(e)}")
                return False
    return False


def run_eval_OCR(model_name, model_path, dataset, nproc, outputs_path, limit=None):
    
    matric_path = os.path.join(outputs_path,"metrics.json")
    if os.path.exists(matric_path):
        metrics = json.load(open(matric_path,'r'))
    else:
        
        if model_name.lower().startswith("gpt") or model_name.lower().startswith("doubao"):
            cfg = default_config_api(model_name, dataset, nproc, outputs_path)
        else:
            cfg = default_config_local(model_name, model_path, dataset, nproc, outputs_path)
        
        if limit is not None:
            cfg["eval_config"]["limit"] = limit
        
        datasets = cfg.get("eval_config", {}).get("data", [])

        # import ipdb; ipdb.set_trace()    
        # Import VLMEvalKit's registry to hook into dataset loading
        try:
            from vlmeval.config import supported_VLM
            from vlmeval.dataset import build_dataset
            from vlmeval.dataset import DATASET_TYPE
    
            # Override the build_dataset function to import custom datasets on demand
            original_build_dataset = build_dataset
            
            def custom_build_dataset(dataset_name, **kwargs):
                # Import custom dataset if needed
                import_custom_dataset(dataset_name)
                # Call the original build_dataset function
                return original_build_dataset(dataset_name, **kwargs)
    
            # Monkey patch the build_dataset function
            import vlmeval.dataset
            vlmeval.dataset.build_dataset = custom_build_dataset
    
        except Exception as e:
            print(f"Warning: Could not hook into VLMEvalKit dataset loading: {e}")
            # Fall back to importing all custom datasets (with the overwriting issue)
            for dataset_name in datasets:
                import_custom_dataset(dataset_name)
        # Compatibility patch: ensure VLMEval receives a `reuse_aux` attribute
        # Some versions of VLMEvalKit's runner access args.reuse_aux but
        # evalscope constructs an Arguments object without that field.
        try:
            import vlmeval.run as _vlm_run
    
            _orig_run_task = _vlm_run.run_task
    
            def _run_task_with_reuse_aux(args):
                if not hasattr(args, "reuse_aux"):
                    setattr(args, "reuse_aux", True)
                return _orig_run_task(args)
    
            _vlm_run.run_task = _run_task_with_reuse_aux
        except Exception as _e:
            print(f"Warning: Failed to patch vlmeval.run.run_task for reuse_aux: {_e}")
    
        run_task(task_cfg=cfg)
    
        
        report_list = Summarizer.get_report_from_cfg(cfg)
        print(f"\n>> The report list: {report_list}")

        #保存指标
        metrics = [report_list[0][name] for name in report_list[0]][0]
        json.dump(metrics, open(matric_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)


    if "Overall_f1_macro" in metrics and "Overall_f1_micro" in metrics:
        metrics["Overall"] = (float(metrics["Overall_f1_macro"])+float(metrics["Overall_f1_micro"]))/2
    
    total_results_path = os.path.join(os.path.dirname(outputs_path),'total_results.json')
    if os.path.exists(total_results_path):
        with open (total_results_path,"r") as f:
            total_results = json.load(f)
    else:
        total_results = {}
        
    total_results[dataset] = metrics

    with open(total_results_path,'w', encoding='utf-8') as f:
        json.dump(total_results,f,indent=4)
                
    return metrics
        

datasets_map = {"OCR_Laboratory_Extract":"JDH_INSPECTION_hard_full_parsing",
                "OCR_Laboratory_SimpleQA":"JDH_INSPECTION_hard_simple_qa",
                "OCR_Laboratory_ComplexQA":"JDH_INSPECTION_hard_abnormality_qa",
                "OCR_Medical_SimpleQA":"JDH_ALLDOC_easyQ_hardImg_test_0828_100",
                "OCR_Medical_ComplexQA":"JDH_ALLDOC_hardQ_hardImg_test_0828_100",
                
                "OCR_Laboratory_Extract_v2":"JDH_INSPECTION_hard_full_parsing_0924_v2",
                "OCR_Laboratory_SimpleQA_v2":"JDH_INSPECTION_hard_simple_qa_0924_v2",
                "OCR_Laboratory_ComplexQA_v2":"JDH_INSPECTION_hard_abnormality_qa_0924_v2",
                "OCR_Medical_SimpleQA_v2":"JDH_ALLDOC_easyQ_hardImg_test_0924_100_v2",
                "OCR_Medical_ComplexQA_v2":"JDH_ALLDOC_hardQ_hardImg_test_0924_100_v2",

                "OCR_LTR_fullparsing":"LTR_fullparsing",
                "OCR_LTR_simpleQA":"LTR_simpleQA",
                "OCR_LTR_abnormalityQA":"LTR_abnormalityQA",
                "OCR_GMD_simpleQA":"GMD_simpleQA",
                "OCR_GMD_complexQA":"GMD_complexQA",
               }

model_map = {"Qwen2.5-VL":{"7":"Qwen2.5-VL-7B-Instruct",
                           "32":"Qwen2.5-VL-32B-Instruct",
                           "72":"Qwen2.5-VL-72B-Instruct",
                        },
             "HealthGPT":{"14":"HealthGPT-14B",
                        "32":"HealthGPT-32B",
                        },
             "HuatuoGPT":{"7":"HuatuoGPT-7B",
                        "34":"HuatuoGPT-34B",
                        },
             "MedGemma":{"4":"MedGemma-4B",
                        "27":"MedGemma-27B",
                        },
             "MedPLIB":{"7":"MedPLIB-7B",
                        },
             "Baichuan":{"32":"Baichuan-M2-32B",
                        },
             "InternVL3_5":{"241":"InternVL3_5-241B-A28B-Instruct",
                        },
             "Qwen3-VL":{"8":"Qwen3-VL-8B-Instruct",
                         "32":"Qwen3-VL-32B-Instruct",
                         "235":"Qwen3-VL-235B-A22B-Instruct",
                        },
             "Hulu":{"7":"Hulu-Med-7B",
                    "14":"Hulu-Med-14B",
                    "32":"Hulu-Med-32B",
                        },
             "LLava_OV":{"4":"LLaVA-OneVision-1.5-4B-Instruct",
                         "8":"LLaVA-OneVision-1.5-8B-Instruct",
                        },
             "Citrus_v":{"8":"Citrus_v-8B",
                         "32":"Citrus_v-32B",
                        },
             "JoyMed":{"8":"JoyMed-8B-v1.0",
                       "32":"JoyMed-32B-v1.0",
                      },
            }

def get_model_name(model_name, model_size):
    if model_name.lower().startswith('gpt') or model_name.lower().startswith('doubao'):
        return model_name
    if model_name not in model_map:
        raise ValueError(f"unknown model_name {model_name} in model_map")
    
    if str(model_size) not in model_map[model_name]:
        raise ValueError(f"unknown model_size {model_size} in model_map")

    return model_map[model_name][str(model_size)]
        
    
    
if __name__ == "__main__":
    
    # "JDH_INSPECTION_hard_full_parsing.tsv"
    # "JDH_INSPECTION_hard_simple_qa.tsv"
    # "JDH_INSPECTION_hard_abnormality_qa.tsv"
    # "JDH_ALLDOC_easyQ_hardImg_test_0828_100.tsv"
    # "JDH_ALLDOC_hardQ_hardImg_test_0828_100.tsv"
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen2.5-VL")
    parser.add_argument('--model_path', type=str, default="/mnt/workspace/offline/shared_model/Qwen2.5-VL-7B-Instruct")
    parser.add_argument('--dataset', type=str, default="OCR_Laboratory_SimpleQA")
    parser.add_argument('--limit', type=int, default=None, help='只评测前 N 条样本，不设则全量')
    parser.add_argument('--model_size', type=int, default=7)
    parser.add_argument('--nproc', type=int, default=10)
    parser.add_argument('--outputs_path', type=str, default="/mnt/workspace/offline/zhoufangru/agent/db_eval/outputs/0_Qwen2.5-VL-7B-Instruct-test/v1")
    parser.add_argument('--local_rank', type=int,default=0)
    parser.add_argument('--prompt_version', type=str,default='v1')

    args = parser.parse_args()

    model_name = get_model_name(args.model_name, args.model_size)
    
    model_path = args.model_path
    datasets = args.dataset.split(',')
    nproc = args.nproc
    
    if os.path.exists(args.prompt_version): 
        prompts_dict = json.load(open(args.prompt_version,'r'))
        os.environ["REASONING"] = str(prompts_dict.get('REASONING',False))

    for dataset in datasets:        
        # outputs_path = os.path.join(args.outputs_path, dataset)
        outputs_path = os.path.join(args.outputs_path, os.path.basename(args.prompt_version).replace('.json',''), dataset) \
                           if os.path.isfile(args.prompt_version) else os.path.join(args.outputs_path, args.prompt_version, dataset)
        
    
        import socket
        ip_adress = socket.gethostbyname(socket.gethostname())
        print(f"ip地址:{ip_adress}  结果保存路径:{outputs_path}")
    
        run_eval_OCR(model_name, model_path, dataset, nproc, outputs_path, limit=args.limit)