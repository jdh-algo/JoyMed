#!/usr/bin/env python3
"""
Dataset configurations for custom evaluation datasets.
"""

DATASET_CONFIGS = {
    # Custom VQA datasets with their own evaluation scripts
    "LTR_fullparsing": {
        "custom_script": "mm_pipeline/mm_eval/data_configs/JDH_INSPECTION_full_parsing.py",
        "custom_eval_class": "LTR_fullparsing",
    },
    "LTR_abnormalityQA": {
        "custom_script": "mm_pipeline/mm_eval/data_configs/JDH_INSPECTION_abnormality_qa.py",
        "custom_eval_class": "LTR_abnormalityQA",
    },
    "LTR_simpleQA": {
        "custom_script": "mm_pipeline/mm_eval/data_configs/JDH_INSPECTION_simple_qa.py",
        "custom_eval_class": "LTR_simpleQA",
    },
    "GMD_simpleQA": {
        "custom_script": "mm_pipeline/mm_eval/data_configs/JDH_ALLDOC_qa.py",
        "custom_eval_class": "GMD_simpleQA",
    },
    "GMD_complexQA": {
        "custom_script": "mm_pipeline/mm_eval/data_configs/JDH_ALLDOC_qa.py",
        "custom_eval_class": "GMD_complexQA",
    },
}


def get_dataset_list():
    """Get list of all available datasets."""
    return list(DATASET_CONFIGS.keys())


def get_custom_datasets():
    """Get list of datasets that have custom evaluation scripts."""
    return [k for k, v in DATASET_CONFIGS.items() if v["custom_script"] is not None]


def get_standard_datasets():
    """Get list of standard VLMEvalKit datasets."""
    return [k for k, v in DATASET_CONFIGS.items() if v["custom_script"] is None]
