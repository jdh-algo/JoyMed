#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
# MMMU-Medical-test,MMMU-Medical-val,PMC_VQA,MedQA_USMLE,MedMCQA,PubMedQA,OmniMedVQA,Medbullets_op4,Medbullets_op5,MedXpertQA-Text,MedXpertQA-MM,SuperGPQA,HealthBench,IU_XRAY,CheXpert_Plus,MIMIC_CXR,CMB,CMExam,CMMLU,MedQA_MCMLE,VQA_RAD,SLAKE,PATH_VQA,MedFrameQA
# EVAL_DATASETS="PMC_VQA,SLAKE,VQA_RAD,MedXpertQA-MM,PATH_VQA,MMMU-Medical-val,PubMedQA,MedMCQA,MedQA_USMLE,MedXpertQA-Text,CMMLU,MedQA_MCMLE,Medbullets_op4,Medbullets_op5,SuperGPQA,CMB,CMExam,CheXpert_Plus,IU_XRAY,HealthBench,MedFrameQA"
EVAL_DATASETS=$4
PROMPT_VERSION=$5
DATASETS_PATH=$8

MODEL_NAME=$1 # Qwen2.5-VL ｜ InternVL | LingShu |
MODEL_PATH="$2" # Qwen2.5-VL-7B-Instruct | Qwen2.5-VL-32B-Instruct | InternVL3-8B | Lingshu-7B | Lingshu-32B ｜ InternVL3-38B
OUTPUT_PATH="$9/$3"

#vllm setting
CUDA_VISIBLE_DEVICES=$6
USE_VLLM="False"
IFS='.' read -r -a GPULIST <<< "$CUDA_VISIBLE_DEVICES"
# TOTAL_GPUS=${#GPULIST[@]}
CHUNKS=$7
# CHUNKS=`expr ${CHUNKS:-$TOTAL_GPUS} / $GPU_NUM_PERINSTANCE`

#Eval setting
SEED=42
REASONING="False"
TEST_TIMES=1


# Eval LLM setting
MAX_NEW_TOKENS=8192
MAX_IMAGE_NUM=6
TEMPERATURE=0
TOP_P=0.0001
REPETITION_PENALTY=1

# LLM judge setting
USE_LLM_JUDGE="True"
# gpt api model name
GPT_MODEL="gpt-4.1" # gpt-4.1-2025-04-14
OPENAI_API_KEY=""


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python eval.py \
    --eval_datasets "$EVAL_DATASETS" \
    --datasets_path "$DATASETS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --seed $SEED \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_image_num "$MAX_IMAGE_NUM" \
    --use_vllm "$USE_VLLM" \
    --num_chunks $CHUNKS \
    --prompt_version $PROMPT_VERSION \
    --chunk_idx $IDX \
    --reasoning $REASONING \
    --temperature "$TEMPERATURE"  \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --use_llm_judge "$USE_LLM_JUDGE" \
    --judge_gpt_model "$GPT_MODEL" \
    --openai_api_key "$OPENAI_API_KEY" \
    --test_times "$TEST_TIMES"  &  # 这里的反斜杠后面不要有空格
done

wait
