#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com


EVAL_DATASETS=$4
PROMPT_VERSION=$5
DATASETS_PATH=$6

MODEL_NAME="API"
SERVED_MODEL=$1 #"gpt-4.1" "gpt-5" "Doubao-Seed-1.6"

MODEL_PATH="$2" #""
OUTPUT_PATH="$7/$3/$5"


#vllm setting
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
TENSOR_PARALLEL_SIZE="1"
USE_VLLM="False"
IFS='.' read -r -a GPULIST <<< "$CUDA_VISIBLE_DEVICES"
# TOTAL_GPUS=${#GPULIST[@]}
NUM_CHUNKS=10
# CHUNKS=`expr ${CHUNKS:-$TOTAL_GPUS} / $GPU_NUM_PERINSTANCE`

#Eval setting
SEED=42
REASONING="False"
TEST_TIMES=1


# Eval LLM setting
MAX_NEW_TOKENS=5000
MAX_IMAGE_NUM=6
TEMPERATURE=1
TOP_P=0.0001
REPETITION_PENALTY=1

# LLM judge setting
USE_LLM_JUDGE="True"
# gpt api model name
GPT_MODEL="gpt-4.1" # gpt-4.1-2025-04-14
OPENAI_API_KEY=""


# 在NUM_CHUNKS个实例上并行运行
for ((i=0; i<NUM_CHUNKS; i++)); do
    setsid python eval_online.py \
        --eval_datasets "$EVAL_DATASETS" \
        --datasets_path "$DATASETS_PATH" \
        --output_path "$OUTPUT_PATH" \
        --model_name "$MODEL_NAME" \
        --served_model_name "$SERVED_MODEL"\
        --seed $SEED \
        --cuda_visible_devices "$CUDA_VISIBLE_DEVICES" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --use_vllm "$USE_VLLM" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --max_image_num "$MAX_IMAGE_NUM" \
        --temperature "$TEMPERATURE"  \
        --top_p "$TOP_P" \
        --repetition_penalty "$REPETITION_PENALTY" \
        --reasoning "$REASONING" \
        --use_llm_judge "$USE_LLM_JUDGE" \
        --judge_gpt_model "$GPT_MODEL" \
        --openai_api_key "$OPENAI_API_KEY" \
        --test_times "$TEST_TIMES" \
        --num_chunks "$NUM_CHUNKS" \
        --chunk_idx $i \
        &
        
done

# 等待所有进程完成
wait
