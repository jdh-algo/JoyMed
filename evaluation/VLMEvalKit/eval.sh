#!/bin/bash

EVAL_DATASETS=$4
PROMPT_VERSION=$5

DATASETS_PATH=$8

MODEL_NAME=$1 # Qwen2.5-VL ｜ InternVL | LingShu |
MODEL_PATH="$2" # Qwen2.5-VL-7B-Instruct | Qwen2.5-VL-32B-Instruct | InternVL3-8B | Lingshu-7B | Lingshu-32B ｜ InternVL3-38B

OUTPUT_PATH="$9/$3"
CHUNKS=$7
#vllm setting
MODEL_SIZE=$6

if [ $MODEL_NAME == "MedPLIB" ]
then
    TRANSFORMERS_OFFLINE=1 deepspeed --include=localhost:1 --master_port=64996 eval.py \
            --model_name "$MODEL_NAME" \
            --model_path "$MODEL_PATH" \
            --dataset $EVAL_DATASETS \
            --nproc "$CHUNKS" \
            --model_size "$MODEL_SIZE" \
            --prompt_version $PROMPT_VERSION \
            --outputs_path "$OUTPUT_PATH"

else
    python eval.py \
            --model_name "$MODEL_NAME" \
            --model_path "$MODEL_PATH" \
            --dataset $EVAL_DATASETS \
            --nproc "$CHUNKS" \
            --model_size "$MODEL_SIZE" \
            --prompt_version $PROMPT_VERSION \
            --outputs_path "$OUTPUT_PATH"
fi