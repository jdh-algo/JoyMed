# This script copies tokenizer files from a base pretrained Llama model directory into a specific checkpoint directory.
# This is typically required when the checkpoint directory does not contain essential tokenizer files,
# allowing downstream inference or evaluation to work without missing file errors.


BASE=/mnt/workspace/offline/xiehuidong.6688/M3D/LaMed/pretrained_model/huggingface_models/Meta-Llama-3.1-8B-Instruct
CKPT=/mnt/workspace/offline/xiehuidong.6688/M3D/LaMed/output/bridge_only/LaMed-Llama3.1-8B-finetune-MAE-CTRATE-no-compress_checkpoint-96000_v3_bridge_only/checkpoint-7000

cp $BASE/tokenizer.json \
   $BASE/tokenizer_config.json \
   $BASE/special_tokens_map.json \
   $BASE/original/tokenizer.model \
   $CKPT/