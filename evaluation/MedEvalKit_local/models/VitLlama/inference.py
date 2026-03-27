import math
import vit_lamma
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor


def _align_config_vocab_size_for_checkpoint(config, tokenizer):
    """
    Align config.llm_config.vocab_size with tokenizer (for old checkpoints saved
    before we synced vocab_size at save time). Supports llm_config as dict or object.
    """
    if not getattr(config, "llm_config", None):
        return
    vocab_from_tokenizer = len(tokenizer)
    lc = config.llm_config
    llm_vocab = lc.get("vocab_size") if isinstance(lc, dict) else getattr(lc, "vocab_size", None)
    if llm_vocab is None or vocab_from_tokenizer <= llm_vocab:
        return
    new_vocab = math.ceil(vocab_from_tokenizer / 128) * 128
    if isinstance(lc, dict):
        lc["vocab_size"] = new_vocab
    else:
        setattr(lc, "vocab_size", new_vocab)
    print(f"[inference] Aligned llm_config.vocab_size for checkpoint: {llm_vocab} -> {new_vocab}")


# Load model and processor

model_path = "/mnt/workspace/offline/tangwen.32/citrus_swift/models/vit_lamma/v51-20260209-211145/checkpoint-3000"

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
_align_config_vocab_size_for_checkpoint(config, processor.tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    trust_remote_code=True,
    torch_dtype="bfloat16",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model.eval()

# Critical: model must use the same image_3d placeholder token id as the tokenizer.
# If they differ, get_placeholder_mask() finds no positions and no visual tokens are injected → gibberish.
pad_token_id_in_vocab = processor.tokenizer.convert_tokens_to_ids(
    getattr(processor, "image_3d_token", "<|image_3d_pad|>")
)
if getattr(model.config, "image_3d_token_id", None) != pad_token_id_in_vocab:
    model.config.image_3d_token_id = pad_token_id_in_vocab
    print(f"[inference] Synced model.config.image_3d_token_id = {pad_token_id_in_vocab}")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_3d",
                "image_3d": "/mnt/workspace/offline/shared_data/CT-RATE/valid_fixed/valid_999/valid_999_a/valid_999_a_1.nii.gz",
            },
            {"type": "text", "text": "Generate a radiology report for this medical image."},
        ],
    },
]

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image_3d",
#                 "image_3d": "/mnt/workspace/offline/shared_data/Medical_Data_3D/M3D-CAP/M3D_Cap_nii/ct_quizze/005322/Axial_C__portal_venous_phase.nii.gz",
#             },
#             {"type": "text", "text": "Question:\nWhere does the narrowed transition point, where the small bowel enters/exits the cluster, occur?\nOptions:\nA) Left lower quadrant\nB) Right lower quadrant\nC) Left upper quadrant\nD) Right upper quadrant"},
#         ],
#     },
# ]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# 调试：确认 generate 能收到视觉输入（若 keys 里没有 pixel_values_volumes/volume_grid_thw，则 forward 里 has_visual 会为 False）
_input_keys = list(inputs.keys()) if hasattr(inputs, "keys") else list(getattr(inputs, "data", {}).keys())
print(f"[inference] inputs keys: {_input_keys}")

# Generation config: use model's native EOS (<|eot_id|>) and let it decide when to stop.
tokenizer = processor.tokenizer
eos_id = tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id
if pad_id is None:
    pad_id = eos_id

generated_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,
    eos_token_id=eos_id,
    pad_token_id=pad_id,
)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(f"===output: \n {output_text}")