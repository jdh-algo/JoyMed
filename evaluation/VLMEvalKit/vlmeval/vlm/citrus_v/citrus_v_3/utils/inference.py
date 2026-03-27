from architectures import citrus_v_3
from transformers import AutoModelForCausalLM, AutoProcessor

# Load model and processor
model_path = "/mnt/workspace/offline/caoxuyang5/code/ms-swift-390/output/v202-20251104-221939/checkpoint-6000"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype="bfloat16",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_3d",
                "image_3d": "/mnt/workspace/offline/shared_data/Medical_Data_3D/M3D-CAP/M3D_Cap_nii/ct_quizze/005322/Axial_C__portal_venous_phase.nii.gz",
            },
            {"type": "text", "text": "Question:\nWhere does the narrowed transition point, where the small bowel enters/exits the cluster, occur?\nOptions:\nA) Left lower quadrant\nB) Right lower quadrant\nC) Left upper quadrant\nD) Right upper quadrant"},
        ],
    },
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(f"===output: \n {output_text}")