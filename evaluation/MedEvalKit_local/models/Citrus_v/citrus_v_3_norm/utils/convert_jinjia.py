import os
import json
from transformers import AutoTokenizer

jinja_path = "/mnt/workspace/offline/caoxuyang5/code/ms-swift-390/output/v110-20251029-153218/checkpoint-6602/chat_template.jinja"
tokenizer_path = "/mnt/workspace/offline/caoxuyang5/code/ms-swift-390/output/v110-20251029-153218/checkpoint-6602"
save_path = "./test_tokenizer"


# load
with open(jinja_path, "r", encoding="utf-8") as f:
    custom_template = f.read()
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.chat_template = custom_template


# save
tokenizer.save_pretrained(save_path)
template_data = {
    "chat_template": custom_template
}
with open(os.path.join(save_path, "chat_template.json"), "w", encoding="utf-8") as f:
    json.dump(template_data, f, ensure_ascii=False, indent=2)