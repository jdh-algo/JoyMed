"""每个text样本格式
sample={"messages"：{"messages":prompt}} 此时prompt是list，里面是多轮对话
sample={"messages"：{"prompt":prompt}} 此时prompt是str，是question
"""

dataset_type = {
    "PMC_VQA": "image_text",
    "SLAKE": "image_text",
    "VQA_RAD": "image_text",
    "MedXpertQA-MM": "image_text",
    "PATH_VQA": "image_text",
    "MMMU-Medical-val": "image_text",
    "OmniMedVQA": "image_text",
    "MedFrameQA": "image_text",
    "GMAI-MMBench-val": "image_text",
    "GMAI-MMBench-test": "image_text",
    "CMMLU": "only_text",
    "PubMedQA": "only_text",
    "MedMCQA": "only_text",
    "MedQA_USMLE": "only_text",
    "MedQA_MCMLE": "only_text",
    "Medbullets_op4": "only_text",
    "Medbullets_op5": "only_text",
    "MedXpertQA-Text": "only_text",
    "SuperGPQA": "only_text",
    "CMB": "only_text",
    "CMExam": "only_text",
    "HealthBench": "only_text",
    "DiagnosisArena": "only_text",
    "CheXpert_Plus": "image_text",
    "IU_XRAY": "image_text",
    "CT-RATE-Report": "image_text",
    "CT-RATE-VQA": "image_text",
    "M3D-Slice": "image_text",
    "M3D-Ori": "image_text",
    "AMOS-MM-Report": "image_text",
    "AMOS-MM-VQA": "image_text",
    "AMOS-Ori-Report": "image_text",
    "Deeptumor-Report": "image_text",
    "Deeptumor-VQA": "image_text",
    "3D-RAD": "image_text",
    "3D-RAD-Ori": "image_text",
    "3D-RAD_ori": "image_text",
    "3D-MIR": "image_text",
    "INSPECT": "image_text",
    "Know2Do": "image_text",
}

system_prompt_v2_en = "You are an expert with extensive experience in medical research and clinical practice, and you are needed to assist users in answering questions related to medical issues. Let's think step by step."
system_prompt_v2_zh = (
    "您是一位在医学研究和临床实践方面拥有丰富经验的专家，需要您协助用户回答与医学相关的问题。让我们一步步思考。"
)

user_prompt_v2_onlytext_en = """
You are tasked with addressing a medical examination question. Please carefully read the question, provide a detailed thought process, and then present your final answer.
Here is the question:
<Question>
{prompt}
</Question>

Please begin your response.

The formatted output should be as follows:
<think>
[Insert Your Detailed Thought Process here]
</think>
<answer>
[Insert Your Final Answer here]
</answer>
"""
user_prompt_v2_onlytext_zh = """
您的任务是解答一道医学检查问题。请仔细阅读问题，提供详细的思考过程，然后给出您的最终答案。
问题如下：
<Question>
{prompt}
</Question>

请开始您的回答。

输出格式应如下：
<think>
[在此处插入您的详细思考过程]
</think>
<answer>
[在此处插入您的最终答案]
</answer>
"""

user_prompt_v2_imagetext_en = """
You are tasked with analysing a medical examination image. Please carefully check and analyse the image in detail, provide a detailed thought process, and then present your final answer.
Here is the question:<Question>
{prompt}
</Question>
Please begin your response.

The formatted output should be as follows:
[Insert Your Detailed Thought Process here]
<Answer>[Insert Your Final Answer here]</Answer>
"""
user_prompt_v2_imagetext_zh = """
你被要求分析一张医学检查图像。请仔细检查并详细分析这张图像，提供详细的思考过程，然后给出你的最终答案。
问题如下：<Question>
{prompt}
</Question>
请开始您的回答。

输出格式应如下：
[在此处插入您的详细思考过程]
<Answer>[在此处插入您的最终答案]</Answer>
"""

RAG_PROMPT = """You are a medical expert with extensive clinical and academic background, specializing in evidence-based reasoning and advanced medical knowledge. Answer the question based on the given information or your internal knowledge. 

Here is the question:<Question>
{prompt}
</Question>

Put your answer in tag <answer>, e.g., <answer> answer here </answer>."""

prompts_dict = {
    "v1": {
        "only_text": {
            "en": {"system_prompt": None, 
                   "CHOICE": "{prompt}\nAnswer with the option's letter from the given choices directly.", 
                   "CLOSED": "{prompt}\nAnswer the question using a single word or phrase.", 
                   "OPEN": "{prompt}\nPlease answer the question concisely.", 
                   "JUDGE": "{prompt}\nPlease output 'yes' or 'no'(no extra output).", 
                   "REPORT": "You are a helpful assistant. Please generate a report for the given images, including both findings and impressions. Return the report in the following format: Findings: {} Impression: {}."
                  },
            "zh": {"system_prompt": None, 
                   "CHOICE": "{prompt}\n请直接使用给定选项中的选项字母来回答该问题。", 
                   "CLOSED": "{prompt}\n请用一个单词或者短语回答该问题。", 
                   "OPEN": "{prompt}\n请简要回答该问题。", 
                   "JUDGE": "{prompt}\n请输出'是'或'否'(不要有任何其它输出)。", 
                   "REPORT": "生成一份报告。"
                  }
        }, 
        "image_text":{
            "en": {"system_prompt": None, 
                   "CHOICE": "{prompt}\nAnswer with the option's letter from the given choices directly.", 
                   "CLOSED": "{prompt}\nAnswer the question using a single word or phrase.", 
                   "OPEN": "{prompt}\nPlease answer the question concisely.", 
                   "JUDGE": "{prompt}\nPlease output 'yes' or 'no'(no extra output).", 
                   "REPORT": "You are a helpful assistant. Please generate a report for the given images, including both findings and impressions. Return the report in the following format: Findings: {} Impression: {}."
                  },
            "zh": {"system_prompt": None, 
                   "CHOICE": "{prompt}\n请直接使用给定选项中的选项字母来回答该问题。", 
                   "CLOSED": "{prompt}\n请用一个单词或者短语回答该问题。", 
                   "OPEN": "{prompt}\n请简要回答该问题。", 
                   "JUDGE": "{prompt}\n请输出'是'或'否'(不要有任何其它输出)。", 
                   "REPORT": "生成一份报告。"
                  }
        }
    },
    "v2": {
        "only_text": {
            "en": {"system_prompt": system_prompt_v2_en, "user_prompt": user_prompt_v2_onlytext_en},
            "zh": {"system_prompt": system_prompt_v2_zh, "user_prompt": user_prompt_v2_onlytext_zh},
        },
        "image_text": {
            "en": {"system_prompt": system_prompt_v2_en, "user_prompt": user_prompt_v2_imagetext_en},
            "zh": {"system_prompt": system_prompt_v2_zh, "user_prompt": user_prompt_v2_imagetext_zh},
        },
    },
    "rag_bench": {
        "only_text": {
            "en": {"system_prompt": None, "user_prompt": RAG_PROMPT},
            "zh": {"system_prompt": None, "user_prompt": None},
        },
        "image_text": {
            "en": {"system_prompt": None, "user_prompt": RAG_PROMPT},
            "zh": {"system_prompt": None, "user_prompt": None},
        },
    },
}
