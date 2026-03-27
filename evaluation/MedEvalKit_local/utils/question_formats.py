from utils.trainfree_prompt import prompt_dict


def get_multiple_choice_prompt(question,choices,is_reasoning = False,lang = "en", type="2D"):
    choices = [str(choice) for choice in choices]
    options = "\n".join(choices)

    if lang == "en":
        if type=="2D":
            prompt = f"""Question: {question}\nOptions: {options}"""
        else:
            prompt = f"""{question}{ "".join(choices)}"""
        if is_reasoning:
            # Lingshu
            # prompt = prompt + "\n" + 'Answer with the option\'s letter from the given choices and put the letter in one "\\boxed{}".'
            # 添加CoT
            prompt = prompt + "\n" + prompt_dict['multipleChoice'].get('Lingshu')
        # else:
        #     prompt = prompt + "\n" + "Answer with the option's letter from the given choices directly." 

    elif lang == "zh":
        prompt = f"""问题： {question}\n选项： {options}"""
        if is_reasoning:
            # Lingshu
            # prompt = prompt + "\n" + '请直接使用给定选项中的选项字母来回答该问题,并将答案包裹在"\\boxed{}"里'
            # 添加CoT
            prompt = prompt + "\n" + "请直接使用给定选项中的选项字母来回答该问题。" + """为了得到正确答案，你应该：首先，仔细分析图像和问题，并逻辑而有条理地评估每个答案选项。接着，像医学专家一样逐步推理，结合视觉和文本线索，得出符合临床逻辑的结论。推理过程应简洁、清晰且经得起临床验证。"""

        # else:
        #     prompt = prompt + "\n" +  "请直接使用给定选项中的选项字母来回答该问题。"
    # print(f">>>[get_multiple_choice_prompt]\n is_reasoning:{is_reasoning}\tprompt:{prompt}")
    return prompt

def get_judgement_prompt(question,is_reasoning = False, lang = "en"):
    if lang == "en":
        if is_reasoning:
            # Lingshu
            # prompt = question + "\n" + 'Please output "yes" or "no" and put the answer in one "\\boxed{}".'
            # 添加CoT
            prompt = question + "\n" + prompt_dict['judgement'].get('Lingshu')
        else:
        #     prompt = question + "\n" + "Please output 'yes' or 'no'(no extra output)."
            prompt = question

    elif lang == "zh":
        if is_reasoning:
            # Lingshu
            # prompt = question + "\n" + "请输出'是'或'否'，并将答案放在一个'\\boxed{}'中。"
            # 添加CoT
            prompt = question + "\n" + "请输出'是'或'否'(不要有任何其它输出)。" + """为了得到正确答案，你应该：首先，仔细分析图像和问题。接着，像医学专家一样逐步推理，结合视觉和文本线索，得出符合临床逻辑的结论。推理过程应简洁、清晰且经得起临床验证。"""
        else:
        #     prompt = question + "\n" + "请输出'是'或'否'(不要有任何其它输出)。"
            prompt = question

    # print(f">>>[get_judgement_prompt]\n is_reasoning:{is_reasoning}\tprompt:{prompt}")
    return prompt

def get_close_ended_prompt(question,is_reasoning = False, lang = "en"):
    if lang == "en":
        if is_reasoning:
            # Lingshu
            # prompt = question + "\n" + 'Answer the question using a single word or phrase and put the answer in one "\\boxed{}".'
            # 添加CoT
            prompt = question + "\n" + prompt_dict['closeEnded'].get('Lingshu')
        else:
        #     prompt = question + "\n" + "Answer the question using a single word or phrase."
            prompt = question

    elif lang == "zh":
        if is_reasoning:
            # Lingshu
            # prompt = question + "\n" + "请用一个单词或者短语回答该问题，并将答案放在一个'\\boxed{}'中。"
            # 添加CoT
            prompt = question + "\n" + "请用一个单词或者短语回答该问题。" + """在回答问题之前，你应该：首先，仔细分析图像和问题。接着，像医学专家一样逐步推理，结合视觉和文本线索，得出符合临床逻辑的结论。推理过程应简洁、清晰且经得起临床验证。"""
        else:
        #     prompt = question + "\n" + "请用一个单词或者短语回答该问题。"
            prompt = question

    # print(f">>>[get_close_ended_prompt]\n is_reasoning:{is_reasoning}\tprompt:{prompt}")
    return prompt

def get_open_ended_prompt(question,is_reasoning = False, lang = "en"):
    if lang == "en":
        if is_reasoning:
            # Lingshu
            # prompt = question + "\n" + 'Please answer the question concisely and put the answer in one "\\boxed{}".'
            # 添加CoT
            prompt = question + "\n" + prompt_dict['openEnded'].get('Lingshu')
        else:
        #     prompt = question + "\n" + "Please answer the question concisely."
            prompt = question

    elif lang == "zh":
        if is_reasoning:
            # Lingshu
            # prompt = question + "\n" + "请简要回答该问题，并将答案放在一个'\\boxed{}'中。"
            # 添加CoT
            prompt = question + "\n" + "请简要回答该问题。" + """为了得到正确答案，你应该：首先，仔细分析图像和问题。接着，像医学专家一样逐步推理，结合视觉和文本线索，得出符合临床逻辑的结论。推理过程应简洁、清晰且经得起临床验证。"""

        else:
        #     prompt = question + "\n" + "请简要回答该问题。"
            prompt = question

    # print(f">>>[get_open_ended_prompt]\n is_reasoning:{is_reasoning}\tprompt:{prompt}")
    return prompt

# def get_report_generation_prompt():
#     # Lingshu
#     # prompt = "You are a helpful assistant. Please generate a report for the given images, including both findings and impressions. Return the report in the following format: Findings: {} Impression: {}."
#     # 添加CoT
#     prompt = "You are a helpful assistant. Please generate a report for the given images, including both findings and impressions. Return the report in the following format: Findings: {} Impression: {}." + """To achieve exactly answer, you should，First, think between <think> and </think> while output necessary coordinates needed to answer the question in JSON with key ’bbox_2d’. Then, based on the thinking contents and coordinates, rethink between <rethink> </rethink> and then answer the question between <answer> </answer>."""
    
#     # print(">>>[get_report_generation_prompt]: ", prompt)
    
#     return prompt

def get_report_generation_prompt(is_reasoning=False, question_version="Lingshu", lang="en"):
    # if lang == "en":
    #     if is_reasoning:
    #         prompt = prompt_dict['reportGen'][question_version].get("en") + " Let’s think step by step."
    #     else:

    #         if question_version=="Lingshu":
    #             prompt = "You are a helpful assistant. Please generate a report for the given images, including both findings and impressions. Return the report in the following format: Findings: {} Impression: {}."
            
    #         elif question_version=="DerectReport":
    #             prompt = "You are a helpful assistant. Please generate a report for the given images."

    #         else:
                
    #             prompt = prompt_dict['reportGen'][question_version].get("en")

    # elif lang == "zh":
    #     if is_reasoning:
    #         prompt = prompt_dict['reportGen'][question_version].get("zh") + " 让我们逐步分析。"  
    #     else:
    #         prompt = prompt_dict['reportGen'][question_version].get("zh")

    # # print(">>>[get_report_generation_prompt]: ", prompt)
    # return prompt
    return ""
