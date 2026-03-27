import os
import sys
import csv
import time
import base64
from openai import OpenAI
from service_conf import model_conf
from parse_think_anwser import parser_answer


prompts = {
    'X': '请详细阅读上述题目，选择多个正确答案，给出对应答案选项，不要输出其它任何信息。请按照下面格式给出：\n【答案】:[选项]',
    'A1': '请详细阅读上述题目，选择一个最佳答案，给出对应答案选项，不要给其他任何信息。请按照下面格式给出：\n【答案】:[选项]',
    'A2': '请详细阅读上述题目，选择一个最佳答案，给出对应答案选项，不要给其他任何信息。请按照下面格式给出：\n【答案】:[选项]',
    'A3': '请详细阅读上述题目，选择一个最佳答案，给出对应答案选项，不要给其他任何信息。请按照下面格式给出：\n【答案】:[选项]',
    'B': '请详细阅读上述题目，选择一个最佳答案，给出对应答案选项，不要给其他任何信息。请按照下面格式给出：\n【答案】:[选项]',
    'other': '请详细阅读上述题目，选择一个或多个正确答案，给出对应答案选项，不要输出其它任何信息。请按照下面格式给出：\n【答案】:[选项]',
    # 'std': '请详细阅读上述题目，给出对应答案选项，不要给其他任何信息。请按照下面格式给出：\n【答案】:[选项]'
}


def read_byte(file_path):
    with open(file_path, "rb") as f:
        img_byte = f.read()
    encoded_img_byte = base64.b64encode(img_byte).decode("utf-8")
    return encoded_img_byte


def req(client, model_args, ocr_res, prompt, think_prompt='', imgs_data=[]):
    messages=[
        {
            "role": "system",
            # "content": "You are a helpful assistant."
            "content": "你是医学专家"
        },
        {
            "role": "user",
            "content": [],
        },
    ]

    if len(imgs_data) > 0:
        for img_data in imgs_data:
            messages[1]['content'].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_data}"},
            })
    
    messages[1]['content'].append({
        "type": "text",
        "text": f'{ocr_res}\n{prompt}'
    })


    try:
        start_time = time.time()
        if 'Authorization' in model_args.keys():
            chat_response = client.chat.completions.create(
                model=model_args['model_name'],
                messages=messages,
                temperature=model_args['temperature'],
                extra_headers={
                    "Authorization":  f"Bearer {model_args['Authorization']}"
                }
            )
        else:
            chat_response = client.chat.completions.create(
                model=model_args['model_name'],
                messages=messages,
                temperature=model_args['temperature'],
            )
        take_time = time.time() - start_time
        result = chat_response.choices[0].message.content
        print(f'time: {take_time:.3f} s\tprompt_tokens: {chat_response.usage.prompt_tokens}\tcompletion_tokens: {chat_response.usage.completion_tokens}')
    except:
        import traceback
        traceback.print_exc()
        return 'error'

    return result


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print(f'python {__file__} excel_csv save_file model_type img_dir')
        sys.exit()

    excel_csv = sys.argv[1].rstrip('/')
    save_file = sys.argv[2].rstrip('/')
    model_type = sys.argv[3].rstrip('/')
    img_dir = sys.argv[4].rstrip('/')

    model_args = model_conf.get(model_type, None)
    if model_args == None:
        print(f'无效的model_type')
        sys.exit()

    src_res_dir = f'{save_file}.src_res'
    os.makedirs(src_res_dir, exist_ok=True)

    client = OpenAI(
        api_key=model_args['api_key'],
        base_url=model_args['base_url'],
        timeout=600,
        max_retries=3,
    )


    data_num = 0
    with open(excel_csv, mode='r', encoding='utf-8', newline='') as file:
        csv_reader = csv.reader(file)

        for i_line, excel_line in enumerate(csv_reader):
            # 跳过首行
            if i_line == 0:
                continue
            data_num += 1
            print(excel_line)
            # 若原始结果存在，且原结果请求无错，则跳过
            save_src_res_file = f'{src_res_dir}/{i_line}.txt'
            if os.path.exists(save_src_res_file):
                with open(save_src_res_file) as f:
                    f_data = f.read().strip()
                if f_data not in ['error', 'EMPTY']:
                    continue

            # 组合问题
            question_type = excel_line[4]
            question = excel_line[5]
            option = excel_line[6]
            imgs_name = excel_line[11]

            input = f'{question}。选项: {option}'
            print(f'第{i_line}行.\ninput: {input}')

            # 选取prompt
            prompt = prompts['other']
            for key, tmp_prompt in prompts.items():
                if key in question_type:
                    prompt = tmp_prompt
                    break
            post_prompt = model_args.get('post_prompt', '')
            prompt = f'{prompt}{post_prompt}'

            imgs_data = []
            if len(imgs_name) != 0:
                img_name_lst = imgs_name.split(';')
                for img_name in img_name_lst:
                    img_path = os.path.join(img_dir, img_name) + '.jpg'
                    if os.path.exists(img_path):
                        img_data = read_byte(img_path)
                        imgs_data.append(img_data)
                    else:
                        print(f'Warning: image not found ! {img_path}')

            chat_response = req(client, model_args, input, prompt, imgs_data=imgs_data)
            print(f'原始结果: \n{chat_response}')

            # 保存原始结果
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            with open(save_src_res_file, 'w') as f_w:
                f_w.write(f'{chat_response}')


    f_w = open(f'{save_file}', 'w')
    for i_file in range(1, data_num+1):
        save_src_res_file = f'{src_res_dir}/{i_file}.txt'
        # print(res_file)
        with open(save_src_res_file) as f:
            chat_response = f.read()

        norm_res = parser_answer(chat_response)
        f_w.write(f'{norm_res}\n')
        # break
    f_w.close()
