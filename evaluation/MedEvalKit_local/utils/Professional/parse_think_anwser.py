import os
import sys
import re
import pdb



def load_files(source_dir):
    # 遍历模型文件夹中的所有txt文件
    source_files_path = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if os.path.splitext(file)[-1] in ['.txt']:
                file_path = os.path.join(root, file)
                source_files_path.append(file_path)
    return source_files_path


def get_content_lst_between_a_b(start_tag, end_tag, text):
    extracted_text = []
    start_index = text.find(start_tag)
    while start_index != -1:
        end_index = text.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            extracted_text.append(text[start_index + len(start_tag) : end_index].strip())
            start_index = text.find(start_tag, end_index + len(end_tag))
        else:
            break

    return extracted_text


def extract_option(text):
    '''
    ABCD or A.5ml
    '''
    # char_num = 0
    first_char = -1
    char_set = set()
    for char in text:
        if char in 'ABCDEFGHIJK':
            char_set.add(char)
            if first_char == -1:
                first_char = char
            # char_num += 1
    
    if len(char_set) == len(text):
        return text
    elif first_char != -1:
        return first_char
    else:
        text


def process_single_line(text):

    pred_anwsers = get_content_lst_between_a_b('[', ']', text)
    # pdb.set_trace()


    # ABCD or A.5ml or A,B,C or A.5ml,B.6ml
    if len(pred_anwsers) == 0:
        pred_anwsers = [text]

    # print(f'pred_anwsers: {pred_anwsers}')

    # [A,B,C] or [A][B][C] or [A.5ml][B.6ml][A.5ml,B.6ml]
    split_answers = []
    for i, pred_anwser in enumerate(pred_anwsers):
        flag = False
        for split_char in [',', '、', '，', ' ']:
            if split_char in pred_anwser:
                split_answers += pred_anwser.split(split_char)
                flag = True
        
        if not flag:
            split_answers.append(pred_anwser)
            

    # print(f'split_answers: {split_answers}')
    # pdb.set_trace()
    ret = set()
    for split_answer in split_answers:
        # print(f'split_answer: {split_answer}')
        options = extract_option(split_answer)
        if options != None:
        # print(f'after split_answer: {options}')
            ret.add(options)
    
    # print(f'ret: {ret}')

    return ret


def parser_answer(chat_response):
    if '</think>' in chat_response:
        chat_response = chat_response.split('</think>')[-1]
        
    # 解析结果
    chat_response_item = chat_response.split('【答案】')
    # pdb.set_trace()
    if len(chat_response_item) > 1:
        chat_response = chat_response_item[1].replace('：', ':').replace(':', '').strip()
    elif len(chat_response_item) == 1:
        chat_response = chat_response_item[0]
    else:
        return 'EMPTY'

    ret = set()
    f_lines = chat_response.split('\n')
    for i, f_line in enumerate(f_lines):
        f_line = f_line.strip()
        if len(f_line) != 0:
            ret.update(process_single_line(f_line))

    if len(ret) > 0:
        sorted_ret = sorted(ret)
        return ''.join(sorted_ret)
    else:
        return 'EMPTY'


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'python {__file__} res_dir save_file')
        sys.exit()

    res_dir = sys.argv[1].rstrip('/')
    save_file = sys.argv[2].rstrip('/')

    res_files = load_files(res_dir)

    f_w = open(f'{save_file}', 'w')
    for i_file in range(1, len(res_files)+1):
        res_file = os.path.join(res_dir, f'{i_file}.txt')
        # print(res_file)
        with open(res_file) as f:
            chat_response = f.read()

        norm_res = parser_answer(chat_response)

        f_w.write(f'{norm_res}\n')
        # break
    f_w.close()

