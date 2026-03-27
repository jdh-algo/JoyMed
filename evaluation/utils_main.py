import os
import json
import pynvml
pynvml.nvmlInit()
gpuDeviceCount = pynvml.nvmlDeviceGetCount()

def cal_gpu(model_size, dataset):
    #返回单个实例需要gpu卡数（L20-46G）,限制1/2/4/8
    model_size = int(model_size)
    if model_size <20:
        gpu_num_perinstance = 1
    elif model_size <40:
        gpu_num_perinstance = 2
    elif model_size <80:
        gpu_num_perinstance = 4
    else:
        gpu_num_perinstance = 8

    if "MedFrameQA" in dataset and model_size>15:
        gpu_num_perinstance *= 2
        
    if gpu_num_perinstance > gpuDeviceCount:
        print("gpu 不足 ！")
        return False,"",0
    
    gpuDeviceStr = ""
    for i in range(gpuDeviceCount):
        if i % gpu_num_perinstance ==gpu_num_perinstance-1:
            gpuDeviceStr += f'{i}.'
        else:
            gpuDeviceStr += f'{i},'
    gpuDeviceStr = gpuDeviceStr[:-1]
    chunks = int(gpuDeviceCount/gpu_num_perinstance)
    return True, gpuDeviceStr, chunks

def start_running(judge_info=None):
    judge_info = judge_info.replace('/','_')
    file_root = "running_file"
    if os.path.exists(file_root):
        running_file_list = os.listdir(file_root)
    else:
        running_file_list = []
    
    print("running_file_list: ",running_file_list)
    if judge_info is not None:
        if judge_info in running_file_list:
            #正在运行，返回False表示不要再重复run了
            return False
        else:
            os.makedirs(os.path.join(file_root, judge_info), exist_ok=True)
            return True
def end_running(judge_info):
    judge_info = judge_info.replace('/','_')
    file_root = "running_file"
    file_name = os.path.join(file_root, judge_info)
    if os.path.exists(file_name):
        os.rmdir(file_name)
    
    

if __name__ == "__main__":
    print(start_running("30026_stage123-600k-43k-v4.6.1-ep3/v1/CMMLU"))
    print(end_running("30026_stage123-600k-43k-v4.6.1-ep3/v1/CMMLU"))