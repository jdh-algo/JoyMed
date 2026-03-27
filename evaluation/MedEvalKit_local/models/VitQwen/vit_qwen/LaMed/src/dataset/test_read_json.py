import json

#with open("/mnt/workspace/offline/shared_data/Medical_Data_3D/M3D-CAP/M3D_Cap.json", 'r') as file:
#        json_file = json.load(file)
#data_list = json_file[mode]
#print(json_file["train"][0]["image"])

# with open("/mnt/workspace/offline/shared_data/Medical_Data_3D/CT-RATE/dataset/01_processed_jdh/train_ct_rate_caption_full_47k_251114.jsonl", 'r') as file:
#         json_file = json.load(file)
# print(json_file)

json_file = []
with open("/mnt/workspace/offline/shared_data/Medical_Data_3D/M3D-CAP/01_process_jdh/train_m3d_cap_caption_revised_filtered_93k_1117.jsonl", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        json_file.append(obj)
        
print(json_file[0])
print(json_file[0]['conversations'][1]['value'])

report = json_file[0]['conversations'][1]['value']