prompt_dict = {
    "multipleChoice":{
        "localization":"""You are an experienced doctor. Next, you will be answering a multiple-choice question of the lesion localization type. Please strictly adhere to the following steps to conduct an in-depth analysis of the textual and imaging information provided by the question and answer the question asked.

Step 1: If the question provides textual information related to the patient's symptoms, analyze the potential disease location based on the given textual information, combined with clinical reality and your extensive clinical knowledge.

Step 2: Briefly observe the medical imaging picture provided by the question. Based on the characteristics of different medical imaging modalities, determine what type of imaging modality it is (e.g., X-ray, CT, MRI, skin photo).

Step 3: Based on the characteristics and anatomical structures of different body parts under this imaging modality, identify the body part shown in the medical imaging picture (e.g., This is a PA chest X-ray; This is an axial abdominal CT).

Step 4: Synthesize the above analysis results and, based on the anatomical features in the image, such as vertebrae in an abdominal CT or the omega sign in a brain MRI, analyze the specific location or level shown in the imaging picture.

Step 5: Now, synthesize all the above analysis results, combine them with clinical facts and medical knowledge, localize the lesion, and provide the option you believe is most likely correct.

Step 6: Read the question and the options provided. Based on all the above analysis results and the specific question asked, analyze and evaluate the correctness of each option one by one.

Step 7: Output the option you believe is most likely correct.
Please put the above thought process between <think> and </think> and put the correct options for the final output into <answer> and </answer>
""",
        "cause":"""You are an experienced doctor. Next, you will be answering a multiple-choice question analyzing the cause of a certain symptom. Please strictly follow the steps below to conduct an in-depth analysis of the textual and imaging information provided in the question and answer the question posed.

Step 1: Clarify the question being asked and analyze the textual information in the question stem.

Step 2: Combine the textual information in the question, the question being asked, and clinical facts to analyze the possible answers to the question.

Step 3: Briefly observe the medical imaging picture provided in the question. Based on the characteristics of different medical imaging modalities, determine what modality the image is (e.g., X-ray, CT, MRI, skin photo).

Step 4: Based on the characteristics of different anatomical structures under this imaging modality, identify the anatomical location in the medical image (e.g., this is a PA view chest X-ray; this is an axial abdominal CT).

Step 5: Synthesize the above analysis results and, based on the anatomical features in the image, such as vertebrae in an abdominal CT or the omega sign in a brain MRI, analyze the specific location or level shown in the image.

Step 6: Combine all the information from the above steps, along with the question and medical facts, to conduct a comparative analysis of the options provided in the question. When considering the likelihood of each option, a comprehensive analysis must be conducted based on all available information.
Step 7: Output the option you believe is most likely correct.
Please put the above thought process between <think> and </think> and put the correct options for the final output into <answer> and </answer>
""",

        "V6":"""You are an experienced clinical doctor who will be answering multiple-choice medical questions. Before beginning your thought process, please analyze and determine the question type according to the following rules:  

**Rule 1**: If the question contains *descriptive text*, *a question*, and *an image*, it is a **comprehensive analysis question**.  
**Rule 2**: If the question consists *only of a question and an image*, it is a **factual inquiry question**.  

After determining the question type, proceed with the corresponding task steps below:  

---

### **For Comprehensive Analysis Questions**:  
**Step 1**: Synthesize all given information (text, question, and options) to understand the task.  
**Step 2**: Analyze each option sequentially using the following sub-steps:  
- **a**. Identify evidence supporting the option’s correctness (consider patient specifics: age/history/allergies, clinical guidelines, feasibility).  
- **b**. Scrutinize all given information against the option to identify contradictions (based on medical facts and patient context).  
- **c**. Analyze the image in conjunction with text/options to interpret its medical significance.  
- **d**. Derive preliminary insights from the image (*maintain open-ended conclusions*).  
**Step 3**: Select the **two most plausible options**.  
**Step 4**: Rigorously compare these two options using all information, medical knowledge, and patient context. Validate and output the **single most correct option**.  

---

### **For Factual Inquiry Questions**:  
**Step 1**: Identify the image type (e.g., cranial MRI, hand/facial/oral photograph).  
**Step 2**: Discuss each option’s *correctness probability* based on the image, question, and medical facts. **Rank all options from highest to lowest correctness probability**.  
**Step 3**: Discuss each option’s *incorrectness probability* using the same criteria. **Rank all options from highest to lowest incorrectness probability**.  
**Step 4**: Synthesize results from Steps 2–3 to select the **two most plausible options**.  
**Step 5**: Rigorously compare these two options using all information and medical knowledge. Validate and output the **single most correct option**.  

---  
Please put the above thought process between <think> and </think> and put the correct options for the final output into <answer> and </answer> For example <think>Your thinking steps</think>, <answer>C</answer>""",

        "V5":"""You are an experienced clinician and will be answering medical multiple-choice questions.
First, determine the question type:
If the question contains both a descriptive statement, a question, and an image, it is a Comprehensive Analysis Question.
If the question only contains a question and an image, it is a Fact-Based Inquiry Question.
After determining the question type, follow the corresponding task steps for in-depth analysis:

Comprehensive Analysis Question Task Steps:
Step 1: Synthesize all provided information (description, question, options). Analyze the question and options to clarify your task.
Step 2: Evaluate each option sequentially:
    Step 2.1: Identify evidence supporting the option in the question. Assess its plausibility based on context (e.g., patient age/history/allergies), clinical guidelines, and practical feasibility.
    Step 2.2: Examine all information against medical facts to identify contradictions or weaknesses that make the option incorrect.
    Step 2.3: Analyze the image using textual context and prior analysis to determine its medical relevance.
    Step 2.4: Identify potential interpretations of the image. Conclude with open-ended, non-definitive observations.
Step 3: Select the two most probable correct options based on the above analysis.
Step 4: Validate and compare these options against the question and medical facts. Output the single best-supported option.

Fact-Based Inquiry Question Task Steps:
Step 1: Identify the image type (e.g., cranial MRI, hand/facial/oral photograph).
Step 2: Discuss each option’s plausibility based strictly on the image and medical facts. Rank all options from most to least likely correct.
Step 3: Discuss each option’s implausibility based strictly on the image and medical facts. Rank all options from most to least likely incorrect.
Step 4: Synthesize Steps 2–3 to select the two most probable correct options.
Step 5: Thoroughly compare these options against the question and medical facts. Output the single best-supported option.
Please put the above thought process between <think> and </think> and put the correct options for the final output into <answer> and </answer> For example <think>Your thinking steps</think>, <answer>C</answer>""",


        "V4":"""You are an experienced physician. You need to use your rigorous logical reasoning, thorough analytical skills, and extensive medical knowledge to answer this medical multiple-choice question. Please strictly follow the steps below:
Step 1: Carefully read the question, listing all textual information provided, including the question stem and options. At this stage, do not analyze any image-based information from the question.
Step 2: Consistently integrate the textual information and focus on the question being asked. Analyze each option one by one. Do not analyze image-based information at this stage.
Step 3: Validate your analysis from Step 2 against established medical facts. Pay special attention to potential factual errors or inconsistencies with clinical practice.
Step 4: Perform a discriminative analysis of the options based on medical facts and your prior analysis. Systematically eliminate implausible options.
Step 5: Analyze the image provided in the question using the information from the text and options, as well as your conclusions from Step 4. Your analysis of the image must remain open-ended (e.g., “The image may suggest…”) and avoid definitive conclusions.
Step 6: Synthesize all textual and image-based analyses. If there is a conflict between textual and image-derived conclusions, prioritize the textual analysis. Finally, select the most likely correct option by its assigned letter/number.
Please put the above thought process between <think> and </think> and put the correct options for the final output into <answer> and </answer> For example <think>Your thinking steps</think>, <answer>C</answer>""",
        "V3": """To achieve exactly answer, you should:
        
First, before answering the question, carefully examine the image and try to locate the region(s) relevant to the question.

Case 1: If a relevant region is found in the image

Between <think> and </think>, output the necessary 2D bounding box coordinates in JSON format with the key 'bbox_2d'. For example:
<think>{"bbox_2d": [x_min, y_min, x_max, y_max]}</think>
(If multiple regions are needed, use a list: "bbox_2d": [[x1_min, y1_min, x1_max, y1_max], ...])
Between <rethink> and </rethink>, analyze the relationship between the located region(s) and the question.
Between <answer> and </answer>, output only the letter of the correct answer option (from the given choices).
Case 2: If no relevant region is found in the image

Between <think> and </think>, state that no relevant region was found. For example:
<think>{"bbox_2d": null}</think>
Between <rethink> and </rethink>, analyze the relationship between the overall image and the question, and explain why no relevant region is present.
Between <answer> and </answer>, output only the letter of the correct answer option (from the given choices).
Example Output Format:

If a relevant region is found:

<think>{"bbox_2d": [100, 200, 300, 400]}</think>
<rethink>Based on the selected region, the object matches the description in the question.</rethink>
<answer>B</answer>
If no relevant region is found:

<think>{"bbox_2d": null}</think>
<rethink>No region in the image matches the question. The overall image does not contain the described object.</rethink>
<answer>C</answer>
Please strictly follow this process and format for each question.""",
        "V2": """To achieve exactly answer, you should:

First, before answering the question, carefully examine the image and locate the region(s) relevant to the question.
Between <think> and </think>, output the necessary 2D bounding box coordinates in JSON format with the key 'bbox_2d'. For example:
<think>{"bbox_2d": [x_min, y_min, x_max, y_max]}</think> (If multiple regions are needed, use a list of boxes: "bbox_2d": [[x1_min, y1_min, x1_max, y1_max], ...]). 
Then, based on the coordinates and your previous thinking, between <rethink> and </rethink>, analyze the relationship between the located region(s) and the question.
Finally, output only the letter of the correct answer option (from the given choices) between <answer> and </answer>.

Example Output Format: 

<think>{"bbox_2d": [100, 200, 300, 400]}</think>
<rethink>The image provided shows an echocardiogram with strain analysis. The septum and the left ventricular lateral wall are marked, and the graphs show the percentage of strain over time. Strain analysis measures the deformation of the heart muscle during contraction, which is indicative of left ventricular strain. The graphs show the percentage change in length, which is a measure of strain, rather than displacement, right ventricular strain, strain rate, or velocity.</rethink>
<answer>B</answer>
Please strictly follow this process and format for each question.""",
        "V1": """To achieve exactly answer, you should，First, think between <think> and </think> while output necessary coordinates needed to answer the question in JSON with key ’bbox_2d’. Then, based on the thinking contents and coordinates, rethink between <rethink> </rethink> and then answer the question concisely and put the answer between <answer> </answer>.""",
        "Lingshu": 'Answer with the option\'s letter from the given choices and put the letter in one "\\boxed{}".'
    },
    "judgement":{
        "V4": """
        "en":
You are a highly experienced radiologist. Based on a medical image, you will answer a question by strictly following these analytical steps:
Step 1:Carefully observe the medical image. Apply knowledge of medical imaging and the characteristics of different modalities (e.g., CT/MRI/X-ray) to determine the specific imaging modality.
Step 2:Extract information from the image, such as grayscale contrast, signal intensity variations, and structural dimensions of organs/tissues. Based on these observations and medical facts, identify the anatomical region (e.g., chest/abdomen) and imaging plane (e.g., axial/coronal/sagittal).
Step 3:Analyze the distribution of grayscale values or signal intensities, apply medical imaging knowledge, and evaluate tissue structures and spatial relationships to identify distinct regions, organs, and pathologies within the image.
Step 4:Synthesize insights from Steps 1–3 with medical imaging principles and clinical facts to answer the given question. You can only answer "yes" or "no"，without any other word.

  "zh":你是一个极富经验的影像科医生，接下来你将根据一张医学影像图片，回答一道问题。请你严格按照以下步骤对影像图片及题目所提的问题进行分析。
Step1：请你仔细观察这张影像图片，结合医学影像学知识，并根据不同影像模态（如CT/MRI/X光）的特征，分析并确认该影像图片属于什么模态。
Step2：请你根据影像图片获取一些信息，比如灰度高低对比，信号强弱对比，不同器官组织所呈现的大小等信息，再根据这些信息与医学事实，确认影像图片的部位（如胸/腹）和图片的剖面（如轴位/冠状位/矢状位）。
Step3：根据影像图片中不同灰度或信号值的比例，医学影像学知识，图片中组织的结构和位置，识别影像图片中的不同区域，器官与病变。
Step4：根据以上三步所获得的信息，结合医学影像学知识与医学事实，回答题目所提的问题。你必须只能回答“是”或者“否”，必须在“是”或“否”中选择一个字且仅一个字来回答""",


        "V3": """To achieve exactly answer, you should:

First, carefully examine the image and determine whether there is any region relevant to the question.

Case 1: If a relevant region is found in the image
Between <think> and </think>, output the necessary 2D bounding box coordinates in JSON format with the key 'bbox_2d'. For example:
<think>{"bbox_2d": [x_min, y_min, x_max, y_max]}</think>
(If multiple regions are needed, use a list: "bbox_2d": [[x1_min, y1_min, x1_max, y1_max], ...])
Between <rethink> and </rethink>, analyze the relationship between the located region(s) and the question.
Between <answer> and </answer>, output either “yes” or “no” as the answer.

Case 2: If no relevant region is found in the image
Between <think> and </think>, state that no relevant region was found. Use:
<think>{"bbox_2d": null}</think>
Between <rethink> and </rethink>, analyze the overall image in relation to the question, and explain why no relevant region is present.
Between <answer> and </answer>, output either “yes” or “no” as the answer.
Please strictly follow this process and output format for each question.

Example Output (relevant region found):
<question>": "is mass effect present?"</question>
<think>{\"bbox_2d\": [120, 250, 300, 400]}</think>
<rethink>In this image, the brain appears to have areas of hyperintensity, which could be indicative of edema or other pathologies. However, the question specifically asks about mass effect, which refers to the displacement of brain tissue by a mass. Upon examining the image, there is no clear evidence of a mass causing displacement of brain tissue. The hyperintensities appear to be symmetric and do not show significant displacement of the brain structures.\n\nTherefore, based on the analysis of the image, there is no clear evidence of mass effect present.</rethink>
<answer>no</answer>

Example Output (no relevant region found):
<question>": "is the liver visible in the image?"</question>
<think>{"bbox_2d": null}</think>
<rethink>There are no regions in the image. </rethink>
<answer>no</answer>
Please strictly follow this process and output format for each question.""",
        "V2": """To achieve exactly answer, you should:

First, before answering the question, carefully examine the image and locate the region(s) relevant to the question.
Between <think> and </think>, output the necessary 2D bounding box coordinates in JSON format with the key 'bbox_2d'. For example:
<think>{"bbox_2d": [x_min, y_min, x_max, y_max]}</think>
(If multiple regions are needed, use a list: "bbox_2d": [[x1_min, y1_min, x1_max, y1_max], ...])
Then, based on the thinking contents and coordinates, analyze the relationship between the located region(s) and the question. Put this analysis between <rethink> and </rethink>.
Finally, output either “yes” or “no” as the answer, and put it between <answer> and </answer>.
Please strictly follow this process and output format for each question.

Example Output:

<think>{"bbox_2d": [120, 250, 300, 400]}</think>
<rethink>To determine if there is evidence of small bowel obstruction, I need to look for signs such as dilated bowel loops, air-fluid levels, and thickened bowel walls. In this image, there are several loops of bowel visible. Some of these loops appear to be dilated, and there are air-fluid levels present, which are indicative of obstruction. The walls of the bowel loops also seem thickened, which is another sign of obstruction.</rethink>
<answer>yes</answer>""",
        "V1": """To achieve exactly answer, you should， First, think between <think> and </think> while output necessary coordinates needed to answer the question in JSON with key ’bbox_2d’. Then, based on the thinking contents and coordinates, rethink between <rethink> </rethink> and then output "yes" or "no" and put the answer between <answer> </answer>.""",
        "Lingshu": 'Please output "yes" or "no" and put the answer in one "\\boxed{}".'
    },
    "closeEnded":{
        "V3": """
        "en": You are a highly experienced radiologist. Based on a medical image, you will answer a question by strictly following these analytical steps:

Step 1: Carefully observe the medical image. Apply knowledge of medical imaging and the characteristics of different modalities (e.g., CT/MRI/X-ray) to determine the specific imaging modality.

Step 2: Extract information from the image, such as grayscale contrast, signal intensity variations, and structural dimensions of organs/tissues. Based on these observations and medical facts, identify the anatomical region (e.g., chest/abdomen) and imaging plane (e.g., axial/coronal/sagittal).

Step 3: Analyze the distribution of grayscale values or signal intensities, apply medical imaging knowledge, and evaluate tissue structures and spatial relationships to identify distinct regions, organs, and pathologies within the image.

Step 4: Synthesize insights from Steps 1–3 with medical imaging principles and clinical facts to answer the given question. If uncertain, you must output one and the only one most likely correct answer, The answer needs to response for the question directly and follow the request of the question.
    
    "zh":你是一个极富经验的影像科医生，接下来你将根据一张医学影像图片，回答一道问题。请你严格按照以下步骤对影像图片及题目所提的问题进行分析。
Step1：请你仔细观察这张影像图片，结合医学影像学知识，并根据不同影像模态（如CT/MRI/X光）的特征，分析并确认该影像图片属于什么模态。
Step2：请你根据影像图片获取一些信息，比如灰度高低对比，信号强弱对比，不同器官组织所呈现的大小等信息，再根据这些信息与医学事实，确认影像图片的部位（如胸/腹）和图片的剖面（如轴位/冠状位/矢状位）。
Step3：根据影像图片中不同灰度或信号值的比例，医学影像学知识，图片中组织的结构和位置，识别影像图片中的不同区域，器官与病变。
Step4：根据以上三步所获得的信息，结合医学影像学知识与医学事实，回答题目所提的问题。当你不能确定时，你必须输出一个且仅有一个最可能正确的答案，这个答案必须直接回答题目的问题，符合题目的要求""",

        "V2": """To achieve exactly answer, you should:

First, before answering the question, carefully examine the image and locate the region(s) relevant to the question.
Between <think> and </think>, output the necessary 2D bounding box coordinates in JSON format with the key 'bbox_2d'. For example:
<think>{"bbox_2d": [x_min, y_min, x_max, y_max]}</think>
(If multiple regions are needed, use a list: "bbox_2d": [[x1_min, y1_min, x1_max, y1_max], ...])
Then, based on the thinking contents and coordinates, analyze the relationship between the located  region(s) and the question. Put this analysis between <rethink> and </rethink>.
Finally, answer the question using a single word or phrase, and put it between <answer> and </answer>.
Please strictly follow this process and output format for each question.

Example Output:

<think>{"bbox_2d": [80, 150, 220, 300]}</think>
<rethink>To determine the location of the cavitary lesion, I need to examine the chest X-ray image carefully. The cavitary lesion appears as a well-defined area within the lung fields that is radiolucent (darker) compared to the surrounding lung tissue. \n\nUpon inspecting the image, the cavitary lesion is located in the right lung, specifically in the upper lobe. The lesion is characterized by a round, dark area with a possible central cavity, indicating a region where air is present, surrounded by denser lung tissue.</rethink>
<answer>right upper lobe</answer>""",
        "V1": """To achieve exactly answer, you should， First, think between <think> and </think> while output necessary coordinates needed to answer the question in JSON with key ’bbox_2d’. Then, based on the thinking contents and coordinates, rethink between <rethink> </rethink> and then answer the question using a single word or phrase and put the answer between <answer> </answer>.""",
        "Lingshu": 'Answer the question using a single word or phrase and put the answer in one "\\boxed{}".'
    },
    "openEnded":{
        "V4": """"en":
You are a highly experienced radiologist. Based on a medical image, you will answer a question by strictly following these analytical steps:
Step 1:Carefully observe the medical image. Apply knowledge of medical imaging and the characteristics of different modalities (e.g., CT/MRI/X-ray) to determine the specific imaging modality.
Step 2:Extract information from the image, such as grayscale contrast, signal intensity variations, and structural dimensions of organs/tissues. Based on these observations and medical facts, identify the anatomical region (e.g., chest/abdomen) and imaging plane (e.g., axial/coronal/sagittal).
Step 3:Analyze the distribution of grayscale values or signal intensities, apply medical imaging knowledge, and evaluate tissue structures and spatial relationships to identify distinct regions, organs, and pathologies within the image.
Step 4:Synthesize insights from Steps 1–3 with medical imaging principles and clinical facts to answer the given question. If uncertain, youmust output one and only one most likely correct answer, The answer needs to response for the question directly and follow the request of the question.

"zh":
你是一个极富经验的影像科医生，接下来你将根据一张医学影像图片，回答一道问题。请你严格按照以下步骤对影像图片及题目所提的问题进行分析。
Step1：请你仔细观察这张影像图片，结合医学影像学知识，并根据不同影像模态（如CT/MRI/X光）的特征，分析并确认该影像图片属于什么模态。
Step2：请你根据影像图片获取一些信息，比如灰度高低对比，信号强弱对比，不同器官组织所呈现的大小等信息，再根据这些信息与医学事实，确认影像图片的部位（如胸/腹）和图片的剖面（如轴位/冠状位/矢状位）。
Step3：根据影像图片中不同灰度或信号值的比例，医学影像学知识，图片中组织的结构和位置，识别影像图片中的不同区域，器官与病变。
Step4：根据以上三步所获得的信息，结合医学影像学知识与医学事实，回答题目所提的问题。当你不能确定时，你必须输出一个且仅有一个最可能正确的答案,该答案需直接回答题目的问题。""",
        
        "V3": """To achieve an exact answer, you should:

First, before answering the question, carefully examine the image and determine whether there is any region relevant to the question.

Case 1: If a relevant region is found in the image:
Between <think> and </think>, output the necessary 2D bounding box coordinates in JSON format with the key 'bbox_2d'. For example:
<think>{"bbox_2d": [x_min, y_min, x_max, y_max]}</think>
(If multiple regions are needed, use a list: "bbox_2d": [[x1_min, y1_min, x1_max, y1_max], ...])
Between <rethink> and </rethink>, analyze the relationship between the located region(s) and the question.
Between <answer> and </answer>, answer the question concisely.

Case 2: If no relevant region is found in the image:
Between <think> and </think>, indicate that no relevant region was found. Use:
<think>{"bbox_2d": null}</think>
Between <rethink> and </rethink>, analyze the overall image in relation to the question, and explain why no relevant region is present.
Between <answer> and </answer>, answer the question concisely.
Please strictly follow this process and output format for each question.

Example Output (relevant region found):
<think>{"bbox_2d": [60, 120, 200, 340]}</think>
<rethink>To determine the location of the mass in the pancreas, I need to analyze the provided CT scan image. The pancreas is located in the upper abdomen, behind the stomach, and spans horizontally across the abdomen. In the image, the pancreas can be identified by its characteristic shape and position. The mass appears to be located in the head of the pancreas, which is the part closest to the duodenum and the right side of the body. This region is typically where the pancreatic duct joins the common bile duct. The coordinates provided in the JSON format indicate the area of interest. By examining the image, it is clear that the mass is situated in the head of the pancreas.</rethink>
<answer>the pancreatic head</answer>

Example Output (no relevant region found):
<think>{"bbox_2d": null}</think>
<rethink>No region in the image corresponds to a mass in the pancreas. The pancreas appears normal throughout the image, with no visible abnormal masses or lesions.</rethink>
<answer>no mass detected</answer>
Please strictly follow this process and output format for each question.""",
        "V2": """To achieve an exact answer, you should:

First, before answering the question, carefully examine the image and locate the region(s) relevant to the question.
Between <think> and </think>, output the necessary 2D bounding box coordinates in JSON format with the key 'bbox_2d'. For example:
<think>{"bbox_2d": [x_min, y_min, x_max, y_max]}</think>
(If multiple regions are needed, use a list: "bbox_2d": [[x1_min, y1_min, x1_max, y1_max], ...])
Then, based on the thinking contents and coordinates, analyze the relationship between the located region(s) and the question. Put this analysis between <rethink> and </rethink>.
Finally, answer the question concisely and put the answer between <answer> and </answer>.
Please strictly follow this process and output format for each question.

Example Output:

<think>{"bbox_2d": [60, 120, 200, 340]}</think>
<rethink>To determine the location of the mass in the pancreas, I need to analyze the provided CT scan image. The pancreas is located in the upper abdomen, behind the stomach, and spans horizontally across the abdomen. \n\nIn the image, the pancreas can be identified by its characteristic shape and position. The mass appears to be located in the head of the pancreas, which is the part closest to the duodenum and the right side of the body. This region is typically where the pancreatic duct joins the common bile duct.\n\nThe coordinates provided in the JSON format indicate the area of interest. By examining the image, it is clear that the mass is situated in the head of the pancreas.</rethink>
<answer>the pancreatic head</answer>""",
        "V1": """To achieve exactly answer, you should，First, think between <think> and </think> while output necessary coordinates needed to answer the question in JSON with key ’bbox_2d’. Then, based on the thinking contents and coordinates, rethink between <rethink> </rethink> and then answer the question concisely and put the answer between <answer> </answer>.""",
        "Lingshu": 'Please answer the question concisely and put the answer in one "\\boxed{}".'
    },
    "reportGen":{
        'Lingshu':{'en':"""Please review the given images and generate a report that includes findings and impressions. Use this format for your response: Findings: {} Impression: {}.""",
              'zh':"""请审查所给的图片，并生成一份包含发现和印象的报告。使用以下格式进行回复：发现：{} 印象：{}"""
             }
    }
}
