<h1 align='center'>
  JoyMed: A Leading Medical Foundation Model with Adaptive Reasoning
</h1>

<div align='center'>
    <img src="assets/JoyMed.png" alt="JoyMed" title="JoyMed" width="150">  
</div>

<div align='center'>
    <a target="_blank" href="" onclick="return false;">Ju&nbsp;Huang</a>&nbsp
    <a target="_blank" href="" onclick="return false;">Xinyi&nbsp;Liu</a>&nbsp
    <a target="_blank" href="" onclick="return false;">Sheng&nbsp;Shi</a>&nbsp
    <a target="_blank" href="" onclick="return false;">Fangru&nbsp;Zhou</a>&nbsp
    <a target="_blank" href="" onclick="return false;">Jun&nbsp;Zhao</a>&nbsp
    <a target="_blank" href="" onclick="return false;">Jun&nbsp;Xu</a>&nbsp
</div>

<br>

<div align='center'>
    JDH Algo, JD Health International Inc.  
</div>

<br>

<div align='center'>
    <a href='https://github.com/jdh-algo/JoyMed'><img src='https://img.shields.io/github/stars/jdh-algo/JoyMed?style=social'></a>
    <a href='https://huggingface.co/jdh-algo/JoyMed-8B-v1.0'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-JoyMed%208B-yellow'></a>
    <a href='https://huggingface.co/jdh-algo/JoyMed-32B-v1.0'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-JoyMed%2032B (Coming soon)-yellow'></a>
</div>

<br>

## 🔥 News

- [2026-03-27]: 📝 We open-source the evaluation script for the related benchmark today, please refer to [evaluation/Readme.md](https://github.com/jdh-algo/JoyMed/blob/main/evaluation/Readme.md)!
- [2026-03-26]: 📚 We release [📂 MedDocBench](https://huggingface.co/datasets/jdh-algo/MedDocBench), a self-collected medical benchmark dataset to facilitate research and evaluation in medical multimodal learning!
- [2026-03-26]: ⚙️ We have released [🤗 JoyMed-8B-v1.0](https://huggingface.co/jdh-algo/JoyMed-8B-v1.0) for research and application! Welcome to use and explore!
- [2026-03-26]: 🎉 We propose JoyMed, a medical foundation model with adaptive reasoning that balances reasoning accuracy and efficiency, achieves SOTA across multiple benchmarks, and advances the translation of medical multimodal large models to clinical applications!


## 📖 Overview

Reasoning capabilities are foundational to medical multimodal large models (MMLMs), as they enable trustworthy diagnosis, interpretable decision-making, and effective management of complex clinical cases. However, mainstream MMLMs either lack explicit reasoning capacities or rely on fixed, end-to-end, undifferentiated mandatory reasoning paradigms. This not only introduces chain-of-thought redundancy and wastes computational resources but also degrades performance in pure perceptual tasks due to unnecessary reasoning overhead. The core bottleneck lies in the inherent trade-off between reasoning accuracy and computational efficiency: thorough reasoning ensures rigor for complex tasks but imposes redundant costs on simple tasks, whereas efficiency-oriented simplified output paradigms lack the sufficient reasoning completeness critical for complex clinical decision-making. 
To address these challenges, we propose JoyMed, a leading medical foundation model with adaptive reasoning. Building upon direct output and reasoning-augmented output paradigms, we introduce an adaptive reasoning mechanism that directly outputs results for trivially simple tasks to optimize efficiency, while generating stepwise reasoning traces for complex tasks to balance accuracy and interpretability. Experimental results demonstrate that JoyMed achieves state-of-the-art performance across multiple benchmarks, which effectively balances the core clinical requirements of comprehensive accuracy and efficient result acquisition, marking an exploratory step toward translating MMLMs from laboratory research to practical clinical applications.

![Network](assets/Overview.png "Network")


### Key Features

- **Superior Performance**: Our model achieves leading performance across multiple core medical benchmarks,encompassing medical image understanding, text-based question answering, medical document comprehension, and medical report generation, demonstrating its robust capabilities. This outstanding performance stems from our carefully designed two-stage training strategy. First, fine-grained vision-language alignment training significantly enhances the model’s perceptual ability for key regions such as lesions and anatomical structures. Subsequently, reinforcement on complex tasks like report generation and case analysis establishes precise associations between visual regions and textual descriptions, providing solid support for high-level medical visual understanding and question answering.
- **Adaptive Reasoning**: To strike an optimal balance between computational efficiency and deep reasoning, we innovatively propose an adaptive reasoning mechanism. Its core involves constructing a difficulty-tiered dataset and designing corresponding training strategies to mitigate potential mode collapse during the model’s autonomous reasoning process, enabling independent assessment of the intrinsic complexity of problems. Our proposed model operates in three modes: direct output, chain-of-thought reasoning, and adaptive thinking. This design allows the model to intelligently allocate computational resources based on task demands while maintaining high accuracy and interpretability, thereby achieving an optimal trade-off between effectiveness and efficiency.


## 🚀 Model Zoo
JoyMed comes in two variants with different parameter configurations:

|  Model  |  Parameters  |                     Hugging Face                     |
| :------: | :-------: | :--------------------------------------------------: |
| JoyMed-8B-v1.0 | 8B | [🤗 JoyMed-8B-v1.0](https://huggingface.co/jdh-algo/JoyMed-8B-v1.0) |
| JoyMed-32B-v1.0 | 32B | [🤗 JoyMed-32B-v1.0 (Coming soon)](https://huggingface.co/jdh-algo/JoyMed-32B-v1.0) |


## 🏆 Performance

### Medical Textual Question Answering Benchmarks
The best results on each benchmark and average accuracy are highlighted in **bold**, and the scores with <u>underline</u> indicate the second best. Note that MedXQA and SGPQA denote MedXpertQA-Text and SuperGPQA-medical benchmarks.

| **Model** | **PubMedQA** | **MedMCQA** | **MedXQA** | **CMMLU** | **SGPQA** | **MedQA (USMLE)** | **MedQA (MCMLE)** | **Medbullets (op-4)** | **Medbullets (op-5)** | **Avg.** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Proprietary Models** | | | | | | | | | | |
| GPT 4.1 | 76.00 | 87.98 | 30.82 | 81.02 | 50.6 | 77.07 | 81.73 | 78.90 | 73.38 | 70.83 |
| GPT 5 | 78.00 | 62.99 | 40.75 | 82.93 | 49.54 | 76.96 | 74.00 | 88.93 | 87.30 | 71.27 |
| Doubao Seed 1.6 | 76.00 | 75.06 | 30.67 | 91.67 | 55.19 | 93.48 | 94.02 | 82.79 | 76.62 | 75.06 |
| **Open-Source Models (<10B)** | | | | | | | | | | |
| MedGemma 4B | 73.00 | 52.26 | 13.1 | 43.96 | 21.52 | 55.54 | 41.10 | 48.05 | 42.53 | 43.45 |
| Qwen3-VL 8B | 73.20 | 60.05 | 14.98 | 79.07 | 35.68 | 65.67 | 85.61 | 55.84 | 48.70 | 57.64 |
| HuatuoGPT-V 7B | 73.60 | 51.95 | 10.33 | 71.12 | 22.11 | 52.95 | 73.09 | 43.51 | 37.66 | 48.48 |
| Lingshu 7B | 75.40 | 56.13 | 16.45 | 69.02 | 27.51 | 63.39 | 75.98 | 62.66 | 52.92 | 55.50 |
| Citrus-V 8B | 74.80 | 55.10 | 16.90 | 71.19 | 29.47 | 64.89 | 76.94 | 59.09 | 54.22 | 55.84 |
| Hulu-Med 7B | 77.20 | **67.51** | 18.53 | 71.72 | 31.10 | 73.45 | 78.93 | 64.94 | 57.47 | 60.09 |
| **JoyMed 8B** | 78.20 | 65.36 | <u>23.67</u> | 82.75 | 37.10 | 82.64 | 92.06 | 73.05 | 68.18 | 67.00 |
| **JoyMed 8B auto** | **79.40** | 66.58 | 23.55 | <u>83.05</u> | <u>38.04</u> | <u>84.84</u> | 91.42 | **74.35** | **70.46** | <u>67.96</u> |
| **JoyMed 8B thinking** | <u>79.00</u> | <u>66.89</u> | **24.37** | **83.35** | **39.78** | **85.23** | <u>91.77</u> | <u>73.38</u> | <u>70.13</u> | **68.21** |

### Medical Visual Question Answering Benchmarks
The best results on each benchmark and average accuracy are highlighted in **bold**, and the scores with <u>underline</u> indicate the second best. Note that MedXQA and GMAI-MMB denote MedXpertQA-mm and GMAI-MMBench-test benchmarks.

| **Model** | **VQA-RAD** | **MedXQA** | **SLAKE** | **PATH-VQA** | **PMC-VQA** | **OmniMedVQA** | **GMAI-MMB** | **Avg.** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Proprietary Models** | | | | | | | | |
| GPT 4.1 | 62.53 | 43.35 | 72.54 | 54.97 | 38.76 | 55.14 | 58.52 | 55.12 |
| GPT 5 | 68.37 | 51.48 | 65.82 | 31.74 | 36.10 | 38.44 | - | - |
| Doubao Seed 1.6 | 33.49 | 45.75 | 67.28 | 47.58 | 49.94 | 61.68 | 48.50 | 50.60 |
| **Open-Source Models (<10B)** | | | | | | | | |
| MedGemma 4B | 72.06 | 22.05 | 78.32 | 48.64 | 48.02 | 70.04 | 45.59 | 54.96 |
| Qwen3-VL 7B | 63.41 | 25.00 | 72.11 | 43.65 | 54.01 | 76.90 | 54.31 | 55.63 |
| HuatuoGPT-V 7B | 67.85 | 22.30 | 69.39 | 44.29 | 53.84 | 75.14 | 51.56 | 54.91 |
| Lingshu 7B | 68.74 | 26.90 | 82.90 | 60.23 | 55.77 | 82.41 | 54.02 | 61.57 |
| Citrus-V 8B | 64.30 | 25.10 | 84.91 | 62.00 | 55.64 | 72.69 | 45.43 | 57.45 |
| Hulu-Med 7B | 74.50 | 27.70 | 82.66 | 62.57 | **66.95** | **83.70** | 54.28 | 64.62 |
| **JoyMed 8B** | 75.83 | 32.60 | 86.53 | 74.16 | 57.19 | <u>82.36</u> | 59.85 | 66.93 |
| **JoyMed 8B auto** | <u>76.50</u> | **33.25** | **87.97** | <u>75.06</u> | 58.34 | 81.47 | **60.37** | <u>67.56</u> |
| **JoyMed 8B thinking** | **79.16** | <u>33.20</u> | <u>86.82</u> | **75.34** | <u>58.52</u> | 81.43 | <u>60.35</u> | **67.83** |

### Medical Document Understanding Benchmarks
The best results on each benchmark and average accuracy are highlighted in **bold**, and the scores with <u>underline</u> indicate the second best.

|  | **Laboratory Test Report** | | | **GMD** | |  |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Model** | **abnormalityQA** | **fullparsing** | **simpleQA** | **Simple QA** | **Complex QA** | **Avg.** |
| **Proprietary Models** | | | | | | |
| GPT 4.1 | 45.39 | 73.97 | 66.00 | 45.60 | 55.85 | 57.36 |
| GPT 5 | 71.41 | 71.87 | 78.75 | 59.35 | 57.04 | 67.68 |
| Doubao Seed 1.6 | 82.16 | 81.09 | 85.00 | 73.60 | 79.25 | 80.22 |
| **Open-Source Models (<10B)** | | | | | | |
| MedGemma 4B | 13.31 | 36.39 | 17.75 | 18.10 | 20.05 | 21.12 |
| Qwen3-VL 8B | 48.17 | 79.71 | 84.50 | 78.72 | 79.05 | 74.03 |
| HuatuoGPT-V 7B | 7.54 | 32.14 | 9.50 | 17.40 | 9.60 | 15.24 |
| Lingshu 7B | 29.50 | 62.70 | 70.25 | 60.47 | 53.70 | 55.32 |
| Citrus-V 8B | **91.16** | <u>92.57</u> | <u>92.57</u> | 83.22 | **89.45** | **90.18** |
| Hulu-Med 7B | 11.35 | 43.07 | 19.75 | 19.30 | 15.72 | 21.84 |
| **JoyMed 8B** | 88.99 | **93.39** | 92.00 | 83.47 | <u>86.72</u> | <u>88.91</u> |
| **JoyMed 8B auto** | <u>90.80</u> | 88.87 | **94.00** | **85.20** | 85.67 | <u>88.91</u> |
| **JoyMed 8B thinking** | <u>90.80</u> | 88.87 | **94.00** | <u>85.10</u> | 85.47 | 88.85 |

### Medical Image Report Generation Benchmarks
The best results on each benchmark and average accuracy are highlighted in **bold**, and the scores with <u>underline</u> indicate the second best.

|  | **CheXpert Plus** |  |  | **IU XRAY** |  |  |  |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Model** | **ROUGE-L** | **CIDEr** | **RaTE** | **ROUGE-L** | **CIDEr** | **RaTE** | **Avg.** |
| **Proprietary Models** | | | | | | | |
| GPT 4.1 | 24.50 | 78.80 | 45.50 | 32.63 | 124.42 | 50.91 | 59.44 |
| GPT 5 | 24.48 | 86.46 | 51.26 | 31.72 | 131.44 | 56.64 | 64.06 |
| Doubao Seed 1.6 | 19.27 | 61.92 | 45.49 | 22.67 | 92.72 | 53.76 | 49.31 |
| **Open-Source Models (<10B)** | | | | | | | |
| MedGemma 4B | 26.01 | 85.86 | 51.23 | 39.51 | 151.86 | 61.99 | 69.41 |
| Qwen3-VL 7B | 21.64 | 65.12 | 46.51 | 25.22 | 93.56 | 52.88 | 50.82 |
| HuatuoGPT-V 7B | 21.40 | 65.00 | 46.58 | 29.96 | 110.51 | 54.91 | 54.73 |
| Lingshu 7B | 26.50 | 79.00 | 45.40 | <u>44.52</u> | **200.47** | 60.30 | 75.90 |
| Citrus-V 8B | 28.94 | 94.57 | 51.07 | 24.78 | 118.94 | 56.25 | 62.40 |
| Hulu-Med 7B | 28.94 | 94.57 | 51.07 | 36.15 | 158.91 | 63.50 | 68.37 |
| **JoyMed 8B** | **32.54** | **120.90** | 55.68 | 42.85 | 190.94 | 64.90 | <u>84.63</u> |
| **JoyMed 8B auto** | 31.68 | <u>120.55</u> | <u>55.73</u> | **44.94** | <u>195.59</u> | **65.96** | **85.74** |
| **JoyMed 8B thinking** | <u>31.99</u> | 119.69 | **55.89** | 44.40 | 188.00 | <u>65.70</u> | 84.28 |


## 🛠️ Installation

1. Installing vLLM
    ```shell
    uv venv
    source .venv/bin/activate
    uv pip install -U vllm --torch-backend=auto

    # Update transformers to support the latest models.
    pip install -U "transformers==5.2.*"
    ```

2. Running JoyMed-8B-v1.0
    ```bash
    vllm serve jdh-algo/JoyMed-8B-v1.0 \
        --tensor-parallel-size 8 \
        --mm-encoder-tp-mode data \
        --mm-processor-cache-type shm \
        --enable-prefix-caching \
        --trust-remote-code \
        --gpu-memory-utilization 0.9
    ```

## 💻 Quick Start

### **Instruct** mode

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="jdh-algo/JoyMed-8B-v1.0",
    messages=[{"role": "user", "content": "What are the common causes of hypertension in adults? /no_think"}] # end with '/no_think' or nothing
)

print(response.choices[0].message.content)
```

### **Thinking** mode

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="jdh-algo/JoyMed-8B-v1.0",
    messages=[{"role": "user", "content": "What are the common causes of hypertension in adults? /think"}]
)

print(response.choices[0].message.content)
```

### **Auto** mode

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="jdh-algo/JoyMed-8B-v1.0",
    messages=[{"role": "user", "content": "What are the common causes of hypertension in adults? /auto_think"}]
)

print(response.choices[0].message.content)
```

## 🏛 License
This project is licensed under the Apache License (Version 2.0). For models and datasets, please refer to the original resource page and follow the corresponding License.