<h1 align='center'>
  JoyMed: A Leading Medical Foundation Model with Adaptive Reasoning
</h1>

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
    <a href='https://huggingface.co/jdh-algo/JoyMed-8B-v1.0'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-JoyMed%208B-yellow'></a>
</div>

<br>



## 📝 Introduction
Reasoning capabilities are foundational to medical multimodal large models (MMLMs), as they enable trustworthy diagnosis, interpretable decision-making, and effective management of complex clinical cases. However, mainstream MMLMs either lack explicit reasoning capacities or rely on fixed, end-to-end, undifferentiated mandatory reasoning paradigms. This not only introduces chain-of-thought redundancy and wastes computational resources but also degrades performance in pure perceptual tasks due to unnecessary reasoning overhead. The core bottleneck lies in the inherent trade-off between reasoning accuracy and computational efficiency: thorough reasoning ensures rigor for complex tasks but imposes redundant costs on simple tasks, whereas efficiency-oriented simplified output paradigms lack the sufficient reasoning completeness critical for complex clinical decision-making. 
To address these challenges, we propose JoyMed, a leading medical foundation model with adaptive reasoning. Building upon direct output and reasoning-augmented output paradigms, we introduce an adaptive reasoning mechanism that directly outputs results for trivially simple tasks to optimize efficiency, while generating stepwise reasoning traces for complex tasks to balance accuracy and interpretability. Experimental results demonstrate that JoyMed achieves state-of-the-art performance across multiple benchmarks, which effectively balances the core clinical requirements of comprehensive accuracy and efficient result acquisition, marking an exploratory step toward translating MMLMs from laboratory research to practical clinical applications.


## 🛠️ Installation

1. Installing vLLM
    ```shell
    uv venv
    source .venv/bin/activate
    uv pip install -U vllm --torch-backend=auto
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

## 🏛 License
This project is licensed under the Apache License (Version 2.0). For models and datasets, please refer to the original resource page and follow the corresponding License.