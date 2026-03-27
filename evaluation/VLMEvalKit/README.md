# MM-Eval 多模态模型评测流水线

一个完整的多模态模型评测流水线系统，基于evalscope，支持批量评测本地部署模型和远程API模型。

## 目录结构

```
mm_eval/
├── eval_pipeline.py      # 主程序
├── all_config.yaml       # 配置文件
├── data_configs/         # 数据集配置（需要配置自定义评估脚本时使用）
├── eval_V.py             # 评测执行器（VLMEvalKit的基本评估函数）
├── evalscope_configs/    # EvalScope配置（基本配置，可以用eval_V.py跑）
├── pipeline_logs/        # 流水线日志（自动生成）
├── vllm_logs/            # vLLM服务日志（自动生成）
└── outputs/              # 推理和评估结果（自动生成）
```

## 核心文件说明

### 1. eval_pipeline.py - 评测流水线主程序

这是一个功能完备的评测流水线管理系统，支持：

- **两阶段评测流程**：
  - **推理阶段(inference)**：对所有模型和数据集进行推理，生成预测结果
  - **评估阶段(evaluation)**：使用评判模型对推理结果进行评分和分析

- **支持两种模型类型**：
  - **本地模型**：通过vLLM自动部署和管理，支持多实例负载均衡
  - **远程模型**：通过API调用，如GPT-4、Doubao等

- **守护进程模式**：
  - 支持后台运行，自动管理进程生命周期
  - 提供状态查询、日志查看、优雅停止等管理功能

- **智能重试机制**：
  - 自动检测torch编译错误并重试
  - GPU内存不足时的错误处理

### 2. all_config.yaml - 统一配置文件

采用YAML格式的配置文件，包含以下配置项：

#### 模型配置 (models)
```yaml
models:
  local_models:     # 本地部署模型
    - name: "模型名称"
      vllm_config:  # vLLM部署配置
        gpus: 8     # GPU数量
        instances: 4 # 实例数量
        base_port: 7800  # 起始端口
        proxy_port: 7888 # 代理端口
        additional_args:  # 可选：额外的vLLM参数
          - "--gpu-memory-utilization"
          - "0.90"
          - "--enforce-eager"
      eval_config:  # 评测配置
        api_base: "http://localhost:7888/v1/chat/completions"
        
  remote_models:    # 远程API模型
    - name: "gpt-4.1"
      eval_config:
        api_base: "API地址"
        key: "API密钥"
```

#### 数据集配置 (datasets)
```yaml
datasets:
  - "JDH_SKIN_lesion_type_vqa"  # 自定义数据集
  - "MMBench_DEV_EN"             # 标准评测集
```

#### 评测设置 (evaluation)
```yaml
evaluation:
  nproc: 3  # 每个vLLM实例的并发数
  max_remote_concurrency: 16  # 远程模型最大并发数
  limit: null  # 样本数限制，null表示全部，一版测试时使用
```

#### 配置覆盖机制说明

注：（这部分后续可优化）由于Pipeline调用`vllm_service`，需要对当前`all_config.yaml`与`vllm_service`中的`model_configs.py`配合时的机制进行说明。

1. 默认配置`vllm_service/model_configs.py`中参数的生效情况：
- 生效的默认参数：model_path, max_model_len, served_model_name
- 不生效的默认参数：port, cuda_devices, tensor_parallel_size。`all_config.yaml`通过新的参数gpus, instances, base_port, proxy_port，基于`vllm_service/vllm_mangager.py`提供的端口和GPU并行机制覆盖（详见[文档](../vllm_service/README.md)）。

2. `all_config.yaml`中特殊参数：
   - `model`：必须匹配`model_configs.py`中的键名
   - `additional_args`：如果指定则完全替换默认值（整个list），否则使用默认值。


## 使用方法

### 1. 启动评测流水线

```bash
# 后台启动完整流水线（推理+评估）
python eval_pipeline.py start --config all_config.yaml

# 前台运行（便于调试）
python eval_pipeline.py start --config all_config.yaml --foreground

# 仅运行推理阶段
python eval_pipeline.py start --config all_config.yaml --phase infer

# 仅运行评估阶段（需要已有推理结果）
python eval_pipeline.py start --config all_config.yaml --phase eval
```

### 2. 管理流水线

```bash
# 查看运行状态
python eval_pipeline.py status

# 查看日志
python eval_pipeline.py logs
python eval_pipeline.py logs --tail  # 实时跟踪

# 停止流水线
python eval_pipeline.py stop

# 清理资源
python eval_pipeline.py cleanup

# 清理IPC资源
python eval_pipeline.py cleanup-ipc --force
```

## 工作流程

1. **配置准备**：
   - 编辑`all_config.yaml`，配置评测模型和数据集
   - 模型文件统一存放在`/mnt/afs/xuyangcao/shared_models`
   - 数据集统一存放在`/mnt/afs/xuyangcao/benchmarks`
   - 自定义评估脚本配置在`data_configs/`

2. **推理阶段**：
   - 自动启动本地模型的vLLM服务
   - 并行处理本地模型和远程模型
   - 对每个数据集进行推理，保存结果

3. **评估阶段**：
   - 启动评判模型（如需要）
   - 对推理结果进行评分
   - 生成评测报告

4. **结果输出**：
   - 推理结果保存在`outputs/模型名/`目录
   - 评测报告生成在相应目录
   - 汇总报告：`outputs/evaluation_summary.json`

## 特性说明

### 智能并发控制
- 本地模型：总并发数 = nproc × instances
- 远程模型：使用max_remote_concurrency限制并发

### 断点续跑
- 自动检测已完成的推理结果
- 支持中断后继续运行：重新运行相同命令

### 资源管理
- 自动清理vLLM进程
- GPU内存自动回收
- IPC资源泄漏检测和清理

### 错误处理机制

#### 1. Torch编译错误（自动重试）
只有torch.compile相关的错误会触发自动重试机制：
- **错误特征**：
  - `"failed to get the hash of the compiled graph"`
  - `"torch.compile failed"`
  - `"compilation failed"`
  - `"AssertionError: failed to get the hash of the compiled graph"`
- **处理方式**：自动禁用编译优化(`compilation_config=0`)并重试失败的实例

#### 2. GPU内存错误（仅记录，不重试）
以下GPU相关错误会被识别但不会触发重试：
- **CUDA内存不足**：
  - `"OutOfMemoryError"`
  - `"CUDA out of memory"`
  - 日志提示：`"Instance on port {port} failed due to CUDA OOM (not retriable)"`
  
- **缓存块内存不足**：
  - `"No available memory for the cache blocks"`
  - 日志提示：`"GPU memory error: Insufficient memory for cache blocks (increase gpu_memory_utilization)"`

- **GPU内存配置问题**：
  - `"gpu_memory_utilization"`相关错误
  - 日志提示：`"Instance on port {port} failed due to GPU memory config (not retriable)"`

#### 3. 其他运行时错误（仅记录）
- **一般运行时错误**：提取并显示具体的`RuntimeError`信息
- **未知错误**：记录为`"Unknown failure"`并提示检查日志文件

#### 4. 错误日志位置
- vLLM实例日志：`./vllm_logs/vllm_instance_{n}_port_{port}_{model_name}.log`
- 流水线主日志：`./pipeline_logs/pipeline_{timestamp}.log`


