# Experiment Results

Copy the sections below for every configuration. Preserve raw JSON results when
possible. Do not compare runs unless the important workload conditions match.

## Training experiment

- Lab and algorithm:
- Dataset name, revision, and split:
- Features and target, if applicable:
- Random seed:
- Hardware and selected device:
- Epochs or training steps:
- Learning rate:
- Batch size or gradient accumulation:
- Other hyperparameters:
- Baseline metric:
- Final training metric:
- Validation or test metric:
- Artifact path:
- What changed from the previous run:
- What one variable should be tested next:

For reinforcement learning, also record episodes, reward design, discount
factor, exploration schedule, success rate, and average return. For LoRA, also
record base-model revision, adapter rank, alpha, dropout, target modules,
sequence length, and trainable-parameter count.

## Inference experiment

## Environment

- Date and time:
- Repository commit:
- vLLM / vLLM-Metal version:
- Backend: Metal / CUDA
- Operating system:
- Hardware and memory:
- Driver / CUDA version, if applicable:
- Model and exact revision:
- Weight precision or quantization:
- KV-cache precision:

## Server configuration

- Profile:
- Full launch command:
- Maximum model length:
- Maximum sequences:
- Maximum batched tokens:
- Prefix caching enabled:
- GPU memory utilization setting:
- Other flags:

## Workload

- Prompt source:
- Approximate input tokens:
- Maximum output tokens:
- Request count:
- Concurrency:
- Request arrival rate:
- Streaming enabled:
- Warm-up requests:

## Results

| Run | Concurrency | Requests/s | Mean ms | P50 ms | P95 ms | P99 ms | TTFT P95 ms | Success rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | | | | | | | | |
| 2 | | | | | | | | |
| 3 | | | | | | | | |

## Server observations

- GPU utilization:
- GPU / unified-memory use:
- KV-cache utilization:
- Prefix-cache hit rate:
- Queue depth:
- Preemptions:
- Errors or warnings:

## Interpretation

- Saturation point:
- Most important bottleneck:
- Latency versus throughput tradeoff:
- Proposed service-level indicator and objective:
- What changed from the baseline:
- What single variable should be tested next:
