# Optimization and tuning

Optimization is an experiment, not a list of universally fast flags. Begin with
`learning`, record a baseline, change one setting, and repeat the same workload.

## Tuning map

| Mechanism | What it changes | Often helps | Main risks |
|---|---|---|---|
| Batching | Requests processed together | Throughput, utilization | Queueing and tail latency |
| KV cache | Reuses prior token states | Decode cost | Memory pressure |
| Prefix caching | Reuses shared prompt prefixes | Repeated long contexts | Little gain for unrelated prompts |
| Chunked prefill | Splits long prompt work | Mixed prefill/decode fairness | Needs workload-specific token budget |
| Quantization | Uses fewer bits | Memory and sometimes speed | Quality loss or unsupported kernels |
| Shorter context | Reduces reserved capacity | Memory and concurrency | Rejects longer requests |
| Tensor parallelism | Splits a model across GPUs | Models that do not fit one GPU | Communication overhead |

## KV cache and prefix caching

The normal KV cache helps within one request. **Automatic prefix caching (APC)**
can reuse full cached blocks across requests that share the same prefix.

```text
request A: [same long document][question A]
request B: [same long document][question B]
            ^^^^^^^^^^^^^^^^^^^
            reusable prefix work
```

Configured in this project with:

```toml
enable_prefix_caching = true
```

It reduces repeated prefill work, not the cost of generating a long new answer.
Test it with repeated system prompts or documents, then compare cache-hit metrics
and TTFT.

## Chunked prefill

Long prompts can monopolize compute while other requests are decoding. Chunked
prefill divides prompt processing into smaller token chunks and allows decode
work between them. In modern vLLM V1 it is enabled by default when possible.

`max_num_batched_tokens` is an important control:

- A smaller budget, such as 2048, can improve inter-token latency because a
  large prefill is less likely to delay decode work.
- A larger budget can improve prefill efficiency, TTFT, and aggregate
  throughput, but may hurt tail ITL or use more memory.

The project exposes this difference:

```bash
inference-lab-serve --profile latency --dry-run
inference-lab-serve --profile throughput --dry-run
```

## Where the profiles come from

The three profiles are not built into the Python. They are sections of
`configs/serve.toml`, and the file is the only place they are defined:

```toml
[profile.latency]
max_num_seqs = 16
```

So adding a fourth is a config change, not a code change. Append a section and
use it immediately:

```toml
[profile.batch]
max_num_seqs = 64
```

```bash
inference-lab-serve --profile batch --dry-run
```

Each key becomes a flag by replacing underscores with hyphens, so `max_num_seqs`
emits `--max-num-seqs 64`. A value of `true` emits a bare flag with no value; a
profile setting overrides the same key in `[backend.*]`, which overrides
`[common]`.

Name a profile that does not exist and the command stops with a message listing
the ones it did find, read from the config file rather than from a list kept in
the code:

```text
unknown profile 'typo'; choose from latency, learning, throughput.
```

## Maximum sequences

`max_num_seqs` limits how many sequences can participate in one iteration.
Higher values offer more batching opportunity but require more KV-cache memory
and can increase latency. The profiles set 16 for learning/latency and 128 for
throughput so the difference is visible. These are starting points, not
recommendations for every device.

## Context length

`max_model_len` caps the accepted sequence length. A smaller cap can preserve
memory for more concurrent requests. Do not advertise a context length the
server cannot reliably sustain under expected concurrency.

```bash
inference-lab-serve --model Qwen/Qwen3-0.6B \
  --extra-arg=--max-model-len \
  --extra-arg=8192
```

The explicit configuration file is preferable for repeated experiments.

## Quantization

Quantization represents weights, activations, or the KV cache with fewer bits.
For example, a 4-bit weight representation uses much less storage than 16-bit
weights. Actual memory does not scale perfectly because metadata, temporary
buffers, and unquantized components remain.

The safest beginner method is to select a checkpoint prepared for the intended
backend instead of converting weights during the first experiment:

```bash
# Example Metal model from the vLLM-Metal supported-model matrix
inference-lab-serve --backend metal \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit
```

On NVIDIA, choose a vLLM-supported quantized checkpoint and verify that its
quantization format is efficient on the actual GPU generation. A smaller model
at higher precision can outperform a larger, poorly supported quantized model.

Measure output quality as well as speed. Quantization can change generated
answers, and aggressive formats may be unsuitable for a particular task.

## CUDA memory utilization

The NVIDIA profile contains:

```toml
gpu_memory_utilization = 0.90
```

This controls the fraction of GPU memory vLLM may use for its executor. More
space can support a larger KV cache, but setting it too high leaves insufficient
headroom for the driver, other processes, or transient allocations. Watch for
out-of-memory errors and preemption metrics.

## Experiment sequence

1. Warm the model before measuring.
2. Fix the model, precision, prompt set, and output length.
3. Measure concurrency 1, 4, 8, 16, and 32.
4. Change one profile setting.
5. Repeat at least three times.
6. Record median results and explain outliers.
7. Stop when throughput flattens, errors appear, or p99 becomes unacceptable.

