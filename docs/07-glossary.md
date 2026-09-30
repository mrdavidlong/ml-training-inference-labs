# Glossary

Terms used across this repository, grouped by where you will meet them. Each
entry says what it means, and where it shows up in the labs.

For a fuller summary of each concept — what it is, when to use it, how it
works, and its risks — see [Concepts at a glance](14-concepts-at-a-glance.md).

Jump to: [Training](#training-concepts) · [Inference](#inference-concepts) ·
[Serving and performance](#serving-and-performance) ·
[Measurement](#measurement-and-reliability) ·
[Hardware and frameworks](#hardware-and-frameworks) ·
[Easily confused](#easily-confused-pairs)

## Training concepts

**Feature** — an input value the model reads. Square footage, in
`data/house_prices.csv`.

**Label** (or **target**) — the correct answer for a training example. The sale
price, in the same file. Unsupervised learning has none.

**Parameter** — a number the algorithm learns. The weights and bias in linear
regression; a Q-table entry in Q-learning. You do not choose these.

**Hyperparameter** — a setting *you* choose before training: learning rate,
epochs, number of clusters, discount factor. Tuning means searching these.

**Loss function** — turns prediction errors into one number to minimize.
Squaring errors (see MSE) makes every miss positive and punishes big misses
disproportionately.

**Gradient** — the direction and rate at which loss changes as a parameter
changes. "Which way is downhill, and how steep?" Gradient descent repeatedly
steps downhill.

**Learning rate** — how big each downhill step is. Too small and training
crawls; too large and it overshoots and diverges. `ml-lab-linear` produces `nan`
at a learning rate of 0.9 — see [docs/09](09-linear-regression.md).

**Epoch** — one complete pass through the training data.

**Batch** — a group of examples processed before one parameter update. Distinct
from an inference batch (below), which is a group of *requests*.

**Gradient accumulation** — summing gradients over several small batches before
updating, to imitate a larger batch that would not fit in memory. Used in
`training_labs/lora_finetune.py`.

**Overfitting** — memorizing the training data instead of learning a general
rule. Detected by a train/test gap: low training error, high test error.

**Generalization** — the opposite: performing well on examples never seen
during training. The actual goal.

**Train/test split** — holding data back from training so you can measure
generalization honestly. `ml-lab-linear` uses 16 rows to train and 4 to test.

**Standardization** — rescaling features to comparable ranges so that one large
feature (square footage in thousands) does not swamp a small one (bedrooms,
1–5).

**Supervised / unsupervised / reinforcement / transfer learning** — the four
strategies in this repository, distinguished by what feedback exists. See
[docs/08](08-training-overview.md).

## Inference concepts

**Inference** — using a model with its parameters frozen to produce output for
new input. No learning happens.

**Token** — a chunk of text as the tokenizer sees it: a word, word fragment,
punctuation mark, or space. Models see integer IDs, never characters. `" is"`
and `"is"` are different tokens.

**Tokenizer** — the component converting text to token IDs and back.

**Context length** — the maximum number of tokens a model can attend to at
once. Capped per-server by `max_model_len`.

**Prefill** — processing all prompt tokens, which can happen in parallel.
Compute-bound. Produces the first output token.

**Decode** — generating tokens one at a time, each depending on the last.
Memory-bandwidth-bound. Determines how fast text appears after the first token.

**Attention** — the mechanism comparing the current position against earlier
ones, using a query, key, and value vector per token.

**KV cache** — stored keys and values from earlier tokens, so they are computed
once rather than recomputed at every step. The main consumer of memory on a
busy server.

**Autoregressive** — generating output one token at a time, feeding each result
back in as input.

**Greedy decoding** — always choosing the highest-probability next token. Makes
output deterministic, which is why `training_labs/lora_inference.py` uses it for
before/after comparisons.

**Temperature / top-p / top-k** — settings that control how the next token is
sampled. They affect variety and reproducibility, not really speed.

## Serving and performance

**Serving engine** — software that holds a model in memory and answers requests
over a network. vLLM is one. PyTorch is not.

**vLLM** — the serving engine used here. Started by `inference_labs/serve.py`
as a separate process; nothing in this repository imports it.

**Paged attention** — managing KV-cache memory in small fixed-size blocks
rather than one contiguous reservation per request, so memory is allocated as a
sequence actually grows. See [docs/02](02-vllm-architecture.md#paged-attention).

**Block table** — the per-sequence list mapping its logical token positions to
the physical blocks holding them.

**Continuous batching** — rebuilding the batch after every step so finished
requests leave immediately and queued ones join mid-flight. Contrast with
**static batching**, which holds the group until all members finish.

**Chunked prefill** — splitting a long prompt's prefill across several steps so
it cannot stall other requests' decoding.

**APC — Automatic Prefix Caching** — reusing cached blocks across *different*
requests that begin with identical tokens, such as a shared system prompt or
document. Enabled by `enable_prefix_caching = true`.

**Preemption** — the server evicting in-flight work because it ran out of KV
cache. Any nonzero preemption count is a warning sign; see
[docs/06](06-troubleshooting.md#out-of-memory).

**Quantization** — representing weights, activations, or cache entries with
fewer bits. Saves memory, may change output quality.

**Tensor parallelism** — splitting one model across several GPUs so it fits.
Adds communication overhead.

**OpenAI-compatible API** — the HTTP request/response *shape* popularized by
OpenAI. A compatible server neither calls OpenAI nor uses its models; it just
speaks the same wire format, which is why one client can target any of them.

**SSE — Server-Sent Events** — the streaming format carrying incremental
`data:` chunks over one HTTP response. How token-by-token output arrives.

## Measurement and reliability

**Latency** — how long one request took.

**Throughput** — how much work completed per second across the server. Can
improve while latency worsens.

**Goodput** — throughput counting only work that met the objective. A server
answering 1,000 req/s but missing its target on half has a goodput of 500.

**TTFT — Time to First Token** — delay from request start until the first token
arrives. Dominated by prefill and queueing. Requires `--stream` to measure.

**ITL — Inter-Token Latency** — the gap between generated tokens. Determines
whether output feels smooth or stuttering.

**p50 / p95 / p99 — percentiles** — p95 = 800 ms means 95% of requests finished
within 800 ms. **p50 is the median.** Percentiles expose the tail that an
average conceals.

**Tail latency** — the slow end of the distribution (p95, p99), where user
complaints come from.

**Warm-up** — discarded initial requests that would otherwise include model
initialization and compilation in your steady-state numbers.

**SLI — Service-Level Indicator** — a measured quantity, e.g. successful-request
rate.

**SLO — Service-Level Objective** — a target for an SLI, e.g. "p95 TTFT under
750 ms for this workload."

**SLA — Service-Level Agreement** — an external, often contractual commitment.
An SLO you promised to someone else.

## Hardware and frameworks

**GPU — Graphics Processing Unit** — a highly parallel processor suited to the
large matrix operations models are made of.

**Accelerator** — any such hardware: a GPU, Apple's integrated GPU, a TPU.

**CUDA — Compute Unified Device Architecture** — NVIDIA's programming platform
and software stack.

**Metal** — Apple's GPU programming interface on Apple Silicon.

**MLX** — Apple's array and machine-learning framework, used as vLLM-Metal's
compute backend.

**MPS — Metal Performance Shaders** — the backend PyTorch uses to run on Apple
GPUs. Used by the LoRA labs; deliberately *not* by
`labs/pytorch_inference.py`.

**Unified memory** — Apple Silicon's single memory pool shared by CPU and GPU.
Why there is no `gpu_memory_utilization` setting on Metal.

**PyTorch** — a general framework that runs operations as it meets them.

**JAX** — a framework built on transformations (`jit`, `vmap`, `grad`) that
compiles whole functions before running them.

**XLA — Accelerated Linear Algebra** — the compiler JAX targets.

**JIT — Just-In-Time compilation** — compiling once the needed shapes and types
are known, then reusing the compiled code.

**LLM — Large Language Model** — a model trained to process and generate
language tokens.

**LoRA — Low-Rank Adaptation** — freezing a base model and training small
low-rank matrices alongside it, so a fraction of a percent of the parameters
change. See [docs/12](12-lora-finetuning.md).

**PEFT — Parameter-Efficient Fine-Tuning** — the family of methods LoRA belongs
to; also the library used here.

**Adapter** — the small trained result of LoRA. Not a model on its own; it must
be loaded alongside its base.

**Base model** — the pretrained model an adapter attaches to.

**API — Application Programming Interface** — a contract between programs.

**HTTP — Hypertext Transfer Protocol** — the protocol the serving API speaks.

## Machine-learning metrics

**MSE — Mean Squared Error** — the average squared difference between
prediction and target. In *squared* units, so hard to interpret directly.

**RMSE — Root Mean Squared Error** — the square root of MSE, back in the
target's original units. `ml-lab-linear` reports "test RMSE: 21.33 thousand
dollars", which is directly meaningful.

**Inertia** — total squared distance from points to their assigned cluster
centers. Always falls as K rises, so it **cannot** choose K on its own.

**Silhouette score** — compares how close each point is to its own cluster
versus the nearest other cluster, from −1 to 1. Unlike inertia, it *can* choose
K. See [docs/10](10-kmeans.md).

**Q-value** — an estimate of total future reward for taking an action in a
state. "Q" is for quality.

**Policy** — the agent's rule for choosing an action. With a Q-table, the
policy is simply "take the highest-valued action here."

**Reward** — the environment's immediate feedback signal.

**Discount factor** — how much future rewards count relative to immediate ones.

**Epsilon** — the probability of exploring randomly rather than exploiting the
best known action. Usually decayed over time.

**RL — Reinforcement Learning** — learning behavior from actions and delayed
rewards rather than from labeled answers.

## Easily confused pairs

| These sound alike but differ | |
|---|---|
| **Batch** (training) | A group of *examples* before a parameter update |
| **Batch** (inference) | A group of *requests* processed in one pass |
| **Latency** | Time for one request |
| **Throughput** | Work per second overall — can move the opposite way |
| **MSE** | Squared units; good for optimizing |
| **RMSE** | Original units; good for reporting |
| **Inertia** | Always improves with more clusters; cannot pick K |
| **Silhouette** | Peaks at a good K; can pick K |
| **SLI** | The thing you measure |
| **SLO** | The target for it |
| **SLA** | The promise to someone else |
| **KV cache** | Reuse *within* one request |
| **Prefix caching** | Reuse *across* different requests |
| **Static batching** | Group fixed until all finish |
| **Continuous batching** | Group rebuilt every step |
| **Prefill** | Prompt processing; compute-bound; sets TTFT |
| **Decode** | Token generation; memory-bound; sets ITL |
| **Base model** | Complete and usable alone |
| **Adapter** | Useless without its base model |
| **Metal** | Apple's GPU interface |
| **MPS** | PyTorch's backend that uses it |
| **MLX** | Apple's ML framework, used by vLLM-Metal |
| **Parameter** | Learned by the algorithm |
| **Hyperparameter** | Chosen by you |
