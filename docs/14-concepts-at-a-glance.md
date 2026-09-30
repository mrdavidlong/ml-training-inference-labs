# Concepts at a glance

One card per concept the labs teach. Every card answers the same four
questions: what it is, when to use it, how it works, and its limitations and
risks. Use this page to review what you have studied or to compare
concepts side by side. Each card links to the guide that teaches it in full.

Where the [glossary](07-glossary.md) defines a term in a sentence, this page
explains a whole idea in a few paragraphs.

Jump to: [Training track](#training-track) · [Inference track](#inference-track)

| Track | Concepts |
|---|---|
| Training | [Supervised learning](#1-supervised-learning) · [Unsupervised learning](#2-unsupervised-learning) · [Reinforcement learning](#3-reinforcement-learning) · [LoRA fine-tuning](#4-fine-tuning-with-lora) |
| Inference | [Prefill and decode](#5-large-language-model-inference-prefill-and-decode) · [vLLM serving](#6-serving-with-vllm) · [Paged attention](#7-paged-attention) · [Continuous batching](#8-continuous-batching) · [Chunked prefill](#9-chunked-prefill-and-the-token-budget) · [Prefix caching](#10-prefix-caching) · [Memory knobs](#11-context-length-memory-utilization-and-quantization) · [Measurement](#12-measurement-percentiles-ttft-and-throughput) · [Frameworks](#13-pytorch-jax-and-vllm) |

---

## Training track

### 1. Supervised learning

*Lab:* `ml-lab-linear` · *Guide:* [docs/09](09-linear-regression.md)

**What it is.** Teaching a model to predict an answer by showing it examples
that already have the correct answer attached (labeled data). Linear
regression is the simplest form: it predicts a number as a weighted sum of the
inputs.

**Use it when** you have historical examples with known outcomes and want to
predict the outcome for new cases, such as house prices, demand, or a risk
score. Linear regression suits relationships that are roughly a straight line.
It also makes a baseline that is easy to explain before you try anything more
complex.

**How it works.**

- **Features** are the input measurements, such as square footage and
  bedrooms. The **label** (or target) is the answer to predict, such as the
  sale price.
- The model learns **weights** and a **bias**. A prediction is
  `w₁·x₁ + w₂·x₂ + … + b`.
- A **loss function** turns all the prediction errors into one number to
  minimize. The lab uses mean squared error (MSE).
- **Gradient descent** repeats one loop:
  1. **Predict** with the current weights.
  2. **Measure** the loss against the true labels.
  3. **Compute the gradient**: which direction each weight should move to
     reduce the loss.
  4. **Update**: move each weight a small step in that direction. The
     **learning rate** sets the step size.
  5. **Repeat** for many **epochs** (full passes over the data).
- The lab normalizes each feature to a similar scale before training. It keeps
  the means and scales so that new inputs can be rescaled the same way.

**Limitations and risks.**

- It can only fit straight-line relationships unless you engineer features by
  hand.
- It is sensitive to outliers. Features on very different scales slow or break
  training, which is why the lab normalizes them.
- A learning rate that is too high diverges; the lab produces `nan` at 0.9. A
  learning rate that is too low crawls.
- **Overfitting**: a model can memorize its training data and still do poorly
  on new data. Always evaluate on data held out from training.
- A large weight shows correlation, not cause.
- Biased or mislabeled data produces a model that is confidently wrong.

### 2. Unsupervised learning

*Lab:* `ml-lab-kmeans` · *Guide:* [docs/10](10-kmeans.md)

**What it is.** Finding structure in data that has no labels. K-means groups
similar points into *k* clusters.

**Use it when** you want to discover groups in unlabeled data, such as
customer segments, document topics, or the main colors in an image. It is also
useful for exploring data before any labels exist.

**How it works (Lloyd's algorithm).**

1. **Initialize** *k* starting centroids (cluster centers). The lab uses the
   k-means++ rule: after a random first pick, points far from existing
   centroids are more likely to be chosen next.
2. **Assign** every point to its nearest centroid.
3. **Update** each centroid to the average of the points assigned to it.
4. **Repeat** steps 2 and 3 until the assignments stop changing
   (**convergence**).

**Inertia** is the total squared distance from every point to its own
centroid. Lower means tighter clusters, but inertia always falls as *k* rises.
To compare different values of *k*, use the **silhouette score** instead.

**Limitations and risks.**

- You must choose *k* in advance. Heuristics such as the elbow method and the
  silhouette score help, but the choice is still a judgment call.
- It assumes clusters are roughly round and similar in size. It fails on
  elongated, nested, or very uneven groups.
- Different starting points can give different answers. Fix `--seed` for
  repeatable runs, and try several seeds before trusting a result.
- It is sensitive to feature scaling and outliers.
- K-means always returns *k* groups, even when the data has no real groups.
  A name like "high-value customers" is your interpretation, not something the
  algorithm found.

### 3. Reinforcement learning

*Lab:* `ml-lab-q-learning` · *Guide:* [docs/11](11-reinforcement-learning.md)

**What it is.** Training an agent to make a sequence of decisions through trial
and error. The agent is guided by rewards instead of labeled answers.

**Use it when** actions affect future situations and the payoff may come much
later. Examples include games, robotics, scheduling, and recommendation
sequences. The same idea underlies reinforcement learning from human
feedback (RLHF), which is used to align large language models.

**How it works.**

- **Core pieces**:
  - **agent**: the learner.
  - **environment**: the world it acts in.
  - **state**: the current situation.
  - **action**: a move the agent can make.
  - **reward**: feedback after each move.
  - **policy**: the agent's rule for choosing actions.
- In Q-learning, the policy comes from a **Q-table**. It estimates the
  long-term reward of taking each action in each state.
- **The loop**:
  1. **Observe** the current state.
  2. **Act**. With probability ε (epsilon), take a random action to
     **explore**. Otherwise **exploit** the best action the table knows. This
     is the ε-greedy rule. The lab decays ε from 0.8 toward 0.02: it explores
     widely early and exploits later.
  3. **Evaluate**. The environment returns a reward and a new state.
  4. **Update** the table toward the reward plus the discounted value of the
     best next action:
     `Q(s,a) ← Q(s,a) + α · [r + γ · max Q(s′,·) − Q(s,a)]`.
     - α (alpha) is the learning rate. The lab uses 0.2.
     - γ (gamma) is the discount factor, which sets how much future reward
       counts. The lab uses 0.95.
- An **episode** is one attempt from start to finish. Over many episodes,
  value propagates backward from the goal.
- At inference time the agent always takes its best-known action and never
  explores.

**Limitations and risks.**

- **Reward hacking**: the agent maximizes exactly what you reward, not what
  you meant. A poorly designed reward produces strange or unsafe shortcuts.
- It is sample-inefficient: it may need thousands or millions of episodes.
- A table works only for small, discrete state spaces. Larger problems need
  function approximation, such as deep Q-networks, which train much less
  stably.
- Exploration in the real world can be expensive or dangerous, which is why
  agents are usually trained in simulation.
- Results are sensitive to α, γ, the ε schedule, and the random seed.

### 4. Fine-tuning with LoRA

*Labs:* `ml-lab-lora`, `ml-lab-lora-infer` · *Guide:* [docs/12](12-lora-finetuning.md)

**What it is.** Adapting a pretrained language model to a new task or style
by training a small add-on called an **adapter**, while the model's own
weights stay unchanged. LoRA stands for low-rank adaptation.

**Use it when** a base model is close to what you need but prompting alone is
not enough, such as for a fixed output format, a domain's vocabulary, or a
consistent tone. It also fits when GPU memory is limited, or when you want
several task-specific variants that share one base model.

**How it works.**

- **Freeze** the base model's weights.
- Beside selected weight matrices, **inject two small matrices, A and B**. The
  lab attaches them to every linear layer (`target_modules="all-linear"`).
  Their product `B·A`, scaled by `lora_alpha / r`, is added to the frozen
  layer's output.
- The **rank** `r` is the inner width of A and B. It sets how much the adapter
  can learn. The lab uses `r=8` and `lora_alpha=16`.
- **The training loop**:
  1. Run a **forward pass** on your examples.
  2. Compute the **loss**: the model's next-token prediction error.
  3. Run a **backward pass**. Gradients reach only A and B.
  4. Take an **optimizer step**. **Gradient accumulation** sums gradients over
     several small batches before each step, which simulates a larger batch in
     limited memory.
- Only the adapter is saved, which is usually a few megabytes. To generate,
  load the base model plus the adapter. You can also merge the adapter into
  the base weights, but the lab keeps them separate so you can compare the two.

**Limitations and risks.**

- It teaches behavior and format far better than new facts. For knowledge,
  retrieval-augmented generation (RAG) is usually the better tool.
- A small or narrow dataset overfits, or causes **catastrophic forgetting**,
  where the model loses general abilities it had before.
- The adapter learns your examples faithfully, including their mistakes.
- The rank, alpha, learning rate, and choice of target layers all need
  experimentation.
- Fine-tuning can weaken a base model's safety behavior. The base model's
  license still applies to anything built on it.

---

## Inference track

### 5. Large language model inference: prefill and decode

*Guides:* [docs/01](01-inference-basics.md), [docs/02](02-vllm-architecture.md#prefill-decode-and-chunked-prefill)

**What it is.** Running a trained model to produce output. For a large
language model (LLM), that means generating text one token at a time.

**Use it when** the model answers any request. These ideas start to matter as
soon as you care about speed, cost, or serving more than one user.

**How it works.**

1. **Tokenize**: the prompt is split into tokens.
2. **Prefill**: the whole prompt is processed in one parallel pass. This phase
   is compute-bound and mostly determines **time to first token (TTFT)**.
3. **Decode**: the model generates one token per step. Each new token attends
   to every earlier token, whose intermediate results are kept in the
   **KV cache** (key/value cache). This phase is limited by memory bandwidth
   and determines **inter-token latency (ITL)**.
4. **Stop** at an end-of-sequence token or a length limit. The tokens are then
   turned back into text and streamed back to the caller.

**Limitations and risks.**

- Latency grows with output length because decode steps run one after another.
- The KV cache grows with context length multiplied by the number of
  concurrent requests. That makes GPU memory, not compute, the usual
  bottleneck.
- Sampling settings such as temperature change the output. The same prompt can
  produce different answers.

### 6. Serving with vLLM

*Lab:* `inference-lab-serve` · *Guide:* [docs/02](02-vllm-architecture.md)

**What it is.** A high-throughput inference engine that serves a model behind
an OpenAI-compatible HTTP API. In this repository, `inference-lab-serve`
detects your hardware and builds the `vllm serve …` command. With `--dry-run`,
it prints the command without running it.

**Use it when** one model must serve many concurrent users efficiently on an
NVIDIA GPU (CUDA) or Apple Silicon (Metal). For a one-off script, plain
PyTorch or `transformers` is simpler.

**How it works.** It combines the techniques in cards 7–10 to keep the
hardware busy and memory well used. This lab layers its settings from
`configs/serve.toml`: `[common]` first, then the backend table, then the
profile (`learning`, `latency`, or `throughput`).

**Limitations and risks.**

- It is a large, fast-moving dependency with version- and hardware-specific
  quirks. Metal support is newer than CUDA support.
- Settings tuned for throughput can hurt single-user latency, and the reverse
  is also true.
- It is not hardened for the open internet. Keep the default `127.0.0.1`
  binding, or put an authenticating gateway in front of it.

### 7. Paged attention

*Guide:* [docs/02](02-vllm-architecture.md#paged-attention)

**What it is.** A way to store the KV cache in fixed-size **blocks** allocated
on demand, much as an operating system pages memory. Without it, each request
reserves one large contiguous region.

**Use it when** many requests of different lengths share one GPU. vLLM always
uses it, so there is nothing to switch on.

**How it works.** Each request has a **block table** that maps its token
positions to physical memory blocks. Blocks are allocated as a sequence grows
and freed when the request finishes. Requests with identical prefixes can
share blocks.

**Limitations and risks.**

- It removes wasted memory but does not add memory. When blocks run out,
  requests wait or are **preempted**, which shows up as latency spikes.
- It needs specialized attention kernels and adds some bookkeeping overhead.

### 8. Continuous batching

*Guide:* [docs/02](02-vllm-architecture.md#continuous-batching) · *Setting:* `max_num_seqs`

**What it is.** A scheduler that adds new requests to the running batch and
removes finished ones at every decode step. Static batching, by contrast,
waits for the whole batch to finish.

**Use it when** traffic is concurrent and output lengths vary. vLLM always
uses it.

**How it works.** At each step the scheduler drops finished sequences and
fills the freed slots from the waiting queue. It admits at most
`max_num_seqs` sequences, within the token budget from card 9. The lab's
`throughput` profile allows 128 sequences; `learning` and `latency` allow 16.

**Limitations and risks.**

- More concurrency raises total throughput but also each request's latency,
  because the GPU is shared with more requests.
- Tail latency (p99) is sensitive to bursts of traffic.

### 9. Chunked prefill and the token budget

*Guide:* [docs/03](03-tuning.md#the-token-budget-and-chunked-prefill) · *Setting:* `max_num_batched_tokens`

**What it is.** Splitting a long prompt's prefill into chunks and interleaving
them with other requests' decode steps.

**Use it when** workloads are mixed: some users send very long prompts while
others are partway through generating.

**How it works.** Each scheduler step processes at most
`max_num_batched_tokens` tokens. A long prefill spans several steps instead of
stalling everyone for one large step. The `latency` profile sets a budget of
2048 tokens; `throughput` sets 16384.

**Limitations and risks.**

- The long prompt gets a slower TTFT so that everyone else sees smoother
  latency.
- A budget that is too small lowers throughput; one that is too large brings
  the stalls back.

### 10. Prefix caching

*Guide:* [docs/03](03-tuning.md#prefix-caching) · *Setting:* `enable_prefix_caching`

**What it is.** Reusing KV cache blocks across requests that start with
identical tokens. The ordinary KV cache, by contrast, is reused only within a
single request.

**Use it when** many requests share a long common beginning, such as a system
prompt, few-shot examples, or a document that is asked about repeatedly. All
three lab profiles enable it.

**How it works.** Each block is identified by a hash of its tokens and the
tokens before it. If a new request's prefix matches blocks already in the
cache, prefill skips recomputing them.

**Limitations and risks.**

- It helps only when prefixes match exactly, token for token. A timestamp at
  the top of the prompt defeats it.
- Cached blocks compete with active requests for memory.
- On a server shared by several tenants, cache timing can in principle reveal
  that someone else sent the same prefix. Treat that as a side channel.

### 11. Context length, memory utilization, and quantization

*Guide:* [docs/03](03-tuning.md#context-length) · *Settings:* `max_model_len`, `gpu_memory_utilization`, `quantization`

**What they are.** The main memory settings.

- **Context length** is the longest prompt plus output that is allowed.
- **GPU memory utilization** is the fraction of GPU memory vLLM may claim.
- **Quantization** stores weights at lower precision, such as 8 or 4 bits.

**Use them when** memory is limiting you.

- Lower the context length to fit more concurrent requests.
- Raise memory utilization on a GPU dedicated to serving.
- Quantize to fit a larger model on smaller hardware, or to cut cost.

**How they work.** GPU memory holds the model weights plus the KV cache.
A shorter context, a higher utilization, or smaller weights all leave room for
more KV blocks, and so for more concurrent requests. The lab uses a
4096-token context and sets utilization to 0.90 on CUDA.

**Limitations and risks.**

- Too short a context rejects or truncates real requests.
- Utilization close to 1.0 risks out-of-memory crashes and starves other
  processes on the same GPU.
- Quantization can lower answer quality, especially for reasoning and
  arithmetic, and the effect varies by model and method. Re-check quality, not
  just speed.

### 12. Measurement: percentiles, TTFT, and throughput

*Lab:* `inference-lab-loadtest` · *Guide:* [docs/05](05-measurement.md)

**What it is.** Measuring how an inference service behaves under realistic
load. You report distributions, not averages.

**Use it when** you run before and after every tuning change, compare
hardware or models, or set service-level objectives (SLOs).

**How it works.**

- **Percentiles**: p50, p95, and p99 latency are the times that 50%, 95%, and
  99% of requests beat. The lab computes them with linear interpolation.
- **TTFT**: measured from the first streamed token, so it appears only with
  `--stream`.
- **Throughput**: completed requests per second. **Success rate**: the share
  of requests that did not fail. Read success rate first; a server that is
  fast because it rejected half the traffic is not fast.
- **Procedure**: warm the server up first, change one variable at a time, and
  collect enough requests for the percentile you care about.
- **Pass/fail gates**:
  - Exit code `2` means the success rate fell below `--min-success-rate`.
  - Exit code `3` means p95 latency exceeded `--max-p95-ms`.

**Limitations and risks.**

- p99 from 20 requests is noise. Tail percentiles need many samples.
- A good average can hide a terrible p99.
- Client-side numbers include network and queueing time; server metrics do
  not. Know which view you are reading.
- Synthetic prompts may not resemble real traffic, so results may not transfer.
- Numbers from `inference_labs/mock_server.py` are `time.sleep` constants for
  tests. Never quote them as performance results.

### 13. PyTorch, JAX, and vLLM

*Guide:* [docs/04](04-pytorch-jax-vllm.md)

**What they are.**

- **PyTorch** is a general deep-learning framework that runs operations
  eagerly, line by line.
- **JAX** is a functional framework built around compiled transformations
  (`jit`, `grad`, `vmap`).
- **vLLM** is a specialized serving engine built on top of PyTorch.

**Use each when:**

- **PyTorch**: for research, training, and fine-tuning. It has the largest
  ecosystem, and the LoRA lab uses it.
- **JAX**: for large-scale training on TPUs (tensor processing units), or
  numerical research that benefits from compilation.
- **vLLM**: for serving a language model to many users. You serve models with
  it; you do not write or train them in it.

**How they relate.** The frameworks provide tensors, automatic
differentiation, and hardware kernels. vLLM adds the serving machinery you
would otherwise have to build yourself: a scheduler, a KV cache manager,
batching, and an HTTP API.

**Limitations and risks.**

- PyTorch's eager mode leaves speed unused unless you compile.
- JAX is harder to learn, and debugging compiled, functional code takes
  practice.
- vLLM is narrow by design: it does not train.
- All three depend on specific hardware and driver versions.

---

## Next

- [Glossary](07-glossary.md) — one-line definitions of every term used above.
- [Training overview](08-training-overview.md) — run all four training labs.
- [Inference fundamentals](01-inference-basics.md) — start the inference track.
