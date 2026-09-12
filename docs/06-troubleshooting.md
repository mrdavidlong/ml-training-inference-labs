# Troubleshooting

## Start here: what does `doctor` say?

Nearly every setup problem is visible in one command, and it runs in a second:

```bash
uv run inference-lab-doctor
```

```json
{
  "operating_system": "Darwin",
  "machine": "arm64",
  "backend": "metal",
  "reason": "Apple Silicon on macOS detected",
  "nvidia_smi": false,
  "apple_silicon": true,
  "python": "3.11.15",
  "python_is_native_arm64": true,
  "commands": {
    "vllm": null,
    "uv": "/opt/homebrew/bin/uv",
    "nvidia-smi": null
  },
  "ready": false
}
```

Follow the field that looks wrong:

```text
backend: "cpu"                 -> "Backend detected as cpu"
python_is_native_arm64: false  -> "Apple Silicon reports x86-64"
commands.vllm: null            -> "vllm is not installed"
ready: false, everything else ok -> vLLM runtime not installed (docs/00 step 2)
```

`ready: false` with `backend: metal` and `vllm: null` is the **normal** state
after installing only this repository. The whole training track works in that
state. Only the serving labs need step 2.

## Symptom index

| What you see | Section |
|---|---|
| `No supported accelerator was detected` | [Backend detected as cpu](#backend-detected-as-cpu) |
| `vllm is not installed in the active environment` | [vllm is not installed](#vllm-is-not-installed) |
| `unknown profile 'x'; choose from ...` | [Unknown profile](#unknown-profile) |
| Wrong backend chosen | [Detection chose the wrong backend](#detection-chose-the-wrong-backend) |
| `machine: x86_64` on a Mac | [Apple Silicon reports x86-64](#apple-silicon-reports-x86-64) |
| Server starts, then dies | [Out of memory](#out-of-memory) |
| Model refuses to load on Metal | [The model does not load on Metal](#the-model-does-not-load-on-metal) |
| Client cannot connect | [Healthy server, no connection](#healthy-server-but-the-client-cannot-connect) |
| `"mean_ttft_ms": null` | [TTFT is null](#ttft-is-null) |
| `ModuleNotFoundError: torch` | [Missing optional dependency](#missing-optional-dependency) |
| CI fails on `uv sync --locked` | [The lock file is stale](#the-lock-file-is-stale) |
| Results vary wildly between runs | [Unstable measurements](#unstable-measurements) |

## Backend detected as cpu

```text
No supported accelerator was detected. Use an Apple Silicon Mac, an NVIDIA GPU,
or pass --backend mock only for the test fixture. See docs/06-troubleshooting.md.
```

`inference-lab-serve` refuses to run on CPU on purpose. A CPU run would take
many seconds per response, teach nothing accurate about latency or throughput,
and quietly hide a broken accelerator setup. Refusing is kinder than complying.

Detection order is Apple Silicon → usable `nvidia-smi` → `cpu`. Landing on
`cpu` means neither was found. Check:

```bash
uname -m                # arm64 on Apple Silicon
nvidia-smi              # should list an actual device
```

To exercise the client plumbing without an accelerator, use the mock server —
never for performance numbers:

```bash
uv run python -m inference_labs.demo --requests 40 --concurrency 8
```

## vllm is not installed

```text
vllm is not installed in the active environment; follow docs/00-installation.md
```

The detector found a supported backend, but no `vllm` executable on `PATH`.
Almost always this means you are in the wrong terminal — the vLLM runtime lives
in its own environment:

```bash
source ~/.venv-vllm-metal/bin/activate     # Apple Silicon
vllm --version
```

Note this check is skipped for `--dry-run`, so you can always preview the
command on a machine with no vLLM installed:

```bash
inference-lab-serve --dry-run       # works regardless
```

## Unknown profile

```text
unknown profile 'typo'; choose from latency, learning, throughput. See docs/03-tuning.md.
```

The list in that message is read from `configs/serve.toml` at runtime, not
hardcoded — so if a profile you just added is missing from it, your TOML edit
did not land where you think. Confirm which file is being read:

```bash
inference-lab-serve --config configs/serve.toml --profile yourprofile --dry-run
```

## Detection chose the wrong backend

Preview what each would do, without starting anything:

```bash
inference-lab-doctor
inference-lab-serve --backend metal --dry-run
inference-lab-serve --backend cuda --dry-run
```

Inside a container, `nvidia-smi` may be installed even though no GPU device was
passed through. Verify it lists a device *from the same environment* that will
start vLLM — not from the host.

One trap worth knowing: **the `INFERENCE_BACKEND` environment variable
overrides the `--backend` flag**, not the other way round. If an explicit flag
seems to be ignored, check for a leftover variable:

```bash
echo "$INFERENCE_BACKEND"
unset INFERENCE_BACKEND
```

## Apple Silicon reports x86-64

Your terminal or Python is running under Rosetta translation. Metal will not
work. Check both:

```bash
uname -m
uv run python -c 'import platform; print(platform.machine())'
```

Both must print `arm64`. If Python disagrees with the shell, the Python install
is the problem — not the terminal. Install a native ARM64 Python and rebuild
the environment:

```bash
rm -rf .venv
uv sync
```

## The model does not load on Metal

Check the current vLLM-Metal supported-model matrix. Some checkpoints use a
weight format the MLX path does not accept, and the error can be obscure.

Start from a known-good small model and change one thing at a time:

```bash
inference-lab-serve --backend metal --model Qwen/Qwen3-0.6B
```

MLX-community checkpoints are generally a safe next step:

```bash
inference-lab-serve --backend metal --model mlx-community/Qwen2.5-7B-Instruct-4bit
```

## Out of memory

Usually the KV cache, not the weights — the model loads fine and the server
dies once real requests arrive. Try these **in order**, changing one at a time:

1. Stop other GPU-heavy applications (browsers count).
2. Reduce `max_num_seqs` — fewer concurrent sequences, less cache.
3. Reduce `max_model_len` — less reserved capacity per request.
4. Use a smaller model.
5. Use a supported quantized checkpoint.
6. On CUDA only, adjust `gpu_memory_utilization`. Both directions are
   plausible, so measure rather than guess:

```text
lower it  -> if the driver or transient allocations need headroom
raise it  -> if vLLM allocated too little KV cache and memory sits free
```

Check `/metrics` for preemptions before changing anything. **Preemptions above
zero means vLLM ran out of KV cache and evicted work** — that points squarely
at steps 2 and 3.

```bash
curl -s http://127.0.0.1:8000/metrics | grep -i preempt
```

## Healthy server, but the client cannot connect

Check in this order:

```bash
curl http://127.0.0.1:8000/health          # from the SERVER machine first
```

- If that fails, the server is not actually up — read its terminal output.
- If it succeeds locally but not remotely, it is bound to `127.0.0.1`, which
  accepts connections only from the same machine. That is the safe default.
- On a remote host, forward the port rather than rebinding:

```bash
ssh -L 8000:127.0.0.1:8000 user@remote-host
```

- Then check security groups and local firewall rules.

Do not solve a connectivity problem by exposing an unauthenticated endpoint to
the internet. If you must bind publicly, set `VLLM_API_KEY` first — see
[installation](00-installation.md#network-safety).

## TTFT is null

```json
  "mean_ttft_ms": null,
  "p95_ttft_ms": null
```

Not a bug. TTFT is measured from the first streamed chunk, and without
streaming there are no chunks to time. Add `--stream`:

```bash
inference-lab-loadtest --requests 40 --concurrency 8 --stream
```

## Missing optional dependency

```text
Install the LoRA dependencies with: uv sync --extra lora
```

Heavy dependencies are imported lazily inside `main()`, so a base install never
breaks at import time — but the lab exits when you actually run it:

```bash
uv sync --extra lora
```

`labs/pytorch_inference.py` points at https://pytorch.org/get-started/locally/
and `labs/jax_inference.py` at https://docs.jax.dev/en/latest/installation.html
instead, since neither is a project extra.

To confirm your arguments before a multi-gigabyte download:

```bash
uv run ml-lab-lora --dry-run
```

## The lock file is stale

CI runs `uv sync --locked`, which fails rather than silently updating. If you
edited `pyproject.toml`, refresh the lock and commit it:

```bash
uv lock
git add uv.lock
```

## Unstable measurements

If two identical runs disagree substantially, suspect the measurement before
the server:

| Cause | Fix |
|---|---|
| No warm-up | Discard the first few requests ([docs/05](05-measurement.md#warm-up)) |
| Too few requests | p99 needs hundreds; report `n` alongside |
| Thermal throttling | Let the machine cool; watch for sustained-load droop |
| Other processes | Close browsers and other GPU users |
| One run only | Run at least three; compare medians |
| Model still loading | Wait for the server's ready line before starting |

## Reading a failure correctly

Failures in this project surface as a message pointing at a guide, not a
traceback:

```text
SystemExit: <what went wrong>. See docs/0N-....md.
```

If you get a bare Python traceback instead, that is a genuine bug rather than a
configuration problem — the message is the intended interface.

## Still stuck

Collect this before asking for help; it answers most first questions:

```bash
uv run inference-lab-doctor
uv run inference-lab-serve --dry-run
uv --version && uv run python -V
uname -m
vllm --version           # in the vLLM environment
```

## Next

- [Installation](00-installation.md) — the two-install layout.
- [Tuning](03-tuning.md) — changing settings deliberately.
- [Measurement](05-measurement.md) — making the numbers trustworthy.
