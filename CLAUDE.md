# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Run everything from the repository root (the directory holding `pyproject.toml`); `uv` creates `.venv/` there.

```bash
uv sync                                              # base install: no third-party runtime deps
uv sync --extra lora                                 # adds torch, transformers, peft, datasets, accelerate
uv run python -m unittest discover -s tests -v       # full suite (what CI runs)
uv run python -m unittest tests.test_serve -v        # one module
uv run python -m unittest tests.test_serve.ServeCommandTest.test_metal_omits_cuda_memory_flag
```

There is no linter, formatter, or type checker configured. CI (`.github/workflows/tests.yml`) runs `uv sync --locked` then the unittest suite on Python 3.12 — if you touch `pyproject.toml`, refresh `uv.lock` or CI fails on the lock check.

Lab entry points are declared in `[project.scripts]`: `inference-lab-doctor`, `inference-lab-serve`, `inference-lab-loadtest`, `ml-lab-linear`, `ml-lab-kmeans`, `ml-lab-q-learning`, `ml-lab-lora`, `ml-lab-lora-infer`.

## Architecture

This is a teaching repository. The code exists to be read by beginners, and the numbered guides in `docs/` are as much the product as the Python is. Two independent tracks share only the packaging.

### Dependency policy

The base install has **zero third-party runtime dependencies**. `inference_labs` uses `urllib`, `http.server`, `asyncio`, and `tomllib`; `training_labs/{linear_regression,kmeans,q_learning}.py` implement gradient descent, Lloyd's algorithm, and Q-learning against the standard library alone. That is deliberate — a reader can follow every line without knowing NumPy.

Heavy dependencies live behind extras and are imported **lazily inside `main()`**, raising `SystemExit` with an install hint rather than failing at import time (`training_labs/lora_finetune.py:262`, `labs/pytorch_inference.py:53`). Preserve this pattern; a module-level `import torch` would break `uv sync` users and the test suite.

### Inference track

`inference_labs/serve.py` does not serve anything. It detects hardware, builds a `vllm serve ...` argv, prints it, and `subprocess.run`s it — so `--dry-run` shows the exact command a reader could type by hand.

`build_command` layers three TOML tables from `configs/serve.toml` in precedence order: `[common]` → `[backend.<metal|cuda>]` → `[profile.<learning|latency|throughput>]`, then converts each key to a flag by `"--" + name.replace("_", "-")`. Booleans become bare flags when true. **New tuning knobs belong in the TOML, not in Python** — that is what makes the profile comparison the lab teaches (`docs/03-tuning.md`) a data change rather than a code change.

`inference_labs/hardware.py:112` (`detect_backend`) is the single source of backend truth, shared by `serve` and `doctor`. Note that `INFERENCE_BACKEND` overrides the *argument*, so the env var beats an explicit `--backend` flag. Detection order is Apple Silicon → usable `nvidia-smi` → `cpu`; `serve` refuses to run on `cpu`.

`loadtest.py` drives any OpenAI-compatible endpoint. TTFT is measured by timing the first non-`[DONE]` SSE `data:` line, so it is only populated with `--stream`. `metrics.summarize` does linear-interpolation percentiles and reports p50/p95/p99 — the lab's point is that a single fast run proves nothing. Exit codes are meaningful: `2` for success rate below `--min-success-rate`, `3` for p95 above `--max-p95-ms`.

`inference_labs/mock_server.py` exists **only** so tests can exercise streaming, error handling, and metric math without downloading a model. Its latencies are `time.sleep` constants. Never quote numbers from it as performance results, and never point a learning command at it (README "Mock server policy").

### Training track

Each lab is a self-contained script with a `main()` and an argparse CLI, reading a tiny inspectable dataset from `data/`. The LoRA lab is the bridge to the inference track: `lora_finetune.py` freezes a small base model, injects a PEFT adapter, runs an explicit visible training loop with gradient accumulation, and saves to `artifacts/` (gitignored); `lora_inference.py` loads base + adapter to generate. Device selection there is independent of `inference_labs.hardware` — it prefers CUDA → MPS → CPU via `torch` directly.

### Documentation coupling

`docs/` is a numbered study path referenced from the README's table of contents, and the README duplicates key command sequences. Behavior changes to a lab's flags, defaults, or output usually need a matching edit in its guide (`docs/09`–`docs/13` for training, `docs/00`–`docs/07` for inference) and possibly the README. `RESULTS.md` is a worksheet template for recording experiment runs.

## Conventions

- Every module starts with `from __future__ import annotations` and uses builtin generics (`list[float]`, `dict[str, str]`).
- Frozen dataclasses with a `to_dict()` for anything that gets serialized to JSON (`HardwareInfo`, `Summary`, `Result`).
- Failures surface as `SystemExit` with a message pointing at the relevant doc, not as tracebacks.
- Defaults are conservative and local-only (`127.0.0.1`, small context, low concurrency) because a beginner runs them first.
- Prose in docs and help text spells out acronyms on first use and avoids hype; keep new text in that register.
