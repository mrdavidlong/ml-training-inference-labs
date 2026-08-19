# Installation and `uv`

There are **two separate installations** in this project, and conflating them
is the most common setup mistake:

```text
1. THIS REPOSITORY              2. THE vLLM RUNTIME
   uv sync                         vLLM-Metal installer, or uv pip install vllm
   |                               |
   pure standard library           a large accelerator-specific package
   runs: the training labs,        runs: the actual model server
         the load tester,
         the tests
                                   Needed ONLY for the real serving labs.
```

You can complete every training lab, the measurement workflow, and the whole
test suite with **only step 1**. Step 2 is required just to serve a real model,
and it can live in a different environment — even on a different machine —
because the client talks to it over HTTP.

## Quick start

```bash
git clone https://github.com/mrdavidlong/ml-training-inference-labs
cd ml-training-inference-labs
uv sync
uv run ml-lab-linear
```

If that last command prints a training summary, you are ready for the entire
training track. Nothing else is needed:

```text
rows: 20 (train=16, test=4)
training MSE: 16205.03 -> 83.66
test RMSE: 21.33 thousand dollars
sample prediction for 1,800 sq ft, 3 bedrooms, age 12: $612k
```

## Why `uv`?

`uv` is a fast Python package and environment manager. It replaces the usual
combination of `python -m venv`, `pip`, and hand-maintained requirements files,
while still producing an ordinary `.venv` directory you can inspect.

Two ways to use it:

```bash
uv run ml-lab-linear          # runs in the project environment, no activation
                              # this is what the guides use throughout

source .venv/bin/activate     # or activate once, then use plain commands
ml-lab-linear
```

`uv run` is recommended while learning because it cannot silently use the wrong
environment. `source` only edits the current terminal's `PATH`; it starts no
background service and affects no other window.

Useful commands:

| Command | What it does |
|---|---|
| `uv sync` | Creates `.venv` if needed and installs the base project |
| `uv sync --extra lora` | Adds PyTorch, Transformers, PEFT, Accelerate, Datasets |
| `uv sync --locked` | Fails if `uv.lock` is stale — this is what CI runs |
| `uv run <command>` | Runs inside the project environment |
| `uv run --with scikit-learn python ...` | Adds a package for one command only |

## What the base install contains

Nothing. That is deliberate:

```bash
uv sync
uv run python -c "import urllib.request, http.server, asyncio, tomllib; print('all standard library')"
```

The training labs implement gradient descent, Lloyd's algorithm, and Q-learning
using only Python's standard library, so you can read every line without
knowing NumPy. The inference client uses `urllib` and `asyncio`. No NumPy, no
PyTorch, no vLLM.

Heavier dependencies live behind an extra and are imported *lazily inside the
program*, so a base install never breaks:

```bash
uv run ml-lab-lora --help     # works without the extra installed
uv run ml-lab-lora            # exits with an install hint if it is missing
```

## Requirements

| | Minimum | Notes |
|---|---|---|
| Python | 3.11 | `tomllib` arrived in 3.11; CI tests on 3.12 |
| Disk | ~1 GB base | The LoRA extra adds several GB for PyTorch |
| Accelerator | none for training | Apple Silicon or NVIDIA for the serving labs |

Check what you have:

```bash
uv --version
uv run python -V
uv run inference-lab-doctor
```

## The `doctor` command

`inference-lab-doctor` answers "will the serving labs work on this machine?"
before you spend ten minutes on a download that fails. It prints JSON:

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

Read it in this order:

- **`backend`** — `metal`, `cuda`, or `cpu`. `cpu` means no supported
  accelerator was found, and the serving labs will refuse to start.
- **`python_is_native_arm64`** — on a Mac this must be `true`. `false` means
  Python is running under Rosetta translation and Metal will not work. See
  [troubleshooting](06-troubleshooting.md#apple-silicon-reports-x86-64).
- **`commands.vllm`** — the path to the `vllm` executable, or `null` if it is
  not on your `PATH`. Above it is `null`, which is exactly why `ready` is
  `false`.
- **`ready`** — `true` only when both a supported backend and a `vllm`
  executable were found.

`ready: false` with `backend: metal` is the normal state after step 1 and
before step 2. The training track does not care.

## Step 2a — Apple Silicon (Metal)

Use the official upstream installer. vLLM-Metal needs a carefully matched macOS
wheel plus native Metal components, so installing it into this project's
environment is not the supported path:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

This verifies you are on Apple Silicon ARM64, ensures `uv` is present, creates
`~/.venv-vllm-metal`, and installs prebuilt vLLM and vLLM-Metal packages. It
leaves no source checkout behind. Like any installer it downloads and executes
code — use only the official URL above.

Then, in the terminal where you want to serve:

```bash
source ~/.venv-vllm-metal/bin/activate
vllm --version
```

Requirements: macOS on Apple Silicon, native ARM64 Python 3.12 (not Rosetta),
Apple command-line developer tools, and enough unified memory and disk for the
model. Start with `Qwen/Qwen3-0.6B`; it is deliberately tiny.

## Step 2b — NVIDIA (CUDA)

On a Linux host with a working NVIDIA driver:

```bash
nvidia-smi                                  # must list your GPU
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
vllm --version
```

`nvidia-smi` succeeding is not proof of a usable GPU — inside a container it
can be present with no device passed through. Confirm it lists an actual device
from the same environment that will start vLLM.

## Two terminals, two environments

This is the layout that avoids most trouble. The serving runtime and the client
never need to be in the same environment, because they communicate over HTTP:

```text
TERMINAL 1 — the server                TERMINAL 2 — the client
source ~/.venv-vllm-metal/bin/activate cd ml-training-inference-labs
   (or the CUDA venv)                  uv run inference-lab-loadtest
                                                     |
cd ml-training-inference-labs                        | HTTP
uv run inference-lab-serve --profile learning        v
   -> listening on 127.0.0.1:8000  <-----------------+
```

If installing this project into the vLLM environment would change or remove the
vLLM packages, keep them separate exactly as above. The client needs no vLLM
install at all.

## Verify the whole setup

```bash
uv run python -m unittest discover -s tests -v    # should end in OK
uv run inference-lab-doctor                       # check backend and ready
uv run inference-lab-serve --dry-run              # prints the command only
```

`--dry-run` works even with no vLLM installed, which makes it a safe way to
confirm the configuration before committing to a model download.

No accelerator at all? This runs the complete measurement workflow against the
built-in mock server:

```bash
uv run python -m inference_labs.demo --requests 40 --concurrency 8
```

The mock server's latencies are `time.sleep` constants. It teaches the
workflow; its numbers are meaningless as performance results.

## Network safety

The default bind address is `127.0.0.1`, which accepts connections only from
the same machine. Do not bind `0.0.0.0` on a cloud instance without
authentication, firewall rules, and a reason to expose it. For a short
experiment on a remote host, forward the port over SSH instead:

```bash
ssh -L 8000:127.0.0.1:8000 user@remote-host
```

Your local client then uses `http://127.0.0.1:8000` while the server stays
private on the remote machine.

If you must expose the port, set an API key. `inference-lab-serve` reads it
from the environment and appends `--api-key`, so the secret never appears in
your shell history:

```bash
export VLLM_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
inference-lab-serve --host 0.0.0.0
```

## Common installation problems

| Symptom | Likely cause | Fix |
|---|---|---|
| `ready: false`, `vllm: null` | Step 2 not done, or wrong terminal | Activate the vLLM environment |
| `python_is_native_arm64: false` | Rosetta/x86-64 Python | Install native ARM64 Python, recreate `.venv` |
| `backend: cpu` on a GPU machine | Driver or container passthrough | [Troubleshooting](06-troubleshooting.md) |
| CI fails on `uv sync --locked` | `pyproject.toml` edited, lock stale | Run `uv lock` and commit `uv.lock` |
| `ModuleNotFoundError: torch` | LoRA extra not installed | `uv sync --extra lora` |

More detail in [troubleshooting](06-troubleshooting.md).

## Next

- [Training overview](08-training-overview.md) — start here if you have no GPU.
- [Inference fundamentals](01-inference-basics.md) — start here for the serving
  track.
