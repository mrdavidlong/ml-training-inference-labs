# Installation and `uv`

## Why use `uv`?

`uv` is a fast Python package and environment manager. It can replace the most
common combination of `python -m venv`, `pip`, and hand-maintained requirement
files while still producing an ordinary `.venv` directory.

For this project:

```bash
uv sync
source .venv/bin/activate
```

`uv sync` reads `pyproject.toml`, creates `.venv` if necessary, and synchronizes
the environment. `source` only adjusts the current terminal's `PATH`; it does
not install a background service.

The linear regression, K-means, Q-learning, inference client, and tests use the
base environment. The LoRA lab has larger optional dependencies:

```bash
uv sync --extra lora
```

This installs PyTorch, Transformers, PEFT, Accelerate, and Datasets. Model
weights are downloaded only when the real LoRA command runs, not during
`uv sync`.

Use the upstream vLLM-Metal installer for its runtime because the plugin needs a
carefully matched vLLM macOS wheel and native Metal components. The current
installer itself uses `uv` internally.

## Official vLLM-Metal installer

For a Mac user who wants to run inference rather than develop vLLM-Metal, use
the official upstream command:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

This downloads and executes the installer maintained in the official
`vllm-project/vllm-metal` repository. It verifies Apple Silicon ARM64, ensures
`uv` is present, creates `~/.venv-vllm-metal`, and installs prebuilt vLLM and
vLLM-Metal packages. It does not leave a source checkout behind.

Like any software installer, it downloads executable packages and
dependencies. Use only the official repository URL shown above. The complete
install, run, and uninstall commands are in the root `README.md`.

## Apple Silicon Metal

The complete commands are in the root `README.md`. Requirements include:

- macOS on Apple Silicon;
- native ARM64 Python 3.12, not Rosetta/x86-64 Python;
- Apple command-line developer tools;
- enough unified memory and disk space for the selected model.

Use `Qwen/Qwen3-0.6B` first. It is deliberately small. Larger models require
more unified memory and take longer to load.

## NVIDIA CUDA

On a Linux host with a working NVIDIA driver:

```bash
nvidia-smi
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
vllm --version
```

Then return to this repository and run:

```bash
uv sync
source .venv/bin/activate
inference-lab-doctor
inference-lab-serve --backend cuda --profile learning
```

If installing this project into a different environment would remove or change
the vLLM packages, keep the serving runtime and client project in separate
terminals and environments. The client communicates over HTTP and does not need
vLLM installed.

## Network safety

The default bind address is `127.0.0.1`, which accepts connections only from
the same machine. Do not bind to `0.0.0.0` on a public cloud instance without
authentication, firewall rules, transport encryption, and a reason to expose
it. SSH port forwarding is safer for a short experiment:

```bash
ssh -L 8000:127.0.0.1:8000 user@remote-host
```

The local client can then use `http://127.0.0.1:8000` while the server remains
private on the remote host.
