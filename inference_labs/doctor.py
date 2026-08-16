"""Report what this machine can run, before you try to run anything.

Setting up GPU inference has many ways to go subtly wrong: the driver is
missing, the wrong Python is active, vLLM was installed into a different
environment. Each failure normally shows up as a confusing error halfway
through a long startup. This command answers the question up front and prints
everything as JSON, which is easy to paste into a bug report.

    $ inference-lab-doctor
    {
      "operating_system": "Darwin",
      "machine": "arm64",
      "backend": "metal",
      "reason": "Apple Silicon on macOS detected",
      ...
      "ready": false          <- look here first
    }

If "ready" is false, the fields above it say why.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys

from inference_labs.hardware import BACKENDS, detect_backend


def main() -> None:
    """Inspect the environment and print a JSON report.

    Nothing is returned and nothing is validated or refused -- this command is
    purely diagnostic, and is expected to be useful precisely on machines that
    cannot serve.
    """
    parser = argparse.ArgumentParser(description="Inspect the local inference environment")
    # Accepting --backend lets you preview what *would* happen under an
    # override, without setting an environment variable and forgetting it.
    parser.add_argument("--backend", choices=BACKENDS, default="auto")
    args = parser.parse_args()
    info = detect_backend(args.backend)

    report = {
        # `**` splats the HardwareInfo fields in as top-level keys.
        **info.to_dict(),
        # Just the version, e.g. "3.12.4", dropping the compiler details that
        # sys.version also carries.
        "python": sys.version.split()[0],
        # A specific Mac trap: an Intel-built Python running under Rosetta
        # translation on an Apple Silicon machine. It works, but cannot reach
        # the GPU, so the Metal backend silently will not function. This is
        # trivially true on non-Apple machines, hence the `not apple_silicon`.
        "python_is_native_arm64": not info.apple_silicon or platform.machine().lower() == "arm64",
        # Where each tool was found, or null if it is not on PATH. The path
        # itself is the useful part: it reveals a vllm installed into a
        # different virtual environment than the active one.
        "commands": {
            "vllm": shutil.which("vllm"),
            "uv": shutil.which("uv"),
            "nvidia-smi": shutil.which("nvidia-smi"),
        },
        # The bottom line: both a real accelerator and an installed vLLM are
        # needed before `inference-lab-serve` can do anything.
        "ready": shutil.which("vllm") is not None and info.backend in {"metal", "cuda"},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
