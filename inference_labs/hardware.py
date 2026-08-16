"""Work out which accelerator this machine has, so the other tools can adapt.

Running a language model is arithmetic-heavy, and which chip does that
arithmetic changes both the speed and the software needed:

    Apple Silicon Mac  ->  "metal"  ->  vLLM with the MLX/Metal backend
    NVIDIA graphics card -> "cuda"  ->  vLLM with the CUDA backend
    anything else      ->  "cpu"    ->  too slow to serve; the tools stop here

This module is the single place that decision is made. Both `doctor` (which
reports it) and `serve` (which acts on it) call `detect_backend`, so they can
never disagree about what hardware you have.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


# Every value the --backend flag accepts.
#   auto  - decide by inspecting the machine (the default)
#   metal - force Apple GPU
#   cuda  - force NVIDIA GPU
#   cpu   - no accelerator; serving refuses to start
#   mock  - the fake server in mock_server.py, for tests only
BACKENDS = ("auto", "metal", "cuda", "cpu", "mock")


@dataclass(frozen=True)
class HardwareInfo:
    """The result of inspecting the machine: what was found, what was chosen,
    and why.

    Frozen (immutable) because it is a snapshot of a decision already made.
    Nothing downstream should be able to quietly alter it.
    """

    # The operating system name from Python, e.g. "Darwin" for macOS, "Linux".
    # Note "Darwin" is macOS's internal name -- it is not a separate OS.
    operating_system: str
    # The processor architecture, lowercased, e.g. "arm64" on Apple Silicon or
    # "x86_64" on an Intel or AMD machine.
    machine: str
    # The chosen backend: one of BACKENDS, but never "auto" -- by the time this
    # exists, "auto" has been resolved into a concrete answer.
    backend: str
    # A short human-readable explanation of why that backend was chosen, printed
    # so a surprising result can be understood rather than just accepted.
    reason: str
    # Whether a working NVIDIA GPU was found. Recorded even when it was not the
    # one chosen, which is what makes "I have a GPU but it picked cpu"
    # diagnosable.
    nvidia_smi: bool
    # Whether this is an Apple Silicon Mac (M-series), as opposed to an older
    # Intel Mac, which has no usable GPU backend here.
    apple_silicon: bool

    def to_dict(self) -> dict[str, str | bool]:
        """Convert to a plain dictionary so it can be printed as JSON.

        Returns:
            One key per field above, with the same values.
        """
        return asdict(self)


def _nvidia_is_usable() -> bool:
    """Check for an NVIDIA GPU that can actually be used, not merely present.

    The distinction matters. A machine can have the NVIDIA tools installed
    while the driver is broken, or (commonly) be inside a container that can
    see the program but was not started with GPU access. Asking the GPU to name
    itself is a real end-to-end check; finding the file is not.

    Returns:
        True if an NVIDIA GPU responds, or failing that, if the driver device
        file exists.
    """
    # `which` finds nvidia-smi, the tool the NVIDIA driver installs. Absent
    # means no driver, which means no usable GPU.
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            # Ask the driver to list GPU names. Success plus non-empty output
            # proves a GPU is present and responding.
            completed = subprocess.run(
                [executable, "--query-gpu=name", "--format=csv,noheader"],
                # check=False: a non-zero exit is an expected answer ("no GPU
                # here"), not a crash, so handle it rather than raising.
                check=False,
                capture_output=True,
                text=True,
                # Never hang the whole tool on a wedged driver.
                timeout=5,
            )
            if completed.returncode == 0 and completed.stdout.strip():
                return True
        except (OSError, subprocess.TimeoutExpired):
            # The program could not be run, or took too long. Either way it is
            # not a usable GPU -- fall through to the file check below.
            pass
    # Last resort: the driver's device file. This catches minimal container
    # images that expose the GPU but omit the nvidia-smi command-line tool.
    return Path("/dev/nvidia0").exists()


def detect_backend(requested: str = "auto") -> HardwareInfo:
    """Decide which backend to use and explain the decision.

    Args:
        requested: What the caller asked for, normally from a --backend flag.
            "auto" inspects the machine; any other value is honored as-is.

    Returns:
        A HardwareInfo describing both what was found and what was chosen.

    Raises:
        ValueError: If the requested backend is not in BACKENDS.

    Warning:
        The INFERENCE_BACKEND environment variable takes precedence over the
        `requested` argument, not the other way round. So with
        INFERENCE_BACKEND=cuda set, even an explicit `--backend metal` yields
        cuda. If a backend choice seems to be ignored, check that variable
        first.
    """
    requested = os.environ.get("INFERENCE_BACKEND", requested).lower()
    if requested not in BACKENDS:
        raise ValueError(f"backend must be one of: {', '.join(BACKENDS)}")

    operating_system = platform.system()
    machine = platform.machine().lower()
    # Apple Silicon means macOS on an ARM chip. Both halves are required: an
    # Intel Mac is macOS but not ARM, and a Linux ARM server is ARM but not
    # macOS. Neither can use the Metal backend.
    apple_silicon = operating_system == "Darwin" and machine in {"arm64", "aarch64"}
    nvidia_smi = _nvidia_is_usable()

    # An explicit request short-circuits detection, but the detected facts are
    # still recorded so `doctor` can show a mismatch between what you asked for
    # and what is actually installed.
    if requested != "auto":
        return HardwareInfo(
            operating_system,
            machine,
            requested,
            f"explicit override selected {requested}",
            nvidia_smi,
            apple_silicon,
        )

    # Apple Silicon is checked before NVIDIA simply because a Mac cannot have a
    # usable NVIDIA GPU, so the two never genuinely compete.
    if apple_silicon:
        return HardwareInfo(
            operating_system,
            machine,
            "metal",
            "Apple Silicon on macOS detected",
            nvidia_smi,
            apple_silicon,
        )
    if nvidia_smi:
        return HardwareInfo(
            operating_system,
            machine,
            "cuda",
            "a usable NVIDIA device was detected",
            nvidia_smi,
            apple_silicon,
        )
    # No accelerator. This is reported rather than raised: `doctor` should still
    # be able to describe an unsuitable machine. It is `serve` that refuses to
    # start, because a CPU-only run would be too slow to teach anything about
    # serving performance.
    return HardwareInfo(
        operating_system,
        machine,
        "cpu",
        "no Apple Metal or NVIDIA CUDA accelerator was detected",
        nvidia_smi,
        apple_silicon,
    )
