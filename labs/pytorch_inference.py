"""Measure how batching and compilation change PyTorch inference speed.

WHAT THIS SHOWS

vLLM does a lot of clever work to serve models efficiently. This lab strips
that away and demonstrates the single most important idea underneath it, using
a small made-up network instead of a real model.

The idea is that a GPU is not one fast calculator but thousands of slow ones
working in parallel. Handing it one item at a time leaves nearly all of them
idle. Handing it 32 items at once costs barely more time than one. Sketched
(these are illustrative shapes, NOT measurements -- your hardware decides the
real ones):

    batch=1    latency 2.0ms    ->    500 items/second
    batch=8    latency 2.4ms    ->  3,333 items/second
    batch=32   latency 4.1ms    ->  7,800 items/second

Latency per *call* rises slightly, but items processed per second climbs
steeply. That trade -- accept a little more latency, get far more throughput --
is the reason inference servers batch requests together, and it is what vLLM's
"continuous batching" automates.

Run this on your own hardware and watch where the curve flattens. That point is
where the GPU has run out of idle capacity, and batching further stops helping.

Requires PyTorch, which is not a base dependency of this repository:
    uv sync --extra lora        (installs torch among others)
    uv run python labs/pytorch_inference.py --batch-sizes 1 8 32 --compile
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    """Time a small network at several batch sizes and print the results.

    Nothing is returned; a table goes to standard output.
    """
    parser = argparse.ArgumentParser(description="PyTorch batching and compilation inference lab")
    # `nargs="+"` accepts several values: --batch-sizes 1 4 16 64
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    # Imported here, not at the top of the file, because PyTorch is optional.
    # A module-level import would break anyone running the base install.
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install PyTorch from https://pytorch.org/get-started/locally/") from exc

    # Note this checks only for CUDA, unlike the LoRA lab's three-way choice.
    # Apple's MPS backend works but its timing behavior differs enough that the
    # comparison this lab is making would be muddier.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # A stand-in for a real model: expand 1024 numbers to 4096, apply a
    # non-linear function, then compress back to 1024. This is the same shape
    # as the "feed-forward" block found inside every transformer layer, which
    # is why it is a fair thing to benchmark.
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 4096),
        torch.nn.GELU(),
        torch.nn.Linear(4096, 1024),
    ).eval().to(device)  # .eval() disables training-only behavior such as dropout

    if args.compile:
        # torch.compile traces the model and generates optimized code, fusing
        # separate operations into single passes over memory. The first call is
        # slow because compilation happens then -- which is exactly why the
        # warmup below matters. https://docs.pytorch.org/docs/stable/torch.compiler.html
        model = torch.compile(model)

    def synchronize() -> None:
        """Wait for the GPU to actually finish its queued work.

        GPU calls are asynchronous: PyTorch queues the work and returns
        immediately, before the arithmetic is done. Timing without this would
        measure how fast Python can queue instructions, which is meaningless
        and typically reports absurdly fast results. On CPU there is no queue,
        so this does nothing.
        """
        if device == "cuda":
            torch.cuda.synchronize()

    print(f"torch={torch.__version__} device={device} compiled={args.compile}")
    for batch_size in args.batch_sizes:
        # Random input of the right shape. The values are irrelevant; only the
        # amount of arithmetic matters for timing.
        inputs = torch.randn(batch_size, 1024, device=device)

        # `inference_mode` turns off the bookkeeping PyTorch keeps for training.
        # It is a stronger version of `no_grad` and gives slightly faster,
        # lower-memory execution.
        with torch.inference_mode():
            # WARMUP: ten untimed calls. The first calls are unrepresentatively
            # slow because of one-time costs -- memory allocation, kernel
            # selection, and torch.compile's actual compilation. Including them
            # would distort the average badly.
            for _ in range(10):
                model(inputs)
            synchronize()

            # The measured section. Many iterations are averaged because a
            # single call is too short and noisy to time reliably.
            started = time.perf_counter()
            for _ in range(args.iterations):
                model(inputs)
            synchronize()  # must finish before stopping the clock

        seconds = time.perf_counter() - started
        items = batch_size * args.iterations
        # Two views of the same run. Latency is per call and tends to rise with
        # batch size; throughput counts individual items and tends to rise much
        # faster. Watching them diverge is the whole point of the lab.
        print(
            f"batch={batch_size:>3} latency_ms={seconds / args.iterations * 1000:>8.3f} "
            f"items_per_second={items / seconds:>10.1f}"
        )


if __name__ == "__main__":
    main()
