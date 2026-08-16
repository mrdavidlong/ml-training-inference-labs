"""The same batching experiment as the PyTorch lab, written in JAX.

Run this alongside labs/pytorch_inference.py. The measurements should tell the
same story -- throughput climbs with batch size -- because the story is about
how GPUs work, not about which library you use.

WHAT IS DIFFERENT ABOUT JAX

PyTorch executes operations as it meets them. JAX instead builds a description
of the whole computation and compiles it before running, which lets it optimize
across the entire function rather than one operation at a time. Two ideas do
most of the work:

    jax.vmap  Write the function for ONE item; vmap turns it into a version
              that handles a whole batch. You never write the batch dimension
              by hand.

    jax.jit   Compile the function to fast machine code the first time it runs,
              then reuse that code for every later call with the same shapes.

Also, JAX arrays are immutable and its functions are meant to be pure -- no
hidden state being modified. That is a real constraint, and it is what makes
the aggressive compilation safe.

    PyTorch                          JAX
    ------------------------------   ------------------------------
    runs operations immediately      compiles the function first
    model holds its own weights      weights are values you pass in
    batching is written into shapes  vmap adds the batch dimension
    torch.compile is opt-in          jit is the normal way to work

Neither is better. PyTorch is more forgiving to debug; JAX's structure suits
compilation and large-scale parallelism. Both are general frameworks -- neither
is a serving engine like vLLM. See docs/04-pytorch-jax-vllm.md.

Requires JAX, which is not a dependency of this repository:
    https://docs.jax.dev/en/latest/installation.html
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    """Time a small JAX network at several batch sizes and print the results.

    Nothing is returned; a table goes to standard output, in the same format as
    the PyTorch lab so the two can be compared directly.
    """
    parser = argparse.ArgumentParser(description="JAX vmap and jit inference lab")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    # Imported here rather than at the top because JAX is optional.
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise SystemExit("Install JAX from https://docs.jax.dev/en/latest/installation.html") from exc

    # JAX handles randomness explicitly: instead of hidden global state, you
    # carry an explicit "key" and split it whenever you need more randomness.
    # Verbose, but it makes results exactly reproducible across machines and
    # keeps functions pure. https://docs.jax.dev/en/latest/random-numbers.html
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)

    # The weight matrices, matching the PyTorch lab's 1024 -> 4096 -> 1024
    # shape. Note they are plain values here, not owned by a model object --
    # the function below closes over them. Scaled by 0.01 to keep the numbers
    # small, which avoids overflow through the layers.
    w1 = jax.random.normal(k1, (1024, 4096), dtype=jnp.float32) * 0.01
    w2 = jax.random.normal(k2, (4096, 1024), dtype=jnp.float32) * 0.01

    def one(x):
        """The network for a SINGLE input vector of length 1024.

        Note there is no batch dimension anywhere here -- `x` is one vector,
        and the code reads as if only one item will ever exist. vmap supplies
        the batching, which is the point being demonstrated.

        Args:
            x: One input vector, shape (1024,).

        Returns:
            One output vector, shape (1024,). `@` is matrix multiplication and
            gelu is the same activation function the PyTorch lab uses.
        """
        return jax.nn.gelu(x @ w1) @ w2

    # Read inside out: vmap makes `one` accept a stack of vectors, and jit
    # compiles the batched result. Compare with PyTorch, where the batch
    # dimension is baked into the layer shapes from the start.
    batched = jax.jit(jax.vmap(one))

    print(f"jax={jax.__version__} devices={jax.devices()}")
    for batch_size in args.batch_sizes:
        inputs = jnp.ones((batch_size, 1024), dtype=jnp.float32)

        # WARMUP, and more importantly compilation. jit compiles separately for
        # each input shape it sees, so every batch size pays a one-time cost on
        # its first call. Timing that would measure the compiler, not the model.
        batched(inputs).block_until_ready()

        started = time.perf_counter()
        output = None
        for _ in range(args.iterations):
            # JAX dispatches asynchronously and returns a placeholder straight
            # away, so this loop mostly just queues work.
            output = batched(inputs)
        assert output is not None
        # Wait for the queued work to actually finish before stopping the
        # clock. This is JAX's equivalent of torch.cuda.synchronize(); without
        # it the loop would appear impossibly fast. Blocking on the final
        # result is enough, because the calls complete in order.
        output.block_until_ready()

        seconds = time.perf_counter() - started
        items = batch_size * args.iterations
        print(
            f"batch={batch_size:>3} latency_ms={seconds / args.iterations * 1000:>8.3f} "
            f"items_per_second={items / seconds:>10.1f}"
        )


if __name__ == "__main__":
    main()
