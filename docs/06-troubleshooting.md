# Troubleshooting

## Automatic detection chose the wrong backend

Preview and override:

```bash
inference-lab-doctor
inference-lab-serve --backend metal --dry-run
inference-lab-serve --backend cuda --dry-run
```

In a container, `nvidia-smi` may be installed even though no GPU device was
passed through. Verify that `nvidia-smi` lists a device from inside the same
environment that starts vLLM.

## Apple Silicon reports x86-64

The terminal or Python is running through Rosetta. Check:

```bash
uname -m
python3 -c 'import platform; print(platform.machine())'
```

Both should report `arm64`. Install a native ARM64 Python and recreate the
environment.

## The model does not load on Metal

Check the current vLLM-Metal supported-model matrix. Some checkpoints use a
weight format the MLX path does not accept. Start with `Qwen/Qwen3-0.6B` or an
explicitly listed MLX-community checkpoint.

## Out of memory

Try, in order:

1. Stop other GPU-heavy applications.
2. Reduce `--max-num-seqs`.
3. Reduce `--max-model-len`.
4. Use a smaller model.
5. Use a supported quantized checkpoint.
6. On CUDA, cautiously reduce `--gpu-memory-utilization` if transient headroom
   is the problem, or increase it if vLLM allocated too little KV cache and the
   system otherwise has free memory. Measure rather than guess.

## The server is healthy but the client cannot connect

- Confirm the port and bind address.
- On a remote host, use SSH port forwarding.
- Check security groups and local firewall rules.
- Do not solve a connectivity problem by exposing an unauthenticated endpoint
  to the entire internet.

## CPU fallback

The detector does not silently start CPU vLLM because platform support and
installation differ, performance can be extremely slow, and that fallback can
hide a broken accelerator setup. The error explains the detected state. Use
`--backend mock` only to test the client plumbing, not model performance.

