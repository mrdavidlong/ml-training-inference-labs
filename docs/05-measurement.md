# Measurement and observability

## Core measurements

- **TTFT (time to first token):** delay before streaming begins.
- **ITL (inter-token latency):** delay between generated tokens.
- **End-to-end latency:** request start through final response byte.
- **Throughput:** completed requests or generated tokens per second.
- **Goodput:** work completed while meeting the chosen service objective.
- **Error rate:** failed requests divided by attempted requests.
- **Queue depth:** requests waiting for service.
- **KV-cache utilization:** how much cache capacity is occupied.

## Percentiles

If p95 latency is 800 milliseconds, approximately 95 percent of observations
were at or below 800 milliseconds, and 5 percent were slower. p99 exposes rarer
tail behavior that an average can hide.

Always report the request count. A p99 from 20 requests is not stable evidence
of rare behavior.

## Client and server views

The client observes network time, queueing, server work, and response transfer.
Server metrics observe internal phases and resource state. Both are necessary.

```text
client latency = network + queue + prefill + decode + response transfer
```

## A useful first matrix

| Run | Concurrency | Input tokens | Output tokens | TTFT p95 | ITL p95 | E2E p99 | Tokens/s | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1 | 128 | 64 | | | | | |
| B | 4 | 128 | 64 | | | | | |
| C | 16 | 128 | 64 | | | | | |

Use `RESULTS.md` to record software versions, hardware, and configuration.

## Warm-up

The first request may include model initialization, graph compilation, kernel
selection, or cache population. Decide whether startup is part of the question.
For steady-state serving, run warm-up requests and exclude them explicitly.
For cold-start analysis, measure startup separately.

## Service objectives

An SLI is a **service-level indicator**, such as successful-request rate. An
SLO is a **service-level objective**, such as “99 percent successful and p95
TTFT below 750 ms for this workload.” Define the workload because latency is
meaningless without input/output length and arrival pattern.

