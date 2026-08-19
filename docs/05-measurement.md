# Measurement and observability

Measuring an inference server is easy to do and easy to do wrong. This guide
covers what to measure, how to avoid the standard traps, and how to turn
numbers into a pass/fail answer.

The tools are `inference_labs/loadtest.py` (generates load, times it) and
`inference_labs/metrics.py` (turns timings into percentiles).

## Why the average is the wrong number

Start here, because it justifies everything else. Ten requests: nine took
100 ms, one took 5,000 ms.

```bash
uv run python -c "
from inference_labs.metrics import percentile
values = [100]*9 + [5000]
print('mean', sum(values)/len(values))
for p in (50, 95, 99):
    print(f'p{p}', round(percentile(values, p), 1))
"
```

```text
mean 590.0
p50 100.0
p95 2795.0
p99 4559.0
```

The mean is 590 ms — a number that describes **none** of the ten requests. No
user waited 590 ms. Nine had a fast experience, one had a terrible one, and the
average erases both facts. The median (p50) says most requests were fine; p99
says the bad case is genuinely bad.

**A change that improves the mean while worsening p99 has usually made the
service worse.** That is why every number in this repository is a percentile.

## Core measurements

| Metric | Question it answers | Where it comes from |
|---|---|---|
| **TTFT** (time to first token) | How long before something appears? | First streamed chunk |
| **ITL** (inter-token latency) | How smoothly does text arrive? | Gaps between chunks — *not* reported by this repo's load tester; see below |
| **End-to-end latency** | How long did the whole request take? | Start to final byte |
| **Throughput** | How much finished per second? | Requests or tokens / wall time |
| **Goodput** | How much finished *within the objective*? | Throughput minus too-slow work |
| **Error rate** | How often did it fail? | Failures / attempts |
| **Queue depth** | How many are waiting? | Server `/metrics` |
| **KV-cache utilization** | How close to full is the cache? | Server `/metrics` |

TTFT and ITL map onto the two phases from
[inference fundamentals](01-inference-basics.md#prefill-and-decode): TTFT is
dominated by prefill, ITL by decode. If TTFT is bad, look at prompt length and
queueing. If ITL is bad, look at batch size and decode contention.

**Goodput** deserves attention. A server that answers 1,000 requests per second
but misses your latency target on half of them has a real throughput of 500.
Throughput alone can reward exactly the wrong behavior.

## How percentiles are computed here

`metrics.percentile` sorts the values and reads off the position, interpolating
between neighbors — the same convention NumPy uses by default:

```text
values [10, 20, 30, 40], p=95
  position = (4 - 1) * 95/100 = 2.85
  lower = index 2 (value 30), upper = index 3 (value 40)
  fraction = 0.85
  result = 30*0.15 + 40*0.85 = 38.5
```

```bash
uv run python -c "from inference_labs.metrics import percentile; print(percentile([10,20,30,40], 95))"
# 38.5
```

Percentile definitions differ slightly between tools. Small disagreements at
p99 on a few dozen samples are normal and not worth chasing.

## The sample-count trap

This is the most common beginner error:

```text
20 requests, p99 = ???

p99 of 20 samples is interpolated between the two slowest observations.
You have ONE sample in the top 5%. That is an anecdote, not a percentile.
```

Rough guidance:

| Requests | p50 | p95 | p99 |
|---|---|---|---|
| 20 | usable | shaky | meaningless |
| 100 | good | usable | shaky |
| 1,000 | good | good | usable |

Always report the request count next to the percentile. `p99=812ms (n=40)` is
honest; `p99=812ms` alone invites the reader to trust it too much.

## Running a measurement

Two terminals. Server in one:

```bash
inference-lab-serve --profile learning
```

Client in the other:

```bash
inference-lab-loadtest --requests 40 --concurrency 8 --stream
```

```json
{
  "requests": 12,
  "succeeded": 12,
  "failed": 0,
  "success_rate": 1.0,
  "wall_seconds": 0.28060279198689386,
  "requests_per_second": 42.765076979563645,
  "mean_latency_ms": 92.83528458278549,
  "p50_latency_ms": 91.74616649397649,
  "p95_latency_ms": 96.37233530374942,
  "p99_latency_ms": 96.40769988152897,
  "mean_ttft_ms": 14.180538082049074,
  "p95_ttft_ms": 16.816958162235096
}
```

(That output is from the mock server, so the *values* are meaningless — but the
shape is what you will get from a real one.)

### `--stream` is not optional for TTFT

TTFT is measured by timing the first non-`[DONE]` SSE `data:` line. Without
`--stream` there are no intermediate chunks, so there is nothing to time:

```bash
inference-lab-loadtest --requests 12 --concurrency 4        # no --stream
```

```json
  "mean_ttft_ms": null,
  "p95_ttft_ms": null
```

`null`, not zero. If you see nulls where you expected TTFT, you forgot
`--stream`.

### ITL is a concept here, not a reported field

`inference_labs/loadtest.py` records TTFT and end-to-end latency, but does not
currently time the gaps *between* streamed chunks, so there is no `itl` key in
the JSON. You can approximate it from what is reported:

```text
approximate mean ITL = (end_to_end_latency - TTFT) / (output_tokens - 1)
```

With `--max-tokens` fixed across runs, that estimate is good enough to compare
two profiles. Extending the load tester to record per-chunk timestamps and
report a true ITL percentile is a worthwhile exercise.

## Warm-up

The first request may include model initialization, kernel selection, graph
compilation, and cache population. Decide which question you are answering:

```text
STEADY-STATE SERVING              COLD-START ANALYSIS
run warm-up requests first        measure the first request deliberately
discard them explicitly           report it separately
report the steady state           never blend the two
```

```bash
inference-lab-loadtest --requests 5  --concurrency 1 --stream   # discard
inference-lab-loadtest --requests 60 --concurrency 8 --stream   # measure
```

Blending them produces a number that answers neither question.

## A first experiment matrix

Vary one thing — concurrency — and hold everything else fixed:

| Run | Concurrency | Input tokens | Output tokens | TTFT p95 | E2E p95 | E2E p99 | Req/s | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1 | 128 | 64 | | | | | |
| B | 4 | 128 | 64 | | | | | |
| C | 16 | 128 | 64 | | | | | |
| D | 32 | 128 | 64 | | | | | |

Every column above maps to a key the load tester actually prints
(`p95_ttft_ms`, `p95_latency_ms`, `p99_latency_ms`, `requests_per_second`,
`failed`), so you can fill it in straight from the JSON.

The expected shape:

```text
throughput                      latency
    |          ___________          |                    /
    |        /                      |                  /
    |      /                        |          ______/
    |    /                          |    _____/
    |  /                            |___/
    +---------------------> conc    +---------------------> conc
      rises, then flattens            flat, then climbs sharply

              the useful operating point is near the knee of both
```

Past the knee you are adding queueing delay without adding work. That point,
not a flag, is what you are actually looking for.

Record results in `RESULTS.md` together with hardware, software versions, and
configuration. A latency number without its context cannot be reproduced or
compared later.

## Saving results

```bash
mkdir -p results        # --json-output will not create the directory
inference-lab-loadtest --requests 60 --concurrency 8 --stream \
  --json-output results/learning-c8-run1.json
```

Name files so the configuration is visible without opening them:
`<profile>-c<concurrency>-run<n>.json`. `results/*.json` is gitignored, since
measurements are specific to one machine and one moment.

## Pass/fail thresholds

The load tester can act as a check rather than a report. Exit codes are
meaningful, so this works in a script or in CI:

| Exit | Meaning |
|---|---|
| `0` | All thresholds met |
| `2` | Success rate below `--min-success-rate` (default 0.99) |
| `3` | p95 latency above `--max-p95-ms` |

```bash
inference-lab-loadtest --requests 60 --concurrency 8 --stream \
  --min-success-rate 0.99 --max-p95-ms 750
echo "exit=$?"
```

Verified behavior, using the mock server as the target:

```text
--max-p95-ms 10     -> exit=3   (p95 was ~96 ms)
--max-p95-ms 5000   -> exit=0
server unreachable  -> exit=2   (every request failed)
```

Note that a threshold breach exits non-zero but prints no extra explanation —
the JSON summary itself is the evidence. Check `$?`, not stderr.

## Client view versus server view

You need both. They see different things:

```text
client latency = network + queue + prefill + decode + response transfer
                 ^^^^^^^^^^^^^^^                      ^^^^^^^^^^^^^^^^^
                 invisible to the server              invisible to the server

server /metrics = queue depth, KV-cache use, prefix-cache hits, preemptions
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                  invisible to the client
```

If client latency is high but server-side time is low, the problem is queueing,
the network, or your client — not the model. vLLM exposes Prometheus metrics:

```bash
curl http://127.0.0.1:8000/metrics
```

Useful counters: KV-cache utilization, number of running versus waiting
requests, prefix-cache hit rate, and preemptions. **Preemptions above zero
means vLLM ran out of KV cache and had to evict work** — a strong signal to
reduce `max_num_seqs` or `max_model_len`.

## Service objectives

- **SLI** (service-level indicator): something you measure, e.g. the fraction
  of successful requests.
- **SLO** (service-level objective): a target for an SLI, e.g. "99% successful
  and p95 TTFT under 750 ms for this workload."
- **SLA** (service-level agreement): an external, often contractual promise.

The phrase "for this workload" is what makes an SLO meaningful. Latency has no
value without input length, output length, concurrency, and arrival pattern —
the same server can be well within target on short prompts and hopeless on long
ones.

## Practice without a GPU

```bash
uv run python -m inference_labs.demo --requests 40 --concurrency 8
```

This starts the mock server, runs the full measurement loop, prints a summary,
and writes JSON to `results/`. It exercises streaming, concurrency, percentile
math, and the report format.

**Its latencies are `time.sleep` constants.** Use it to learn the workflow and
to test your own tooling. Never quote its numbers as performance results, and
never point a learning command at it.

## Measurement mistakes to avoid

| Mistake | Why it misleads |
|---|---|
| Reporting the mean | Hides the tail, as shown at the top |
| p99 from 20 requests | One sample is not a percentile |
| No warm-up | Measures initialization, not serving |
| Forgetting `--stream` | TTFT is `null` |
| No accelerator sync | Times dispatch, not execution ([docs/04](04-pytorch-jax-vllm.md)) |
| Changing two settings | Cannot attribute the difference |
| One run | No idea of variance; run at least three |
| Uniform prompts | Flatters batching; real traffic is uneven |
| Comparing across hardware | Only valid if everything else is fixed |
| Quoting the mock server | Those are `sleep` calls |

## Next

- [Tuning](03-tuning.md) — change a setting and compare properly.
- [Troubleshooting](06-troubleshooting.md) — when the numbers say something is
  broken.
