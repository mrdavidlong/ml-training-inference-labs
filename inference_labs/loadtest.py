"""Send many requests at an inference server and measure how it copes.

A single request tells you almost nothing. Servers behave differently under
load: queues form, batches fill up, memory pressure builds. This tool sends a
controlled burst of traffic and reports the resulting timings.

TWO NUMBERS THAT ARE NOT THE SAME

    request sent
        |
        |<---- TTFT ---->|                      time to first token:
        |                |                      how long until text starts
        |                v                      appearing. What a user
        |          "Continuous"                 perceives as responsiveness.
        |                 " batching"
        |                 " lets"     <- tokens stream in one by one
        |                 " a"           (the gaps are inter-token latency)
        |                 " server..."
        |<-------- total latency ------->|      the full end-to-end time
        v
    response complete

A server can have excellent total latency but feel sluggish because nothing
appears for two seconds, or start instantly and then trickle. Measuring only
one of them hides half the story.

WHY "CONCURRENCY" MATTERS

`--requests 40 --concurrency 4` means 40 requests total, at most 4 in flight at
any moment. Concurrency is the realistic part: real servers face several users
at once, and how gracefully throughput and tail latency hold up as concurrency
rises is the thing worth measuring.

This talks to any server implementing OpenAI's chat API, so the identical
command works against vLLM on Metal, vLLM on CUDA, or the mock server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

from inference_labs.metrics import Summary, summarize


@dataclass(frozen=True)
class Result:
    """The outcome of one single request."""

    # Total time from sending the request to the response completing, in
    # seconds. Recorded even for failures, though it is not used in that case.
    latency_seconds: float
    # Seconds until the first token of the reply arrived. None when the request
    # was not streamed, or when it failed before any token appeared.
    ttft_seconds: float | None
    # The failure description, or None if the request succeeded. Storing the
    # message rather than a bare flag keeps the reason available for debugging.
    error: str | None


def _post_once(url: str, api_key: str, payload: dict[str, object], stream: bool) -> Result:
    """Send one chat request and time it.

    This is deliberately written with `urllib` from the standard library rather
    than a friendlier HTTP package, so the repository needs no third-party
    dependencies. It is blocking; `run_load` handles the concurrency.

    Args:
        url: The full endpoint, ending in /v1/chat/completions.
        api_key: Sent as a bearer token. Local servers usually ignore it, but
            it must be present or some reject the request outright.
        payload: The JSON request body -- model, messages, and options.
        stream: Whether the response arrives incrementally. Only streamed
            requests can produce a time-to-first-token measurement.

    Returns:
        A Result with the timings, or with `error` set if anything went wrong.
        Failures are returned rather than raised so that one bad request cannot
        abort the whole test run.
    """
    started = time.perf_counter()
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # A unique id per request. Servers commonly log it, which lets a
            # slow request here be matched to its entry in the server's logs.
            "X-Request-ID": str(uuid.uuid4()),
        },
        method="POST",
    )
    first_token: float | None = None
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            if stream:
                # Streaming uses server-sent events: the server holds the
                # connection open and writes lines as tokens are produced.
                #
                #     data: {"choices":[{"delta":{"content":"Continuous"}}]}
                #     data: {"choices":[{"delta":{"content":" batching"}}]}
                #     data: [DONE]
                #
                # Reading line by line lets us stamp the clock the instant the
                # first real content line arrives.
                while True:
                    line = response.readline()
                    # An empty read means the connection closed: end of stream.
                    if not line:
                        break
                    # Three conditions: it must be a data line, not the
                    # end-of-stream marker, and the first one seen -- later
                    # tokens must not overwrite the measurement.
                    if line.startswith(b"data:") and b"[DONE]" not in line and first_token is None:
                        first_token = time.perf_counter()
            else:
                # Non-streamed: one complete JSON body. Parsed rather than
                # discarded so that a malformed response counts as a failure.
                json.loads(response.read())
        finished = time.perf_counter()
        return Result(finished - started, first_token - started if first_token else None, None)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        # URLError covers connection refused and HTTP error statuses,
        # TimeoutError the 120-second cap, and ValueError a body that is not
        # valid JSON. All are ordinary outcomes of a load test, not bugs.
        return Result(time.perf_counter() - started, None, str(exc))


async def run_load(
    base_url: str,
    api_key: str,
    model: str,
    requests: int,
    concurrency: int,
    max_tokens: int,
    stream: bool,
) -> Summary:
    """Fire the whole workload and summarize the results.

    Args:
        base_url: Root address of the server, e.g. "http://127.0.0.1:8000".
        api_key: Bearer token to send with every request.
        model: Model name, which must match what the server reports at
            /v1/models. A mismatch is usually rejected.
        requests: Total number of requests to send.
        concurrency: Maximum number in flight at once.
        max_tokens: Cap on the reply length. Longer replies take longer, so
            holding this fixed is essential when comparing two configurations.
        stream: Whether to stream responses, which enables TTFT measurement.

    Returns:
        A Summary with success rate, throughput, and latency percentiles.

    Note:
        Every request sends the identical prompt with temperature 0, so the work
        per request is as close to constant as possible. That isolates the
        variable being tested. It also means a server with prefix caching
        enabled will look unusually good here, since it can reuse work across
        identical prompts -- realistic traffic would vary the prompts.
    """
    # A semaphore is a counter that permits only N holders at a time. Others
    # wait until a slot frees. This is what enforces the concurrency limit.
    semaphore = asyncio.Semaphore(concurrency)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Explain continuous batching in one sentence."}],
        # 0 makes the model always pick its most likely next token, so output
        # length stays consistent between runs and does not add noise.
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    async def bounded_call() -> Result:
        """Wait for a free concurrency slot, then run one request."""
        async with semaphore:
            # `_post_once` blocks, which would stall the whole event loop.
            # `to_thread` runs it on a worker thread so other requests continue.
            # This is fine despite Python's global interpreter lock, because the
            # threads spend nearly all their time waiting on the network.
            return await asyncio.to_thread(
                _post_once,
                f"{base_url.rstrip('/')}/v1/chat/completions",
                api_key,
                payload,
                stream,
            )

    wall_start = time.perf_counter()
    # Start every request at once; the semaphore throttles how many actually
    # proceed. `gather` waits for all of them and returns results in order.
    results = await asyncio.gather(*(bounded_call() for _ in range(requests)))
    wall_seconds = time.perf_counter() - wall_start

    successes = [r.latency_seconds for r in results if r.error is None]
    ttfts = [r.ttft_seconds for r in results if r.ttft_seconds is not None]
    failures = sum(r.error is not None for r in results)
    return summarize(successes, failures, wall_seconds, ttfts)


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line interface.

    Split out from `main` so tests can inspect the arguments without running
    a load test.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description="Load-test an OpenAI-compatible chat endpoint")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=os.environ.get("INFERENCE_API_KEY", "local-lab-key"))
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--stream", action="store_true")
    # The two thresholds below turn this tool into a pass/fail check, which is
    # what makes it usable as an automated gate rather than just a report.
    parser.add_argument("--max-p95-ms", type=float, default=None)
    parser.add_argument("--min-success-rate", type=float, default=0.99)
    parser.add_argument("--json-output")
    return parser


def write_json_output(path: str, payload: dict[str, object]) -> None:
    """Write one summary to disk as JSON, creating any missing directories.

    The obvious destination is something like `results/latency-run1.json`, but
    `results/` is untracked and absent on a fresh clone, so a plain `open`
    would fail with FileNotFoundError after the load test had already run and
    the measurements would be lost. Creating the parents first makes the flag
    behave the way the docs use it.

    Args:
        path: Destination file path. Its parent directories are created if
            they do not already exist.
        payload: The summary dictionary to serialize.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> None:
    """Run one load test, print the summary, and exit with a meaningful code.

    Exit codes:
        0: the test ran and met any thresholds given.
        1: bad arguments.
        2: success rate fell below --min-success-rate.
        3: p95 latency exceeded --max-p95-ms.

    Distinct codes let a script or CI job react to *why* it failed. A server
    that is slow needs different attention from one that is dropping requests.
    """
    args = build_parser().parse_args()
    if args.requests < 1 or args.concurrency < 1 or args.max_tokens < 1:
        raise SystemExit("requests, concurrency, and max-tokens must be positive")

    # `asyncio.run` starts the event loop, runs the async workload, and cleans
    # up. It is the bridge from ordinary blocking code into async code.
    summary = asyncio.run(
        run_load(args.base_url, args.api_key, args.model, args.requests, args.concurrency, args.max_tokens, args.stream)
    )
    output = summary.to_dict()
    print(json.dumps(output, indent=2))
    if args.json_output:
        # Saved so runs can be compared later. RESULTS.md is the worksheet for
        # recording what hardware and settings produced each file.
        write_json_output(args.json_output, output)

    # Checked after writing the file, so a failing run still leaves its data
    # behind for inspection.
    if summary.success_rate < args.min_success_rate:
        raise SystemExit(2)
    if args.max_p95_ms is not None and summary.p95_latency_ms > args.max_p95_ms:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
