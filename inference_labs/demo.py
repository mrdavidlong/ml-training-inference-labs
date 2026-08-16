"""Run the whole measurement pipeline end to end against the mock server.

This is the zero-setup walkthrough: it starts the fake server, load-tests it,
prints a summary, and saves the result to results/. No GPU, no model download,
no second terminal. The point is to see the *shape* of the workflow --

    start a server -> send load -> collect timings -> save for comparison

-- before doing it for real, where each of those steps involves more setup.

The numbers it produces are meaningless as performance data, because the server
being measured is a stack of `time.sleep` calls. See mock_server.py. To measure
something real, start vLLM with `inference-lab-serve` and point
`inference-lab-loadtest` at it instead.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from inference_labs.loadtest import run_load
from inference_labs.mock_server import make_server


def main() -> None:
    """Start the mock server, load-test it, print and save the summary.

    Nothing is returned. A JSON file is written to results/ named by timestamp.
    """
    parser = argparse.ArgumentParser(description="Run the laptop inference-service lab")
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=32)
    args = parser.parse_args()

    # Port 0 lets the operating system pick a free port, so this never collides
    # with a real server already using 8000.
    server = make_server()
    # The server must run alongside the load test, not before it, so it goes on
    # a background thread. `daemon=True` means this thread cannot keep the
    # process alive if the main thread exits unexpectedly.
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # Read the port back, since the OS chose it.
        base_url = f"http://127.0.0.1:{server.server_port}"
        # The trailing True is `stream=True`, which is what makes
        # time-to-first-token appear in the summary.
        summary = asyncio.run(
            run_load(base_url, "local-lab-key", "mock-model", args.requests, args.concurrency, args.output_tokens, True)
        )
    finally:
        # `finally` guarantees shutdown even if the load test raises, so a
        # failed run cannot leave an orphaned server holding the port.
        server.shutdown()      # stop the serve_forever loop
        server.server_close()  # release the socket
        thread.join(timeout=2)  # wait briefly for the thread to finish

    results = Path("results")
    results.mkdir(exist_ok=True)
    # UTC and a sortable format, so files from different machines and time
    # zones still line up chronologically when listed.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = results / f"mock-{timestamp}.json"
    output.write_text(json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary.to_dict(), indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
