"""A fake inference server that returns canned text, for testing only.

WHY THIS EXISTS

The test suite needs to check that the load tester handles streaming, parses
responses, counts errors, and computes metrics correctly. Verifying that should
not require downloading a multi-gigabyte model or owning a GPU. This server
speaks just enough of the OpenAI API to exercise all of that in milliseconds.

WHAT IT IS NOT

It is not a model. It returns the word "ok" repeatedly, and its "thinking time"
is a `time.sleep` of a fixed duration. Any latency or throughput number
measured against it describes this file's sleep constants and nothing else.

    Never quote performance results obtained from this server.

Every learning exercise in this repository points at a real vLLM server for
exactly that reason. See the "Mock server policy" section of the README.

The endpoints implemented are a small subset of the real API:
    GET  /health               - liveness check
    GET  /v1/models            - lists one fake model
    GET  /metrics              - Prometheus-style counters
    POST /v1/chat/completions  - the actual completion, streamed or not
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock


class Counters:
    """Request and error tallies shared across all connections.

    The server handles requests on multiple threads at once, and `+=` in Python
    is not atomic -- it reads, adds, and writes back as separate operations. Two
    threads can therefore interleave and lose a count. The lock makes each
    update indivisible.
    """

    def __init__(self) -> None:
        # Held while modifying either counter below.
        self.lock = Lock()
        # Total chat completion requests accepted.
        self.requests = 0
        # Requests that failed, e.g. an unparseable body.
        self.errors = 0


# Module-level so every request handler instance shares one set of tallies.
# `BaseHTTPRequestHandler` constructs a fresh object per request, so per-instance
# state would be lost immediately.
COUNTERS = Counters()


class Handler(BaseHTTPRequestHandler):
    """Handles one HTTP request. Python creates a new instance for each."""

    # HTTP/1.1 keeps the connection open between requests, which avoids a fresh
    # TCP handshake every time and keeps the fake server's overhead low enough
    # that it does not distort what the tests are checking.
    protocol_version = "HTTP/1.1"
    server_version = "InferenceLab/0.1"

    def log_message(self, format: str, *args: object) -> None:
        """Silence the default per-request logging.

        The base class writes a line to stderr for every request, which would
        bury the test output under dozens of lines. Overriding with an empty
        body discards them.
        """
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        """Write a complete, non-streamed response.

        Args:
            status: HTTP status code, e.g. 200 or 404.
            body: The already-encoded response body.
            content_type: MIME type for the body.
        """
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        # Required by HTTP/1.1 keep-alive: without it the client cannot tell
        # where this response ends and the next begins.
        self.send_header("Content-Length", str(len(body)))
        # Echo the client's request id back, mimicking what real servers do so
        # a client can correlate its own logs with the server's.
        self.send_header("X-Request-ID", self.headers.get("X-Request-ID", str(uuid.uuid4())))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Route GET requests. Called automatically by the HTTP server."""
        if self.path == "/health":
            # Liveness check. Real orchestrators poll this to decide whether an
            # instance should receive traffic.
            self._send(HTTPStatus.OK, b'{"status":"ok"}', "application/json")
            return
        if self.path == "/v1/models":
            # Clients call this to discover the served model name. The load
            # tester's --model must match what appears here.
            body = json.dumps({"object": "list", "data": [{"id": "mock-model"}]}).encode()
            self._send(HTTPStatus.OK, body, "application/json")
            return
        if self.path == "/metrics":
            # Prometheus text format: one "metric_name value" per line. Real
            # vLLM exposes far richer metrics at this same path.
            with COUNTERS.lock:
                body = (
                    f"inference_lab_requests_total {COUNTERS.requests}\n"
                    f"inference_lab_errors_total {COUNTERS.errors}\n"
                ).encode()
            self._send(HTTPStatus.OK, body, "text/plain; version=0.0.4")
            return
        self._send(HTTPStatus.NOT_FOUND, b'{"error":"not found"}', "application/json")

    def do_POST(self) -> None:
        """Route POST requests, i.e. chat completions."""
        if self.path != "/v1/chat/completions":
            self._send(HTTPStatus.NOT_FOUND, b'{"error":"not found"}', "application/json")
            return
        try:
            # The body must be read by exact length; the socket has no
            # end-of-file until the connection closes, which keep-alive prevents.
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            # Capped at 256 so a client asking for an enormous reply cannot make
            # a test hang for minutes.
            max_tokens = min(int(request.get("max_tokens", 16)), 256)
            stream = bool(request.get("stream", False))
            with COUNTERS.lock:
                COUNTERS.requests += 1
            if stream:
                self._stream(max_tokens)
            else:
                # Fake "work". Roughly mimics the shape of real inference: a
                # small fixed startup cost plus time proportional to the number
                # of tokens produced. The constants are invented.
                time.sleep(0.006 + max_tokens * 0.001)
                body = json.dumps(
                    {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion",
                        "model": request.get("model", "mock-model"),
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok " * max_tokens}, "finish_reason": "stop"}],
                        # Real servers report token counts here. These are made
                        # up, but present so clients that read them do not break.
                        "usage": {"prompt_tokens": 8, "completion_tokens": max_tokens, "total_tokens": max_tokens + 8},
                    }
                ).encode()
                self._send(HTTPStatus.OK, body, "application/json")
        except Exception as exc:
            # Broad by design: this is a test fixture, and any malformed request
            # should become a 400 that the load tester counts as a failure,
            # rather than a traceback that kills the handler thread. Production
            # code should not catch this broadly.
            with COUNTERS.lock:
                COUNTERS.errors += 1
            self._send(HTTPStatus.BAD_REQUEST, json.dumps({"error": str(exc)}).encode(), "application/json")

    def _stream(self, max_tokens: int) -> None:
        """Send the reply incrementally as server-sent events.

        This is the path the tests care about most, because it is what lets the
        load tester measure time to first token. The wire format is a sequence
        of "data:" lines, terminated by a sentinel:

            data: {"choices":[{"index":0,"delta":{"content":"ok "}}]}
            data: {"choices":[{"index":0,"delta":{"content":"ok "}}]}
            data: [DONE]

        Args:
            max_tokens: How many chunks to emit, one per token.
        """
        self.send_response(HTTPStatus.OK)
        # The MIME type for server-sent events.
        self.send_header("Content-Type", "text/event-stream")
        # Stops proxies from buffering, which would defeat streaming entirely.
        self.send_header("Cache-Control", "no-cache")
        # A streamed body has no known length, so the connection close is what
        # signals the end. Keep-alive cannot be used here.
        self.send_header("Connection", "close")
        self.send_header("X-Request-ID", self.headers.get("X-Request-ID", str(uuid.uuid4())))
        self.end_headers()

        for index in range(max_tokens):
            # The first token is slower than the rest (10ms versus 2ms),
            # imitating real inference: a model must process the whole prompt
            # before it can emit anything, then produces subsequent tokens
            # comparatively quickly. This gap is what makes TTFT a distinct
            # measurement, and it is what the smoke test verifies is captured.
            time.sleep(0.002 if index else 0.01)
            chunk = {"choices": [{"index": 0, "delta": {"content": "ok "}, "finish_reason": None}]}
            # The blank line after each event is required by the SSE format.
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            # Without flushing, output would sit in a buffer and arrive in one
            # burst, making the stream indistinguishable from a normal response.
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


def make_server(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Create the server without starting it.

    Args:
        host: Interface to bind. Defaults to 127.0.0.1, so it is reachable only
            from this machine.
        port: TCP port. The default of 0 asks the operating system to pick any
            free port, which lets tests run in parallel without colliding. Read
            the actual port back from `server.server_port` afterwards.

    Returns:
        A configured server. Call `serve_forever()` to start it, normally on a
        background thread -- see demo.py or tests/test_smoke.py.

    Note:
        "Threading" means each connection is handled on its own thread, so the
        server can hold several requests at once. A single-threaded server would
        serialize them and make the concurrency setting meaningless.
    """
    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    """Run the mock server in the foreground until interrupted with Control-C."""
    parser = argparse.ArgumentParser(description="OpenAI-compatible mock inference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = make_server(args.host, args.port)
    # `server.server_port` rather than `args.port`, so the real port is shown
    # even when 0 was requested and the OS chose one.
    print(f"mock server listening on http://{args.host}:{server.server_port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
