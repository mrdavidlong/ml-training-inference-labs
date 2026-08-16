import asyncio
import threading
import unittest

from inference_labs.loadtest import run_load
from inference_labs.mock_server import make_server


class SmokeTest(unittest.TestCase):
    def test_streaming_load(self) -> None:
        server = make_server()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = asyncio.run(
                run_load(
                    f"http://127.0.0.1:{server.server_port}",
                    "test-key",
                    "mock-model",
                    requests=12,
                    concurrency=4,
                    max_tokens=4,
                    stream=True,
                )
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertEqual(result.requests, 12)
        self.assertEqual(result.failed, 0)
        self.assertEqual(result.success_rate, 1.0)
        self.assertIsNotNone(result.p95_ttft_ms)


if __name__ == "__main__":
    unittest.main()

