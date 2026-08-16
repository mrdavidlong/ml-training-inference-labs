import unittest

from inference_labs.metrics import percentile, summarize


class MetricsTest(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0], 50), 2.0)
        self.assertAlmostEqual(percentile([1.0, 2.0], 95), 1.95)

    def test_summary(self) -> None:
        result = summarize([0.01, 0.02, 0.03], failures=1, wall_seconds=0.1, ttft_seconds=[0.005, 0.006])
        self.assertEqual(result.requests, 4)
        self.assertEqual(result.succeeded, 3)
        self.assertEqual(result.failed, 1)
        self.assertAlmostEqual(result.success_rate, 0.75)
        self.assertAlmostEqual(result.requests_per_second, 30.0)


if __name__ == "__main__":
    unittest.main()

