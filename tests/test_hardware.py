import unittest
from unittest.mock import patch

from inference_labs.hardware import detect_backend


class HardwareDetectionTest(unittest.TestCase):
    @patch("inference_labs.hardware._nvidia_is_usable", return_value=False)
    @patch("inference_labs.hardware.platform.machine", return_value="arm64")
    @patch("inference_labs.hardware.platform.system", return_value="Darwin")
    def test_detects_apple_silicon(self, _system, _machine, _nvidia) -> None:
        self.assertEqual(detect_backend().backend, "metal")

    @patch("inference_labs.hardware._nvidia_is_usable", return_value=True)
    @patch("inference_labs.hardware.platform.machine", return_value="x86_64")
    @patch("inference_labs.hardware.platform.system", return_value="Linux")
    def test_detects_nvidia(self, _system, _machine, _nvidia) -> None:
        self.assertEqual(detect_backend().backend, "cuda")

    @patch("inference_labs.hardware._nvidia_is_usable", return_value=False)
    def test_explicit_override_wins(self, _nvidia) -> None:
        self.assertEqual(detect_backend("cuda").backend, "cuda")


if __name__ == "__main__":
    unittest.main()

