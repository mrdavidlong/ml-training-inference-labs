import json
import os
import tempfile
import unittest
from pathlib import Path

from inference_labs.loadtest import write_json_output


SUMMARY = {"requests": 4, "failed": 0, "success_rate": 1.0, "p95_latency_ms": 12.5}


class WriteJsonOutputTest(unittest.TestCase):
    def test_creates_missing_parent_directories(self) -> None:
        # The regression this guards: `--json-output results/run1.json` used to
        # raise FileNotFoundError on a fresh clone, because `results/` is
        # untracked -- losing the measurements after the load test had run.
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "results" / "nested" / "run1.json"
            write_json_output(str(destination), SUMMARY)
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), SUMMARY)

    def test_writes_into_an_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "run1.json"
            write_json_output(str(destination), SUMMARY)
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), SUMMARY)

    def test_bare_filename_has_no_directory_to_create(self) -> None:
        # Path("run1.json").parent is ".", which mkdir must tolerate.
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                write_json_output("run1.json", SUMMARY)
                self.assertEqual(json.loads(Path("run1.json").read_text(encoding="utf-8")), SUMMARY)
            finally:
                os.chdir(previous)

    def test_overwrites_an_earlier_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "results" / "run1.json"
            write_json_output(str(destination), {"requests": 99})
            write_json_output(str(destination), SUMMARY)
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), SUMMARY)

    def test_file_ends_with_a_newline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "results" / "run1.json"
            write_json_output(str(destination), SUMMARY)
            self.assertTrue(destination.read_text(encoding="utf-8").endswith("}\n"))


if __name__ == "__main__":
    unittest.main()
