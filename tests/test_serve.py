import unittest
from pathlib import Path

from inference_labs.serve import build_command, load_config


CONFIG = Path(__file__).resolve().parent.parent / "configs" / "serve.toml"


class ServeCommandTest(unittest.TestCase):
    def test_metal_omits_cuda_memory_flag(self) -> None:
        command = build_command("metal", "learning", load_config(CONFIG))
        self.assertIn("Qwen/Qwen3-0.6B", command)
        self.assertNotIn("--gpu-memory-utilization", command)

    def test_cuda_includes_memory_flag(self) -> None:
        command = build_command("cuda", "throughput", load_config(CONFIG))
        self.assertIn("--gpu-memory-utilization", command)
        self.assertIn("--max-num-batched-tokens", command)

    def test_every_config_profile_is_usable(self) -> None:
        # The regression this guards: --profile used to carry a hardcoded
        # `choices=` list, so a [profile.*] section added to the TOML -- the
        # documented way to extend this lab -- was rejected before
        # build_command ever saw it. The config file is the source of truth.
        config = load_config(CONFIG)
        profiles = sorted(config["profile"])
        self.assertGreater(len(profiles), 0)
        for profile in profiles:
            with self.subTest(profile=profile):
                command = build_command("cuda", profile, config)
                self.assertEqual(command[:3], ["vllm", "serve", "Qwen/Qwen3-0.6B"])

    def test_profile_added_to_config_is_accepted(self) -> None:
        # Simulates a reader following docs/03-tuning.md and inventing a new
        # profile, without editing the real file on disk.
        config = load_config(CONFIG)
        config["profile"]["batch"] = {"max_num_seqs": 64}
        command = build_command("cuda", "batch", config)
        self.assertIn("--max-num-seqs", command)
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "64")

    def test_overrides_replace_rather_than_duplicate(self) -> None:
        # The regression this guards: `model_override or config.pop("model")`
        # short-circuited when an override was given, so the key stayed in the
        # dictionary and the generic flag loop emitted it again. The command
        # then carried both values, and vLLM's last-wins parsing silently used
        # the config's rather than the caller's.
        command = build_command(
            "metal",
            "learning",
            load_config(CONFIG),
            model_override="my/model",
            host_override="0.0.0.0",
            port_override=9000,
        )
        self.assertEqual(command[:7], ["vllm", "serve", "my/model", "--host", "0.0.0.0", "--port", "9000"])
        for flag in ("--model", "--host", "--port"):
            with self.subTest(flag=flag):
                self.assertEqual(command.count(flag), 1 if flag != "--model" else 0)
        # The config's values must not survive anywhere in the command.
        self.assertNotIn("Qwen/Qwen3-0.6B", command)
        self.assertNotIn("127.0.0.1", command)
        self.assertNotIn("8000", command)

    def test_config_values_used_when_no_override(self) -> None:
        # The other side of the same fix: with nothing overridden, the config
        # still supplies all three, and still only once each.
        command = build_command("metal", "learning", load_config(CONFIG))
        self.assertEqual(command[:7], ["vllm", "serve", "Qwen/Qwen3-0.6B", "--host", "127.0.0.1", "--port", "8000"])
        self.assertNotIn("--model", command)
        self.assertEqual(command.count("--host"), 1)
        self.assertEqual(command.count("--port"), 1)

    def test_unknown_profile_names_the_available_ones(self) -> None:
        with self.assertRaises(ValueError) as caught:
            build_command("metal", "nonexistent", load_config(CONFIG))
        message = str(caught.exception)
        self.assertIn("nonexistent", message)
        # The message has to list the real options, because it is now the only
        # place a reader is told what the valid profiles are.
        for profile in load_config(CONFIG)["profile"]:
            self.assertIn(profile, message)


if __name__ == "__main__":
    unittest.main()
