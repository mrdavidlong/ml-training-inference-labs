import unittest
from pathlib import Path

from training_labs.kmeans import fit_kmeans, silhouette_hint
from training_labs.linear_regression import fit_linear_regression
from training_labs.lora_finetune import (
    choose_device,
    format_prompt,
    format_record,
    load_local_records,
)
from training_labs.q_learning import GridWorld, greedy_path, train_q_learning


class LinearRegressionTest(unittest.TestCase):
    def test_loss_decreases_and_prediction_is_close(self):
        features = [[0.0], [1.0], [2.0], [3.0], [4.0]]
        targets = [1.0, 3.0, 5.0, 7.0, 9.0]
        model, losses = fit_linear_regression(features, targets, epochs=500)
        self.assertLess(losses[-1], losses[0] * 0.01)
        self.assertAlmostEqual(model.predict([5.0]), 11.0, places=1)


class KMeansTest(unittest.TestCase):
    def test_separates_obvious_clusters(self):
        points = [(0.0, 0.0), (0.2, 0.1), (10.0, 10.0), (10.2, 9.9)]
        centers, assignments, inertia = fit_kmeans(points, 2, seed=1)
        self.assertEqual(len(centers), 2)
        self.assertEqual(sorted(assignments.count(label) for label in set(assignments)), [2, 2])
        self.assertLess(inertia, 0.2)
        self.assertGreater(silhouette_hint(points, assignments), 0.9)


class QLearningTest(unittest.TestCase):
    def test_policy_reaches_goal(self):
        q_values = train_q_learning(episodes=2_000, seed=3)
        path = greedy_path(q_values)
        self.assertEqual(path[0], GridWorld.start)
        self.assertEqual(path[-1], GridWorld.goal)
        self.assertLessEqual(len(path) - 1, 12)


class LoraPreparationTest(unittest.TestCase):
    # Stands in for tokenizer.eos_token, which needs the optional `lora` extra.
    EOS = "</s>"

    def test_bundled_records_load_and_format(self):
        path = Path(__file__).parents[1] / "data" / "lora_tiny_instructions.jsonl"
        records = load_local_records(path, 3)
        self.assertEqual(len(records), 3)
        formatted = format_record(records[0], self.EOS)
        self.assertIn("### User:", formatted)
        self.assertIn("### Assistant:", formatted)

    def test_prompt_ends_after_assistant_marker(self):
        # Generation must start on the answer itself, so nothing may follow the
        # marker except the newline that separates it.
        self.assertEqual(
            format_prompt("Expand the acronym GPU."),
            "### User:\nExpand the acronym GPU.\n\n### Assistant:\n",
        )

    def test_training_text_is_prompt_plus_answer(self):
        # The regression this guards: lora_inference.py used to send the bare
        # prompt, so the model saw a shape it never trained on and stopped
        # immediately. Training text and generation prompt must share a prefix.
        record = {
            "instruction": "Expand the acronym GPU.",
            "input": "",
            "output": "GPU means graphics processing unit.",
        }
        formatted = format_record(record, self.EOS)
        prompt = format_prompt(record["instruction"])
        self.assertTrue(formatted.startswith(prompt))
        self.assertEqual(formatted[len(prompt):], record["output"] + self.EOS)

    def test_input_field_stays_with_the_request(self):
        # Extra context belongs above the assistant marker; if it leaked below,
        # the model would be trained to produce it rather than react to it.
        formatted = format_record(
            {
                "instruction": "Summarize the paragraph.",
                "input": "GPUs run many small operations at once.",
                "output": "GPUs are parallel.",
            },
            self.EOS,
        )
        request, answer = formatted.split("### Assistant:\n")
        self.assertIn("GPUs run many small operations at once.", request)
        self.assertEqual(answer, "GPUs are parallel." + self.EOS)

    def test_training_text_ends_with_the_stop_token(self):
        # The regression this guards: without a trailing end-of-sequence token
        # the model is never shown where an answer ends, so generation runs on
        # until it hits the token cap, usually repeating the marker.
        formatted = format_record(
            {"instruction": "Expand the acronym GPU.", "input": "", "output": "Done."},
            self.EOS,
        )
        self.assertTrue(formatted.endswith(self.EOS))
        # And exactly once -- a stray extra copy would train the model to emit
        # a visible token rather than to stop.
        self.assertEqual(formatted.count(self.EOS), 1)

    def test_generation_prompt_carries_no_stop_token(self):
        # format_prompt feeds generation, so a stop token anywhere inside it
        # would tell the model to finish before it has written anything.
        self.assertNotIn(self.EOS, format_prompt("Expand the acronym GPU."))

    def test_empty_prompt_still_produces_the_markers(self):
        # An empty --prompt is user error, but it must not silently collapse
        # into text with no assistant cue at all.
        self.assertEqual(format_prompt(""), "### User:\n\n\n### Assistant:\n")

    def test_device_prefers_cuda_then_mps(self):
        class Available:
            @staticmethod
            def is_available():
                return True

        class TorchWithCuda:
            cuda = Available()
            backends = type("Backends", (), {"mps": Available()})()

        class Unavailable:
            @staticmethod
            def is_available():
                return False

        class TorchWithMps:
            cuda = Unavailable()
            backends = type("Backends", (), {"mps": Available()})()

        self.assertEqual(choose_device(TorchWithCuda()), "cuda")
        self.assertEqual(choose_device(TorchWithMps()), "mps")


if __name__ == "__main__":
    unittest.main()
