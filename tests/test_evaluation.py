import contextlib
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import RUBRICS, build_message, load_tasks
from evaluate import evaluate_batches, main
from metrics import calculate_metrics, parse_score, read_results, summarize


class ScoreTests(unittest.TestCase):
    def test_accept_scores(self):
        for value in (1, "1", '"1"', "'1'", " 1 ", "<think>2 or 3?</think><answer>1</answer>"):
            with self.subTest(value=value):
                self.assertEqual(parse_score(value), 1)

    def test_reject_ambiguous_or_out_of_range_scores(self):
        for value in (True, 1.0, None, {}, 0, 5, "10", "3.5", "Score 3", "1 or 4",
                      "<think>4</think>", "<answer>3 or 4</answer>",
                      "<answer>2</answer><answer>3</answer>", "<answer>2", "2</answer>"):
            with self.subTest(value=value):
                self.assertIsNone(parse_score(value))

    def test_metrics_and_denominators(self):
        records = [{"true_score": truth, "model_output_score": prediction}
                   for truth, prediction in ((1, "1"), (2, 3), (4, 2), (3, None))]
        report = calculate_metrics(records)
        self.assertEqual(report["total_samples"], 4)
        self.assertEqual(report["valid_samples"], 3)
        self.assertEqual(report["exact_accuracy"], 0.25)
        self.assertEqual(report["adjacent_accuracy"], 0.5)
        self.assertEqual(report["valid_exact_accuracy"], 1 / 3)
        self.assertEqual(report["mae"], 1)
        self.assertAlmostEqual(report["mse"], 5 / 3)
        self.assertAlmostEqual(report["rmse"], math.sqrt(5 / 3))
        self.assertEqual(sum(map(sum, report["confusion_matrix"])), 3)

    def test_all_invalid_is_reported_not_nan(self):
        report = calculate_metrics([{"true_score": 2, "model_output_score": None}])
        self.assertIsNone(report["mae"])
        self.assertEqual(report["exact_accuracy"], 0)
        self.assertEqual(report["invalid_predictions"], 1)
        json.dumps(report, allow_nan=False)

    def test_inference_error_is_invalid_even_with_stale_score(self):
        report = calculate_metrics([{"true_score": 1, "model_output_score": 1, "error": "RuntimeError"}])
        self.assertEqual(report["inference_errors"], 1)
        self.assertEqual(report["valid_samples"], 0)

    def test_invalid_ground_truth_fails(self):
        for truth in (0, 5, "1", 1.0, True, None):
            with self.subTest(truth=truth), self.assertRaises(ValueError):
                calculate_metrics([{"true_score": truth, "model_output_score": 1}])
        with self.assertRaises(ValueError):
            calculate_metrics([])

    def test_per_category_metrics(self):
        report = summarize([
            {"category": "BAP", "true_score": 1, "model_output_score": 1},
            {"category": "EST", "true_score": 4, "model_output_score": 2},
        ])
        self.assertEqual(report["overall"]["exact_accuracy"], 0.5)
        self.assertEqual(report["by_category"]["BAP"]["exact_accuracy"], 1)

    def test_json_jsonl_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = {"id": "one", "true_score": 1, "model_output_score": "1"}
            for extension, text in (("json", json.dumps([item])), ("jsonl", json.dumps(item) + "\n")):
                path = root / f"input.{extension}"
                path.write_text(text, encoding="utf-8")
                self.assertEqual(read_results(path), [item])
            path.write_text(json.dumps(item) + "\n" + json.dumps(item), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_results(path)


class FolderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def scene(self, code="BAP", name=None):
        config = RUBRICS[code]
        scene = self.root / "nested" / (name or config["prefixes"][0] + "_001")
        scene.mkdir(parents=True)
        names = [f"Model_{score}.wav" for score in range(1, 5)]
        names += (["user_speech1.wav", "model_speech2.wav", "user_speech10.wav"]
                  if config["multi_turn"] else ["user_neutral.wav"])
        for name in names:
            (scene / name).touch()
        (scene / "user_metadata.json").write_text("{}", encoding="utf-8")
        return scene

    def test_all_ten_categories(self):
        for code in RUBRICS:
            self.scene(code)
        tasks = load_tasks(self.root, list(RUBRICS))
        self.assertEqual(len(tasks), 40)
        self.assertEqual(len({task["id"] for task in tasks}), 40)
        self.assertEqual({task["category"] for task in tasks}, set(RUBRICS))
        self.assertTrue(all(not Path(task["model_response_audio"]).is_absolute() for task in tasks))

    def test_category_subset(self):
        self.scene("BAP")
        self.scene("IAA")
        self.assertEqual({item["category"] for item in load_tasks(self.root, ["IAA"])}, {"IAA"})

    def test_missing_requested_category_fails(self):
        self.scene()
        with self.assertRaises(ValueError):
            load_tasks(self.root, ["BAP", "ECA"])

    def test_typo_and_corrected_alias(self):
        self.scene("IAA", "ImplicitAttitude_001")
        self.assertEqual(len(load_tasks(self.root, ["IAA"])), 4)

    def test_single_scene_as_root(self):
        scene = self.scene()
        self.assertEqual(len(load_tasks(scene, ["BAP"])), 4)

    def test_multi_turn_numeric_order_and_prompt(self):
        self.scene("EST")
        task = load_tasks(self.root, ["EST"])[0]
        self.assertTrue(task["final_user_audio"].endswith("user_speech10.wav"))
        self.assertEqual([item["role"] for item in task["history_context_audios"]], ["user", "model"])
        message = build_message(task, self.root)[0]["content"]
        self.assertEqual(len([part for part in message if part["type"] == "audio"]), 4)
        altered = {**task, "true_score": 4}
        self.assertEqual(build_message(task, self.root), build_message(altered, self.root))

    def test_duplicate_speech_number_fails(self):
        scene = self.scene("EST")
        (scene / "model_speech1.wav").touch()
        with self.assertRaises(ValueError):
            load_tasks(self.root, ["EST"])

    def test_final_model_turn_fails(self):
        scene = self.scene("ECA")
        (scene / "model_speech11.wav").touch()
        with self.assertRaises(ValueError):
            load_tasks(self.root, ["ECA"])

    def test_incomplete_and_extra_score_fail(self):
        scene = self.scene()
        (scene / "Model_4.wav").unlink()
        with self.assertRaises(ValueError):
            load_tasks(self.root, ["BAP"])
        (scene / "Model_4.wav").touch()
        (scene / "Model_10.wav").touch()
        with self.assertRaises(ValueError):
            load_tasks(self.root, ["BAP"])

    def test_duplicate_score_or_user_fails(self):
        scene = self.scene()
        extra = scene / "Model_1_extra.wav"
        extra.touch()
        with self.assertRaises(ValueError):
            load_tasks(self.root, ["BAP"])
        extra.unlink()
        (scene / "user_extra.wav").touch()
        with self.assertRaises(ValueError):
            load_tasks(self.root, ["BAP"])

    def test_prompt_reasoning_options(self):
        self.scene()
        task = load_tasks(self.root, ["BAP"])[0]
        prompt = build_message(task, self.root, think=True, think_max_len=100)[0]["content"][-1]["text"]
        self.assertIn("less than 100 words", prompt)
        prompt = build_message(task, self.root, think=True, think_max_len=0)[0]["content"][-1]["text"]
        self.assertNotIn("less than", prompt)

    def test_dry_run_without_model_dependencies(self):
        self.scene()
        script = Path(__file__).resolve().parents[1] / "evaluate.py"
        result = subprocess.run([sys.executable, "-B", str(script), "--data-root", str(self.root),
                                 "--categories", "BAP", "--dry-run"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["total_samples"], 4)

    def test_mocked_cli_and_overwrite_protection(self):
        self.scene()
        output = self.root / "output" / "results.jsonl"
        argv = ["evaluate.py", "--data-root", str(self.root), "--categories", "BAP",
                "--model", "example/model", "--output", str(output)]
        with patch.object(sys, "argv", argv), patch("evaluate.make_predictor") as factory, \
                patch("evaluate.version", return_value="test"), contextlib.redirect_stdout(io.StringIO()):
            factory.return_value = lambda batch: ["<answer>2</answer>" for _ in batch]
            self.assertEqual(main(), 0)
            self.assertEqual(len(read_results(output)), 4)
            report = json.loads(output.with_suffix(".metrics.json").read_text())
            self.assertEqual(report["overall"]["exact_accuracy"], 0.25)
            metadata = json.loads(output.with_suffix(".run.json").read_text())
            self.assertEqual(metadata["total_samples"], 4)
            original = output.read_bytes()
            with self.assertLogs(level="ERROR"):
                self.assertEqual(main(), 1)
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(factory.call_count, 1)


class BatchTests(unittest.TestCase):
    def test_middle_batch_failure_does_not_shift_predictions(self):
        tasks = [{"id": str(i), "true_score": (i % 4) + 1} for i in range(6)]

        def predict(batch):
            if batch[0]["id"] == "2":
                raise RuntimeError("simulated failure")
            return [str(item["true_score"]) for item in batch]

        with self.assertLogs(level="ERROR"):
            results = list(evaluate_batches(tasks, predict, batch_size=2))
        self.assertEqual([item["id"] for item in results], [item["id"] for item in tasks])
        self.assertEqual([item["model_output_score"] for item in results], [1, 2, None, None, 1, 2])
        self.assertEqual(calculate_metrics(results)["inference_errors"], 2)
        self.assertNotIn("error", tasks[0])

    def test_wrong_output_count_marks_whole_batch_failed(self):
        tasks = [{"id": str(i), "true_score": 1} for i in range(2)]
        with self.assertLogs(level="ERROR"):
            results = list(evaluate_batches(tasks, lambda batch: ["1"], batch_size=2))
        self.assertEqual(len(results), 2)
        self.assertTrue(all(item["error"] == "ValueError" for item in results))

    def test_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            list(evaluate_batches([], lambda batch: [], batch_size=0))


if __name__ == "__main__":
    unittest.main()
