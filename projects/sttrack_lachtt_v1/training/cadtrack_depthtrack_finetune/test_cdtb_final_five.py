"""CPU control-flow checks; synthetic files do not validate model performance."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "cdtb_final_five", Path(__file__).with_name("cdtb_final_five.py")
)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


class BatchTests(unittest.TestCase):
    def fixture(self, root):
        cases = []
        rows = []
        for index in range(80):
            frames = 1274 if index < 79 else 1310
            name = "seq%02d" % index
            boxes = root / (name + ".txt")
            scores = root / (name + "_all_scores.txt")
            boxes.write_text("0,0,10,10\n" * frames)
            scores.write_text("1\n" * frames)
            cases.append(dict(sequence=name, frames=frames, gt_sha256="synthetic-gt"))
            rows.append(dict(
                sequence=name, frames=frames, groundtruth_sha256="synthetic-gt",
                boxes_sha256=batch.sha(boxes), scores_sha256=batch.sha(scores),
            ))
        cases_path = root / "cases.json"
        batch.write(cases_path, cases)
        receipt = dict(
            status="complete_CADTrack_RGBD_OPE_predictions",
            dataset="cdtb", checkpoint_sha256="synthetic-weight",
            cases_sha256=batch.sha(cases_path), runner_sha256="synthetic-runner",
            groundtruth_opened_during_tracking=False, optimizer_steps=0,
            sequences=rows, frames=101956,
        )
        batch.write(root / "receipt.json", receipt)
        batch.write(root / "metrics.json", dict(
            status="complete_CADTrack_RGBD_OPE_metrics",
            dataset="cdtb", checkpoint_sha256="synthetic-weight",
            cases_sha256=batch.sha(cases_path), runner_sha256="synthetic-runner",
            receipt_sha256=batch.sha(root / "receipt.json"),
            metric_source_sha256=batch.METRIC_SHA256,
            metrics=dict(
                sequences=80, frames=101956,
                precision_percent=50, recall_percent=50, f_score_percent=50,
            ),
        ))
        return cases_path

    def test_complete_coverage_and_checkpoint_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = self.fixture(root)
            result = batch.verify_result(root, "cdtb", "synthetic-weight", cases)
            self.assertEqual(result["metrics"]["frames"], 101956)
            with self.assertRaises(AssertionError):
                batch.verify_result(root, "cdtb", "different-weight", cases)

    def test_truncated_trajectory_rejected_even_with_updated_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = self.fixture(root)
            boxes = root / "seq00.txt"
            boxes.write_text("0,0,10,10\n")
            receipt = batch.read(root / "receipt.json")
            receipt["sequences"][0]["boxes_sha256"] = batch.sha(boxes)
            batch.write(root / "receipt.json", receipt)
            metrics = batch.read(root / "metrics.json")
            metrics["receipt_sha256"] = batch.sha(root / "receipt.json")
            batch.write(root / "metrics.json", metrics)
            with self.assertRaises(AssertionError):
                batch.verify_result(root, "cdtb", "synthetic-weight", cases)

    def test_failed_stage_has_exit_receipt_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(AssertionError):
                batch.stage(root, "track", [sys.executable, "-c", "exit(7)"], dict(os.environ), 0)
            self.assertEqual(batch.read(root / "track.receipt.json")["exit_code"], 7)
            self.assertEqual((root / "track.exit").read_text(), "7\n")
            with self.assertRaises(AssertionError):
                batch.stage(root, "track", [sys.executable, "-c", "exit(0)"], dict(os.environ), 0)


if __name__ == "__main__":
    unittest.main()
