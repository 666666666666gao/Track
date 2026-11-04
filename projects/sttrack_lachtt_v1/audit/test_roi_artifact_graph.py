"""Mutation tests for full-sequence cumulative training receipt checks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from roi_artifact_graph import (BUNDLE, FINAL, check_ope_binding, check_shard_partition, check_stage_names,
                                require, sequence_history, sha, strict_cpu_load)


class HistoryTests(unittest.TestCase):
    def fixture(self):
        rows, total = [], 0
        for epoch in (1, 2):
            for index in range(152):
                calls = 1456 if index == 0 else 1446
                total += calls
                rows.append(dict(epoch=epoch, sequence="seq%03d" % index,
                                 sequence_index=index, sequences_per_epoch=152,
                                 track_calls=calls, total_track_calls=total))
        return dict(epochs=2, track_calls=439604, sequence_records=rows)

    def test_complete_history(self):
        trained = self.fixture()
        self.assertEqual(len(sequence_history(trained)), 304)

    def test_substitution_duplicate_and_partial_history(self):
        original = self.fixture()
        for index in (0, 152):
            changed = deepcopy(original)
            changed["sequence_records"][index]["sequence"] = "seq001"
            with self.assertRaises(ValueError):
                sequence_history(changed)
        changed = deepcopy(original)
        changed["sequence_records"].pop()
        with self.assertRaises(ValueError):
            sequence_history(changed)

    def test_inconsistent_cumulative_calls(self):
        changed = self.fixture()
        changed["sequence_records"][200]["total_track_calls"] += 1
        with self.assertRaises(ValueError):
            sequence_history(changed)

    def test_explicit_checks_survive_optimization(self):
        with self.assertRaises(ValueError):
            require(False, "must reject")

    def test_shard_partition_identity_and_cardinality(self):
        shards = [dict(index=i, gpu=(0, 1, 0, 1)[i],
                       anchor_count=(441, 441, 441, 442)[i]) for i in range(4)]
        check_shard_partition(shards)
        for changed in (shards[:3], [shards[0], shards[1], shards[1], shards[3]]):
            with self.assertRaises(ValueError):
                check_shard_partition(changed)

    def test_metrics_must_bind_to_dataset_plan_and_predictions(self):
        prediction = dict(plan_sha256="depth-plan", final_sha256=FINAL,
                          bundle_sha256=BUNDLE, status="complete_M122_official_OPE_predictions",
                          frames=12)
        metrics = dict(plan_sha256="depth-plan", receipt_sha256="receipt",
                       metric_source_sha256="metric", final_sha256=FINAL,
                       bundle_sha256=BUNDLE,
                       metrics=dict(sequences=2, frames=12))
        with tempfile.TemporaryDirectory() as directory:
            cases = Path(directory) / "cases.json"
            cases.write_text(json.dumps([{}, {}]))
            plan = dict(cases_path=str(cases), metric_source_sha256="metric")
            plan_digest = sha(cases)
            prediction["plan_sha256"] = plan_digest
            metrics["plan_sha256"] = plan_digest
            metrics["receipt_sha256"] = plan_digest
            check_ope_binding(str(cases), plan, prediction, metrics, str(cases))
            swapped = deepcopy(metrics)
            swapped["plan_sha256"] = "cdtb-plan"
            with self.assertRaises(ValueError):
                check_ope_binding(str(cases), plan, prediction, swapped, str(cases))

    def test_stage_names_cannot_be_satisfied_by_repetition(self):
        names = (["train_fixed_pass"] * 6 + ["prepare", "depthtrack_test_track",
                 "cdtb_track", "depthtrack_test_analyze", "cdtb_analyze", "vot_track",
                 "vot_official_analysis", "vot_seal", "collect"])
        check_stage_names([dict(stage=name) for name in names])
        names[-1] = names[-2]
        with self.assertRaises(ValueError):
            check_stage_names([dict(stage=name) for name in names])


if __name__ == "__main__":
    unittest.main()
