"""Mutation tests for full-sequence cumulative training receipt checks."""
from copy import deepcopy
import unittest

from roi_artifact_graph import require, sequence_history


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


if __name__ == "__main__":
    unittest.main()
