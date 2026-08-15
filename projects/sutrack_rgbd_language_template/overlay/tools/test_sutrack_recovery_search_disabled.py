import unittest
from unittest.mock import patch

from lib.test.tracker.sutrack import SUTRACK
from lib.test.tracker.sutrack_recovery_search import SUTRACKRecoverySearch


class RecoverySearchDisabledTest(unittest.TestCase):
    def test_disabled_path_returns_exact_parent_prediction(self):
        tracker = object.__new__(SUTRACKRecoverySearch)
        tracker.recovery_search_use = False
        tracker.recovery_search_factor = 6.0
        target_bbox = [
            278.4265251159668,
            119.18964767456052,
            92.67017364501953,
            97.25887298583984,
        ]
        candidate_bbox = list(target_bbox)
        parent_output = {
            "target_bbox": target_bbox,
            "best_score": 0.91,
            "safe_template_decision": object(),
            "online_state_evidence": {
                "candidate_bbox": candidate_bbox,
                "confidence": 0.91,
            },
        }

        with patch.object(
                SUTRACK, "track", return_value=parent_output) as parent_track:
            output = tracker.track("image", {"depth_path": "depth.png"})

        parent_track.assert_called_once_with(
            "image", {"depth_path": "depth.png"})
        self.assertIs(output, parent_output)
        self.assertIs(output["target_bbox"], target_bbox)
        self.assertIs(
            output["online_state_evidence"]["candidate_bbox"],
            candidate_bbox,
        )
        recovery = output["recovery_search_evidence"]
        self.assertIs(recovery["enabled"], False)
        self.assertIs(recovery["second_pass"], False)
        self.assertIs(recovery["recovery_selected"], False)
        self.assertEqual(recovery["baseline_bbox"], candidate_bbox)
        self.assertIsNone(recovery["recovery_bbox"])


if __name__ == "__main__":
    unittest.main()
