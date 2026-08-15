import importlib.util
import math
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ANALYZERS = (
    HERE / "analyze_depthtrack_train_state_trace.py",
    HERE / "analyze_depthtrack_train_state_trace_full152.py",
)


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def anchor_unavailable_record():
    return {
        "prior_bbox": [366.0, 114.0, 137.0, 161.0],
        "candidate_bbox": [369.0, 113.0, 133.0, 164.0],
        "online_evidence": {
            "confidence": 0.96,
            "response_margin": 0.95,
            "identity_similarity": None,
            "normalized_center_jump": None,
            "log_depth_change": None,
            "dynamic_active": False,
            "checked": False,
            "stable_frames": 0,
            "reasons": ["anchor_unavailable"],
        },
    }


class StateTraceAnalyzerFeatureTest(unittest.TestCase):
    def test_anchor_unavailable_is_a_finite_non_checked_feature_row(self):
        for analyzer_path in ANALYZERS:
            with self.subTest(analyzer=analyzer_path.name):
                module = load_module(analyzer_path)
                values = module.features(anchor_unavailable_record())
                self.assertEqual(len(values), len(module.FEATURE_NAMES))
                self.assertTrue(all(math.isfinite(value) for value in values))
                self.assertEqual(values[4], 0.0)
                self.assertEqual(values[10], 0.0)

    def test_checked_evidence_cannot_hide_a_missing_center_jump(self):
        record = anchor_unavailable_record()
        record["online_evidence"]["checked"] = True
        for analyzer_path in ANALYZERS:
            with self.subTest(analyzer=analyzer_path.name):
                module = load_module(analyzer_path)
                with self.assertRaisesRegex(
                        ValueError, "missing center jump"):
                    module.features(record)


if __name__ == "__main__":
    unittest.main()
