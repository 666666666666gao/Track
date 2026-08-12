import sys
import types
import unittest

import torch


if "fvcore.nn" not in sys.modules:
    fvcore = types.ModuleType("fvcore")
    fvcore_nn = types.ModuleType("fvcore.nn")
    fvcore_nn.FlopCountAnalysis = object
    fvcore_nn.parameter_count_table = lambda *args, **kwargs: ""
    fvcore.nn = fvcore_nn
    sys.modules["fvcore"] = fvcore
    sys.modules["fvcore.nn"] = fvcore_nn


class FrameLanguageFusionTest(unittest.TestCase):
    def test_language_feature_modulates_visual_search_feature(self):
        from lib.models.mplt_track.mplt_track import FrameLanguageFusion

        fusion = FrameLanguageFusion(hidden_dim=8, residual_weight=0.2)
        visual = torch.randn(2, 8, 4, 4)
        language = torch.randn(2, 8)

        output, similarity = fusion(visual, language)

        self.assertEqual(output.shape, visual.shape)
        self.assertEqual(similarity.shape, (2, 1, 4, 4))
        self.assertFalse(torch.equal(output, visual))

    def test_missing_language_preserves_baseline_feature(self):
        from lib.models.mplt_track.mplt_track import FrameLanguageFusion

        fusion = FrameLanguageFusion(hidden_dim=8, residual_weight=0.2)
        visual = torch.randn(1, 8, 2, 2)

        output, similarity = fusion(visual, None)

        self.assertIs(output, visual)
        self.assertIsNone(similarity)


if __name__ == "__main__":
    unittest.main()
