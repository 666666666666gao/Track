import sys
import types
import unittest

import numpy as np


if "tensorboardX" not in sys.modules:
    tensorboard_x = types.ModuleType("tensorboardX")
    tensorboard_x.SummaryWriter = object
    sys.modules["tensorboardX"] = tensorboard_x

if "lmdb" not in sys.modules:
    sys.modules["lmdb"] = types.ModuleType("lmdb")


class SequenceFrameDescriptionTest(unittest.TestCase):
    def test_initial_only_description_is_not_exposed_as_per_frame_text(self):
        from lib.test.evaluation.data import Sequence

        sequence = Sequence(
            "cup01_indoor",
            [["rgb0.jpg", "rgb1.jpg"], ["depth0.png", "depth1.png"]],
            "depthtrack",
            np.array([[1, 2, 3, 4], [2, 3, 3, 4]], dtype=np.float32),
            init_data={0: {"bbox": [1, 2, 3, 4], "language_description": "A red cup."}},
        )

        self.assertEqual(sequence.init_info()["init_language_description"], "A red cup.")
        self.assertNotIn("language_description", sequence.frame_info(1))


if __name__ == "__main__":
    unittest.main()
