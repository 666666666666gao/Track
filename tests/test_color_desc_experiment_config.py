import os
import unittest

import yaml


class ColorDescriptionExperimentConfigTest(unittest.TestCase):
    def test_language_config_uses_train_holdout_and_aligned_text_modes(self):
        project_root = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(
            project_root,
            "experiments",
            "mplt_track",
            "vitb_256_mplt_32x1_1e4_depthtrack_15ep_color_desc.yaml",
        )
        with open(path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        self.assertEqual(config["DATA"]["TRAIN"]["DATASETS_NAME"], ["DepthTrack_train_main"])
        self.assertEqual(config["DATA"]["VAL"]["DATASETS_NAME"], ["DepthTrack_train_val"])
        self.assertNotIn("DepthTrack_test", config["DATA"]["VAL"]["DATASETS_NAME"])
        self.assertEqual(config["TRAIN"]["LANGUAGE_TEXT_MODE"], "current")
        self.assertEqual(config["TEST"]["LANGUAGE_TEXT_MODE"], "initial")
        self.assertEqual(config["DATA"]["HORIZONTAL_FLIP_PROBABILITY"], 0.0)
        self.assertNotIn("E:/1111", config["MODEL"]["PRETRAIN_FILE"])
        self.assertNotIn("E:/1111", config["MODEL"]["LANGUAGE"]["MODEL_NAME_OR_PATH"])


if __name__ == "__main__":
    unittest.main()
