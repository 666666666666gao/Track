import os
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np


class RGBDEvalColorDescriptionTest(unittest.TestCase):
    def _write_sequence(self, root, color_root, descriptions=None, sequence="mug_outside"):
        sequence_dir = os.path.join(root, sequence)
        color_dir = os.path.join(sequence_dir, "color")
        depth_dir = os.path.join(sequence_dir, "depth")
        os.makedirs(color_dir)
        os.makedirs(depth_dir)
        for index in range(2):
            cv2.imwrite(
                os.path.join(color_dir, "{:08d}.jpg".format(index + 1)),
                np.full((20, 30, 3), 30 + index, dtype=np.uint8),
            )
            cv2.imwrite(
                os.path.join(depth_dir, "{:08d}.png".format(index + 1)),
                np.full((20, 30), 100 + index, dtype=np.uint16),
            )
        with open(os.path.join(sequence_dir, "groundtruth.txt"), "w", encoding="utf-8") as handle:
            handle.write("1,2,10,8\n2,3,10,8\n")
        output_dir = os.path.join(color_root, sequence)
        os.makedirs(output_dir)
        with open(os.path.join(output_dir, "color_description_ct.txt"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(descriptions or ["A blue mug."]))

    def test_rgbd_all_alias_integrates_three_test_datasets_without_name_collisions(self):
        from lib.test.evaluation.datasets import dataset_groups, get_dataset

        with tempfile.TemporaryDirectory() as root:
            roots = {
                "depthtrack": os.path.join(root, "depthtrack"),
                "cdtb": os.path.join(root, "cdtb"),
                "votrgbd2022": os.path.join(root, "votrgbd2022"),
            }
            colors = {
                name: os.path.join(root, "color_desc", name)
                for name in roots
            }
            self._write_sequence(
                roots["depthtrack"], colors["depthtrack"],
                ["A white adapter."], sequence="adapter_indoor")
            self._write_sequence(
                roots["cdtb"], colors["cdtb"],
                ["A black backpack."], sequence="backpack_blue")
            self._write_sequence(
                roots["votrgbd2022"], colors["votrgbd2022"],
                ["A plush toy."], sequence="cartman_1")
            settings = types.SimpleNamespace(
                depthtrack_path=roots["depthtrack"],
                depthtrack_test_color_desc_root=colors["depthtrack"],
                cdtb_path=roots["cdtb"],
                cdtb_color_desc_root=colors["cdtb"],
                votrgbd2022_path=roots["votrgbd2022"],
                votrgbd2022_color_desc_root=colors["votrgbd2022"],
                prj_dir=root,
            )

            with patch("lib.test.evaluation.data.env_settings", return_value=settings):
                sequences = get_dataset("rgbd_all")
                synonym_sequences = get_dataset("rgbd_all_test")
                with patch.dict(dataset_groups, {"rgbd_nested": ("rgbd_all",)}):
                    nested_sequences = get_dataset("rgbd_nested")
                single_dataset_sequences = get_dataset("cdtb")

            self.assertEqual(len(sequences), 3)
            self.assertEqual(
                [sequence.name for sequence in sequences],
                [
                    "depthtrack__adapter_indoor",
                    "cdtb__backpack_blue",
                    "votrgbd2022__cartman_1",
                ],
            )
            self.assertEqual(
                [sequence.dataset for sequence in sequences],
                ["depthtrack", "cdtb", "votrgbd2022"],
            )
            self.assertEqual(
                [sequence.original_name for sequence in sequences],
                ["adapter_indoor", "backpack_blue", "cartman_1"],
            )
            self.assertEqual(
                [sequence.name for sequence in synonym_sequences],
                [sequence.name for sequence in sequences],
            )
            self.assertEqual(
                [sequence.name for sequence in nested_sequences],
                [sequence.name for sequence in sequences],
            )
            self.assertEqual(
                [sequence.original_name for sequence in nested_sequences],
                ["adapter_indoor", "backpack_blue", "cartman_1"],
            )
            self.assertEqual([sequence.name for sequence in single_dataset_sequences], ["backpack_blue"])
            self.assertFalse(hasattr(single_dataset_sequences[0], "original_name"))

    def test_rgbd_all_test_is_an_explicit_synonym_for_rgbd_all(self):
        from lib.test.evaluation.datasets import dataset_groups

        self.assertIn("rgbd_all", dataset_groups)
        self.assertIn("rgbd_all_test", dataset_groups)
        self.assertEqual(dataset_groups["rgbd_all_test"], dataset_groups["rgbd_all"])

    def test_dataset_group_cycles_fail_with_a_clear_error(self):
        from lib.test.evaluation.datasets import dataset_groups, get_dataset

        with patch.dict(dataset_groups, {"rgbd_cycle_a": ("rgbd_cycle_b",),
                                         "rgbd_cycle_b": ("rgbd_cycle_a",)}):
            with self.assertRaisesRegex(ValueError, "Dataset group cycle"):
                get_dataset("rgbd_cycle_a")

    def test_dataset_group_rejects_duplicate_prefixed_names(self):
        from lib.test.evaluation.datasets import dataset_groups, get_dataset

        with tempfile.TemporaryDirectory() as root:
            dataset_root = os.path.join(root, "cdtb")
            color_root = os.path.join(root, "color_desc")
            self._write_sequence(dataset_root, color_root, sequence="backpack_blue")
            settings = types.SimpleNamespace(
                cdtb_path=dataset_root,
                cdtb_color_desc_root=color_root,
                prj_dir=root,
            )

            with patch("lib.test.evaluation.data.env_settings", return_value=settings), \
                    patch.dict(dataset_groups, {"rgbd_duplicate": ("cdtb", "cdtb")}):
                with self.assertRaisesRegex(ValueError, "duplicate sequence names"):
                    get_dataset("rgbd_duplicate")

    def test_evaluation_sequence_uses_one_initial_description_for_the_whole_sequence(self):
        from lib.test.evaluation.rgbddataset import RGBDDataset

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as color_root:
            self._write_sequence(root, color_root)
            settings = types.SimpleNamespace(dataset_path=root, color_root=color_root, prj_dir=root)
            with patch("lib.test.evaluation.data.env_settings", return_value=settings):
                dataset = RGBDDataset(
                    dataset_name="synthetic",
                    root_attr="dataset_path",
                    color_desc_attr="color_root",
                )
                sequence = dataset.get_sequence_list()[0]

            self.assertEqual(sequence.init_info()["init_language_description"], "A blue mug.")
            self.assertNotIn("language_description", sequence.frame_info(1))

    def test_evaluation_rejects_per_frame_text_for_initial_only_test_data(self):
        from lib.test.evaluation.rgbddataset import RGBDDataset

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as color_root:
            self._write_sequence(root, color_root, ["A blue mug.", "The mug moves right."])
            settings = types.SimpleNamespace(dataset_path=root, color_root=color_root, prj_dir=root)
            with patch("lib.test.evaluation.data.env_settings", return_value=settings):
                dataset = RGBDDataset(
                    dataset_name="synthetic",
                    root_attr="dataset_path",
                    color_desc_attr="color_root",
                )
                with self.assertRaisesRegex(ValueError, "exactly one initial description"):
                    dataset.get_sequence_list()

    def test_evaluation_rejects_ground_truth_frame_count_mismatch(self):
        from lib.test.evaluation.rgbddataset import RGBDDataset

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as color_root:
            self._write_sequence(root, color_root)
            with open(os.path.join(root, "mug_outside", "groundtruth.txt"),
                      "w", encoding="utf-8") as handle:
                handle.write("1,2,10,8\n")
            settings = types.SimpleNamespace(dataset_path=root, color_root=color_root, prj_dir=root)
            with patch("lib.test.evaluation.data.env_settings", return_value=settings):
                dataset = RGBDDataset(
                    dataset_name="synthetic",
                    root_attr="dataset_path",
                    color_desc_attr="color_root",
                )
                with self.assertRaisesRegex(ValueError, "ground-truth frame count mismatch"):
                    dataset.get_sequence_list()

    def test_required_initial_descriptions_fail_fast_when_root_is_missing(self):
        from lib.test.evaluation.rgbddataset import RGBDDataset

        with tempfile.TemporaryDirectory() as root:
            sequence_root = os.path.join(root, "dataset")
            os.makedirs(sequence_root)
            settings = types.SimpleNamespace(dataset_path=sequence_root, prj_dir=root)
            with patch("lib.test.evaluation.data.env_settings", return_value=settings):
                with self.assertRaisesRegex(ValueError, "annotation root is missing"):
                    RGBDDataset(
                        dataset_name="synthetic",
                        root_attr="dataset_path",
                        color_desc_attr="missing_color_root",
                        require_color_descriptions=True,
                    )


if __name__ == "__main__":
    unittest.main()
