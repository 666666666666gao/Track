import os
import sys
import tempfile
import types
import unittest
import warnings

import cv2
import numpy as np


if "tensorboardX" not in sys.modules:
    tensorboard_x = types.ModuleType("tensorboardX")
    tensorboard_x.SummaryWriter = object
    sys.modules["tensorboardX"] = tensorboard_x

if "lmdb" not in sys.modules:
    sys.modules["lmdb"] = types.ModuleType("lmdb")


class DepthTrackColorDescriptionTest(unittest.TestCase):
    def _write_sequence(self, root, color_root):
        sequence = "cup01_indoor"
        sequence_dir = os.path.join(root, sequence)
        color_dir = os.path.join(sequence_dir, "color")
        depth_dir = os.path.join(sequence_dir, "depth")
        os.makedirs(color_dir)
        os.makedirs(depth_dir)
        for index in range(2):
            rgb = np.full((20, 30, 3), 20 + index, dtype=np.uint8)
            depth = np.full((20, 30), 100 + index, dtype=np.uint16)
            cv2.imwrite(os.path.join(color_dir, "{:08d}.jpg".format(index + 1)), rgb)
            cv2.imwrite(os.path.join(depth_dir, "{:08d}.png".format(index + 1)), depth)
        with open(os.path.join(sequence_dir, "groundtruth.txt"), "w", encoding="utf-8") as handle:
            handle.write("1,2,10,8\n2,3,10,8\n")

        description_dir = os.path.join(color_root, sequence)
        os.makedirs(description_dir)
        with open(
            os.path.join(description_dir, "color_description_ct.txt"), "w", encoding="utf-8"
        ) as handle:
            handle.write("A red cup on a desk.\nThe red cup moves beside a book.")
        return sequence

    def test_get_frames_returns_descriptions_aligned_to_requested_ids(self):
        from lib.train.dataset.depthtrack import DepthTrack

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as color_root:
            sequence = self._write_sequence(root, color_root)
            dataset = DepthTrack(root=root, split="train", color_desc_root=color_root)
            frames, annotations, metadata = dataset.get_frames(0, [1, 0])

            self.assertEqual(len(frames), 2)
            self.assertEqual(
                annotations["language_description"],
                ["The red cup moves beside a book.", "A red cup on a desk."],
            )
            self.assertEqual(annotations["language_frame_index"], [1, 0])
            self.assertEqual(metadata["language_source"], "color_description_ct")
            self.assertNotIn("language_description", metadata)
            self.assertEqual(dataset.sequence_list, [sequence])

    def test_required_descriptions_fail_fast_when_a_sequence_file_is_missing(self):
        from lib.train.dataset.depthtrack import DepthTrack
        from lib.utils.color_descriptions import ColorDescriptionError

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as color_root:
            sequence = self._write_sequence(root, color_root)
            os.remove(os.path.join(color_root, sequence, "color_description_ct.txt"))

            with self.assertRaisesRegex(ColorDescriptionError, sequence):
                DepthTrack(
                    root=root,
                    split="train",
                    color_desc_root=color_root,
                    require_color_descriptions=True,
                )

    def test_deterministic_holdout_partitions_are_disjoint_and_complete(self):
        from lib.train.dataset.depthtrack import DepthTrack

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as color_root:
            for index in range(10):
                sequence = "object{:02d}_indoor".format(index)
                sequence_dir = os.path.join(root, sequence)
                os.makedirs(sequence_dir)
                color_dir = os.path.join(sequence_dir, "color")
                depth_dir = os.path.join(sequence_dir, "depth")
                os.makedirs(color_dir)
                os.makedirs(depth_dir)
                cv2.imwrite(
                    os.path.join(color_dir, "00000001.jpg"),
                    np.full((20, 30, 3), index, dtype=np.uint8),
                )
                cv2.imwrite(
                    os.path.join(depth_dir, "00000001.png"),
                    np.full((20, 30), index + 1, dtype=np.uint16),
                )
                with open(
                    os.path.join(sequence_dir, "groundtruth.txt"), "w", encoding="utf-8"
                ) as handle:
                    handle.write("1,2,10,8\n")
                description_dir = os.path.join(color_root, sequence)
                os.makedirs(description_dir)
                with open(
                    os.path.join(description_dir, "color_description_ct.txt"),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    handle.write("A tracked object.")

            train = DepthTrack(
                root=root,
                split="train",
                color_desc_root=color_root,
                sequence_partition="train",
                holdout_ratio=0.2,
                partition_seed=17,
                require_color_descriptions=True,
            )
            val = DepthTrack(
                root=root,
                split="train",
                color_desc_root=color_root,
                sequence_partition="val",
                holdout_ratio=0.2,
                partition_seed=17,
                require_color_descriptions=True,
            )
            val_again = DepthTrack(
                root=root,
                split="train",
                color_desc_root=color_root,
                sequence_partition="val",
                holdout_ratio=0.2,
                partition_seed=17,
                require_color_descriptions=True,
            )

            self.assertEqual(len(train.sequence_list), 8)
            self.assertEqual(len(val.sequence_list), 2)
            self.assertFalse(set(train.sequence_list) & set(val.sequence_list))
            self.assertEqual(
                set(train.sequence_list) | set(val.sequence_list),
                {"object{:02d}_indoor".format(index) for index in range(10)},
            )
            self.assertEqual(val.sequence_list, val_again.sequence_list)

    def test_truncates_extra_ground_truth_to_available_media_frames(self):
        from lib.train.dataset.depthtrack import DepthTrack

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as color_root:
            self._write_sequence(root, color_root)
            with open(
                os.path.join(root, "cup01_indoor", "groundtruth.txt"), "a", encoding="utf-8"
            ) as handle:
                handle.write("3,4,10,8\n")
            dataset = DepthTrack(root=root, split="train", color_desc_root=color_root)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                info = dataset.get_sequence_info(0)

            self.assertEqual(len(info["bbox"]), 2)
            self.assertTrue(any("truncating" in str(item.message) for item in caught))

    def test_accepts_nan_ground_truth_as_an_invalid_frame(self):
        from lib.train.dataset.depthtrack import DepthTrack

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as color_root:
            self._write_sequence(root, color_root)
            with open(
                os.path.join(root, "cup01_indoor", "groundtruth.txt"), "w", encoding="utf-8"
            ) as handle:
                handle.write("1,2,10,8\nnan,nan,nan,nan\n")
            dataset = DepthTrack(root=root, split="train", color_desc_root=color_root)
            info = dataset.get_sequence_info(0)

            self.assertEqual(tuple(info["bbox"].shape), (2, 4))
            self.assertEqual(int(info["valid"][1]), 0)


if __name__ == "__main__":
    unittest.main()
