import os
import tempfile
import unittest


class ColorDescriptionStoreTest(unittest.TestCase):
    def _write_lines(self, root, sequence, lines):
        sequence_dir = os.path.join(root, sequence)
        os.makedirs(sequence_dir)
        path = os.path.join(sequence_dir, "color_description_ct.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))

    def test_returns_text_for_requested_frame_ids(self):
        from lib.utils.color_descriptions import ColorDescriptionStore

        with tempfile.TemporaryDirectory() as root:
            self._write_lines(root, "cup01_indoor", ["A red cup.", "The cup moves left."])
            store = ColorDescriptionStore(root)

            self.assertEqual(
                store.descriptions_for("cup01_indoor", [1, 0]),
                ["The cup moves left.", "A red cup."],
            )
            self.assertEqual(store.frame_count("cup01_indoor"), 2)

    def test_rejects_missing_or_misaligned_descriptions(self):
        from lib.utils.color_descriptions import ColorDescriptionStore

        with tempfile.TemporaryDirectory() as root:
            self._write_lines(root, "ball01_wild", ["A basketball."])
            store = ColorDescriptionStore(root)

            with self.assertRaises(IndexError):
                store.descriptions_for("ball01_wild", [1])

            with self.assertRaises(ValueError):
                store.assert_matches_frame_count("ball01_wild", 2)

    def test_default_roots_keep_all_splits_under_one_annotation_directory(self):
        from lib.utils.color_descriptions import default_color_description_root

        with tempfile.TemporaryDirectory() as project_root:
            expected_root = os.path.join(project_root, "color_desc")
            expected = {
                "depthtrack_train": os.path.join(expected_root, "depthtrack_train"),
                "depthtrack_test": os.path.join(expected_root, "depthtrack_test"),
                "cdtb": os.path.join(expected_root, "cdtb"),
                "votrgbd2022": os.path.join(expected_root, "votrgbd2022"),
            }

            actual = {
                dataset: default_color_description_root(project_root, dataset)
                for dataset in expected
            }

            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
