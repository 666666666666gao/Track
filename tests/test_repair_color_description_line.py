import importlib.util
import os
import tempfile
import unittest
from unittest import mock


def load_tool_module():
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "tools",
        "repair_color_description_line.py",
    )
    spec = importlib.util.spec_from_file_location("repair_color_description_line", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RepairColorDescriptionLineTest(unittest.TestCase):
    def test_default_description_root_uses_the_canonical_train_directory(self):
        tool = load_tool_module()

        self.assertEqual(
            os.path.normpath(tool.DEFAULT_DESCRIPTION_ROOT),
            os.path.normpath(os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "color_desc",
                "depthtrack_train",
            )),
        )

    def test_replaces_one_line_without_changing_alignment(self):
        tool = load_tool_module()
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "color_description_ct.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("First frame.\n\nThird frame.")

            tool.replace_description_line(path, 1, "Second frame.", expected_frame_count=3)

            with open(path, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "First frame.\nSecond frame.\nThird frame.")

    def test_failed_atomic_commit_preserves_the_original_file(self):
        tool = load_tool_module()
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "color_description_ct.txt")
            original = "First frame.\nSecond frame.\nThird frame."
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(original)

            with mock.patch.object(tool.os, "replace", side_effect=OSError("commit failed")):
                with self.assertRaises(OSError):
                    tool.replace_description_line(path, 1, "Replacement.", expected_frame_count=3)

            with open(path, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), original)


if __name__ == "__main__":
    unittest.main()
