import importlib.util
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from PIL import Image


def load_tool_module():
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "tools",
        "annotate_color_descriptions.py",
    )
    spec = importlib.util.spec_from_file_location("annotate_color_descriptions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AnnotationInputTest(unittest.TestCase):
    def _write_sequence(self, root, sequence, boxes):
        sequence_dir = os.path.join(root, sequence)
        color_dir = os.path.join(sequence_dir, "color")
        os.makedirs(color_dir)
        for index in range(len(boxes)):
            Image.new("RGB", (32, 24), color=(10, 20, 30)).save(
                os.path.join(color_dir, "{:08d}.jpg".format(index + 1)))
        with open(os.path.join(sequence_dir, "groundtruth.txt"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(
                "{},{},{},{}".format(*box) for box in boxes))
        return sequence_dir

    def test_collects_naturally_sorted_frames_and_aligned_boxes(self):
        tool = load_tool_module()
        with tempfile.TemporaryDirectory() as root:
            sequence_dir = self._write_sequence(root, "cup01_indoor", [[1, 2, 3, 4], [5, 6, 7, 8]])
            frames, boxes = tool.collect_sequence_inputs(sequence_dir)

            self.assertEqual([os.path.basename(path) for path in frames], ["00000001.jpg", "00000002.jpg"])
            self.assertEqual(boxes, [(1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)])

    def test_rejects_frame_and_ground_truth_count_mismatch(self):
        tool = load_tool_module()
        with tempfile.TemporaryDirectory() as root:
            sequence_dir = self._write_sequence(root, "ball01_wild", [[1, 2, 3, 4]])
            Image.new("RGB", (32, 24), color=(10, 20, 30)).save(
                os.path.join(sequence_dir, "color", "00000002.jpg"))

            with self.assertRaises(ValueError):
                tool.collect_sequence_inputs(sequence_dir)

    def test_rejects_an_invalid_first_frame_ground_truth_box(self):
        tool = load_tool_module()
        with tempfile.TemporaryDirectory() as root:
            sequence_dir = self._write_sequence(
                root, "ball01_wild", [[float("nan"), float("nan"), 0, 0]])

            with self.assertRaisesRegex(ValueError, "invalid.*frame 0"):
                tool.collect_sequence_inputs(sequence_dir)

    def test_test_dataset_dry_run_counts_one_first_frame_per_sequence(self):
        tool = load_tool_module()
        with tempfile.TemporaryDirectory() as root:
            self._write_sequence(root, "ball01_wild", [[1, 2, 3, 4]] * 3)
            self._write_sequence(root, "cup01_indoor", [[1, 2, 3, 4]] * 2)
            output = io.StringIO()

            with redirect_stdout(output):
                result = tool.main([
                    "--dataset", "depthtrack_test",
                    "--data-root", "depthtrack_test={}".format(root),
                    "--stage", "qwen25",
                    "--dry-run",
                ])

            self.assertEqual(result, 0)
            self.assertIn("2 sequences, 2 Qwen2.5 frames", output.getvalue())


class ResumableDescriptionTest(unittest.TestCase):
    def test_resume_discards_an_uncommitted_partial_tail(self):
        tool = load_tool_module()
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "color_description_qwen25.txt")
            metadata = {"dataset": "depthtrack_test", "sequence": "cup01_indoor"}
            output = tool.DescriptionCheckpoint(path, metadata, overwrite=True)
            output.append("First frame description.")

            with open(path, "ab") as handle:
                handle.write(b"interrupted partial sentence")

            resumed = tool.DescriptionCheckpoint(path, metadata)
            self.assertEqual(resumed.lines, ["First frame description."])
            resumed.append("Second frame description.")

            with open(path, "r", encoding="utf-8", newline="") as handle:
                self.assertEqual(
                    handle.read(),
                    "First frame description.\nSecond frame description.\n",
                )

    def test_interrupted_overwrite_recovers_to_the_new_empty_checkpoint(self):
        tool = load_tool_module()
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "color_description_qwen25.txt")
            metadata = {"dataset": "depthtrack_test", "sequence": "cup01_indoor"}
            output = tool.DescriptionCheckpoint(path, metadata, overwrite=True)
            output.append("Old committed description.")
            real_atomic_write = tool._atomic_write_bytes
            call_count = [0]

            def interrupt_second_write(target, content):
                call_count[0] += 1
                if call_count[0] == 2:
                    raise OSError("simulated interruption")
                return real_atomic_write(target, content)

            with mock.patch.object(
                    tool, "_atomic_write_bytes", side_effect=interrupt_second_write):
                with self.assertRaises(OSError):
                    tool.DescriptionCheckpoint(path, metadata, overwrite=True)

            resumed = tool.DescriptionCheckpoint(path, metadata)
            self.assertEqual(resumed.lines, [])
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), b"")

    def test_changed_first_frame_ground_truth_refuses_resume(self):
        tool = load_tool_module()

        class FakeCaptioner(object):
            def __init__(self, *args, **kwargs):
                pass

            def describe(self, image_path, box, sequence_name):
                return "A red cup on a desk."

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as data_root, tempfile.TemporaryDirectory() as output_root:
            sequence_dir = os.path.join(data_root, "cup01_indoor")
            color_dir = os.path.join(sequence_dir, "color")
            os.makedirs(color_dir)
            Image.new("RGB", (32, 24), color=(10, 20, 30)).save(
                os.path.join(color_dir, "00000001.jpg"))
            ground_truth_path = os.path.join(sequence_dir, "groundtruth.txt")
            with open(ground_truth_path, "w", encoding="utf-8") as handle:
                handle.write("1,2,3,4")
            arguments = [
                "--dataset", "depthtrack_test",
                "--data-root", "depthtrack_test={}".format(data_root),
                "--output-root", output_root,
                "--stage", "qwen25",
            ]

            with mock.patch.object(tool, "Qwen25Captioner", FakeCaptioner):
                tool.main(arguments)
                with open(ground_truth_path, "w", encoding="utf-8") as handle:
                    handle.write("5,6,3,4")
                with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                    tool.main(arguments)

    def test_qwen3_retry_receives_the_previous_invalid_attempt(self):
        tool = load_tool_module()

        class FakeCaptioner(object):
            def __init__(self, *args, **kwargs):
                pass

            def describe(self, image_path, box, sequence_name):
                return "A red cup on a desk."

            def close(self):
                pass

        class FakeCorrector(object):
            previous_attempts = []

            def __init__(self, *args, **kwargs):
                pass

            def correct(self, draft, previous_attempt=None, retry_number=0):
                self.previous_attempts.append(previous_attempt)
                if previous_attempt is None:
                    return "bbox"
                return "A red cup rests on a desk."

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as data_root, tempfile.TemporaryDirectory() as output_root:
            sequence_dir = os.path.join(data_root, "cup01_indoor")
            color_dir = os.path.join(sequence_dir, "color")
            os.makedirs(color_dir)
            Image.new("RGB", (32, 24), color=(10, 20, 30)).save(
                os.path.join(color_dir, "00000001.jpg"))
            with open(
                    os.path.join(sequence_dir, "groundtruth.txt"), "w", encoding="utf-8") as handle:
                handle.write("1,2,3,4")
            arguments = [
                "--dataset", "depthtrack_test",
                "--data-root", "depthtrack_test={}".format(data_root),
                "--output-root", output_root,
                "--stage", "all",
            ]

            with mock.patch.object(tool, "Qwen25Captioner", FakeCaptioner), mock.patch.object(
                    tool, "Qwen3Corrector", FakeCorrector):
                tool.main(arguments)

            self.assertEqual(FakeCorrector.previous_attempts, [None, "bbox"])


class RuntimeOptionTest(unittest.TestCase):
    def test_corrected_text_rejects_overlong_or_marked_region_wording(self):
        tool = load_tool_module()
        twenty_one_words = " ".join("word{}".format(index) for index in range(21))

        self.assertFalse(tool._is_valid_corrected_text(twenty_one_words))
        self.assertFalse(tool._is_valid_corrected_text(
            "The pigeon in the marked region has a dark gray body."))
        self.assertTrue(tool._is_valid_corrected_text(
            "The pigeon has a dark gray body and lighter head."))

    def test_retry_feedback_names_word_count_and_forbidden_wording(self):
        tool = load_tool_module()
        overlong = " ".join("word{}".format(index) for index in range(23))
        feedback = tool._correction_retry_feedback(overlong)
        marked_feedback = tool._correction_retry_feedback(
            "The pigeon in the marked region has dark gray feathers.")

        self.assertIn("23 words", feedback)
        self.assertIn("12 to 18 words", feedback)
        self.assertIn("marked region", marked_feedback)
        self.assertIn("remove", marked_feedback.lower())
        aggressive = tool._correction_retry_feedback(overlong, retry_number=2)
        self.assertIn("8 to 12 words", aggressive)
        self.assertIn("color, shape, and material", aggressive)

    def test_four_bit_fixed_device_map_rejects_gpu_memory_cap(self):
        tool = load_tool_module()
        arguments = tool.parse_args(["--load-in-4bit", "--gpu-memory", "4GiB"])

        with self.assertRaisesRegex(ValueError, "--gpu-memory.*--load-in-4bit"):
            tool.validate_runtime_options(arguments)


if __name__ == "__main__":
    unittest.main()
