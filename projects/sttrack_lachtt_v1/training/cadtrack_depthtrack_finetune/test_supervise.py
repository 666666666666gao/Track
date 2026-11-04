import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SUPERVISOR = Path(__file__).with_name("supervise.py")


class SupervisorTests(unittest.TestCase):
    def test_child_exit_is_recorded_outside_empty_run_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "run"
            control_root = root / "control"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SUPERVISOR),
                    "--run-root",
                    str(run_root),
                    "--control-root",
                    str(control_root),
                    "--",
                    sys.executable,
                    "-c",
                    "print('supervised')",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((run_root / "train.log").exists() is False)
            self.assertEqual((control_root / "train.exit").read_text(), "0\n")
            terminal = json.loads(
                (control_root / "terminal.json").read_text(encoding="utf-8")
            )
            self.assertEqual(terminal["exit_code"], 0)
            self.assertEqual(
                (control_root / "train.log").read_text(encoding="utf-8").strip(),
                "supervised",
            )
            self.assertEqual(list(run_root.iterdir()), [])

    def test_nonempty_run_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "run"
            run_root.mkdir()
            (run_root / "preserved").write_text("keep", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SUPERVISOR),
                    "--run-root",
                    str(run_root),
                    "--control-root",
                    str(root / "control"),
                    "--",
                    sys.executable,
                    "-c",
                    "pass",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual((run_root / "preserved").read_text(), "keep")
            self.assertFalse((root / "control").exists())


if __name__ == "__main__":
    unittest.main()
