"""CPU checks of actual child completion/error propagation, without training."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class SupervisorTest(unittest.TestCase):
    def test_receipts_and_exit_codes(self):
        for code in [0, 7]:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                command = [sys.executable, '-c', "import sys; print('child witness'); sys.exit(%d)" % code]
                (root / 'launch.json').write_text(json.dumps(dict(command=command,
                    cwd=str(root), environment={'PYTHONUNBUFFERED': '1'})))
                process = subprocess.run([sys.executable, str(Path(__file__).with_name('supervise.py')), str(root)])
                self.assertEqual(process.returncode, code)
                self.assertEqual((root / 'train.exit').read_text().strip(), str(code))
                self.assertEqual((root / 'train.log').read_text().strip(), 'child witness')
                child = json.loads((root / 'child.json').read_text())
                terminal = json.loads((root / 'terminal.json').read_text())
                self.assertEqual(child['command'], command)
                self.assertEqual(terminal['child_pid'], child['child_pid'])
                self.assertEqual(terminal['exit_code'], code)


if __name__ == '__main__':
    unittest.main()
