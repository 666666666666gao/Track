import importlib.util
import os
import tempfile
import unittest
from unittest.mock import patch


class TrainEnvironmentTest(unittest.TestCase):
    def test_generated_local_file_preserves_windows_paths(self):
        from lib.train.admin import environment

        workspace_dir = r"C:\Users\gb\RGB-D-L"
        data_dir = r"D:\datasets\RGB-D"
        with tempfile.TemporaryDirectory() as root:
            fake_environment_path = os.path.join(root, "environment.py")
            with patch.object(environment, "__file__", fake_environment_path):
                environment.create_default_local_file_ITP_train(workspace_dir, data_dir)

            local_path = os.path.join(root, "local.py")
            spec = importlib.util.spec_from_file_location("generated_train_local", local_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            settings = module.EnvironmentSettings()

            self.assertEqual(settings.workspace_dir, workspace_dir)
            self.assertEqual(
                settings.depthtrack_train_color_desc_root,
                os.path.join(workspace_dir, "color_desc/depthtrack_train"),
            )


if __name__ == "__main__":
    unittest.main()
