import os
import tempfile
import unittest

import torch


class TinyLanguageTracker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = torch.nn.Linear(2, 2)
        self.language_fusion = torch.nn.Linear(2, 2)


class TrackerCheckpointCompatibilityTest(unittest.TestCase):
    def test_old_visual_checkpoint_has_explicit_language_compatibility_error(self):
        from lib.utils.checkpoint import load_tracker_checkpoint

        network = TinyLanguageTracker()
        state = network.state_dict()
        visual_only = {key: value for key, value in state.items() if not key.startswith("language_")}

        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "old_visual_only.pth.tar")
            torch.save({"net": visual_only}, path)

            with self.assertRaisesRegex(RuntimeError, "language-trained checkpoint"):
                load_tracker_checkpoint(network, path, language_enabled=True)

    def test_compatible_checkpoint_is_loaded_strictly(self):
        from lib.utils.checkpoint import load_tracker_checkpoint

        source = TinyLanguageTracker()
        target = TinyLanguageTracker()
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "language.pth.tar")
            torch.save({"net": source.state_dict()}, path)

            load_tracker_checkpoint(target, path, language_enabled=True)

        for source_value, target_value in zip(source.parameters(), target.parameters()):
            self.assertTrue(torch.equal(source_value, target_value))


if __name__ == "__main__":
    unittest.main()
