import types
import unittest

import torch


class RecordingNetwork:
    def __init__(self):
        self.language_description = None

    def __call__(self, **kwargs):
        self.language_description = kwargs["language_description"]
        return {}


class TrainingLanguageTextModeTest(unittest.TestCase):
    @staticmethod
    def _data():
        modality = {
            "template_images": torch.zeros(1, 1, 3, 8, 8),
            "search_images": torch.zeros(2, 1, 3, 8, 8),
            "template_anno": torch.zeros(1, 1, 4),
        }
        return {
            "visible": modality,
            "infrared": dict(modality),
            "initial_language_description": ["The initial red cup."],
            "language_description": ["The current red cup beside a book."],
        }

    def test_actor_uses_configured_language_text_mode(self):
        from lib.train.actors.mplt_track import MPLTTrackActor

        for mode, expected in (
            ("current", ["The current red cup beside a book."]),
        ):
            with self.subTest(mode=mode):
                network = RecordingNetwork()
                cfg = types.SimpleNamespace(
                    MODEL=types.SimpleNamespace(
                        BACKBONE=types.SimpleNamespace(CE_LOC=[]),
                    ),
                    TRAIN=types.SimpleNamespace(LANGUAGE_TEXT_MODE=mode),
                )
                actor = MPLTTrackActor(
                    net=network,
                    objective={},
                    loss_weight={},
                    settings=types.SimpleNamespace(batchsize=1),
                    cfg=cfg,
                )

                actor.forward_pass(self._data())

                self.assertEqual(network.language_description, expected)

    def test_actor_rejects_unknown_language_text_mode(self):
        from lib.train.actors.mplt_track import MPLTTrackActor

        cfg = types.SimpleNamespace(
            MODEL=types.SimpleNamespace(BACKBONE=types.SimpleNamespace(CE_LOC=[])),
            TRAIN=types.SimpleNamespace(LANGUAGE_TEXT_MODE="sometimes"),
        )
        actor = MPLTTrackActor(
            net=RecordingNetwork(),
            objective={},
            loss_weight={},
            settings=types.SimpleNamespace(batchsize=1),
            cfg=cfg,
        )

        with self.assertRaisesRegex(ValueError, "LANGUAGE_TEXT_MODE"):
            actor.forward_pass(self._data())


if __name__ == "__main__":
    unittest.main()
