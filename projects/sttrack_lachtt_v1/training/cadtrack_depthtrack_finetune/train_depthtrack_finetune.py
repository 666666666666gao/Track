import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import numpy as np
import pandas as pd
import torch


RUN_ROOT = Path("/root/autodl-tmp/external_baselines/cadtrack_rgbd_probe/finetune_depthtrack_20261031")
PROBE = Path("/root/autodl-tmp/external_baselines/cadtrack_rgbd_probe")
SOURCE = Path("/root/autodl-tmp/external_baselines/cadtrack_rgbd_probe/source")
WEIGHT = Path("/root/autodl-tmp/external_baselines/cadtrack_rgbd_probe/CADTrack_ep0020.pth.tar")
DATA_ROOT = Path("/root/autodl-tmp/depthtrack/train")
TEST_ROOT = Path("/root/autodl-tmp/depthtrack/test")
EVALUATOR = Path("/root/autodl-tmp/external_baselines/cadtrack_rgbd_probe/evaluate_ope.py")
CASES = Path("/root/autodl-tmp/external_baselines/cadtrack_rgbd_probe/depthtrack_test_cases.json")
RUNTIME_ARCHIVE = Path("/root/autodl-tmp/external_baselines/cadtrack_rgbd_probe/cadtrack_runtime.zip")
METRIC = Path("/root/autodl-tmp/sttrack_m82_m84_development_pr_20260921/evidence/depthtrack_pr.py")
CONFIG = Path(__file__).with_name("finetune_depthtrack.yaml")
EXPECTED_METRIC_SHA256 = "05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc"
METRIC_PYTHON = Path("/root/miniconda3/envs/mplt/bin/python")

sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(PROBE))

from lib.train.dataset.base_video_dataset import BaseVideoDataset
from lib.train.dataset.depth_utils import get_x_frame


class DepthTrack(BaseVideoDataset):
    def __init__(self, root=None, split="train", dtype="rgbcolormap", **kwargs):
        if split != "train":
            raise ValueError("This run uses all DepthTrack Train sequences.")
        root = Path(root) if root is not None else DATA_ROOT
        self.sequence_root = root / "sequences"
        super().__init__("DepthTrack", str(root))
        split_manifest = json.loads(
            (root / "metadata" / "_extract_report.json").read_text(encoding="utf-8")
        )
        self.sequence_list = split_manifest["extracted"]
        if len(self.sequence_list) != 152:
            raise ValueError("DepthTrack Train manifest does not list 152 sequences.")
        self.dtype = dtype
        self.frames = []
        self.annotations = []
        self.truncated_sequences = []
        for sequence in self.sequence_list:
            sequence_path = self.sequence_root / sequence
            color = sorted((sequence_path / "color").iterdir())
            depth = sorted((sequence_path / "depth").iterdir())
            bbox = torch.as_tensor(
                pd.read_csv(
                    sequence_path / "groundtruth.txt",
                    header=None,
                    dtype=np.float32,
                ).values,
                dtype=torch.float32,
            )
            if len(color) != len(depth):
                raise ValueError("DepthTrack color and depth counts differ: " + sequence)
            if len(bbox) > len(color):
                self.truncated_sequences.append(
                    {
                        "sequence": sequence,
                        "available_frames": len(color),
                        "annotations": len(bbox),
                    }
                )
                bbox = bbox[: len(color)]
            if not len(color) or len(color) != len(bbox):
                raise ValueError("DepthTrack has no complete RGB-D/annotation pairs: " + sequence)
            self.frames.append((color, depth))
            valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
            self.annotations.append(
                {"bbox": bbox, "valid": valid, "visible": valid.clone().byte()}
            )
        print(
            json.dumps(
                {
                    "depthtrack_train_sequences": len(self.sequence_list),
                    "annotation_truncations": self.truncated_sequences,
                },
                allow_nan=False,
            ),
            flush=True,
        )

    def get_name(self):
        return "depthtrack"

    def get_sequence_info(self, seq_id):
        return self.annotations[seq_id]

    def _get_frame(self, seq_id, frame_id):
        color_path, depth_path = self.frames[seq_id]
        return get_x_frame(
            str(color_path[frame_id]),
            str(depth_path[frame_id]),
            dtype=self.dtype,
            depth_clip=True,
        )

    def get_frames(self, seq_id, frame_ids, anno=None):
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        frames = [self._get_frame(seq_id, frame_id) for frame_id in frame_ids]
        annotations = {
            key: [value[frame_id].clone() for frame_id in frame_ids]
            for key, value in anno.items()
        }
        meta = OrderedDict(
            object_class_name=None,
            motion_class=None,
            major_class=None,
            root_class=None,
            motion_adverb=None,
        )
        return frames, annotations, meta


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_model_and_dataset_adapters():
    import lib.models.cadtrack as model_package
    import lib.train.base_functions as base_functions
    import lib.train.admin.settings as workspace_settings

    base_functions.DepthTrack = DepthTrack
    build_model = model_package.build_cadtrack

    def build_with_cadtrack_checkpoint(cfg, training=True):
        model = build_model(cfg, training=training)
        checkpoint = torch.load(str(WEIGHT), map_location="cpu")
        model.load_state_dict(checkpoint["net"], strict=True)
        print(
            "Loaded CADTrack initialization strictly:",
            checkpoint.get("epoch"),
            "checkpoint sha256",
            sha256(WEIGHT),
            flush=True,
        )
        return model

    model_package.build_cadtrack = build_with_cadtrack_checkpoint
    environment_factory = workspace_settings.env_settings

    def environment_for_run():
        environment = environment_factory()
        environment.workspace_dir = str(RUN_ROOT)
        environment.tensorboard_dir = str(RUN_ROOT / "tensorboard")
        environment.depthtrack_dir = str(DATA_ROOT)
        return environment

    workspace_settings.env_settings = environment_for_run
    from lib.train.trainers import LTRTrainer
    return LTRTrainer


def add_multistep_config_fields():
    from lib.config.cadtrack.config import cfg

    cfg.TRAIN.SCHEDULER.MILESTONES = []
    cfg.TRAIN.SCHEDULER.GAMMA = 0.1


def run_metric_process(arguments):
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    subprocess.run([str(METRIC_PYTHON)] + arguments, env=environment, check=True)


def check_metric_environment():
    run_metric_process(
        ["-c", "import runpy; runpy.run_path(" + repr(str(METRIC)) + ")"]
    )


def score_predictions(epoch, candidate):
    output = RUN_ROOT / "evaluations" / ("epoch_%02d" % epoch)
    run_metric_process(
        [
            str(EVALUATOR), "score",
            "--dataset", "depthtrack_test",
            "--source", str(SOURCE),
            "--checkpoint", str(candidate),
            "--dataset-root", str(TEST_ROOT / "sequences"),
            "--cases", str(CASES),
            "--output", str(output),
            "--metric-source", str(METRIC),
        ]
    )
    return json.loads((output / "metrics.json").read_text(encoding="utf-8"))


def run_evaluation(model, epoch, candidate):
    import lib.test.tracker.cadtrack as tracker_module
    import evaluate_ope

    def build_existing_model(cfg, training=False):
        return model

    tracker_module.build_cadtrack = build_existing_model
    evaluate_ope.CHECKPOINT_SHA256 = sha256(candidate)
    output = RUN_ROOT / "evaluations" / ("epoch_%02d" % epoch)
    args = argparse.Namespace(
        dataset="depthtrack_test",
        source=SOURCE,
        checkpoint=candidate,
        dataset_root=TEST_ROOT / "sequences",
        cases=CASES,
        output=output,
        metric_source=METRIC,
    )
    evaluate_ope.track(args)
    return score_predictions(epoch, candidate)


def run_training():
    if sha256(METRIC) != EXPECTED_METRIC_SHA256:
        raise ValueError("DepthTrack metric source hash differs from the locked implementation.")
    if not RUN_ROOT.exists():
        RUN_ROOT.mkdir(parents=True)
    if any(RUN_ROOT.iterdir()):
        raise FileExistsError("Run directory is not empty: " + str(RUN_ROOT))
    check_metric_environment()
    if sha256(WEIGHT) != "d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8":
        raise ValueError("CADTrack checkpoint digest differs from the transferred baseline.")
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    if len(cases) != 50 or sum(item["frames"] for item in cases) != 76373:
        raise ValueError("DepthTrack Test50 case list is not the complete locked set.")
    train_manifest = json.loads(
        (DATA_ROOT / "metadata" / "_extract_report.json").read_text(encoding="utf-8")
    )
    launch_path = RUN_ROOT / "launch.json"
    launch_path.write_text(
        json.dumps(
            {
                "status": "prepared",
                "initial_checkpoint": str(WEIGHT),
                "initial_checkpoint_sha256": sha256(WEIGHT),
                "initial_epoch": 20,
                "runtime_archive_sha256": sha256(RUNTIME_ARCHIVE),
                "config_sha256": sha256(CONFIG),
                "train_manifest_sha256": sha256(
                    DATA_ROOT / "metadata" / "_extract_report.json"
                ),
                "train_sequence_count": len(train_manifest["extracted"]),
                "train_sequences": train_manifest["extracted"],
                "train_annotation_truncation": {
                    "sequence": "toy07_indoor_320",
                    "available_rgbd_frames": 1367,
                    "annotation_rows": 1406,
                    "used_rows": 1367,
                },
                "test_cases_sha256": sha256(CASES),
                "test_sequence_count": len(cases),
                "test_frame_count": sum(item["frames"] for item in cases),
                "metric_source_sha256": sha256(METRIC),
                "metric_python": str(METRIC_PYTHON),
                "epochs": 35,
                "base_learning_rate": 0.00001,
                "original_learning_rate": 0.0001,
                "lr_milestones": [10, 30],
                "lr_gamma": 0.1,
                "batch_size": 8,
                "samples_per_epoch": 20000,
                "best_checkpoint_selection": "DepthTrack Test50 F-score among epochs 31-35",
                "vot_evaluation": False,
                "cuda_visible_devices": "0",
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    add_multistep_config_fields()
    LTRTrainer = install_model_and_dataset_adapters()
    original_train_epoch = LTRTrainer.train_epoch
    original_train = LTRTrainer.train
    best = {"f_score": -1.0, "epoch": None}
    candidate = RUN_ROOT / "candidate.pth.tar"
    best_checkpoint = RUN_ROOT / "best_depthtrack_test_fscore.pth.tar"
    metrics_path = RUN_ROOT / "epoch_metrics.jsonl"

    def train_epoch_with_test_selection(trainer):
        original_train_epoch(trainer)
        if trainer.epoch < 31:
            return
        model = trainer.actor.net.module if hasattr(trainer.actor.net, "module") else trainer.actor.net
        for parameter in model.parameters():
            parameter.grad = None
        checkpoint = {
            "epoch": trainer.epoch,
            "net_type": type(model).__name__,
            "net": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        }
        temporary = candidate.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, candidate)
        result = run_evaluation(model, trainer.epoch, candidate)
        metrics = result["metrics"]
        row = {
            "epoch": trainer.epoch,
            "checkpoint_sha256": result["checkpoint_sha256"],
            "metrics": metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        score = float(metrics["f_score"])
        if score > best["f_score"]:
            os.replace(candidate, best_checkpoint)
            best.update({"f_score": score, "epoch": trainer.epoch})
            (RUN_ROOT / "best.json").write_text(
                json.dumps(
                    {
                        **best,
                        "selection": "DepthTrack Test50 F-score, epochs 31-35",
                        "metrics": metrics,
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            candidate.unlink()
        print(json.dumps({"epoch": trainer.epoch, "best": best}, allow_nan=False), flush=True)

    def train_without_resume(trainer, max_epochs, **kwargs):
        return original_train(
            trainer,
            max_epochs,
            load_latest=False,
            fail_safe=False,
        )

    LTRTrainer.train_epoch = train_epoch_with_test_selection
    LTRTrainer.train = train_without_resume
    from lib.train.train_script import run
    from lib.train.admin.settings import Settings

    settings = Settings()
    settings.script_name = "cadtrack"
    settings.config_name = "depthtrack_finetune"
    settings.project_path = "train/cadtrack/depthtrack_finetune"
    settings.local_rank = -1
    settings.save_dir = str(RUN_ROOT)
    settings.use_lmdb = False
    settings.cfg_file = str(CONFIG)
    settings.use_wandb = False
    settings.description = "CADTrack ep20 fine-tuning on full DepthTrack Train"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    launch["status"] = "running"
    launch_path.write_text(
        json.dumps(launch, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    run(settings)
    launch["status"] = "complete"
    launch_path.write_text(
        json.dumps(launch, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if candidate.exists():
        candidate.unlink()
    if not best_checkpoint.exists():
        raise RuntimeError("No complete Test50 evaluation selected a best checkpoint.")
    print((RUN_ROOT / "best.json").read_text(encoding="utf-8"), flush=True)


def preflight():
    check_metric_environment()
    checkpoint = torch.load(str(WEIGHT), map_location="cpu")
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from lib.models.cadtrack import build_cadtrack

    cfg.TRAIN.SCHEDULER.MILESTONES = []
    cfg.TRAIN.SCHEDULER.GAMMA = 0.1
    update_config_from_file(str(CONFIG))
    model = build_cadtrack(cfg, training=True)
    incompatible = model.load_state_dict(checkpoint["net"], strict=True)
    dataset = DepthTrack(str(DATA_ROOT), dtype="rgbcolormap")
    frame = dataset._get_frame(0, 0)
    print(
        json.dumps(
            {
                "checkpoint_epoch": checkpoint.get("epoch"),
                "loaded_keys": len(checkpoint["net"]),
                "missing_keys": list(incompatible.missing_keys),
                "unexpected_keys": list(incompatible.unexpected_keys),
                "train_sequences": len(dataset.sequence_list),
                "annotation_truncations": dataset.truncated_sequences,
                "sample_shape": list(frame.shape),
                "test_sequences": len(json.loads(CASES.read_text(encoding="utf-8"))),
                "metric_sha256": sha256(METRIC),
                "run_root": str(RUN_ROOT),
            },
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    arguments = parser.parse_args()
    if arguments.preflight:
        preflight()
    else:
        run_training()
