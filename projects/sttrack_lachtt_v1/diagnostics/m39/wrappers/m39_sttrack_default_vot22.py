#!/usr/bin/env python3
"""VOT-RGBD2022 entry for the frozen official STTrack update policy."""

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import torch


REPOSITORY = Path("/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1")
CONFIG = REPOSITORY / "experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml"
CHECKPOINT = Path("/root/autodl-tmp/sttrack_checkpoints/STTrack_Vot22.pth.tar")

sys.path.insert(0, str(REPOSITORY))

from lib.config.sttrack.config import cfg, update_config_from_file
from lib.test.tracker.sttrack import STTrack
import m39_vot_bridge as vot
from lib.train.dataset.depth_utils import get_rgbd_frame


torch.set_num_threads(1)
update_config_from_file(str(CONFIG))
if not bool(cfg.MODEL.TSG.FIX_QUERY_WINDOW):
    raise ValueError("fixed query window must remain enabled")
if int(cfg.DATA.TEMPLATE.NUMBER) != 2:
    raise ValueError("official two-template contract drifted")
if int(cfg.TEST.UPDATE_INTERVALS) != 50 or float(cfg.TEST.UPDATE_THRESHOLD) != 0.75:
    raise ValueError("official update contract drifted")

params = SimpleNamespace(
    cfg=cfg,
    checkpoint=str(CHECKPOINT),
    template_factor=float(cfg.TEST.TEMPLATE_FACTOR),
    template_size=int(cfg.TEST.TEMPLATE_SIZE),
    search_factor=float(cfg.TEST.SEARCH_FACTOR),
    search_size=int(cfg.TEST.SEARCH_SIZE),
    save_all_boxes=False,
    debug=0,
)
tracker = STTrack(params)
handle = vot.VOT("rectangle", channels="rgbd")
selection = handle.region()
image_files = handle.frame()
if not image_files:
    raise RuntimeError("VOT initialization frame is missing")
if isinstance(image_files, list) and len(image_files) == 2:
    image = get_rgbd_frame(
        image_files[0], image_files[1], dtype="rgbcolormap", depth_clip=True)
else:
    image = cv2.cvtColor(cv2.imread(image_files), cv2.COLOR_BGR2RGB)
tracker.initialize(image, {"init_bbox": list(selection)})

while True:
    image_files = handle.frame()
    if not image_files:
        break
    if isinstance(image_files, list) and len(image_files) == 2:
        image = get_rgbd_frame(
            image_files[0], image_files[1], dtype="rgbcolormap", depth_clip=True)
    else:
        image = cv2.cvtColor(cv2.imread(image_files), cv2.COLOR_BGR2RGB)
    output = tracker.track(image)
    handle.report(vot.Rectangle(*output["target_bbox"]), output["best_score"])
