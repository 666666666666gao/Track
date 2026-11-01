import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from trax import Image, ImageChannel, Rectangle, Region, Server, TraxStatus

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
sys.path.insert(0, str(SOURCE))

from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.test.tracker.cadtrack import CADTrack


def image_path(images, channel):
    image = images.get(channel)
    if image is None or image.type() != Image.PATH:
        raise RuntimeError("VOT frames must provide RGB and depth file paths")
    return image.path()


def read_rgbd(images):
    color_path = image_path(images, ImageChannel.COLOR)
    depth_path = image_path(images, ImageChannel.DEPTH)
    rgb = cv2.imread(color_path, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        raise FileNotFoundError("Unable to read VOT RGB-D frame")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != depth.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))
    depth = depth.astype(np.float32)
    median_depth = np.median(depth)
    if median_depth > 0:
        limit = min(median_depth * 3.0, 10000.0)
        depth[depth > limit] = limit
    normalized = cv2.normalize(
        depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)
    depth_rgb = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return np.concatenate((rgb, depth_rgb), axis=2)


def build_tracker():
    update_config_from_file(str(SOURCE / "experiments/cadtrack/cadtrack.yaml"))
    params = type("TrackerParams", (), {})()
    params.cfg = cfg
    params.checkpoint = os.environ["CADTRACK_CHECKPOINT"]
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.debug = 0
    params.save_all_boxes = False
    return CADTrack(params)


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    torch.set_grad_enabled(False)
    tracker = None
    with Server(
        [Region.RECTANGLE],
        [Image.PATH],
        image_channels=[ImageChannel.COLOR, ImageChannel.DEPTH],
        tracker_name="CADTrack-RGBD-JET",
        tracker_description="CADTrack RGB-D cross-modal baseline",
        tracker_family="CADTrack",
        metadata={"vot": "python"},
    ) as server:
        while True:
            request = server.wait()
            if request.type in (TraxStatus.QUIT, TraxStatus.ERROR):
                break
            if request.type == TraxStatus.INITIALIZE:
                if tracker is None:
                    tracker = build_tracker()
                bbox = [float(value) for value in request.objects[0][0].bounds()]
                tracker.initialize(read_rgbd(request.image), {"init_bbox": bbox})
                server.status([(Rectangle.create(*bbox), {"confidence": "1.0"})])
            elif request.type == TraxStatus.FRAME:
                if tracker is None:
                    raise RuntimeError("VOT requested tracking before initialization")
                result = tracker.track(read_rgbd(request.image))
                bbox = [float(value) for value in result["target_bbox"]]
                confidence = float(result["best_score"])
                server.status([(
                    Rectangle.create(*bbox),
                    {"confidence": "{:.6f}".format(confidence)},
                )])


if __name__ == "__main__":
    main()
