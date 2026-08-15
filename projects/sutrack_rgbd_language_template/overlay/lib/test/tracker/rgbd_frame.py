"""Lightweight RGB-D frame reader for inference-only entry points."""

import cv2
import numpy as np


def get_rgbd_frame(color_path, depth_path, depth_clip=True):
    """Return the six-channel representation used by official SUTrack RGB-D."""
    rgb_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None or depth is None:
        raise ValueError('Unable to read aligned RGB-D frame')
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2 or rgb_bgr.shape[:2] != depth.shape[:2]:
        raise ValueError('RGB and depth frame geometry must match')

    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    if depth_clip:
        max_depth = min(float(np.median(depth)) * 3.0, 10000.0)
        depth = depth.copy()
        depth[depth > max_depth] = max_depth
    normalized = cv2.normalize(
        depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    normalized = np.asarray(normalized, dtype=np.uint8)
    colormap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return cv2.merge((rgb, colormap))


__all__ = ['get_rgbd_frame']
