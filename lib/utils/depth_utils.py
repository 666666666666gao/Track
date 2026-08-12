import cv2
import numpy as np


TARGET_DEPTH_RADIUS = 500.0
GRABCUT_EXTRA = 50
GRABCUT_RESIZE_THRESHOLD = 300
GRABCUT_RESIZE_FACTOR = 1.5
MIN_TARGET_PIXELS = 16
GRABCUT_ITER = 3
SAFE_MIN_VALID_RATIO = 0.20
SAFE_MIN_DEPTH_CONTRAST = 25.0


def read_rgb_image(color_path):
    rgb = cv2.imread(color_path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(color_path)
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def read_depth_image(depth_path):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(depth_path)
    return depth


def remove_depth_bubbles(depth, min_component_size=200):
    try:
        binary_map = (depth > 0).astype(np.uint8)
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
        sizes = stats[1:, -1]
        mask = np.zeros(depth.shape[:2], dtype=np.uint8)
        for i in range(nb_components - 1):
            if sizes[i] >= min_component_size:
                mask[output == i + 1] = 1
        return depth * mask
    except Exception:
        return depth


def _histogram_target_depth(values):
    valid = values[np.isfinite(values) & (values > 0)]
    if valid.size < MIN_TARGET_PIXELS:
        return None
    hist, bin_edges = np.histogram(valid, bins=20)
    peak_idx = int(np.argmax(hist))
    selected = valid[(valid >= bin_edges[peak_idx]) & (valid <= bin_edges[peak_idx + 1])]
    if selected.size < MIN_TARGET_PIXELS:
        selected = valid
    return float(np.median(selected))


def estimate_target_depth(depth, target_box):
    if target_box is None:
        return None
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    depth = depth.astype(np.float32)
    h, w = depth.shape[:2]
    x, y, bw, bh = [int(round(float(v))) for v in target_box]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + max(bw, 1), w), min(y + max(bh, 1), h)
    if x1 <= x0 or y1 <= y0:
        return None
    target_patch = depth[y0:y1, x0:x1]
    return _histogram_target_depth(target_patch)


def estimate_target_depth_grabcut(depth, target_box):
    if target_box is None:
        return None
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    depth = depth.astype(np.float32)
    h, w = depth.shape[:2]
    x, y, bw, bh = [int(round(float(v))) for v in target_box]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + max(bw, 1), w), min(y + max(bh, 1), h)
    if x1 <= x0 or y1 <= y0:
        return None
    target_patch = depth[y0:y1, x0:x1]
    fallback_depth = _histogram_target_depth(target_patch)
    if fallback_depth is None:
        return None

    try:
        extra_y0 = max(y0 - GRABCUT_EXTRA, 0)
        extra_x0 = max(x0 - GRABCUT_EXTRA, 0)
        extra_y1 = min(y1 + GRABCUT_EXTRA, h)
        extra_x1 = min(x1 + GRABCUT_EXTRA, w)
        local_depth = depth[extra_y0:extra_y1, extra_x0:extra_x1]
        if local_depth.size == 0:
            return fallback_depth

        rect = [
            x0 - extra_x0,
            y0 - extra_y0,
            min(bw, extra_x1 - extra_x0),
            min(bh, extra_y1 - extra_y0),
        ]
        if rect[2] <= 1 or rect[3] <= 1:
            return fallback_depth

        median_depth = fallback_depth + 10.0
        filtered = np.nan_to_num(local_depth.copy(), nan=median_depth * 2.0,
                                 posinf=median_depth * 2.0, neginf=median_depth * 2.0)
        filtered[filtered > median_depth * 2.0] = median_depth * 2.0
        filtered[filtered < 10.0] = median_depth * 2.0

        local_h, local_w = filtered.shape[:2]
        resize_factor = GRABCUT_RESIZE_FACTOR if min(local_w, local_h) > GRABCUT_RESIZE_THRESHOLD else 1.0
        rect_rz = [int(v // resize_factor) for v in rect]
        rz_size = (max(int(local_w // resize_factor), 1), max(int(local_h // resize_factor), 1))
        if rect_rz[2] <= 1 or rect_rz[3] <= 1:
            return fallback_depth

        image = cv2.resize(filtered, rz_size, interpolation=cv2.INTER_AREA)
        min_component = max(int(rect[2] * rect[3] * 0.1), 1)
        image = remove_depth_bubbles(image, min_component_size=min_component)
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = np.asarray(image, dtype=np.uint8)
        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(image, mask, tuple(rect_rz), bgd_model, fgd_model, GRABCUT_ITER, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        fg_mask = remove_depth_bubbles(fg_mask, min_component_size=min_component)
        fg_mask = cv2.resize(fg_mask, (local_w, local_h), interpolation=cv2.INTER_AREA)

        selected_depth = local_depth * fg_mask
        grabcut_depth = _histogram_target_depth(selected_depth.reshape(-1))
        return grabcut_depth if grabcut_depth is not None else fallback_depth
    except Exception:
        return fallback_depth


def layer_depth_around_target(depth, target_box, radius=TARGET_DEPTH_RADIUS, use_grabcut=False):
    estimator = estimate_target_depth_grabcut if use_grabcut else estimate_target_depth
    target_depth = estimator(depth, target_box)
    if target_depth is None:
        return depth
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    layer = depth.astype(np.float32).copy()
    low = max(target_depth - radius, 0.0)
    high = target_depth + radius
    layer[(layer < low) | (layer > high)] = high + 10.0
    return remove_depth_bubbles(layer, min_component_size=200)


def is_safe_to_layer_depth(depth, target_box, min_valid_ratio=SAFE_MIN_VALID_RATIO,
                           min_depth_contrast=SAFE_MIN_DEPTH_CONTRAST):
    if target_box is None:
        return False
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    depth = depth.astype(np.float32)
    h, w = depth.shape[:2]
    x, y, bw, bh = [int(round(float(v))) for v in target_box]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + max(bw, 1), w), min(y + max(bh, 1), h)
    if x1 <= x0 or y1 <= y0:
        return False

    target_patch = depth[y0:y1, x0:x1]
    target_valid = np.isfinite(target_patch) & (target_patch > 0)
    if float(target_valid.mean()) < min_valid_ratio:
        return False

    pad_x = max(int(round(bw * 0.5)), 8)
    pad_y = max(int(round(bh * 0.5)), 8)
    cx0, cy0 = max(x0 - pad_x, 0), max(y0 - pad_y, 0)
    cx1, cy1 = min(x1 + pad_x, w), min(y1 + pad_y, h)
    context = depth[cy0:cy1, cx0:cx1].copy()
    context[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] = 0
    context_valid = np.isfinite(context) & (context > 0)
    if context_valid.sum() < MIN_TARGET_PIXELS:
        return True

    target_depth = float(np.median(target_patch[target_valid]))
    context_depth = float(np.median(context[context_valid]))
    return abs(target_depth - context_depth) >= min_depth_contrast


def depth_to_3ch(depth, dtype="colormap", depth_clip=True, target_box=None, depth_preprocess="rgbcolormap"):
    """DepthTrack/XTrack-style depth conversion.

    `rgbcolormap` is the common DepthTrack/CDTB setting in DeT/XTrack: RGB is
    concatenated with an OpenCV JET colormap generated from the depth map.
    """
    if depth.ndim == 3:
        if depth.dtype == np.uint8 and depth.shape[2] == 3:
            return cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        depth = depth[:, :, 0]

    depth = depth.astype(np.float32)
    if depth_preprocess == "target_layered_safe" and target_box is not None:
        if is_safe_to_layer_depth(depth, target_box):
            depth = layer_depth_around_target(depth, target_box, use_grabcut=False)
    elif depth_preprocess in ("target_layered", "centered_colormap", "target_layered_grabcut") and target_box is not None:
        depth = layer_depth_around_target(
            depth, target_box, use_grabcut=(depth_preprocess == "target_layered_grabcut"))

    valid = np.isfinite(depth)
    if depth_clip and valid.any():
        median_depth = np.median(depth[valid])
        if median_depth > 0:
            max_depth = min(median_depth * 3.0, 10000.0)
            depth = depth.copy()
            depth[depth > max_depth] = max_depth

    depth = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    depth = np.asarray(depth, dtype=np.uint8)
    if dtype in ("3xD", "3x", "rgb3d", "rgb3x"):
        return cv2.merge((depth, depth, depth))
    return cv2.applyColorMap(depth, cv2.COLORMAP_JET)


def get_rgbd_frame(color_path, depth_path, dtype="rgbcolormap", depth_clip=True,
                   target_box=None, depth_preprocess="rgbcolormap"):
    rgb = read_rgb_image(color_path) if color_path else None
    depth = read_depth_image(depth_path) if depth_path else None

    if dtype == "color":
        return rgb
    if dtype in ("colormap", "3xD", "3x"):
        return depth_to_3ch(depth, dtype=dtype, depth_clip=depth_clip,
                            target_box=target_box, depth_preprocess=depth_preprocess)
    if dtype in ("rgbcolormap", "rgb3d", "rgb3x"):
        depth_3ch = depth_to_3ch(depth, dtype=dtype, depth_clip=depth_clip,
                                 target_box=target_box, depth_preprocess=depth_preprocess)
        return np.concatenate((rgb, depth_3ch), axis=2)
    if dtype == "rgbrgb":
        if depth.ndim == 2:
            depth = cv2.merge((depth, depth, depth))
        return np.concatenate((rgb, cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)), axis=2)
    raise ValueError("Unsupported RGB-D dtype: {}".format(dtype))
