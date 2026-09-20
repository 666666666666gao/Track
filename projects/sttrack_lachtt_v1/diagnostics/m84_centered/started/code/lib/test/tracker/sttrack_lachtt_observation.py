"""Candidate-aligned observations for protected STTrack recovery branches.

This module is deliberately inference-only.  It never reads ground truth and
never mutates the public tracker.  Candidate boxes are evaluated in their own
RGB, depth, fused-token and CLIP regions rather than in the public box.
"""

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from lib.utils.box_ops import clip_box


_CLIP_MODEL_CACHE = {}


def finite_bbox(values):
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError):
        return None
    if (len(result) != 4 or not all(math.isfinite(value) for value in result)
            or result[2] <= 0.0 or result[3] <= 0.0):
        return None
    return result


def bbox_iou(first, second):
    first, second = finite_bbox(first), finite_bbox(second)
    if first is None or second is None:
        raise ValueError("invalid bbox")
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0.0 else 0.0


def relative_geometry(candidate, reference):
    candidate, reference = finite_bbox(candidate), finite_bbox(reference)
    if candidate is None or reference is None:
        raise ValueError("invalid relative-geometry bbox")
    cx = candidate[0] + 0.5 * candidate[2]
    cy = candidate[1] + 0.5 * candidate[3]
    rx = reference[0] + 0.5 * reference[2]
    ry = reference[1] + 0.5 * reference[3]
    scale = max(math.sqrt(reference[2] * reference[3]), 1.0)
    return [
        bbox_iou(candidate, reference),
        math.hypot(cx - rx, cy - ry) / scale,
        math.log((candidate[2] * candidate[3]) /
                 (reference[2] * reference[3])),
        math.log(candidate[2] / reference[2]),
        math.log(candidate[3] / reference[3]),
    ]


def clone_query_state(values):
    if values is None:
        return None
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError("query state must contain RGB and depth tensors")
    return [value.detach().clone() for value in values]


def stack_query_states(states):
    if not states or any(state is None for state in states):
        raise ValueError("all recursive branches require query state")
    return [torch.cat([state[index] for state in states], dim=0)
            for index in range(2)]


def split_query_state(values):
    if values is None or len(values) != 2 or values[0].shape[0] != values[1].shape[0]:
        raise ValueError("malformed batched query state")
    return [[values[0][index:index + 1].detach().clone(),
             values[1][index:index + 1].detach().clone()]
            for index in range(values[0].shape[0])]


def _search_origin(prior, search_size, resize_factor):
    crop_size = float(search_size) / float(resize_factor)
    left = round(prior[0] + 0.5 * prior[2] - 0.5 * crop_size)
    top = round(prior[1] + 0.5 * prior[3] - 0.5 * crop_size)
    return float(left), float(top), crop_size


def _map_local_box(local_box, prior, search_size, resize_factor,
                   height, width):
    cx, cy, box_width, box_height = [float(value) for value in local_box]
    previous_x = prior[0] + 0.5 * prior[2]
    previous_y = prior[1] + 0.5 * prior[3]
    half_side = 0.5 * float(search_size) / float(resize_factor)
    mapped = [
        cx + previous_x - half_side - 0.5 * box_width,
        cy + previous_y - half_side - 0.5 * box_height,
        box_width,
        box_height,
    ]
    return [float(value) for value in
            clip_box(mapped, height, width, margin=10)]


def _response_entropy(response):
    values = response.detach().float().reshape(-1).clamp_min(0.0)
    total = values.sum()
    if not torch.isfinite(total) or total <= 0.0:
        return 1.0
    probabilities = values / total
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return float((entropy / math.log(max(2, values.numel()))).item())


def decode_nms_candidates(response, size_map, offset_map, priors,
                          resize_factors, image_shape, search_size,
                          peaks_per_prior, nms_kernel=3):
    """Decode top local response peaks into global boxes.

    The return order is prior-major and peak-rank-minor.  Each record carries
    its source batch index so candidate-aligned tokens can be pooled from the
    exact search crop that proposed it.
    """
    if response.ndim != 4 or response.shape[1] != 1:
        raise ValueError("response must be Bx1xHxW")
    if nms_kernel <= 0 or nms_kernel % 2 == 0:
        raise ValueError("NMS kernel must be positive and odd")
    batch, _, height_cells, width_cells = response.shape
    if batch != len(priors) or batch != len(resize_factors):
        raise ValueError("response/prior batch mismatch")
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    records = []
    for source in range(batch):
        suppressed = response[source, 0].detach().clone()
        peak_values, peak_indexes = [], []
        radius = nms_kernel // 2
        for _ in range(max(peaks_per_prior + 1, 2)):
            flattened = suppressed.reshape(-1)
            index = int(flattened.argmax().item())
            score = float(flattened[index].item())
            if not math.isfinite(score):
                break
            row, column = divmod(index, width_cells)
            peak_values.append(score)
            peak_indexes.append(index)
            suppressed[max(0, row - radius):min(height_cells, row + radius + 1),
                       max(0, column - radius):min(width_cells, column + radius + 1)] = -float("inf")
        if len(peak_values) < peaks_per_prior:
            raise RuntimeError("response has too few greedily suppressed peaks")
        entropy = _response_entropy(response[source])
        for rank in range(peaks_per_prior):
            score = peak_values[rank]
            index = peak_indexes[rank]
            row, column = divmod(index, width_cells)
            box_width = float(size_map[source, 0, row, column].item())
            box_height = float(size_map[source, 1, row, column].item())
            offset_x = float(offset_map[source, 0, row, column].item())
            offset_y = float(offset_map[source, 1, row, column].item())
            local_box = [
                (column + offset_x) / width_cells * search_size /
                resize_factors[source],
                (row + offset_y) / height_cells * search_size /
                resize_factors[source],
                box_width * search_size / resize_factors[source],
                box_height * search_size / resize_factors[source],
            ]
            runner_up = (peak_values[rank + 1]
                         if rank + 1 < len(peak_values) else 0.0)
            records.append({
                "source_index": source,
                "peak_rank": rank,
                "grid_row": row,
                "grid_column": column,
                "bbox": _map_local_box(
                    local_box, priors[source], search_size,
                    resize_factors[source], image_height, image_width),
                "score": score,
                "margin": score - runner_up,
                "entropy": entropy,
            })
    return records


def pool_candidate_tokens(tokens, candidates, priors, resize_factors,
                          search_size, samples=4):
    if tokens.ndim != 3:
        raise ValueError("tokens must be BxNxC")
    side = int(round(math.sqrt(tokens.shape[1])))
    if side * side != tokens.shape[1]:
        raise ValueError("search token count is not square")
    source_indexes = [int(candidate["source_index"])
                      for candidate in candidates]
    feature_maps = tokens[source_indexes].transpose(1, 2).reshape(
        len(candidates), tokens.shape[2], side, side)
    grids = []
    for candidate in candidates:
        source = int(candidate["source_index"])
        bbox = candidate["bbox"]
        origin_x, origin_y, _ = _search_origin(
            priors[source], search_size, resize_factors[source])
        xs = torch.linspace(
            (bbox[0] - origin_x) * resize_factors[source] / search_size,
            (bbox[0] + bbox[2] - origin_x) * resize_factors[source] /
            search_size, samples, device=tokens.device)
        ys = torch.linspace(
            (bbox[1] - origin_y) * resize_factors[source] / search_size,
            (bbox[1] + bbox[3] - origin_y) * resize_factors[source] /
            search_size, samples, device=tokens.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grids.append(torch.stack([xx * 2.0 - 1.0, yy * 2.0 - 1.0], dim=-1))
    grid = torch.stack(grids, dim=0)
    pooled = F.grid_sample(feature_maps, grid, mode="bilinear",
                           padding_mode="zeros", align_corners=True)
    return pooled.mean(dim=(-1, -2))


def raw_depth_rois(raw_depth, bboxes, output_size=16):
    if raw_depth is None or raw_depth.ndim != 2:
        raise ValueError("raw depth must be a single-channel image")
    height, width = raw_depth.shape
    results = []
    for bbox in bboxes:
        x, y, box_width, box_height = finite_bbox(bbox)
        left = max(0, int(math.floor(x)))
        top = max(0, int(math.floor(y)))
        right = min(width, int(math.ceil(x + box_width)))
        bottom = min(height, int(math.ceil(y + box_height)))
        if right <= left or bottom <= top:
            normalized = np.zeros((output_size, output_size), np.float32)
            valid = np.zeros_like(normalized)
        else:
            crop = raw_depth[top:bottom, left:right].astype(np.float32)
            valid_crop = np.isfinite(crop) & (crop > 0.0)
            transformed = np.zeros_like(crop, dtype=np.float32)
            if valid_crop.any():
                log_depth = np.log1p(crop[valid_crop])
                median = float(np.median(log_depth))
                mad = float(np.median(np.abs(log_depth - median)))
                scale = max(1.4826 * mad, 1e-3)
                transformed[valid_crop] = np.clip(
                    (log_depth - median) / scale, -5.0, 5.0)
            normalized = cv2.resize(
                transformed, (output_size, output_size),
                interpolation=cv2.INTER_LINEAR)
            valid = cv2.resize(
                valid_crop.astype(np.float32),
                (output_size, output_size),
                interpolation=cv2.INTER_NEAREST)
        results.append(np.stack([normalized, valid], axis=0))
    return torch.from_numpy(np.stack(results, axis=0))


def _crop_rgb(image, bbox, expansion=1.20):
    bbox = finite_bbox(bbox)
    if bbox is None or image is None or image.ndim != 3:
        raise ValueError("invalid CLIP crop")
    x, y, width, height = bbox
    cx, cy = x + 0.5 * width, y + 0.5 * height
    width, height = width * expansion, height * expansion
    left = max(0, int(math.floor(cx - 0.5 * width)))
    top = max(0, int(math.floor(cy - 0.5 * height)))
    right = min(image.shape[1], int(math.ceil(cx + 0.5 * width)))
    bottom = min(image.shape[0], int(math.ceil(cy + 0.5 * height)))
    if right <= left or bottom <= top:
        raise ValueError("empty CLIP crop")
    rgb = np.ascontiguousarray(image[top:bottom, left:right, :3])
    return Image.fromarray(rgb.astype(np.uint8, copy=False), mode="RGB")


class ClipCandidateEncoder:
    """Actual candidate crop encoding in the same CLIP image/text space."""

    def __init__(self, model_path, initial_image, initial_bbox, text):
        model_path = Path(model_path).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        text = " ".join(str(text or "").split()).strip()
        if not text:
            raise ValueError("identity text is empty")
        import clip
        cache_key = str(model_path)
        if cache_key not in _CLIP_MODEL_CACHE:
            model, preprocess = clip.load(
                cache_key, device="cuda", jit=False)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            _CLIP_MODEL_CACHE[cache_key] = (model, preprocess)
        self.model, self.preprocess = _CLIP_MODEL_CACHE[cache_key]
        initial = self.preprocess(
            _crop_rgb(initial_image, initial_bbox)).unsqueeze(0).cuda()
        tokens = clip.tokenize([text], truncate=True).cuda()
        with torch.no_grad():
            initial_feature = self.model.encode_image(initial).float()
            text_feature = self.model.encode_text(tokens).float()
        self.initial_image_feature = F.normalize(initial_feature, dim=-1)
        self.text_feature = F.normalize(text_feature, dim=-1)

    def anchor_record(self):
        return {
            "initial_image": self.initial_image_feature.detach().cpu().half(),
            "identity_text": self.text_feature.detach().cpu().half(),
        }

    def encode(self, image, bboxes):
        crops = torch.stack([
            self.preprocess(_crop_rgb(image, bbox)) for bbox in bboxes
        ]).cuda()
        with torch.no_grad():
            features = F.normalize(
                self.model.encode_image(crops).float(), dim=-1)
        return features


@dataclass
class RecursiveBranch:
    name: str
    source_name: str
    peak_rank: int
    bbox: list
    query_state: list


__all__ = [
    "ClipCandidateEncoder", "RecursiveBranch", "bbox_iou",
    "clone_query_state", "decode_nms_candidates", "finite_bbox",
    "pool_candidate_tokens", "raw_depth_rois", "relative_geometry",
    "split_query_state", "stack_query_states",
]
