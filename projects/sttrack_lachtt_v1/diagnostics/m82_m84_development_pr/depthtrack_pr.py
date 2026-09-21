"""DepthTrack long-term precision, recall, and F-score evaluation."""

import json
import math
from numbers import Integral
from pathlib import Path

import cv2
import numpy as np
from vot.region import Rectangle, RegionType, Special, calculate_overlaps


def _load_rows(path, columns):
    if not path.is_file():
        raise FileNotFoundError("Missing result file: {}".format(path))
    first_line = next((line for line in path.read_text(encoding='utf-8-sig').splitlines()
                       if line.strip()), '')
    delimiter = ',' if ',' in first_line else None
    values = np.loadtxt(str(path), delimiter=delimiter, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Empty result file: {}".format(path))
    return values.reshape(-1, columns)


def _determine_thresholds(scores, resolution):
    scores = sorted((float(value) for value in scores if math.isfinite(value)), reverse=True)
    if not scores:
        raise ValueError("No finite confidence scores")
    if len(scores) > resolution - 2:
        delta = math.floor(len(scores) / (resolution - 2))
        indices = np.rint(
            np.linspace(delta, len(scores) - delta, num=resolution - 2)).astype(np.int64)
        indices = np.clip(indices, 0, len(scores) - 1)
        thresholds = [scores[index] for index in indices]
    else:
        thresholds = scores
    return np.asarray([math.inf] + thresholds + [-math.inf], dtype=np.float64)


def _vot_overlaps(prediction, groundtruth, width, height):
    """Compute bounded rectangle overlap using the VOT region implementation."""
    if not np.isfinite(prediction).all():
        raise ValueError('Prediction contains non-finite box coordinates')
    if np.any(prediction[:, 2:] < 0):
        raise ValueError('Prediction contains negative box dimensions')

    prediction_regions = [Rectangle(*box) for box in prediction]
    groundtruth_regions = []
    for box in groundtruth:
        if np.isfinite(box).all() and box[2] > 0 and box[3] > 0:
            groundtruth_regions.append(Rectangle(*box))
        else:
            # DepthTrack encodes absent/unknown targets with invalid or NaN boxes.
            groundtruth_regions.append(Special(0))
    overlaps = np.asarray(
        calculate_overlaps(prediction_regions, groundtruth_regions, (width, height)),
        dtype=np.float64)
    visible = np.asarray(
        [region.type is not RegionType.SPECIAL for region in groundtruth_regions], dtype=bool)
    return overlaps, visible


def evaluate_depthtrack_results(dataset_root, results_dir, resolution=100, sequence_names=None):
    """Evaluate tracker TXT output with the VOT long-term PR protocol.

    Precision/recall curves are computed per sequence and then macro-averaged,
    matching the VOT long-term analysis used by DepthTrack.
    """
    if (isinstance(resolution, bool) or not isinstance(resolution, Integral) or
            resolution < 3):
        raise ValueError('resolution must be an integer of at least 3')
    resolution = int(resolution)
    dataset_root = Path(dataset_root)
    results_dir = Path(results_dir)
    available_sequences = {
        path.name: path for path in dataset_root.iterdir() if path.is_dir()
    }
    if sequence_names is None:
        sequence_roots = [available_sequences[name] for name in sorted(available_sequences)]
    else:
        sequence_names = list(sequence_names)
        if not sequence_names:
            raise ValueError('sequence_names must not be empty')
        if len(sequence_names) != len(set(sequence_names)):
            raise ValueError('sequence_names contains duplicates')
        missing = [name for name in sequence_names if name not in available_sequences]
        if missing:
            raise ValueError('Unknown dataset sequences: {}'.format(', '.join(sorted(missing))))
        sequence_roots = [available_sequences[name] for name in sequence_names]
    if not sequence_roots:
        raise ValueError("No DepthTrack sequences found in {}".format(dataset_root))

    sequence_data = []
    all_scores = []
    for sequence_root in sequence_roots:
        name = sequence_root.name
        groundtruth = _load_rows(sequence_root / 'groundtruth.txt', 4)
        prediction = _load_rows(results_dir / '{}.txt'.format(name), 4)
        confidence = _load_rows(results_dir / '{}_all_scores.txt'.format(name), 1).reshape(-1)
        if not (len(groundtruth) == len(prediction) == len(confidence)):
            raise ValueError(
                "Frame count mismatch for {}: GT={}, prediction={}, confidence={}"
                .format(name, len(groundtruth), len(prediction), len(confidence)))
        if not np.isfinite(confidence).all():
            raise ValueError('Non-finite confidence score for {}'.format(name))
        first_image = next(iter(sorted((sequence_root / 'color').glob('*'))), None)
        image = cv2.imread(str(first_image)) if first_image is not None else None
        if image is None:
            raise ValueError("Cannot read an image for sequence {}".format(name))
        height, width = image.shape[:2]
        overlaps, visible = _vot_overlaps(prediction, groundtruth, width, height)
        if not np.any(visible):
            raise ValueError("Sequence {} has no visible ground-truth frames".format(name))
        sequence_data.append((overlaps, visible, confidence))
        all_scores.extend(confidence.tolist())

    thresholds = _determine_thresholds(all_scores, resolution)
    precision_curves = []
    recall_curves = []
    for overlaps, visible, confidence in sequence_data:
        precision = np.zeros(len(thresholds), dtype=np.float64)
        recall = np.zeros(len(thresholds), dtype=np.float64)
        for index, threshold in enumerate(thresholds):
            selected = confidence >= threshold
            if not np.any(selected):
                precision[index] = 1.0
            else:
                precision[index] = overlaps[selected].mean()
                recall[index] = overlaps[selected].sum() / visible.sum()
        precision_curves.append(precision)
        recall_curves.append(recall)

    precision_curve = np.mean(precision_curves, axis=0)
    recall_curve = np.mean(recall_curves, axis=0)
    denominator = precision_curve + recall_curve
    f_curve = np.divide(
        2 * precision_curve * recall_curve, denominator,
        out=np.zeros_like(denominator), where=denominator > 0)
    best_index = int(np.argmax(f_curve))
    precision = float(precision_curve[best_index])
    recall = float(recall_curve[best_index])
    f_score = float(f_curve[best_index])
    return {
        'precision': precision,
        'recall': recall,
        'f_score': f_score,
        'precision_percent': precision * 100.0,
        'recall_percent': recall * 100.0,
        'f_score_percent': f_score * 100.0,
        'threshold': (float(thresholds[best_index])
                      if math.isfinite(thresholds[best_index]) else None),
        'sequences': len(sequence_data),
        'frames': int(sum(len(item[0]) for item in sequence_data)),
    }


def format_metrics(metrics):
    return json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False)
