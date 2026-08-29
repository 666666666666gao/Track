#!/usr/bin/env python3
"""Run a metric-blind CUDA preflight for the low22 transaction tracker."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.test.parameter.sutrack_transaction import parameters
from lib.test.tracker.rgbd_frame import get_rgbd_frame
from lib.test.tracker.rgbd_language_manifest import (
    RGBDAnchorLanguageManifest,
)
from lib.test.tracker.sutrack_transaction import (
    SUTRACKProtectedTransaction,
)


EXPECTED_CHECKPOINT_SHA256 = (
    '2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4')
YAML_NAME = 'sutrack_l384_rgbd_anchor_identity_template_transaction_low22'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-{}'.format(os.getpid()))
    with open(temporary, 'x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def rectangle_from_line(line):
    values = [float(value) for value in line.strip().split(',')]
    if len(values) == 4:
        bbox = values
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs, ys = values[0::2], values[1::2]
        bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
    else:
        raise ValueError('Initial VOT region is neither rectangle nor polygon')
    if (not all(math.isfinite(value) for value in bbox) or
            bbox[2] <= 0.0 or bbox[3] <= 0.0):
        raise ValueError('Initial VOT region is malformed')
    return bbox


def line_at(path, index):
    with open(path, 'r', encoding='utf-8') as stream:
        for line_index, line in enumerate(stream):
            if line_index == index:
                return line
    raise IndexError('{} has no line {}'.format(path, index))


def score_value(value):
    if torch.is_tensor(value):
        value = float(value.detach().reshape(-1).max().cpu().item())
    else:
        value = float(value)
    if not math.isfinite(value):
        raise ValueError('Tracker score is non-finite')
    return value


def validate_bbox(value, width, height):
    bbox = [float(item) for item in value]
    if (len(bbox) != 4 or
            not all(math.isfinite(item) for item in bbox) or
            bbox[2] <= 0.0 or bbox[3] <= 0.0 or
            bbox[0] < 0.0 or bbox[1] < 0.0 or
            bbox[0] + bbox[2] > width + 1.0e-6 or
            bbox[1] + bbox[3] > height + 1.0e-6):
        raise ValueError('Tracker bbox is malformed or outside the image')
    return bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--trace-root', type=Path, required=True)
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            '/root/autodl-tmp/VOT-RGBD2022/sequences/ball06_indoor_2'))
    parser.add_argument('--anchor-index', type=int, default=0)
    parser.add_argument('--frames-after-init', type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable for transaction GPU smoke')
    if args.anchor_index < 0 or args.frames_after_init <= 0:
        raise ValueError('Smoke anchor and frame count must be positive')

    sequence_root = args.sequence_root.resolve()
    sequence_name = sequence_root.name
    anchors = [float(value) for value in (sequence_root / 'anchor.value').read_text(
        encoding='utf-8').splitlines()]
    if (args.anchor_index >= len(anchors) or
            anchors[args.anchor_index] == 0.0):
        raise ValueError('Selected smoke frame is not a VOT anchor')
    last_index = args.anchor_index + args.frames_after_init
    init_bbox = rectangle_from_line(line_at(
        sequence_root / 'groundtruth.txt', args.anchor_index))

    trace_root = args.trace_root.resolve()
    trace_root.mkdir(parents=True, exist_ok=True)
    os.environ['SUTRACK_TRANSACTION_TRACE_ROOT'] = str(trace_root)
    params = parameters(YAML_NAME)
    params.visualization = False
    params.debug = False
    checkpoint = Path(params.checkpoint).resolve()
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError('DepthTrack-trained checkpoint SHA256 changed')

    language_cfg = params.cfg.TEST.RGBD_LANGUAGE
    language_manifest = RGBDAnchorLanguageManifest(
        language_cfg.MANIFEST_PATH, language_cfg.MANIFEST_SHA256,
        language_cfg.EXPECTED_DATASET,
        language_cfg.EXPECTED_RECORD_COUNT)
    init_nlp = language_manifest.language_for(
        sequence_name, args.anchor_index)

    tracker = SUTRACKProtectedTransaction(params, 'depthtrack')
    network_devices = sorted({str(parameter.device)
                              for parameter in tracker.network.parameters()})
    if not network_devices or not all(
            device.startswith('cuda:') for device in network_devices):
        raise RuntimeError('Transaction network was not loaded on CUDA')

    def frame_paths(index):
        stem = '{:08d}'.format(index + 1)
        rgb = sequence_root / 'color' / (stem + '.jpg')
        depth = sequence_root / 'depth' / (stem + '.png')
        if not rgb.is_file() or not depth.is_file():
            raise FileNotFoundError('Missing RGB-D smoke frame {}'.format(stem))
        return rgb, depth

    init_rgb, init_depth = frame_paths(args.anchor_index)
    init_image = get_rgbd_frame(init_rgb, init_depth, depth_clip=True)
    tracker.initialize(init_image, {
        'init_bbox': init_bbox,
        'sequence_name': sequence_name,
        'depth_path': str(init_depth),
        'anchor_index': args.anchor_index,
        'init_nlp': init_nlp,
    })

    frames = []
    for index in range(args.anchor_index + 1, last_index + 1):
        rgb_path, depth_path = frame_paths(index)
        image = get_rgbd_frame(rgb_path, depth_path, depth_clip=True)
        output = tracker.track(image, {'depth_path': str(depth_path)})
        transaction = output.get('protected_transaction')
        if not isinstance(transaction, dict):
            raise ValueError('Transaction tracker omitted diagnostic output')
        if transaction.get('recoverable_error'):
            raise RuntimeError(
                'Transaction CUDA smoke hit a recoverable error: {}'.format(
                    transaction['recoverable_error']))
        frames.append({
            'frame_index': index,
            'target_bbox': validate_bbox(
                output['target_bbox'], image.shape[1], image.shape[0]),
            'best_score': score_value(output['best_score']),
            'event_kind': transaction.get('event_kind'),
            'selected_branch': transaction.get('selected_branch'),
            'transaction_active_after': bool(
                tracker.template_transaction.active),
        })
    torch.cuda.synchronize()

    invalid_event_kinds = sorted({
        str(frame['event_kind']) for frame in frames
        if frame['event_kind'] not in (None, 'template_candidate')
    })
    if invalid_event_kinds:
        raise RuntimeError(
            'Template-only tracker opened a non-template transaction: {}'.format(
                invalid_event_kinds))
    if not any(
            abs(float(current) - float(initial)) > 1.0e-6
            for current, initial in zip(frames[0]['target_bbox'], init_bbox)):
        raise RuntimeError(
            'First recursive bbox was frozen instead of accepting prediction')

    if tracker.transaction_trace_path is None:
        raise RuntimeError('Transaction trace path was not initialized')
    trace_path = Path(tracker.transaction_trace_path).resolve()
    trace_records = [json.loads(line) for line in trace_path.read_text(
        encoding='utf-8').splitlines()]
    if (not trace_records or trace_records[0].get('type') != 'initialize' or
            trace_records[0].get('anchor_index') != args.anchor_index):
        raise ValueError('Transaction smoke trace initialization is malformed')

    sources = (
        Path(__file__).resolve(),
        REPO_ROOT / 'lib/test/tracker/sutrack_transaction.py',
        REPO_ROOT / 'lib/test/tracker/protected_tentative_transaction.py',
        REPO_ROOT / 'lib/test/parameter/sutrack_transaction.py',
        REPO_ROOT / 'experiments/sutrack/'
        'sutrack_l384_rgbd_anchor_identity_template_transaction_low22.yaml',
    )
    payload = {
        'schema': 'sutrack_template_transaction_low22_gpu_smoke_v2',
        'status': 'passed',
        'gpu_inference_exercised': True,
        'public_metric_computed': False,
        'future_ground_truth_read': False,
        'only_initialization_bbox_read': True,
        'bbox_state_freeze_detected': False,
        'allowed_transaction_event_kinds': [None, 'template_candidate'],
        'low22_vot_started': False,
        'sequence_name': sequence_name,
        'anchor_index': args.anchor_index,
        'frames_after_init': args.frames_after_init,
        'initial_bbox': init_bbox,
        'initial_language': init_nlp,
        'network_devices': network_devices,
        'cuda_device_name': torch.cuda.get_device_name(0),
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': checkpoint_sha,
        'language_manifest': str(Path(language_cfg.MANIFEST_PATH).resolve()),
        'language_manifest_sha256': language_manifest.sha256,
        'frames': frames,
        'trace_path': str(trace_path),
        'trace_sha256': sha256_file(trace_path),
        'trace_record_count': len(trace_records),
        'sources': {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in sources
        },
    }
    atomic_json(args.output_json.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
