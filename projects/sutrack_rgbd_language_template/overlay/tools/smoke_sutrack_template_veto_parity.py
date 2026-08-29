#!/usr/bin/env python3
"""Metric-blind CUDA parity check for baseline-first template veto.

The real ``cube02_indoor_2@450B`` prefix opens a template transaction at
tracker frame 10.  The historical old-template-public implementation diverges
on the next frame even though it never promotes the new template.  A correct
baseline-first veto keeps the new-template branch public, so its bbox, score,
and writer decision remain aligned with direct identity-only SUTrack through
the rollback.  Only the initialization ground-truth box is read.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.test.parameter.sutrack import parameters as baseline_parameters
from lib.test.parameter.sutrack_template_veto_transaction import (
    parameters as veto_parameters,
)
from lib.test.tracker.rgbd_frame import get_rgbd_frame
from lib.test.tracker.rgbd_language_manifest import RGBDAnchorLanguageManifest
from lib.test.tracker.sutrack import SUTRACK
from lib.test.tracker.sutrack_template_veto_transaction import (
    SUTRACKBaselineFirstTemplateVeto,
)
from tools.smoke_sutrack_transaction_gpu import (
    atomic_json,
    line_at,
    rectangle_from_line,
    score_value,
    sha256_file,
    validate_bbox,
)


YAML_NAME = 'sutrack_l384_rgbd_anchor_identity_template_veto_low22'
EXPECTED_CHECKPOINT_SHA256 = (
    '2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4')


def frame_paths(sequence_root, index):
    stem = '{:08d}'.format(index + 1)
    rgb = sequence_root / 'color' / (stem + '.jpg')
    depth = sequence_root / 'depth' / (stem + '.png')
    if not rgb.is_file() or not depth.is_file():
        raise FileNotFoundError('Missing RGB-D parity frame {}'.format(stem))
    return rgb, depth


def create_params(loader, trace_root=None):
    if trace_root is not None:
        os.environ['SUTRACK_TRANSACTION_TRACE_ROOT'] = str(trace_root)
    params = loader(YAML_NAME)
    params.visualization = False
    params.debug = False
    checkpoint = Path(params.checkpoint).resolve()
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError('DepthTrack-trained checkpoint SHA256 changed')
    return params


def run_prefix(
        tracker_class, params, sequence_root, anchor_index, frame_count,
        init_bbox, init_nlp, direction, transaction_mode):
    tracker = tracker_class(params, 'depthtrack')
    devices = sorted({str(parameter.device)
                      for parameter in tracker.network.parameters()})
    if not devices or not all(device.startswith('cuda:') for device in devices):
        raise RuntimeError('Parity tracker network was not loaded on CUDA')

    sequence_name = sequence_root.name
    init_rgb, init_depth = frame_paths(sequence_root, anchor_index)
    init_image = get_rgbd_frame(init_rgb, init_depth, depth_clip=True)
    tracker.initialize(init_image, {
        'init_bbox': init_bbox,
        'sequence_name': sequence_name,
        'depth_path': str(init_depth),
        'anchor_index': anchor_index,
        'init_nlp': init_nlp,
    })

    frames = []
    step = -1 if direction == 'backward' else 1
    indices = [anchor_index + step * offset
               for offset in range(1, frame_count + 1)]
    if not indices or min(indices) < 0:
        raise ValueError('Parity prefix leaves the sequence bounds')
    for index in indices:
        rgb_path, depth_path = frame_paths(sequence_root, index)
        image = get_rgbd_frame(rgb_path, depth_path, depth_clip=True)
        output = tracker.track(image, {'depth_path': str(depth_path)})
        bbox = validate_bbox(
            output['target_bbox'], image.shape[1], image.shape[0])
        if transaction_mode:
            transaction = output.get('protected_transaction')
            if not isinstance(transaction, dict):
                raise RuntimeError('Transaction diagnostics are missing')
            if transaction.get('recoverable_error'):
                raise RuntimeError(transaction['recoverable_error'])
            event_kind = transaction.get('event_kind')
            if event_kind not in (None, 'template_candidate'):
                raise RuntimeError('Unexpected veto event kind {}'.format(
                    event_kind))
            if transaction.get('active_before'):
                writer = transaction.get('protected_policy_decision')
            else:
                writer = transaction.get('writer_decision')
            decision = transaction.get('decision')
            action = None if decision is None else decision.get('action')
            selected_branch = transaction.get('selected_branch')
        else:
            decision = output.get('safe_template_decision')
            if decision is None:
                raise RuntimeError('Baseline writer decision is missing')
            writer = asdict(decision)
            event_kind = None
            action = None
            selected_branch = None
        if not isinstance(writer, dict):
            raise RuntimeError('Writer decision is missing at frame {}'.format(
                index))
        frames.append({
            'frame_index': index,
            'target_bbox': bbox,
            'best_score': score_value(output['best_score']),
            'writer_decision': writer,
            'event_kind': event_kind,
            'transaction_action': action,
            'selected_branch': selected_branch,
        })
    torch.cuda.synchronize()
    trace_path = (
        None if not transaction_mode or tracker.transaction_trace_path is None
        else str(Path(tracker.transaction_trace_path).resolve()))
    del tracker
    gc.collect()
    torch.cuda.empty_cache()
    return frames, devices, trace_path


def scalar_equal(left, right, tolerance=1.0e-7):
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


def compare_frames(
        baseline, candidate, bbox_tolerance=1.0e-4,
        score_tolerance=1.0e-6):
    if len(baseline) != len(candidate):
        raise AssertionError('Parity frame counts differ')
    max_bbox_error = 0.0
    max_score_error = 0.0
    for base, current in zip(baseline, candidate):
        if base['frame_index'] != current['frame_index']:
            raise AssertionError('Parity frame indices differ')
        bbox_error = max(abs(a - b) for a, b in zip(
            base['target_bbox'], current['target_bbox']))
        score_error = abs(base['best_score'] - current['best_score'])
        max_bbox_error = max(max_bbox_error, bbox_error)
        max_score_error = max(max_score_error, score_error)
        if bbox_error > bbox_tolerance or score_error > score_tolerance:
            raise AssertionError(
                'Inference parity differs at frame {}: bbox={} score={}'.format(
                    base['frame_index'], bbox_error, score_error))
        if set(base['writer_decision']) != set(current['writer_decision']):
            raise AssertionError('Writer decision keys differ')
        for key in base['writer_decision']:
            if not scalar_equal(
                    base['writer_decision'][key],
                    current['writer_decision'][key]):
                raise AssertionError(
                    'Writer decision differs at frame {} key {}'.format(
                        base['frame_index'], key))
    return max_bbox_error, max_score_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--trace-root', type=Path, required=True)
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            '/root/autodl-tmp/VOT-RGBD2022/sequences/cube02_indoor_2'))
    parser.add_argument('--anchor-index', type=int, default=450)
    parser.add_argument('--frames-after-init', type=int, default=12)
    parser.add_argument(
        '--direction', choices=('forward', 'backward'), default='backward')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable for parity check')
    if args.anchor_index < 0 or args.frames_after_init <= 0:
        raise ValueError('Parity anchor and frame count must be positive')

    sequence_root = args.sequence_root.resolve()
    anchors = [float(value) for value in (
        sequence_root / 'anchor.value').read_text(encoding='utf-8').splitlines()]
    if (args.anchor_index >= len(anchors) or
            anchors[args.anchor_index] == 0.0):
        raise ValueError('Selected parity frame is not a VOT anchor')
    init_bbox = rectangle_from_line(line_at(
        sequence_root / 'groundtruth.txt', args.anchor_index))

    baseline_params = create_params(baseline_parameters)
    language_cfg = baseline_params.cfg.TEST.RGBD_LANGUAGE
    manifest = RGBDAnchorLanguageManifest(
        language_cfg.MANIFEST_PATH, language_cfg.MANIFEST_SHA256,
        language_cfg.EXPECTED_DATASET,
        language_cfg.EXPECTED_RECORD_COUNT)
    init_nlp = manifest.language_for(sequence_root.name, args.anchor_index)
    baseline, baseline_devices, unused_trace = run_prefix(
        SUTRACK, baseline_params, sequence_root, args.anchor_index,
        args.frames_after_init, init_bbox, init_nlp, args.direction, False)
    del baseline_params
    gc.collect()
    torch.cuda.empty_cache()

    trace_root = args.trace_root.resolve()
    trace_root.mkdir(parents=True, exist_ok=True)
    candidate_params = create_params(veto_parameters, trace_root)
    candidate, candidate_devices, trace_path = run_prefix(
        SUTRACKBaselineFirstTemplateVeto, candidate_params, sequence_root,
        args.anchor_index, args.frames_after_init, init_bbox, init_nlp,
        args.direction, True)
    max_bbox_error, max_score_error = compare_frames(baseline, candidate)
    event_count = sum(
        frame['event_kind'] == 'template_candidate' for frame in candidate)
    promotions = sum(
        frame['transaction_action'] == 'promote' for frame in candidate)
    rollbacks = sum(
        frame['transaction_action'] == 'rollback' for frame in candidate)
    if event_count < 1 or promotions != 0 or rollbacks < 1:
        raise AssertionError(
            'Expected a real template event followed by baseline rollback')

    payload = {
        'schema': 'sutrack_template_veto_cuda_parity_v1',
        'status': 'passed',
        'metric_computed': False,
        'future_ground_truth_read': False,
        'only_initialization_bbox_read': True,
        'sequence_name': sequence_root.name,
        'anchor_index': args.anchor_index,
        'direction': args.direction,
        'frames_after_init': args.frames_after_init,
        'checkpoint_sha256': EXPECTED_CHECKPOINT_SHA256,
        'language_manifest_sha256': manifest.sha256,
        'baseline_devices': baseline_devices,
        'candidate_devices': candidate_devices,
        'transaction_events': event_count,
        'veto_promotions': promotions,
        'baseline_rollbacks': rollbacks,
        'max_bbox_absolute_error': max_bbox_error,
        'max_score_absolute_error': max_score_error,
        'bbox_tolerance': 1.0e-4,
        'score_tolerance': 1.0e-6,
        'writer_decisions_exactly_equal': True,
        'trace_path': trace_path,
        'source_sha256': sha256_file(Path(__file__).resolve()),
    }
    atomic_json(args.output_json.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
