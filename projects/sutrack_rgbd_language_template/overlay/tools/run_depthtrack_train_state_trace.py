#!/usr/bin/env python3
"""Collect SUTrack online state evidence without reading future-frame GT."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import torch

from lib.test.evaluation import Tracker
from lib.test.tracker.rgbd_frame import get_rgbd_frame
from lib.test.tracker.rgbd_language_manifest import (
    RGBDLanguageManifest,
    sha256_file,
)


SCHEMA = 'sutrack-depthtrack-train-state-trace/v1'
IMPLEMENTATION_FILES = (
    'lib/config/sutrack/config.py',
    'lib/models/sutrack/encoder.py',
    'lib/test/parameter/sutrack.py',
    'lib/test/tracker/rgbd_frame.py',
    'lib/test/tracker/rgbd_language_manifest.py',
    'lib/test/tracker/safe_template_update.py',
    'lib/test/tracker/sutrack.py',
    'tools/run_depthtrack_train_state_trace.py',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--sequences', required=True,
                        help='comma-separated, pre-registered sequence names')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    return parser.parse_args()


def finite_bbox(values):
    try:
        bbox = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if (len(bbox) != 4 or not all(math.isfinite(value) for value in bbox) or
            bbox[2] <= 0.0 or bbox[3] <= 0.0):
        return None
    return bbox


def first_frame_bbox(path):
    # Deliberately consume only the first initialization row here.  Future GT
    # belongs to the post-inference analyzer, never to this trace producer.
    with path.open('r', encoding='utf-8') as stream:
        raw_line = stream.readline()
    if not raw_line:
        raise ValueError('empty ground-truth initialization file {}'.format(path))
    bbox = finite_bbox(raw_line.strip().replace('\t', ',').split(','))
    if bbox is None:
        raise ValueError('malformed first-frame bbox {}'.format(path))
    return bbox


def aligned_frames(sequence_root):
    rgb = sorted(path for path in (sequence_root / 'color').iterdir()
                 if path.is_file())
    depth = sorted(path for path in (sequence_root / 'depth').iterdir()
                   if path.is_file())
    if not rgb or len(rgb) != len(depth):
        raise ValueError('unaligned RGB/depth count for {}'.format(sequence_root))
    if [path.stem for path in rgb] != [path.stem for path in depth]:
        raise ValueError('unaligned RGB/depth names for {}'.format(sequence_root))
    return rgb, depth


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_record(path):
    return {'path': str(path), 'sha256': sha256_file(path),
            'bytes': path.stat().st_size}


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.cuda.set_device(args.device)
    torch.set_num_threads(1)
    dataset_root = args.dataset_root.resolve()
    sequence_names = [name.strip() for name in args.sequences.split(',')
                      if name.strip()]
    if not sequence_names or len(sequence_names) != len(set(sequence_names)):
        raise ValueError('--sequences must be a non-empty unique list')
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('refusing non-empty output {}'.format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker_info = Tracker('sutrack', args.config, 'depthtrack', None)
    params = tracker_info.get_parameters()
    params.visualization = False
    params.debug = False
    language_config = params.cfg.TEST.RGBD_LANGUAGE
    language_manifest = RGBDLanguageManifest(
        language_config.MANIFEST_PATH,
        language_config.MANIFEST_SHA256,
        language_config.EXPECTED_DATASET,
        language_config.EXPECTED_SEQUENCE_COUNT)
    config_path = (Path('/home/SUTrack_RGBD_L/experiments/sutrack') /
                   (args.config + '.yaml')).resolve()
    repository_root = Path('/home/SUTrack_RGBD_L')

    trace_records = []
    prediction_records = []
    sequence_records = []
    started = time.time()
    tracker = tracker_info.create_tracker(params)
    for sequence_name in sequence_names:
        sequence_root = dataset_root / sequence_name
        if not sequence_root.is_dir():
            raise FileNotFoundError(sequence_root)
        rgb_paths, depth_paths = aligned_frames(sequence_root)
        language = language_manifest.language_for(sequence_name)
        init_bbox = first_frame_bbox(sequence_root / 'groundtruth.txt')
        first_image = get_rgbd_frame(
            str(rgb_paths[0]), str(depth_paths[0]), depth_clip=True)
        tracker.initialize(first_image, {
            'init_bbox': init_bbox,
            'sequence_name': sequence_name,
            'depth_path': str(depth_paths[0]),
            'init_nlp': language,
        })
        prediction_records.append({
            'sequence': sequence_name,
            'frame_index': 0,
            'frame_name': rgb_paths[0].stem,
            'deployed_bbox': init_bbox,
            'initialization': True,
        })
        for frame_index, (rgb_path, depth_path) in enumerate(
                zip(rgb_paths[1:], depth_paths[1:]), start=1):
            image = get_rgbd_frame(
                str(rgb_path), str(depth_path), depth_clip=True)
            output = tracker.track(image, {'depth_path': str(depth_path)})
            deployed = finite_bbox(output['target_bbox'])
            evidence = output.get('online_state_evidence')
            if deployed is None or not isinstance(evidence, dict):
                raise ValueError(
                    'missing finite state evidence at {}:{}'.format(
                        sequence_name, frame_index))
            prior = finite_bbox(evidence.get('prior_bbox'))
            candidate = finite_bbox(evidence.get('candidate_bbox'))
            if prior is None or candidate is None:
                raise ValueError(
                    'malformed action bbox at {}:{}'.format(
                        sequence_name, frame_index))
            decision = output.get('safe_template_decision')
            if decision is None:
                raise ValueError('safe-template decision is absent')
            record = {
                'schema': SCHEMA,
                'sequence': sequence_name,
                'frame_index': frame_index,
                'frame_name': rgb_path.stem,
                'prior_bbox': prior,
                'candidate_bbox': candidate,
                'deployed_bbox': deployed,
                'best_score': float(evidence['confidence']),
                'online_evidence': evidence,
                'safe_template_decision': asdict(decision),
                'ground_truth_available_to_tracker': False,
                'future_frame_text_used': False,
            }
            if not all(math.isfinite(record[key]) for key in ('best_score',)):
                raise ValueError('non-finite score at {}:{}'.format(
                    sequence_name, frame_index))
            trace_records.append(record)
            prediction_records.append({
                'sequence': sequence_name,
                'frame_index': frame_index,
                'frame_name': rgb_path.stem,
                'deployed_bbox': deployed,
                'candidate_bbox': candidate,
                'prior_bbox': prior,
                'initialization': False,
            })
        sequence_records.append({
            'sequence': sequence_name,
            'frame_count': len(rgb_paths),
            'trace_rows': len(rgb_paths) - 1,
            'first_frame_gt_only': True,
            'language_sha256': hashlib.sha256(
                language.encode('utf-8')).hexdigest(),
        })
        print('COMPLETE {} {}/{}'.format(
            sequence_name, len(sequence_records), len(sequence_names)),
              flush=True)

    trace_path = output_dir / 'online_trace.jsonl'
    predictions_path = output_dir / 'predictions.jsonl'
    atomic_write(trace_path, ''.join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n'
        for record in trace_records).encode('utf-8'))
    atomic_write(predictions_path, ''.join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n'
        for record in prediction_records).encode('utf-8'))
    manifest = {
        'schema': SCHEMA,
        'complete': True,
        'dataset': 'DepthTrack Train only',
        'dataset_root': str(dataset_root),
        'sequences': sequence_names,
        'sequence_count': len(sequence_names),
        'frame_count': len(prediction_records),
        'trace_row_count': len(trace_records),
        'ground_truth_consumption': 'first_frame_initialization_only',
        'ground_truth_available_to_tracker': False,
        'future_frame_text_used': False,
        'public_evaluation': False,
        'config': file_record(config_path),
        'checkpoint': file_record(Path(params.checkpoint).resolve()),
        'language_manifest': file_record(Path(language_manifest.path)),
        'implementation_sha256': {
            relative: sha256_file(repository_root / relative)
            for relative in IMPLEMENTATION_FILES
        },
        'sequence_records': sequence_records,
        'trace': file_record(trace_path),
        'predictions': file_record(predictions_path),
        'elapsed_seconds': time.time() - started,
        'cuda_device': torch.cuda.get_device_name(args.device),
    }
    manifest_path = output_dir / 'manifest.json'
    atomic_write(
        manifest_path,
        (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) +
         '\n').encode('utf-8'))
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
