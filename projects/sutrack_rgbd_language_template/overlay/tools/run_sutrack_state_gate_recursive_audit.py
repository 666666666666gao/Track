#!/usr/bin/env python3
"""Run one frozen SUTrack learned-gate shard on held-out Train sequences."""

import argparse
from collections import Counter
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


SCHEMA = 'sutrack-state-gate-recursive-audit-trace/v1'
IMPLEMENTATION_FILES = (
    'lib/config/sutrack/config.py',
    'lib/models/sutrack/encoder.py',
    'lib/test/parameter/sutrack.py',
    'lib/test/tracker/rgbd_frame.py',
    'lib/test/tracker/rgbd_language_manifest.py',
    'lib/test/tracker/safe_template_update.py',
    'lib/test/tracker/sutrack_state_gate.py',
    'lib/test/tracker/sutrack.py',
    'tools/run_sutrack_state_gate_recursive_audit.py',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--sequences', required=True)
    parser.add_argument('--split-plan', type=Path, required=True)
    parser.add_argument('--training-result', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    return parser.parse_args()


def load_json(path):
    with Path(path).open('r', encoding='utf-8') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError('{} is not an object'.format(path))
    return value


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
    with path.open('r', encoding='utf-8') as stream:
        raw_line = stream.readline()
    bbox = finite_bbox(raw_line.strip().replace('\t', ',').split(','))
    if bbox is None:
        raise ValueError('malformed first-frame bbox {}'.format(path))
    return bbox


def aligned_frames(sequence_root):
    rgb = sorted(path for path in (sequence_root / 'color').iterdir()
                 if path.is_file())
    depth = sorted(path for path in (sequence_root / 'depth').iterdir()
                   if path.is_file())
    if (not rgb or len(rgb) != len(depth) or
            [path.stem for path in rgb] !=
            [path.stem for path in depth]):
        raise ValueError('RGB/depth alignment failed {}'.format(sequence_root))
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
    path = Path(path).resolve()
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
    split_path = args.split_plan.resolve()
    training_path = args.training_result.resolve()
    artifact_path = args.artifact.resolve()
    split = load_json(split_path)
    training = load_json(training_path)
    if (split.get('schema') != 'sutrack-state-gate-split-plan/v1' or
            split.get('complete') is not True or
            split.get('audit_consumption_limit') != 1 or
            split.get('public_evaluation') is not False or
            not set(sequence_names).issubset(split['audit_sequences'])):
        raise ValueError('recursive audit split contract failed')
    if (training.get('schema') != 'sutrack-state-gate-training/v1' or
            training.get('complete') is not True or
            training.get('ready_for_recursive_audit') is not True or
            training.get('immediate_audit_policies_evaluated') != 1 or
            training.get('immediate_audit_passed') is not True or
            training.get('public_evaluation') is not False):
        raise ValueError('training result is not ready for recursive audit')
    deployment_seed = int(training['deployment_seed'])
    artifact_record = next((record for record in training['artifacts']
                            if int(record['seed']) == deployment_seed), None)
    if (not isinstance(artifact_record, dict) or
            Path(artifact_record['path']).resolve() != artifact_path or
            artifact_record['sha256'] != sha256_file(artifact_path)):
        raise ValueError('recursive audit artifact binding failed')
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
    gate_config = params.cfg.TEST.LEARNED_STATE_GATE
    if (not bool(gate_config.USE) or
            Path(gate_config.ARTIFACT_PATH).resolve() != artifact_path or
            str(gate_config.ARTIFACT_SHA256).lower() !=
            sha256_file(artifact_path) or
            Path(gate_config.TRAINING_RESULT_PATH).resolve() != training_path or
            str(gate_config.TRAINING_RESULT_SHA256).lower() !=
            sha256_file(training_path)):
        raise ValueError('runtime config does not bind the audit artifact')

    repository_root = Path('/home/SUTrack_RGBD_L')
    config_path = (repository_root / 'experiments' / 'sutrack' /
                   (args.config + '.yaml')).resolve()
    tracker = tracker_info.create_tracker(params)
    predictions = []
    sequence_records = []
    action_counts = Counter()
    started = time.time()
    for sequence_name in sequence_names:
        sequence_root = dataset_root / sequence_name
        rgb_paths, depth_paths = aligned_frames(sequence_root)
        language = language_manifest.language_for(sequence_name)
        init_bbox = first_frame_bbox(sequence_root / 'groundtruth.txt')
        tracker.initialize(
            get_rgbd_frame(str(rgb_paths[0]), str(depth_paths[0]),
                           depth_clip=True),
            {'init_bbox': init_bbox, 'sequence_name': sequence_name,
             'depth_path': str(depth_paths[0]), 'init_nlp': language})
        predictions.append({
            'schema': SCHEMA, 'sequence': sequence_name, 'frame_index': 0,
            'frame_name': rgb_paths[0].stem, 'deployed_bbox': init_bbox,
            'rollback_state': False, 'probability': None,
            'initialization': True,
        })
        sequence_actions = 0
        for frame_index, (rgb_path, depth_path) in enumerate(
                zip(rgb_paths[1:], depth_paths[1:]), start=1):
            output = tracker.track(
                get_rgbd_frame(str(rgb_path), str(depth_path),
                               depth_clip=True),
                {'depth_path': str(depth_path)})
            bbox = finite_bbox(output['target_bbox'])
            decision = output.get('learned_state_gate_decision')
            if bbox is None or decision is None:
                raise ValueError('missing gate output {}:{}'.format(
                    sequence_name, frame_index))
            probability = decision.probability
            if probability is not None:
                probability = float(probability)
                if not math.isfinite(probability):
                    raise ValueError('non-finite gate probability')
            rollback = bool(decision.rollback_state)
            sequence_actions += int(rollback)
            action_counts['rollback_state'] += int(rollback)
            action_counts['checked'] += int(bool(decision.checked))
            action_counts['hard_conflict'] += int(bool(decision.hard_conflict))
            predictions.append({
                'schema': SCHEMA,
                'sequence': sequence_name,
                'frame_index': frame_index,
                'frame_name': rgb_path.stem,
                'deployed_bbox': bbox,
                'rollback_state': rollback,
                'probability': probability,
                'gate_decision': asdict(decision),
                'initialization': False,
                'ground_truth_available_to_tracker': False,
                'future_frame_text_used': False,
            })
        sequence_records.append({
            'sequence': sequence_name,
            'frame_count': len(rgb_paths),
            'rollback_actions': sequence_actions,
        })
        print('COMPLETE {} {}/{}'.format(
            sequence_name, len(sequence_records), len(sequence_names)),
              flush=True)

    predictions_path = output_dir / 'predictions.jsonl'
    atomic_write(predictions_path, ''.join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n'
        for record in predictions).encode('utf-8'))
    manifest = {
        'schema': SCHEMA,
        'complete': True,
        'role': 'single_frozen_policy_recursive_audit_shard',
        'dataset': 'DepthTrack Train audit only',
        'dataset_root': str(dataset_root),
        'sequences': sequence_names,
        'sequence_count': len(sequence_names),
        'frame_count': len(predictions),
        'sequence_records': sequence_records,
        'ground_truth_consumption': 'first_frame_initialization_only',
        'ground_truth_available_to_tracker': False,
        'future_frame_text_used': False,
        'public_evaluation': False,
        'policy_evaluations_on_audit': 1,
        'action_counts': dict(sorted(action_counts.items())),
        'split_plan': file_record(split_path),
        'training_result': file_record(training_path),
        'artifact': file_record(artifact_path),
        'config': file_record(config_path),
        'checkpoint': file_record(Path(params.checkpoint)),
        'language_manifest': file_record(Path(language_manifest.path)),
        'implementation_sha256': {
            relative: sha256_file(repository_root / relative)
            for relative in IMPLEMENTATION_FILES
        },
        'predictions': file_record(predictions_path),
        'elapsed_seconds': time.time() - started,
        'cuda_device': torch.cuda.get_device_name(args.device),
    }
    atomic_write(
        output_dir / 'manifest.json',
        (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) +
         '\n').encode('utf-8'))
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
