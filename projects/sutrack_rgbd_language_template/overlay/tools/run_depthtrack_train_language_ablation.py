#!/usr/bin/env python3
"""Collect paired SUTrack language ON/OFF predictions on DepthTrack Train.

Only the first ground-truth row is read by this producer. Full trajectories are
joined by the independent analyzer after both inference branches are frozen.
"""

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


SCHEMA = 'sutrack-depthtrack-train-language-ablation-trace/v1'
IMPLEMENTATION_FILES = (
    'lib/config/sutrack/config.py',
    'lib/models/sutrack/clip.py',
    'lib/models/sutrack/encoder.py',
    'lib/models/sutrack/fastitpn.py',
    'lib/test/parameter/sutrack.py',
    'lib/test/tracker/rgbd_frame.py',
    'lib/test/tracker/rgbd_language_manifest.py',
    'lib/test/tracker/safe_template_update.py',
    'lib/test/tracker/sutrack.py',
    'tools/run_depthtrack_train_language_ablation.py',
)
CLIP_PATH = Path('/root/.cache/clip/ViT-L-14.pt')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--language-mode', choices=('on', 'off'), required=True)
    parser.add_argument('--sequences', required=True)
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
    with path.open('r', encoding='utf-8') as stream:
        raw_line = stream.readline()
    if not raw_line:
        raise ValueError('empty initialization GT {}'.format(path))
    bbox = finite_bbox(raw_line.strip().replace('\t', ',').split(','))
    if bbox is None:
        raise ValueError('malformed initialization GT {}'.format(path))
    return bbox


def aligned_frames(sequence_root):
    rgb = sorted(path for path in (sequence_root / 'color').iterdir()
                 if path.is_file())
    depth = sorted(path for path in (sequence_root / 'depth').iterdir()
                   if path.is_file())
    if not rgb or len(rgb) != len(depth):
        raise ValueError('unaligned RGB/depth count {}'.format(sequence_root))
    if [path.stem for path in rgb] != [path.stem for path in depth]:
        raise ValueError('unaligned RGB/depth names {}'.format(sequence_root))
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
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        'path': str(resolved),
        'sha256': sha256_file(resolved),
        'bytes': resolved.stat().st_size,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.cuda.set_device(args.device)
    torch.set_num_threads(1)
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    sequences = [item.strip() for item in args.sequences.split(',')
                 if item.strip()]
    if not sequences or len(sequences) != len(set(sequences)):
        raise ValueError('--sequences must be non-empty and unique')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('refusing non-empty output {}'.format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker_info = Tracker('sutrack', args.config, 'depthtrack', None)
    params = tracker_info.get_parameters()
    params.visualization = False
    params.debug = False
    language_enabled = bool(params.cfg.TEST.RGBD_LANGUAGE.USE)
    nlp_enabled = bool(params.cfg.TEST.USE_NLP.DEPTHTRACK)
    expected_enabled = args.language_mode == 'on'
    if language_enabled != expected_enabled or nlp_enabled != expected_enabled:
        raise ValueError(
            'config language contract differs from --language-mode')
    if not bool(params.cfg.TEST.SAFE_TEMPLATE_UPDATE.USE):
        raise ValueError('paired experiment requires safe-template v1 enabled')

    language_manifest = None
    if expected_enabled:
        language_config = params.cfg.TEST.RGBD_LANGUAGE
        language_manifest = RGBDLanguageManifest(
            language_config.MANIFEST_PATH,
            language_config.MANIFEST_SHA256,
            language_config.EXPECTED_DATASET,
            language_config.EXPECTED_SEQUENCE_COUNT)

    repository_root = Path('/home/SUTrack_RGBD_L')
    config_path = (repository_root / 'experiments' / 'sutrack' /
                   (args.config + '.yaml')).resolve()
    tracker = tracker_info.create_tracker(params)
    predictions = []
    sequence_records = []
    started = time.time()

    for sequence_name in sequences:
        sequence_root = dataset_root / sequence_name
        if not sequence_root.is_dir():
            raise FileNotFoundError(sequence_root)
        rgb_paths, depth_paths = aligned_frames(sequence_root)
        init_bbox = first_frame_bbox(sequence_root / 'groundtruth.txt')
        language = (language_manifest.language_for(sequence_name)
                    if language_manifest is not None else None)
        first_image = get_rgbd_frame(
            str(rgb_paths[0]), str(depth_paths[0]), depth_clip=True)
        tracker.initialize(first_image, {
            'init_bbox': init_bbox,
            'sequence_name': sequence_name,
            'depth_path': str(depth_paths[0]),
            'init_nlp': language,
        })
        predictions.append({
            'schema': SCHEMA,
            'language_mode': args.language_mode,
            'sequence': sequence_name,
            'frame_index': 0,
            'frame_name': rgb_paths[0].stem,
            'bbox': init_bbox,
            'best_score': None,
            'initialization': True,
            'ground_truth_available_to_tracker': True,
            'future_frame_text_used': False,
        })
        for frame_index, (rgb_path, depth_path) in enumerate(
                zip(rgb_paths[1:], depth_paths[1:]), start=1):
            image = get_rgbd_frame(
                str(rgb_path), str(depth_path), depth_clip=True)
            output = tracker.track(image, {'depth_path': str(depth_path)})
            bbox = finite_bbox(output.get('target_bbox'))
            if bbox is None:
                raise ValueError('non-finite bbox {}:{}'.format(
                    sequence_name, frame_index))
            score = output.get('best_score')
            if torch.is_tensor(score):
                score = float(score.detach().reshape(-1).max().cpu().item())
            else:
                score = float(score)
            if not math.isfinite(score):
                raise ValueError('non-finite score {}:{}'.format(
                    sequence_name, frame_index))
            decision = output.get('safe_template_decision')
            if decision is None:
                raise ValueError('safe-template decision absent')
            predictions.append({
                'schema': SCHEMA,
                'language_mode': args.language_mode,
                'sequence': sequence_name,
                'frame_index': frame_index,
                'frame_name': rgb_path.stem,
                'bbox': bbox,
                'best_score': score,
                'initialization': False,
                'safe_template_decision': asdict(decision),
                'ground_truth_available_to_tracker': False,
                'future_frame_text_used': False,
            })
        sequence_records.append({
            'sequence': sequence_name,
            'frame_count': len(rgb_paths),
            'prediction_rows': len(rgb_paths),
            'first_frame_gt_only': True,
            'language_enabled': expected_enabled,
            'language_sha256': (
                hashlib.sha256(language.encode('utf-8')).hexdigest()
                if language is not None else None),
        })
        print('COMPLETE {} {}/{}'.format(
            sequence_name, len(sequence_records), len(sequences)), flush=True)

    predictions_path = output_dir / 'predictions.jsonl'
    atomic_write(predictions_path, ''.join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n'
        for record in predictions).encode('utf-8'))
    manifest = {
        'schema': SCHEMA,
        'complete': True,
        'dataset': 'DepthTrack Train only',
        'dataset_root': str(dataset_root),
        'language_mode': args.language_mode,
        'language_enabled': expected_enabled,
        'sequences': sequences,
        'sequence_count': len(sequences),
        'frame_count': len(predictions),
        'prediction_row_count': len(predictions),
        'ground_truth_consumption': 'first_frame_initialization_only',
        'ground_truth_available_to_tracker_after_initialization': False,
        'future_frame_text_used': False,
        'public_evaluation': False,
        'config': file_record(config_path),
        'checkpoint': file_record(Path(params.checkpoint)),
        'clip_checkpoint': file_record(CLIP_PATH),
        'language_manifest': (
            file_record(language_manifest.path)
            if language_manifest is not None else None),
        'implementation_sha256': {
            relative: sha256_file(repository_root / relative)
            for relative in IMPLEMENTATION_FILES
        },
        'sequence_records': sequence_records,
        'predictions': file_record(predictions_path),
        'elapsed_seconds': time.time() - started,
        'cuda_device': torch.cuda.get_device_name(args.device),
    }
    manifest_path = output_dir / 'manifest.json'
    atomic_write(manifest_path, (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) +
        '\n').encode('utf-8'))
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
