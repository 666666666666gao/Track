#!/usr/bin/env python3
"""Run a bounded RGB-D+language inference smoke without scoring ground truth."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import cv2
import torch

from lib.test.evaluation import Tracker
from lib.test.tracker.rgbd_frame import get_rgbd_frame
from lib.test.tracker.rgbd_language_manifest import RGBDLanguageManifest, sha256_file


IMPLEMENTATION_FILES = (
    'lib/config/sutrack/config.py',
    'lib/models/sutrack/encoder.py',
    'lib/test/parameter/sutrack.py',
    'lib/test/tracker/rgbd_frame.py',
    'lib/test/tracker/rgbd_language_manifest.py',
    'lib/test/tracker/safe_template_update.py',
    'lib/test/tracker/temporal_depth_identity.py',
    'lib/test/tracker/sutrack.py',
    'lib/test/vot/sutrack_class.py',
    'tools/smoke_rgbd_language_safe_template.py',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path('/root/autodl-tmp/VOT-RGBD2022/sequences/adapter01_indoor_1'))
    parser.add_argument(
        '--config', default='sutrack_l384_rgbd_language_safe_template')
    parser.add_argument('--frames', type=int, default=6)
    return parser.parse_args()


def initial_bbox(path):
    first_line = path.read_text(encoding='utf-8').splitlines()[0]
    values = [float(value) for value in first_line.replace('\t', ',').split(',')]
    if len(values) == 4:
        bbox = values
    elif len(values) >= 8 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
    else:
        raise ValueError('Unsupported first-frame ground-truth region')
    if (not all(math.isfinite(value) for value in bbox) or
            bbox[2] <= 0.0 or bbox[3] <= 0.0):
        raise ValueError('Malformed first-frame bounding box')
    return bbox


def main():
    args = parse_args()
    if args.frames < 2:
        raise ValueError('--frames must be at least 2')
    sequence_root = args.sequence_root.resolve()
    sequence_name = sequence_root.name
    rgb_paths = sorted((sequence_root / 'color').glob('*'))[:args.frames]
    depth_paths = sorted((sequence_root / 'depth').glob('*'))[:args.frames]
    if len(rgb_paths) != args.frames or len(depth_paths) != args.frames:
        raise ValueError('Requested smoke frames are unavailable')
    if [path.stem for path in rgb_paths] != [path.stem for path in depth_paths]:
        raise ValueError('RGB/depth frame names are not aligned')

    tracker_info = Tracker('sutrack', args.config, 'depthtrack', None)
    params = tracker_info.get_parameters()
    params.visualization = False
    params.debug = False
    language_config = params.cfg.TEST.RGBD_LANGUAGE
    manifest = RGBDLanguageManifest(
        language_config.MANIFEST_PATH,
        language_config.MANIFEST_SHA256,
        language_config.EXPECTED_DATASET,
        language_config.EXPECTED_SEQUENCE_COUNT)
    language = manifest.language_for(sequence_name)
    tracker = tracker_info.create_tracker(params)

    first_image = get_rgbd_frame(
        str(rgb_paths[0]), str(depth_paths[0]),
        depth_clip=True)
    tracker.initialize(first_image, {
        'init_bbox': initial_bbox(sequence_root / 'groundtruth.txt'),
        'sequence_name': sequence_name,
        'depth_path': str(depth_paths[0]),
        'init_nlp': language,
    })

    reason_counts = Counter()
    replace_frames = []
    drop_frames = []
    checked_frames = []
    last_output = None
    for frame_number, (rgb_path, depth_path) in enumerate(
            zip(rgb_paths[1:], depth_paths[1:]), start=2):
        image = get_rgbd_frame(
            str(rgb_path), str(depth_path),
            depth_clip=True)
        last_output = tracker.track(image, {'depth_path': str(depth_path)})
        bbox = [float(value) for value in last_output['target_bbox']]
        confidence = float(last_output['best_score'].detach().max().cpu().item())
        if (len(bbox) != 4 or not all(math.isfinite(value) for value in bbox) or
                bbox[2] <= 0.0 or bbox[3] <= 0.0 or
                not math.isfinite(confidence)):
            raise ValueError('Non-finite tracker output at frame {}'.format(frame_number))
        decision = last_output['safe_template_decision']
        reason_counts.update(decision.reasons)
        if decision.checked:
            checked_frames.append(frame_number)
        if decision.replace_dynamic:
            replace_frames.append(frame_number)
        if decision.drop_dynamic:
            drop_frames.append(frame_number)

    config_path = Path('/home/SUTrack_RGBD_L/experiments/sutrack') / (
        args.config + '.yaml')
    repository_root = Path('/home/SUTrack_RGBD_L')
    receipt = {
        'schema': 'sutrack_rgbd_language_safe_template_smoke_v1',
        'sequence': sequence_name,
        'frames_consumed': args.frames,
        'ground_truth_consumption': 'first_frame_initialization_only',
        'config_path': str(config_path),
        'config_sha256': sha256_file(config_path),
        'checkpoint_path': params.checkpoint,
        'checkpoint_sha256': sha256_file(params.checkpoint),
        'language_manifest_path': manifest.path,
        'language_manifest_sha256': manifest.sha256,
        'language_sha256': hashlib.sha256(language.encode('utf-8')).hexdigest(),
        'implementation_sha256': {
            relative_path: sha256_file(repository_root / relative_path)
            for relative_path in IMPLEMENTATION_FILES
        },
        'safe_template': {
            'checked_frames': checked_frames,
            'replace_frames': replace_frames,
            'drop_frames': drop_frames,
            'reason_counts': dict(sorted(reason_counts.items())),
            'dynamic_active_at_end': bool(tracker.safe_template_policy.dynamic_active),
        },
        'final_bbox': [float(value) for value in last_output['target_bbox']],
        'final_confidence': float(
            last_output['best_score'].detach().max().cpu().item()),
        'cuda_device': torch.cuda.get_device_name(torch.cuda.current_device()),
    }
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
