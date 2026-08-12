#!/usr/bin/env python3
import argparse
import contextlib
import json
import os
import sys

import numpy as np
import torch
from trax import Image, ImageChannel, Rectangle, Region, Server, TraxStatus


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from lib.test.parameter.mplt_track import parameters
from lib.test.tracker.mplt_track import MPLTTrack
from lib.utils.rgbd_language import load_language_annotations
from lib.utils.depth_utils import get_rgbd_frame, read_rgb_image


def _image_path(images, channel):
    image = images.get(channel)
    if image is None:
        return None
    if image.type() != Image.PATH:
        raise RuntimeError('Only TraX path images are supported, got {}'.format(image.type()))
    return image.path()


def _combine_rgbd(images, target_box=None, depth_preprocess='rgbcolormap'):
    rgb_path = _image_path(images, ImageChannel.COLOR)
    depth_path = _image_path(images, ImageChannel.DEPTH)
    if rgb_path is None:
        raise RuntimeError('VOT frame does not contain a color channel')
    if depth_path is None:
        rgb = read_rgb_image(rgb_path)
        depth = np.zeros_like(rgb)
        frame = np.concatenate([rgb, depth], axis=2)
    else:
        frame = get_rgbd_frame(
            rgb_path, depth_path, dtype='rgbcolormap', depth_clip=True,
            target_box=target_box, depth_preprocess=depth_preprocess)
    return frame, rgb_path


def _region_to_bbox(region):
    if region.type == Region.RECTANGLE:
        return list(region.bounds())
    if region.type == Region.POLYGON:
        points = np.asarray(list(region), dtype=np.float32)
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
    raise RuntimeError('Unsupported initialization region type: {}'.format(region.type))


def _infer_sequence_name(rgb_path):
    parent = os.path.basename(os.path.dirname(rgb_path))
    if parent.lower() in ('color', 'rgb', 'visible', 'images', 'img'):
        return os.path.basename(os.path.dirname(os.path.dirname(rgb_path)))
    return parent


def _annotation_paths(args):
    paths = []
    if args.lang_jsonl:
        paths.append(args.lang_jsonl)
    env_path = os.environ.get('MPLT_LANG_JSONL', '')
    if env_path:
        paths.extend(env_path.split(os.pathsep))
    paths.extend([
        os.path.join(PROJECT_ROOT, 'annotations_cleaned', 'votrgbd2022_language.jsonl'),
        os.path.join(PROJECT_ROOT, 'annotations_cleaned', 'cdtb_language.jsonl'),
        os.path.join(PROJECT_ROOT, 'annotations', 'vot_rgbd2022_first_qwen3_corrected.jsonl'),
        os.path.join(PROJECT_ROOT, 'annotations', 'depthtrack_test_first_qwen3_corrected.jsonl'),
    ])
    return [p for p in paths if p and os.path.isfile(p)]


def _language_init_fields(language_meta, sequence_name):
    fields = {}
    meta = language_meta.get(sequence_name, {})
    for key, value in meta.items():
        if key.startswith('language_'):
            fields['init_{}'.format(key)] = value
    return fields


def _build_tracker(args):
    config = args.config or os.environ.get(
        'MPLT_CONFIG', 'vitb_256_mplt_32x1_1e4_depthtrack_15ep_roberta')
    epoch = int(args.epoch or os.environ.get('MPLT_EPOCH', '10'))
    checkpoint_root = args.checkpoint_root or os.environ.get(
        'MPLT_CHECKPOINT_ROOT', os.path.join(PROJECT_ROOT, 'output', 'depthtrack_roberta'))

    log_dir = os.path.join(PROJECT_ROOT, 'output', 'vot_logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'vot_mplt_{}_ep{:04d}.log'.format(config, epoch))
    with open(log_path, 'a') as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            params = parameters(config)
            params.debug = 0
            params.save_all_boxes = False
            params.checkpoint = os.path.join(
                checkpoint_root,
                'checkpoints/train/mplt_track/{}/MPLTTrack_ep{:04d}.pth.tar'.format(config, epoch))
            if not os.path.isfile(params.checkpoint):
                raise FileNotFoundError(params.checkpoint)
            tracker = MPLTTrack(params, args.dataset_name)

    return tracker, config, epoch


def main():
    parser = argparse.ArgumentParser(description='TraX wrapper for MPLT RGB-D/RGB-D-L tracker.')
    parser.add_argument('--config', default='')
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--checkpoint-root', default='')
    parser.add_argument('--dataset-name', default=os.environ.get('MPLT_DATASET_NAME', 'votrgbd2022'))
    parser.add_argument('--lang-jsonl', default='')
    args = parser.parse_args()

    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    torch.set_grad_enabled(False)

    tracker = None
    previous_image = None
    previous_bbox = None
    depth_preprocess = 'rgbcolormap'
    language_meta = load_language_annotations(paths=_annotation_paths(args), rag_roots=[])

    with Server(
        [Region.RECTANGLE],
        [Image.PATH],
        image_channels=[ImageChannel.COLOR, ImageChannel.DEPTH],
        tracker_name='DaLa-MPLT-RGBD-L',
        tracker_description='MPLT RGB-D tracker with optional RoBERTa language guidance',
        tracker_family='mplt',
        metadata={'vot': 'python'},
    ) as server:
        while True:
            request = server.wait()
            if request.type in (TraxStatus.QUIT, TraxStatus.ERROR):
                break

            if request.type == TraxStatus.INITIALIZE:
                tracker, _, _ = _build_tracker(args)
                bbox = _region_to_bbox(request.objects[0][0])
                cfg = getattr(tracker.params, 'cfg', None)
                data_cfg = getattr(cfg, 'DATA', None)
                depth_preprocess = getattr(data_cfg, 'DEPTH_PREPROCESS', 'rgbcolormap')
                current_image, rgb_path = _combine_rgbd(
                    request.image, target_box=bbox, depth_preprocess=depth_preprocess)
                previous_image = current_image
                previous_bbox = bbox
                sequence_name = _infer_sequence_name(rgb_path)
                init_info = {'init_bbox': bbox}
                init_info.update(_language_init_fields(language_meta, sequence_name))
                tracker.initialize([current_image, current_image], init_info)
                server.status([(Rectangle.create(*bbox), {'confidence': '1.0'})])
                continue

            if request.type == TraxStatus.FRAME:
                if tracker is None or previous_image is None:
                    raise RuntimeError('Received frame before initialization')
                current_image, _ = _combine_rgbd(
                    request.image, target_box=previous_bbox, depth_preprocess=depth_preprocess)
                output = tracker.track([previous_image, current_image], {})
                previous_image = current_image
                bbox = [float(v) for v in output['target_bbox']]
                previous_bbox = bbox
                confidence = float(output.get('best_score', 1.0))
                server.status([(Rectangle.create(*bbox), {'confidence': '{:.6f}'.format(confidence)})])


if __name__ == '__main__':
    main()
