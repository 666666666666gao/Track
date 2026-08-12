import os
import csv
import random
from collections import OrderedDict

import cv2
import numpy as np
import pandas
import torch

from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings
from lib.utils.depth_utils import get_rgbd_frame, read_rgb_image
from lib.utils.rgbd_language import load_language_annotations, sequence_list_from_jsonl


class DepthTrack(BaseVideoDataset):
    """DepthTrack RGB-D loader with optional first-frame language annotations.

    Supported sequence layouts:
      root/<sequence>/color/*.jpg + root/<sequence>/depth/*.png
      root/sequences/<sequence>/color/*.jpg + root/sequences/<sequence>/depth/*.png

    Supported language annotation layouts:
      annotations_cleaned/depthtrack_train_language.jsonl
      annotations_cleaned/depthtrack_test_language.jsonl

    The language is stored in obj_meta and propagated by the sampler. The RGB-D
    baseline still trains without using it, while RGB-D-L variants can consume
    `language_description`, `language_appearance`, `language_depth_relation`, etc.
    """

    RGB_DIR_CANDIDATES = ('color', 'rgb', 'visible', 'img', 'imgs', 'image', 'images')
    DEPTH_DIR_CANDIDATES = ('depth', 'depths', 'depth_colormap', 'depth_color', 'infrared')
    GT_FILE_CANDIDATES = ('groundtruth.txt', 'groundtruth_rect.txt', 'init.txt', 'rgb.txt')
    SPLIT_FILE_CANDIDATES = {
        'train': ('train.txt', 'trainlist.txt', 'training.txt', 'trainingsetList.txt'),
        'test': ('test.txt', 'testlist.txt', 'testing.txt', 'testingsetList.txt'),
        'val': ('val.txt', 'vallist.txt', 'validation.txt'),
    }
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

    def __init__(self, root=None, image_loader=None, split='train', seq_ids=None, data_fraction=None,
                 lang_jsonl=None, lang_root=None, dtype='rgbcolormap', depth_preprocess='rgbcolormap'):
        settings = env_settings()
        if root is None:
            root = getattr(settings, 'depthtrack_dir', '')
        super().__init__('DepthTrack', root, image_loader)
        self.split = split or 'train'
        self.dtype = dtype
        self.depth_preprocess = depth_preprocess
        self.data_root = self._resolve_data_root(self.root)
        self.lang_jsonl = lang_jsonl or self._default_lang_jsonl(settings, self.split)
        self.lang_root = lang_root or getattr(settings, 'depthtrack_lang_root', '')
        self.language_meta = load_language_annotations(
            paths=[self.lang_jsonl] if self.lang_jsonl else [],
            rag_roots=[self.lang_root] if self.lang_root else [])
        self.sequence_list = self._get_sequence_list(self.split)

        if seq_ids is not None:
            self.sequence_list = [self.sequence_list[i] for i in seq_ids]
        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

        self.sequence_meta_info = self._load_meta_info()
        self.seq_per_class = self._build_seq_per_class()
        self.class_list = sorted(list(self.seq_per_class.keys()))

    def get_name(self):
        return 'depthtrack'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    @staticmethod
    def _project_annotations_dir():
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'annotations_cleaned'))

    def _default_lang_jsonl(self, settings, split):
        attr_names = [
            'depthtrack_{}_lang_path'.format(split),
            'depthtrack_lang_path',
        ]
        for attr in attr_names:
            path = getattr(settings, attr, '')
            if path and os.path.isfile(path):
                return path
        filename = 'depthtrack_{}_language.jsonl'.format('test' if split in ('test', 'val') else 'train')
        candidates = [
            os.path.join(self._project_annotations_dir(), filename),
            os.path.join(self.root, 'annotations', filename),
            os.path.join(os.path.dirname(self.root), 'annotations', filename),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return ''

    @staticmethod
    def _resolve_data_root(root):
        if root and os.path.isdir(os.path.join(root, 'sequences')):
            return os.path.join(root, 'sequences')
        return root

    def _load_meta_info(self):
        return {s: self._read_meta(self._get_sequence_path_by_name(s), s) for s in self.sequence_list}

    def _read_meta(self, seq_path, seq_name):
        class_name = seq_name.split('_')[0]
        meta_file = os.path.join(seq_path, 'meta_info.ini')
        if os.path.isfile(meta_file):
            try:
                with open(meta_file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    low = line.lower()
                    if 'object' in low and ':' in line:
                        class_name = line.split(':', 1)[-1].strip()
                        break
            except Exception:
                pass
        meta = OrderedDict({'object_class_name': class_name})
        if seq_name in self.language_meta:
            meta.update(self.language_meta[seq_name])
        return meta

    def _build_seq_per_class(self):
        seq_per_class = {}
        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s].get('object_class_name', None)
            seq_per_class.setdefault(object_class, []).append(i)
        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self, split):
        candidates = self.SPLIT_FILE_CANDIDATES.get(split, ())
        for base in (self.root, self.data_root):
            for filename in candidates:
                path = os.path.join(base, filename)
                if os.path.isfile(path):
                    with open(path, 'r') as f:
                        return [line.strip().split(',')[0] for line in f if line.strip()]

        # If a corrected annotation file is provided, use it as the canonical sequence list.
        if self.lang_jsonl and os.path.isfile(self.lang_jsonl):
            seqs = sequence_list_from_jsonl(self.lang_jsonl)
            if seqs:
                return seqs

        # Fallback: every directory containing a ground-truth file is treated as a sequence.
        seqs = []
        if os.path.isdir(self.data_root):
            for name in sorted(os.listdir(self.data_root)):
                seq_path = os.path.join(self.data_root, name)
                if not os.path.isdir(seq_path):
                    continue
                if self._find_gt_file(seq_path) is not None:
                    seqs.append(name)
        return seqs

    def _find_dir(self, seq_path, candidates):
        for dirname in candidates:
            path = os.path.join(seq_path, dirname)
            if os.path.isdir(path):
                return path
        raise FileNotFoundError('Cannot find any of {} under {}'.format(candidates, seq_path))

    def _find_gt_file(self, seq_path):
        for filename in self.GT_FILE_CANDIDATES:
            path = os.path.join(seq_path, filename)
            if os.path.isfile(path):
                return path
        return None

    def _list_frames(self, frame_dir):
        frames = [f for f in os.listdir(frame_dir) if os.path.splitext(f)[1].lower() in self.IMG_EXTENSIONS]
        frames.sort(key=self._natural_key)
        return [os.path.join(frame_dir, f) for f in frames]

    @staticmethod
    def _natural_key(name):
        stem = os.path.splitext(os.path.basename(name))[0]
        digits = ''.join(ch if ch.isdigit() else ' ' for ch in stem).split()
        return int(digits[-1]) if digits else stem

    def _read_bb_anno(self, seq_path):
        gt_file = self._find_gt_file(seq_path)
        if gt_file is None:
            raise FileNotFoundError('Cannot find ground-truth file under {}'.format(seq_path))
        gt = pandas.read_csv(gt_file, sep=r'[,\t ]+', header=None, engine='python',
                             dtype=np.float32, na_filter=False).values
        return torch.tensor(gt[:, :4], dtype=torch.float32)

    def _read_target_visible(self, seq_path, num_frames):
        for filename in ('absence.label', 'absent.txt', 'absence.txt'):
            path = os.path.join(seq_path, filename)
            if os.path.isfile(path):
                with open(path, 'r', newline='') as f:
                    absence = torch.ByteTensor([int(float(v[0])) for v in csv.reader(f) if v])
                visible = (~absence.bool()).byte()
                return visible[:num_frames], visible[:num_frames].float()
        visible = torch.ByteTensor([1 for _ in range(num_frames)])
        return visible, visible.float()

    def _get_sequence_path_by_name(self, seq_name):
        for base in (self.data_root, self.root, os.path.join(self.root, 'sequences')):
            path = os.path.join(base, seq_name)
            if os.path.isdir(path):
                return path
        return os.path.join(self.data_root, seq_name)

    def _get_sequence_path(self, seq_id):
        return self._get_sequence_path_by_name(self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible, visible_ratio = self._read_target_visible(seq_path, len(bbox))
        visible = visible & valid.byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}

    def _get_frame_path(self, seq_path, frame_id):
        rgb_dir = self._find_dir(seq_path, self.RGB_DIR_CANDIDATES)
        depth_dir = self._find_dir(seq_path, self.DEPTH_DIR_CANDIDATES)
        rgb_frames = self._list_frames(rgb_dir)
        depth_frames = self._list_frames(depth_dir)
        return rgb_frames[frame_id], depth_frames[frame_id]

    @staticmethod
    def _read_rgb(path):
        return read_rgb_image(path)

    @staticmethod
    def _read_depth(path):
        return get_rgbd_frame(None, path, dtype='colormap', depth_clip=True)

    def _get_frame(self, seq_path, frame_id, target_box=None):
        rgb_path, depth_path = self._get_frame_path(seq_path, frame_id)
        return get_rgbd_frame(rgb_path, depth_path, dtype=self.dtype, depth_clip=True,
                              target_box=target_box, depth_preprocess=self.depth_preprocess)

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
        return obj_meta.get('object_class_name', None)

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        bbox = anno.get('bbox', None)
        target_boxes = [bbox[f_id, ...].tolist() if bbox is not None else None for f_id in frame_ids]
        frame_list = [self._get_frame(seq_path, f_id, target_box=target_box)
                      for f_id, target_box in zip(frame_ids, target_boxes)]

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames, obj_meta
