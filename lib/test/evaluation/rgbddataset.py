import os
import cv2
import numpy as np

from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
from lib.utils.rgbd_language import load_language_annotations, sequence_list_from_jsonl


class RGBDDataset(BaseDataset):
    """Generic RGB-D benchmark dataset with optional language annotations.

    It supports DepthTrack, CDTB and VOT-RGBD style directories:
      root/<sequence>/color/*.jpg, root/<sequence>/depth/*.png, root/<sequence>/groundtruth.txt
      root/sequences/<sequence>/color/*.jpg, root/sequences/<sequence>/depth/*.png, ...
    """

    RGB_DIR_CANDIDATES = ('color', 'rgb', 'visible', 'img', 'imgs', 'image', 'images')
    DEPTH_DIR_CANDIDATES = ('depth', 'depths', 'depth_colormap', 'depth_color', 'infrared')
    GT_FILE_CANDIDATES = ('groundtruth.txt', 'groundtruth_rect.txt', 'init.txt', 'rgb.txt')
    SPLIT_FILE_CANDIDATES = ('test.txt', 'testlist.txt', 'testing.txt', 'testingsetList.txt', 'val.txt')
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

    def __init__(self, dataset_name='depthtrack', root_attr='depthtrack_path', split='test',
                 lang_jsonl_attr='depthtrack_test_lang_path', lang_jsonl_name='depthtrack_test_language.jsonl',
                 rag_root_attr='', rag_root_name=''):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.base_path = self._get_base_path(root_attr)
        self.data_root = self._resolve_data_root(self.base_path, split)
        self.lang_jsonl = self._get_lang_jsonl(lang_jsonl_attr, lang_jsonl_name)
        self.rag_root = self._get_rag_root(rag_root_attr, rag_root_name)
        self.language_meta = load_language_annotations(
            paths=[self.lang_jsonl] if self.lang_jsonl else [],
            rag_roots=[self.rag_root] if self.rag_root else [])
        self.sequence_list = self._get_sequence_list()

    @staticmethod
    def _project_annotations_dir():
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'annotations_cleaned'))

    def _get_base_path(self, root_attr):
        path = getattr(self.env_settings, root_attr, '')
        if not path and root_attr.endswith('_path'):
            path = getattr(self.env_settings, root_attr.replace('_path', '_dir'), '')
        if not path:
            path = os.path.join(getattr(self.env_settings, 'prj_dir', ''), 'data', self.dataset_name)
        return path

    @staticmethod
    def _resolve_data_root(root, split=None):
        if split and root:
            split_root = os.path.join(root, split)
            if os.path.isdir(os.path.join(split_root, 'sequences')):
                return os.path.join(split_root, 'sequences')
            if os.path.isdir(split_root):
                return split_root
        if root and os.path.isdir(os.path.join(root, 'sequences')):
            return os.path.join(root, 'sequences')
        return root

    def _get_lang_jsonl(self, attr, filename):
        for name in (attr, '{}_lang_path'.format(self.dataset_name)):
            if not name:
                continue
            path = getattr(self.env_settings, name, '')
            if path and os.path.isfile(path):
                return path
        candidates = [
            os.path.join(self._project_annotations_dir(), filename),
            os.path.join(self.base_path, 'annotations', filename),
            os.path.join(os.path.dirname(self.base_path), 'annotations', filename),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return ''

    def _get_rag_root(self, attr, dirname):
        for name in (attr, '{}_lang_root'.format(self.dataset_name)):
            if not name:
                continue
            path = getattr(self.env_settings, name, '')
            if path and os.path.isdir(path):
                return path
        candidates = [
            os.path.join(self._project_annotations_dir(), dirname) if dirname else '',
            os.path.join(self.base_path, 'annotations', dirname) if dirname else '',
            os.path.join(os.path.dirname(self.base_path), 'annotations', dirname) if dirname else '',
        ]
        for path in candidates:
            if path and os.path.isdir(path):
                return path
        return ''

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        seq_path = self._get_sequence_path(sequence_name)
        anno_path = self._find_gt_file(seq_path)
        if anno_path is None:
            raise FileNotFoundError('Cannot find ground-truth file under {}'.format(seq_path))
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', '\t', ' '), dtype=np.float64)

        rgb_dir = self._find_dir(seq_path, self.RGB_DIR_CANDIDATES)
        depth_dir = self._find_dir(seq_path, self.DEPTH_DIR_CANDIDATES)
        frames_list_v = self._list_frames(rgb_dir)
        frames_list_d = self._list_frames(depth_dir)
        frames_list = [frames_list_v, frames_list_d]
        object_class = sequence_name.split('_')[0]
        init_data = {0: {'bbox': ground_truth_rect.reshape(-1, 4)[0]}}
        if sequence_name in self.language_meta:
            meta = self.language_meta[sequence_name]
            object_class = meta.get('object_class_name', object_class)
            for key, value in meta.items():
                if key.startswith('language_'):
                    init_data[0][key] = value
        return Sequence(sequence_name, frames_list, self.dataset_name, ground_truth_rect.reshape(-1, 4),
                        init_data=init_data, object_class=object_class)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        for base in (self.base_path, self.data_root):
            for filename in self.SPLIT_FILE_CANDIDATES:
                path = os.path.join(base, filename)
                if os.path.isfile(path):
                    with open(path, 'r') as f:
                        return [line.strip().split(',')[0] for line in f if line.strip()]
        if self.lang_jsonl and os.path.isfile(self.lang_jsonl):
            seqs = sequence_list_from_jsonl(self.lang_jsonl)
            if seqs:
                return seqs
        if self.language_meta:
            return list(self.language_meta.keys())
        if not os.path.isdir(self.data_root):
            return []
        return [name for name in sorted(os.listdir(self.data_root))
                if os.path.isdir(os.path.join(self.data_root, name))]

    def _get_sequence_path(self, sequence_name):
        for base in (self.data_root, self.base_path, os.path.join(self.base_path, 'sequences')):
            path = os.path.join(base, sequence_name)
            if os.path.isdir(path):
                return path
        return os.path.join(self.data_root, sequence_name)

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


class DepthTrackEvalDataset(RGBDDataset):
    def __init__(self, split='test'):
        super().__init__(dataset_name='depthtrack', root_attr='depthtrack_path', split=split,
                         lang_jsonl_attr='depthtrack_test_lang_path',
                         lang_jsonl_name='depthtrack_test_language.jsonl')


class CDTBEvalDataset(RGBDDataset):
    def __init__(self, split='test'):
        super().__init__(dataset_name='cdtb', root_attr='cdtb_path', split=split,
                         lang_jsonl_attr='cdtb_lang_path',
                         lang_jsonl_name='cdtb_language.jsonl',
                         rag_root_attr='cdtb_lang_root',
                         rag_root_name='CDTBLang_Qwen3_RAGStyle')


class VOTRGBD2022EvalDataset(RGBDDataset):
    def __init__(self, split='test'):
        super().__init__(dataset_name='votrgbd2022', root_attr='votrgbd2022_path', split=split,
                         lang_jsonl_attr='votrgbd2022_lang_path',
                         lang_jsonl_name='votrgbd2022_language.jsonl',
                         rag_root_attr='votrgbd2022_lang_root',
                         rag_root_name='VOTRGBD2022Lang_Qwen3_RAGStyle')
