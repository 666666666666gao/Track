"""VOT RGB-D adapter that binds the exact multi-start anchor to the tracker."""

import sys

import cv2
import torch

from lib.test.tracker.rgbd_frame import get_rgbd_frame
from lib.test.vot.sutrack_class import (
    SUTrack as BaseSUTrack,
    _anchor_index,
    _frame_paths,
    _sequence_name,
)
import lib.test.vot.vot as vot


class TransactionSUTrack(BaseSUTrack):
    """Pass the VOT anchor index through to identity and trace state."""

    def __init__(self, tracker_name, para_name):
        super().__init__(tracker_name=tracker_name, para_name=para_name)
        language_config = self.tracker.cfg.TEST.RGBD_LANGUAGE
        if not (language_config.USE and
                language_config.ANCHOR_SPECIFIC):
            raise ValueError(
                'Transaction VOT adapter requires anchor-specific language')

    def initialize(
            self, img_rgb, selection, sequence_name, depth_path=None,
            anchor_index=None):
        if (isinstance(anchor_index, bool) or
                not isinstance(anchor_index, int) or anchor_index < 0):
            raise ValueError('Transaction VOT anchor index is malformed')
        x, y, width, height = selection
        bbox = [x, y, width, height]
        self.H, self.W, _ = img_rgb.shape
        init_info = {
            'init_bbox': bbox,
            'sequence_name': sequence_name,
            'depth_path': depth_path,
            'anchor_index': anchor_index,
        }
        init_info['init_nlp'] = self.language_manifest.language_for(
            sequence_name, anchor_index)
        return self.tracker.initialize(img_rgb, init_info)


def run_vot_exp(
        tracker_name, para_name, vis=False, out_conf=False,
        channel_type='rgbd'):
    if vis:
        raise ValueError('Transaction VOT adapter requires vis=False')
    torch.set_num_threads(1)
    tracker = TransactionSUTrack(tracker_name, para_name)
    handle = vot.VOT('rectangle', channels=channel_type)
    selection = handle.region()
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)

    rgb_path, depth_path = _frame_paths(imagefile)
    sequence_name = _sequence_name(rgb_path)
    if depth_path is not None:
        image = get_rgbd_frame(rgb_path, depth_path, depth_clip=True)
    else:
        image = cv2.cvtColor(
            cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    tracker.initialize(
        image, selection, sequence_name, depth_path,
        anchor_index=_anchor_index(rgb_path))

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        rgb_path, depth_path = _frame_paths(imagefile)
        if depth_path is not None:
            image = get_rgbd_frame(rgb_path, depth_path, depth_clip=True)
        else:
            image = cv2.cvtColor(
                cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        bbox, max_score = tracker.track(image, depth_path)
        if out_conf:
            handle.report(vot.Rectangle(*bbox), max_score)
        else:
            handle.report(vot.Rectangle(*bbox))


__all__ = ['TransactionSUTrack', 'run_vot_exp']
