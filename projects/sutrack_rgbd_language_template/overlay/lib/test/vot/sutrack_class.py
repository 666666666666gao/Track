from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pdb
import cv2
import torch
# import vot
import sys
import time
import os
from pathlib import Path
from lib.test.evaluation import Tracker
import lib.test.vot.vot as vot
from lib.test.vot.vot_utils import *
from lib.test.tracker.rgbd_language_manifest import (
    RGBDAnchorLanguageManifest,
    RGBDLanguageManifest,
)
from lib.test.tracker.rgbd_frame import get_rgbd_frame


class SUTrack(object):
    def __init__(self, tracker_name='sutrack', para_name='sutrack_b224'):
        # create tracker
        tracker_info = Tracker(tracker_name, para_name, "depthtrack", None)
        params = tracker_info.get_parameters()
        params.visualization = False
        params.debug = False
        language_config = params.cfg.TEST.RGBD_LANGUAGE
        self.language_manifest = None
        self.anchor_specific_language = False
        if language_config.USE:
            self.anchor_specific_language = bool(
                getattr(language_config, 'ANCHOR_SPECIFIC', False))
            if self.anchor_specific_language:
                self.language_manifest = RGBDAnchorLanguageManifest(
                    language_config.MANIFEST_PATH,
                    language_config.MANIFEST_SHA256,
                    language_config.EXPECTED_DATASET,
                    language_config.EXPECTED_RECORD_COUNT)
            else:
                self.language_manifest = RGBDLanguageManifest(
                    language_config.MANIFEST_PATH,
                    language_config.MANIFEST_SHA256,
                    language_config.EXPECTED_DATASET,
                    language_config.EXPECTED_SEQUENCE_COUNT)
        self.tracker = tracker_info.create_tracker(params)

    def write(self, str):
        txt_path = ""
        file = open(txt_path, 'a')
        file.write(str)

    def initialize(self, img_rgb, selection, sequence_name, depth_path=None,
                   anchor_index=None):
        # init on the 1st frame
        # region = rect_from_mask(mask)
        x, y, w, h = selection
        bbox = [x,y,w,h]
        self.H, self.W, _ = img_rgb.shape
        init_info = {
            'init_bbox': bbox,
            'sequence_name': sequence_name,
            'depth_path': depth_path,
        }
        if self.language_manifest is not None:
            if self.anchor_specific_language:
                init_info['init_nlp'] = self.language_manifest.language_for(
                    sequence_name, anchor_index)
            else:
                init_info['init_nlp'] = self.language_manifest.language_for(
                    sequence_name)
        _ = self.tracker.initialize(img_rgb, init_info)

    def track(self, img_rgb, depth_path=None):
        # track
        outputs = self.tracker.track(img_rgb, {'depth_path': depth_path})
        pred_bbox = outputs['target_bbox']
        best_score = outputs['best_score']
        if torch.is_tensor(best_score):
            max_score = float(best_score.detach().max().cpu().item())
        else:
            max_score = float(best_score)
        return pred_bbox, max_score


def _frame_paths(imagefile):
    if isinstance(imagefile, (list, tuple)):
        if len(imagefile) != 2:
            raise ValueError('RGB-D VOT frames must contain exactly RGB and depth paths')
        return os.fspath(imagefile[0]), os.fspath(imagefile[1])
    return os.fspath(imagefile), None


def _sequence_name(rgb_path):
    path = Path(rgb_path)
    if path.parent.name.lower() in ('color', 'rgb'):
        sequence_name = path.parent.parent.name
    else:
        sequence_name = path.parent.name
    if not sequence_name:
        raise ValueError('Unable to derive sequence name from {}'.format(rgb_path))
    return sequence_name


def _anchor_index(rgb_path):
    """Map VOT's one-based RGB filename to its zero-based anchor index."""
    stem = Path(rgb_path).stem
    if not stem.isdigit():
        raise ValueError(
            'Unable to derive numeric VOT anchor index from {}'.format(rgb_path))
    frame_number = int(stem)
    if frame_number <= 0:
        raise ValueError('VOT RGB frame numbers must be one-based')
    return frame_number - 1


def run_vot_exp(tracker_name, para_name, vis=False, out_conf=False, channel_type='color'):

    torch.set_num_threads(1)
    save_root = os.path.join('', para_name)
    if vis and (not os.path.exists(save_root)):
        os.mkdir(save_root)
    tracker = SUTrack(tracker_name=tracker_name, para_name=para_name)

    if channel_type=='rgb':
        channel_type=None
    handle = vot.VOT("rectangle", channels=channel_type)

    selection = handle.region()
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)
    if vis:
        '''for vis'''
        rgb_path, _ = _frame_paths(imagefile)
        seq_name = _sequence_name(rgb_path)
        save_v_dir = os.path.join(save_root,seq_name)
        if not os.path.exists(save_v_dir):
            os.mkdir(save_v_dir)
        cur_time = int(time.time() % 10000)
        save_dir = os.path.join(save_v_dir, str(cur_time))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # read rgbd data
    rgb_path, depth_path = _frame_paths(imagefile)
    sequence_name = _sequence_name(rgb_path)
    if depth_path is not None:
        image = get_rgbd_frame(rgb_path, depth_path, depth_clip=True)
    else:
        image = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB) # Right

    tracker.initialize(
        image, selection, sequence_name, depth_path,
        anchor_index=_anchor_index(rgb_path))

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break

        # read rgbd data
        rgb_path, depth_path = _frame_paths(imagefile)
        if depth_path is not None:
            image = get_rgbd_frame(rgb_path, depth_path, depth_clip=True)
        else:
            image = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)  # Right

        b1, max_score = tracker.track(image, depth_path)

        if out_conf:
            handle.report(vot.Rectangle(*b1), max_score)
        else:
            handle.report(vot.Rectangle(*b1))
        if vis:
            '''Visualization'''
            # original image
            image_ori = image[:,:,::-1].copy() # RGB --> BGR
            image_name = os.path.basename(rgb_path)
            save_path = os.path.join(save_dir, image_name)
            image_b = image_ori.copy()
            cv2.rectangle(image_b, (int(b1[0]), int(b1[1])),
                          (int(b1[0] + b1[2]), int(b1[1] + b1[3])), (0, 0, 255), 2)
            image_b_name = image_name.replace('.jpg','_bbox.jpg')
            save_path = os.path.join(save_dir, image_b_name)
            cv2.imwrite(save_path, image_b)
