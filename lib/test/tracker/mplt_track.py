import math

from lib.models.mplt_track import build_mplt_track
from lib.test.tracker.basetracker import BaseTracker
import torch
import numpy as np

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.test.utils.tracker_utils import vis_attn_maps
from lib.train.data.processing_utils import sample_target
from scipy.ndimage import gaussian_filter
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class MPLTTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(MPLTTrack, self).__init__(params)
        network = build_mplt_track(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.language_info = {}
        self.velocity = [0.0, 0.0]
        self.stable_frames = 0
        self.last_update_frame = 0
        self.lost_frames = 0
        self.template_signature = None
        self.template_rgb_signature = None
        self.template_depth_signature = None
        self.template_rgb_spatial_signature = None
        self.template_depth_spatial_signature = None
        self.last_template_consistency = 1.0
        self.last_template_rgb_consistency = 1.0
        self.last_template_depth_consistency = 1.0
        self.template_consistency_ema = 1.0
        self.template_rgb_consistency_ema = 1.0
        self.template_depth_consistency_ema = 1.0
        self.last_template_spatial_rgb_consistency = 1.0
        self.last_template_spatial_depth_consistency = 1.0
        self.last_update_ratio = 1.0
        self.score_ema = None
        self.last_effective_lost_thr = 0.0
        self.last_effective_damping_thr = 0.0
        self.last_response_entropy = 0.0
        self.last_response_peak_ratio = 0.0
        self.last_response_center_distance = 0.0
        self.last_response_guard = 0.0
        self.last_depth_guard = 0.0
        self.last_center_follow = 0.0
        self.last_window_penalty = 0.0
        self.last_search_factor = 0.0
        self.last_confident_mismatch_search = 0.0
        self.last_candidate_rerank = 0.0
        self.last_candidate_rank = 0.0
        self.last_candidate_score_gain = 0.0
        self.last_candidate_consistency_gain = 0.0
        self.last_candidate_selected_score_ratio = 0.0
        self.last_candidate_selected_rgb_consistency = 0.0
        self.last_candidate_selected_depth_consistency = 0.0
        self.last_candidate_selected_motion_score = 0.0
        self.last_candidate_selected_iou = -1.0
        self.last_candidate_oracle_iou = -1.0
        self.last_candidate_oracle_rank = -1.0
        self.last_candidate_oracle_score_ratio = 0.0
        self.last_candidate_oracle_rgb_consistency = 0.0
        self.last_candidate_oracle_depth_consistency = 0.0
        self.last_candidate_oracle_consistency = 0.0
        self.last_candidate_oracle_motion_score = 0.0
        self.candidate_rerank_streak = 0
        self.last_candidate_probation = 0.0
        self.last_candidate_recovery_trigger = 0.0
        self.last_candidate_stable_block = 0.0
        self.last_candidate_ordinary_trigger = 0.0
        self.last_language_candidate_rerank = 0.0
        self.last_language_candidate_rank = 0.0
        self.last_language_candidate_best = 0.0
        self.last_language_candidate_selected = 0.0
        self.last_language_candidate_gain = 0.0
        self.last_template_update = 0.0
        self.last_template_update_reason = 0.0
        self.last_template_update_gate_score = 0.0
        self.last_template_update_gate_stable = 0.0
        self.last_template_update_gate_consistency = 0.0
        self.last_template_update_gate_rgb = 0.0
        self.last_template_update_gate_depth = 0.0
        self.last_language_runtime_gate = 1.0
        self.last_raw_score = 1.0
        self.last_score_calibration = 1.0
        self.last_raw_score = 1.0
        self.last_score_calibration = 1.0

    @staticmethod
    def _box_center(box):
        return [box[0] + 0.5 * box[2], box[1] + 0.5 * box[3]]

    @staticmethod
    def _box_iou_xywh(a, b):
        if a is None or b is None:
            return -1.0
        ax1, ay1, aw, ah = [float(v) for v in a]
        bx1, by1, bw, bh = [float(v) for v in b]
        if aw <= 0.0 or ah <= 0.0 or bw <= 0.0 or bh <= 0.0:
            return -1.0
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        if union <= 0.0:
            return -1.0
        return inter / union

    @staticmethod
    def _crop_box(image, box):
        h, w = image.shape[:2]
        x, y, bw, bh = [int(round(float(v))) for v in box]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + max(bw, 1), w), min(y + max(bh, 1), h)
        if x1 <= x0 or y1 <= y0:
            return None
        return image[y0:y1, x0:x1]

    @staticmethod
    def _hist_feature(values, bins=16):
        hist, _ = np.histogram(values.reshape(-1), bins=bins, range=(0, 255))
        hist = hist.astype(np.float32)
        denom = float(hist.sum())
        if denom <= 0.0:
            return hist
        return hist / denom

    @staticmethod
    def _normalize_signature(signature):
        if signature is None:
            return None
        norm = float(np.linalg.norm(signature))
        if norm <= 1e-6:
            return None
        return signature / norm

    @staticmethod
    def _spatial_feature(values, size):
        if values is None or values.size == 0:
            return None
        patch = cv2.resize(values.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
        patch = patch.reshape(-1).astype(np.float32)
        patch = patch - float(patch.mean())
        std = float(patch.std())
        if std <= 1e-6:
            return None
        patch = patch / std
        norm = float(np.linalg.norm(patch))
        if norm <= 1e-6:
            return None
        return patch / norm

    def _extract_spatial_signatures(self, image, box):
        crop = self._crop_box(image, box)
        if crop is None or crop.size == 0:
            return None, None
        size = int(getattr(self.cfg.TEST, "SPATIAL_CONSISTENCY_SIZE", 32))
        size = max(size, 8)
        rgb_signature = None
        if crop.shape[2] >= 3:
            rgb = cv2.cvtColor(crop[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
            rgb_signature = self._spatial_feature(rgb, size)
        depth_signature = None
        if crop.shape[2] >= 6:
            depth = crop[:, :, 3:6].astype(np.float32).mean(axis=2)
            depth_signature = self._spatial_feature(depth, size)
        return rgb_signature, depth_signature

    @staticmethod
    def _spatial_similarity(template_signature, current_signature):
        if template_signature is None or current_signature is None:
            return 1.0
        corr = float(np.dot(template_signature, current_signature))
        return min(max(0.5 * (corr + 1.0), 0.0), 1.0)

    def _extract_box_signatures(self, image, box):
        crop = self._crop_box(image, box)
        if crop is None or crop.size == 0:
            return None, None, None
        rgb_features = []
        rgb = crop[:, :, :3]
        for channel in range(3):
            rgb_features.append(self._hist_feature(rgb[:, :, channel]))
        rgb_signature = self._normalize_signature(np.concatenate(rgb_features, axis=0))
        depth_signature = None
        joint_parts = list(rgb_features)
        if crop.shape[2] >= 6:
            depth = crop[:, :, 3:6].astype(np.float32).mean(axis=2)
            depth_feature = self._hist_feature(depth)
            depth_signature = self._normalize_signature(depth_feature)
            joint_parts.append(depth_feature)
        joint_signature = self._normalize_signature(np.concatenate(joint_parts, axis=0))
        return joint_signature, rgb_signature, depth_signature

    def _extract_box_signature(self, image, box):
        signature, _, _ = self._extract_box_signatures(image, box)
        return signature

    def _template_consistency_parts(self, image, box, record=False):
        _, rgb_signature, depth_signature = self._extract_box_signatures(image, box)
        rgb_score = 1.0
        depth_score = 1.0
        if self.template_rgb_signature is not None and rgb_signature is not None:
            rgb_score = float(np.dot(self.template_rgb_signature, rgb_signature))
        if self.template_depth_signature is not None and depth_signature is not None:
            depth_score = float(np.dot(self.template_depth_signature, depth_signature))
        rgb_score = min(max(rgb_score, 0.0), 1.0)
        depth_score = min(max(depth_score, 0.0), 1.0)
        spatial_rgb_score = 1.0
        spatial_depth_score = 1.0
        if bool(getattr(self.cfg.TEST, "SPATIAL_CONSISTENCY", False)):
            rgb_spatial, depth_spatial = self._extract_spatial_signatures(image, box)
            spatial_rgb_score = self._spatial_similarity(self.template_rgb_spatial_signature, rgb_spatial)
            spatial_depth_score = self._spatial_similarity(self.template_depth_spatial_signature, depth_spatial)
            rgb_weight = min(max(float(getattr(self.cfg.TEST, "SPATIAL_CONSISTENCY_RGB_WEIGHT", 0.0)), 0.0), 1.0)
            depth_weight = min(max(float(getattr(self.cfg.TEST, "SPATIAL_CONSISTENCY_DEPTH_WEIGHT", 0.0)), 0.0), 1.0)
            rgb_score = (1.0 - rgb_weight) * rgb_score + rgb_weight * spatial_rgb_score
            depth_score = (1.0 - depth_weight) * depth_score + depth_weight * spatial_depth_score
        if record:
            self.last_template_spatial_rgb_consistency = spatial_rgb_score
            self.last_template_spatial_depth_consistency = spatial_depth_score
        return rgb_score, depth_score

    def _template_consistency(self, image, box):
        if self.template_signature is None or image is None:
            return 1.0
        signature = self._extract_box_signature(image, box)
        if signature is None:
            return 1.0
        score = float(np.dot(self.template_signature, signature))
        return min(max(score, 0.0), 1.0)

    def _apply_score_window(self, score_map):
        test_cfg = getattr(self.cfg, "TEST", None)
        mode = str(getattr(test_cfg, "WINDOW_PENALTY_MODE", "multiply")).lower()
        penalty = min(max(float(getattr(test_cfg, "WINDOW_PENALTY", 0.0)), 0.0), 1.0)
        self.last_window_penalty = penalty if mode in ("blend", "dtp", "dtp_blend") else 0.0
        if mode in ("blend", "dtp", "dtp_blend") and penalty > 0.0:
            return score_map * (1.0 - penalty) + self.output_window.view(1, 1, self.feat_sz, self.feat_sz) * penalty
        return self.output_window * score_map

    def _update_response_diagnostics(self, response):
        flat = response.detach().float().reshape(-1)
        if flat.numel() == 0:
            self.last_response_entropy = 0.0
            self.last_response_peak_ratio = 0.0
            self.last_response_center_distance = 0.0
            return
        values = flat - flat.min()
        denom = values.sum().clamp_min(1e-6)
        prob = values / denom
        entropy = -(prob * (prob + 1e-6).log()).sum() / math.log(float(max(flat.numel(), 2)))
        topk = torch.topk(flat, k=min(2, flat.numel()))
        top1 = topk.values[0].abs().clamp_min(1e-6)
        top2 = topk.values[1] if topk.values.numel() > 1 else torch.zeros_like(top1)
        peak_ratio = top2 / top1
        peak_idx = int(topk.indices[0].item())
        row, col = divmod(peak_idx, self.feat_sz)
        center = 0.5 * (self.feat_sz - 1)
        max_dist = math.sqrt(2.0 * center * center) if center > 0.0 else 1.0
        center_distance = math.hypot(float(col) - center, float(row) - center) / max_dist
        self.last_response_entropy = float(entropy.item())
        self.last_response_peak_ratio = float(peak_ratio.item())
        self.last_response_center_distance = float(center_distance)

    def _bbox_from_response_index(self, index, size_map, offset_map):
        idx = torch.as_tensor([int(index)], device=size_map.device, dtype=torch.long)
        idx_y = torch.div(idx, self.feat_sz, rounding_mode='floor')
        idx_x = idx % self.feat_sz
        gather_idx = idx.view(1, 1, 1).expand(1, 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=gather_idx)
        offset = offset_map.flatten(2).gather(dim=2, index=gather_idx).squeeze(-1)
        return torch.cat([(idx_x.to(torch.float32).view(1, 1) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float32).view(1, 1) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1).squeeze(0)

    def _candidate_motion_score(self, box):
        prev = self.state[1]
        prev_cx, prev_cy = self._box_center(prev)
        cand_cx, cand_cy = self._box_center(box)
        jump = math.hypot(cand_cx - prev_cx, cand_cy - prev_cy)
        ref = max(prev[2], prev[3], 1.0)
        sigma = max(float(getattr(self.cfg.TEST, "CANDIDATE_RERANK_MOTION_SIGMA", 1.50)), 1e-3)
        return math.exp(-jump / (sigma * ref))

    @staticmethod
    def _flatten_language_map(language_map, target_shape):
        if language_map is None:
            return None
        if language_map.dim() == 4:
            if language_map.shape[-2:] != target_shape:
                language_map = torch.nn.functional.interpolate(
                    language_map.float(), size=target_shape, mode='bilinear', align_corners=False)
            return language_map.flatten(1)
        if language_map.dim() == 3:
            return language_map.flatten(1)
        if language_map.dim() == 2:
            return language_map
        return None

    def _select_candidate_box(self, response, size_map, offset_map, resize_factor, image, H, W,
                              gt_box=None, language_maps=None):
        test_cfg = getattr(self.cfg, "TEST", None)
        self.last_candidate_rerank = 0.0
        self.last_candidate_rank = 0.0
        self.last_candidate_score_gain = 0.0
        self.last_candidate_consistency_gain = 0.0
        self.last_candidate_selected_score_ratio = 0.0
        self.last_candidate_selected_rgb_consistency = 0.0
        self.last_candidate_selected_depth_consistency = 0.0
        self.last_candidate_selected_motion_score = 0.0
        self.last_candidate_selected_iou = -1.0
        self.last_candidate_oracle_iou = -1.0
        self.last_candidate_oracle_rank = -1.0
        self.last_candidate_oracle_score_ratio = 0.0
        self.last_candidate_oracle_rgb_consistency = 0.0
        self.last_candidate_oracle_depth_consistency = 0.0
        self.last_candidate_oracle_consistency = 0.0
        self.last_candidate_oracle_motion_score = 0.0
        self.last_candidate_probation = 0.0
        self.last_candidate_recovery_trigger = 0.0
        self.last_candidate_stable_block = 0.0
        self.last_candidate_ordinary_trigger = 0.0
        self.last_language_candidate_rerank = 0.0
        self.last_language_candidate_rank = 0.0
        self.last_language_candidate_best = 0.0
        self.last_language_candidate_selected = 0.0
        self.last_language_candidate_gain = 0.0
        flat = response.flatten(1)
        best_score, best_idx = torch.max(flat, dim=1, keepdim=True)
        best_score_value = float(best_score[0, 0].item())
        best_box = self._bbox_from_response_index(best_idx[0, 0].item(), size_map, offset_map)
        best_box = (best_box * self.params.search_size / resize_factor).tolist()
        best_mapped = clip_box(self.map_box_back(best_box, resize_factor), H, W, margin=10)
        self.last_candidate_selected_iou = self._box_iou_xywh(best_mapped, gt_box)

        language_flat = None
        language_trigger = False
        if bool(getattr(test_cfg, "LANGUAGE_CANDIDATE_RERANK", False)) and language_maps is not None:
            language_flat = self._flatten_language_map(
                language_maps.get('language_target_map', language_maps.get('language_targetness_map', None)),
                (self.feat_sz, self.feat_sz))
            if language_flat is not None and language_flat.shape[1] == flat.shape[1]:
                best_lang = float(language_flat[0, best_idx[0, 0]].item())
                top_lang = float(language_flat.max(dim=1).values[0].item())
                lang_gain = top_lang - best_lang
                self.last_language_candidate_best = best_lang
                self.last_language_candidate_gain = lang_gain
                language_trigger = (
                    self.frame_id > int(getattr(test_cfg, "CANDIDATE_RERANK_WARMUP_FRAMES", 3)) and
                    lang_gain >= float(getattr(test_cfg, "LANGUAGE_CANDIDATE_TRIGGER_GAP", 0.08))
                )

        if not bool(getattr(test_cfg, "CANDIDATE_RERANK", False)):
            self.candidate_rerank_streak = 0
            self._update_candidate_oracle(flat, size_map, offset_map, resize_factor, H, W, gt_box, image=image)
            return best_mapped, best_score_value
        if self.frame_id <= int(getattr(test_cfg, "CANDIDATE_RERANK_WARMUP_FRAMES", 3)):
            self.candidate_rerank_streak = 0
            self._update_candidate_oracle(flat, size_map, offset_map, resize_factor, H, W, gt_box, image=image)
            return best_mapped, best_score_value
        max_frame = int(getattr(test_cfg, "CANDIDATE_RERANK_MAX_FRAME", 0))
        if max_frame > 0 and self.frame_id > max_frame:
            self.candidate_rerank_streak = 0
            self._update_candidate_oracle(flat, size_map, offset_map, resize_factor, H, W, gt_box, image=image)
            return best_mapped, best_score_value

        trigger_entropy = float(getattr(test_cfg, "CANDIDATE_RERANK_TRIGGER_ENTROPY", 0.68))
        trigger_peak = float(getattr(test_cfg, "CANDIDATE_RERANK_TRIGGER_PEAK_RATIO", 0.82))
        trigger_consistency = float(getattr(test_cfg, "CANDIDATE_RERANK_TRIGGER_MIN_CONSISTENCY", 0.0))
        best_rgb_cons, best_depth_cons = self._template_consistency_parts(image, best_mapped)
        best_consistency = 0.5 * (best_rgb_cons + best_depth_cons)
        low_template_consistency = trigger_consistency > 0.0 and best_consistency < trigger_consistency
        unstable_response = (self.last_response_entropy >= trigger_entropy or
                             self.last_response_peak_ratio >= trigger_peak)
        selective_recovery = bool(getattr(test_cfg, "CANDIDATE_RERANK_SELECTIVE_RECOVERY", False))
        recovery_augment = bool(getattr(test_cfg, "CANDIDATE_RERANK_RECOVERY_AUGMENT", False))
        score_ref = self.score_ema if self.score_ema is not None else best_score_value
        stable_response = (
            best_score_value >= float(getattr(test_cfg, "CANDIDATE_RERANK_STABLE_SCORE_THR", 0.72)) and
            score_ref >= float(getattr(test_cfg, "CANDIDATE_RERANK_STABLE_EMA_THR", 0.65)) and
            self.last_response_entropy <= float(getattr(test_cfg, "CANDIDATE_RERANK_STABLE_ENTROPY_MAX", 0.42)) and
            self.last_response_peak_ratio <= float(getattr(test_cfg, "CANDIDATE_RERANK_STABLE_PEAK_RATIO_MAX", 0.42)) and
            (best_consistency >= float(getattr(test_cfg, "CANDIDATE_RERANK_STABLE_CONSISTENCY_MIN", 0.58)) or
             best_rgb_cons >= float(getattr(test_cfg, "CANDIDATE_RERANK_STABLE_RGB_MIN", 0.58)))
        )
        recovery_trigger = False
        if (selective_recovery or recovery_augment) and not stable_response:
            low_score = best_score_value <= float(getattr(test_cfg, "CANDIDATE_RERANK_RECOVERY_MAX_SCORE", 0.62))
            low_ema = score_ref <= float(getattr(test_cfg, "CANDIDATE_RERANK_RECOVERY_MAX_EMA", 0.70))
            ambiguous = (
                self.last_response_entropy >= float(getattr(test_cfg, "CANDIDATE_RERANK_RECOVERY_MIN_ENTROPY", 0.30)) or
                self.last_response_peak_ratio >= float(getattr(test_cfg, "CANDIDATE_RERANK_RECOVERY_MIN_PEAK_RATIO", 0.25)) or
                self.last_response_center_distance >= float(getattr(test_cfg, "CANDIDATE_RERANK_RECOVERY_MIN_CENTER_DISTANCE", 0.08))
            )
            weak_consistency = (
                best_consistency <= float(getattr(test_cfg, "CANDIDATE_RERANK_RECOVERY_MAX_CONSISTENCY", 0.78)) or
                best_depth_cons <= float(getattr(test_cfg, "CANDIDATE_RERANK_RECOVERY_MAX_DEPTH_CONSISTENCY", 0.55))
            )
            recovery_trigger = (low_score or low_ema or low_template_consistency) and ambiguous and weak_consistency
        self.last_candidate_recovery_trigger = 1.0 if recovery_trigger else 0.0
        self.last_candidate_stable_block = 1.0 if stable_response else 0.0
        self.last_candidate_ordinary_trigger = 1.0 if (unstable_response or low_template_consistency) else 0.0
        if language_trigger:
            min_lang_base_score = float(getattr(test_cfg, "LANGUAGE_CANDIDATE_MIN_BASE_SCORE", 0.0))
            min_lang_base_ema = float(getattr(test_cfg, "LANGUAGE_CANDIDATE_MIN_BASE_EMA", 0.0))
            min_lang_base_consistency = float(getattr(test_cfg, "LANGUAGE_CANDIDATE_MIN_BASE_CONSISTENCY", 0.0))
            min_lang_base_rgb = float(getattr(test_cfg, "LANGUAGE_CANDIDATE_MIN_BASE_RGB", 0.0))
            min_lang_base_depth = float(getattr(test_cfg, "LANGUAGE_CANDIDATE_MIN_BASE_DEPTH", 0.0))
            max_lang_base_score = float(getattr(test_cfg, "LANGUAGE_CANDIDATE_MAX_BASE_SCORE", 0.0))
            max_lang_base_ema = float(getattr(test_cfg, "LANGUAGE_CANDIDATE_MAX_BASE_EMA", 0.0))
            max_lang_base_consistency = float(getattr(test_cfg, "LANGUAGE_CANDIDATE_MAX_BASE_CONSISTENCY", 0.0))
            max_lang_base_rgb = float(getattr(test_cfg, "LANGUAGE_CANDIDATE_MAX_BASE_RGB", 0.0))
            max_lang_base_depth = float(getattr(test_cfg, "LANGUAGE_CANDIDATE_MAX_BASE_DEPTH", 0.0))
            max_lang_entropy = float(getattr(test_cfg, "LANGUAGE_CANDIDATE_MAX_RESPONSE_ENTROPY", 0.0))
            max_lang_peak_ratio = float(getattr(test_cfg, "LANGUAGE_CANDIDATE_MAX_RESPONSE_PEAK_RATIO", 0.0))
            max_lang_base = float(getattr(test_cfg, "LANGUAGE_CANDIDATE_MAX_BASE_LANG", 0.0))
            language_trigger = (
                (min_lang_base_score <= 0.0 or best_score_value >= min_lang_base_score) and
                (min_lang_base_ema <= 0.0 or score_ref >= min_lang_base_ema) and
                (min_lang_base_consistency <= 0.0 or best_consistency >= min_lang_base_consistency) and
                (min_lang_base_rgb <= 0.0 or best_rgb_cons >= min_lang_base_rgb) and
                (min_lang_base_depth <= 0.0 or best_depth_cons >= min_lang_base_depth) and
                (max_lang_base_score <= 0.0 or best_score_value <= max_lang_base_score) and
                (max_lang_base_ema <= 0.0 or score_ref <= max_lang_base_ema) and
                (max_lang_base_consistency <= 0.0 or best_consistency <= max_lang_base_consistency) and
                (max_lang_base_rgb <= 0.0 or best_rgb_cons <= max_lang_base_rgb) and
                (max_lang_base_depth <= 0.0 or best_depth_cons <= max_lang_base_depth) and
                (max_lang_entropy <= 0.0 or self.last_response_entropy <= max_lang_entropy) and
                (max_lang_peak_ratio <= 0.0 or self.last_response_peak_ratio <= max_lang_peak_ratio) and
                (max_lang_base <= 0.0 or self.last_language_candidate_best <= max_lang_base)
            )
            if language_trigger and bool(getattr(test_cfg, "LANGUAGE_CANDIDATE_STABLE_BASE_BLOCK", False)):
                stable_base = (
                    best_consistency >= float(getattr(test_cfg, "LANGUAGE_CANDIDATE_STABLE_BASE_CONSISTENCY", 0.82)) and
                    best_rgb_cons >= float(getattr(test_cfg, "LANGUAGE_CANDIDATE_STABLE_BASE_RGB", 0.82)) and
                    best_depth_cons >= float(getattr(test_cfg, "LANGUAGE_CANDIDATE_STABLE_BASE_DEPTH", 0.80))
                )
                if stable_base:
                    language_trigger = False
        if selective_recovery and not recovery_trigger and not language_trigger:
            self.candidate_rerank_streak = 0
            self._update_candidate_oracle(flat, size_map, offset_map, resize_factor, H, W, gt_box, image=image)
            return best_mapped, best_score_value
        if not unstable_response and not low_template_consistency and not recovery_trigger and not language_trigger:
            self.candidate_rerank_streak = 0
            self._update_candidate_oracle(flat, size_map, offset_map, resize_factor, H, W, gt_box, image=image)
            return best_mapped, best_score_value
        max_score = float(getattr(test_cfg, "CANDIDATE_RERANK_MAX_SCORE", 0.0))
        if max_score > 0.0 and best_score_value > max_score and not recovery_trigger and not language_trigger:
            self.candidate_rerank_streak = 0
            self._update_candidate_oracle(flat, size_map, offset_map, resize_factor, H, W, gt_box, image=image)
            return best_mapped, best_score_value
        min_base_rgb = float(getattr(test_cfg, "CANDIDATE_RERANK_REQUIRE_MIN_RGB_CONSISTENCY", 0.0))
        min_base_depth = float(getattr(test_cfg, "CANDIDATE_RERANK_REQUIRE_MIN_DEPTH_CONSISTENCY", 0.0))
        min_base_consistency = float(getattr(test_cfg, "CANDIDATE_RERANK_REQUIRE_MIN_CONSISTENCY", 0.0))
        min_base_rgb_ema = float(getattr(test_cfg, "CANDIDATE_RERANK_REQUIRE_MIN_RGB_CONSISTENCY_EMA", 0.0))
        min_base_depth_ema = float(getattr(test_cfg, "CANDIDATE_RERANK_REQUIRE_MIN_DEPTH_CONSISTENCY_EMA", 0.0))
        min_base_consistency_ema = float(getattr(test_cfg, "CANDIDATE_RERANK_REQUIRE_MIN_CONSISTENCY_EMA", 0.0))
        if ((min_base_rgb > 0.0 and best_rgb_cons < min_base_rgb) or
                (min_base_depth > 0.0 and best_depth_cons < min_base_depth) or
                (min_base_consistency > 0.0 and best_consistency < min_base_consistency) or
                (min_base_rgb_ema > 0.0 and self.template_rgb_consistency_ema < min_base_rgb_ema) or
                (min_base_depth_ema > 0.0 and self.template_depth_consistency_ema < min_base_depth_ema) or
                (min_base_consistency_ema > 0.0 and self.template_consistency_ema < min_base_consistency_ema)):
            self.candidate_rerank_streak = 0
            self._update_candidate_oracle(flat, size_map, offset_map, resize_factor, H, W, gt_box, image=image)
            return best_mapped, best_score_value

        topk_cfg = "CANDIDATE_RERANK_RECOVERY_TOPK" if recovery_trigger else "CANDIDATE_RERANK_TOPK"
        topk = min(int(getattr(test_cfg, topk_cfg, getattr(test_cfg, "CANDIDATE_RERANK_TOPK", 5))), flat.numel())
        if language_trigger:
            topk = min(max(topk, int(getattr(test_cfg, "LANGUAGE_CANDIDATE_RESPONSE_TOPK", topk))), flat.numel())
        if topk <= 1:
            self.candidate_rerank_streak = 0
            self._update_candidate_oracle(flat, size_map, offset_map, resize_factor, H, W, gt_box, image=image)
            return best_mapped, best_score_value
        values, indices = torch.topk(flat, k=topk, dim=1)
        self._update_candidate_oracle(flat, size_map, offset_map, resize_factor, H, W, gt_box,
                                      values, indices, image=image)
        if language_trigger:
            lang_topk = min(int(getattr(test_cfg, "LANGUAGE_CANDIDATE_TOPK", 8)), flat.numel())
            if language_flat is not None and lang_topk > 0:
                _, lang_indices = torch.topk(language_flat, k=lang_topk, dim=1)
                ordered = []
                seen = set()
                for idx_value in torch.cat([best_idx[0], indices[0], lang_indices[0]], dim=0).tolist():
                    idx_value = int(idx_value)
                    if idx_value not in seen:
                        ordered.append(idx_value)
                        seen.add(idx_value)
                indices = torch.as_tensor(ordered, device=flat.device, dtype=torch.long).view(1, -1)
                values = flat.gather(dim=1, index=indices)
            min_score_ratio = float(getattr(test_cfg, "LANGUAGE_CANDIDATE_MIN_SCORE_RATIO", 0.02))
            score_w = float(getattr(test_cfg, "LANGUAGE_CANDIDATE_SCORE_WEIGHT", 0.25))
            rgb_w = float(getattr(test_cfg, "LANGUAGE_CANDIDATE_RGB_WEIGHT", 0.10))
            depth_w = float(getattr(test_cfg, "LANGUAGE_CANDIDATE_DEPTH_WEIGHT", 0.10))
            motion_w = float(getattr(test_cfg, "LANGUAGE_CANDIDATE_MOTION_WEIGHT", 0.10))
            lang_w = float(getattr(test_cfg, "LANGUAGE_CANDIDATE_LANG_WEIGHT", 0.45))
        elif recovery_trigger:
            min_score_ratio = float(getattr(test_cfg, "CANDIDATE_RERANK_RECOVERY_MIN_SCORE_RATIO",
                                            getattr(test_cfg, "CANDIDATE_RERANK_MIN_SCORE_RATIO", 0.92)))
            score_w = float(getattr(test_cfg, "CANDIDATE_RERANK_RECOVERY_SCORE_WEIGHT",
                                    getattr(test_cfg, "CANDIDATE_RERANK_SCORE_WEIGHT", 0.55)))
            rgb_w = float(getattr(test_cfg, "CANDIDATE_RERANK_RECOVERY_RGB_WEIGHT",
                                  getattr(test_cfg, "CANDIDATE_RERANK_RGB_WEIGHT", 0.25)))
            depth_w = float(getattr(test_cfg, "CANDIDATE_RERANK_RECOVERY_DEPTH_WEIGHT",
                                    getattr(test_cfg, "CANDIDATE_RERANK_DEPTH_WEIGHT", 0.15)))
            motion_w = float(getattr(test_cfg, "CANDIDATE_RERANK_RECOVERY_MOTION_WEIGHT",
                                     getattr(test_cfg, "CANDIDATE_RERANK_MOTION_WEIGHT", 0.05)))
            lang_w = 0.0
        else:
            min_score_ratio = float(getattr(test_cfg, "CANDIDATE_RERANK_MIN_SCORE_RATIO", 0.92))
            score_w = float(getattr(test_cfg, "CANDIDATE_RERANK_SCORE_WEIGHT", 0.55))
            rgb_w = float(getattr(test_cfg, "CANDIDATE_RERANK_RGB_WEIGHT", 0.25))
            depth_w = float(getattr(test_cfg, "CANDIDATE_RERANK_DEPTH_WEIGHT", 0.15))
            motion_w = float(getattr(test_cfg, "CANDIDATE_RERANK_MOTION_WEIGHT", 0.05))
            lang_w = 0.0
        best_ref = max(best_score_value, 1e-6)
        lang_values = None
        lang_min = 0.0
        lang_den = 1.0
        if language_trigger and language_flat is not None:
            lang_values = language_flat.gather(dim=1, index=indices)
            lang_min = float(lang_values.min().item())
            lang_den = max(float(lang_values.max().item()) - lang_min, 1e-6)

        candidates = []
        for rank in range(values.shape[1]):
            raw_score = float(values[0, rank].item())
            if raw_score < best_ref * min_score_ratio:
                continue
            bbox = self._bbox_from_response_index(indices[0, rank].item(), size_map, offset_map)
            bbox = (bbox * self.params.search_size / resize_factor).tolist()
            mapped = clip_box(self.map_box_back(bbox, resize_factor), H, W, margin=10)
            rgb_cons, depth_cons = self._template_consistency_parts(image, mapped)
            motion_score = self._candidate_motion_score(mapped)
            relative_score = raw_score / best_ref
            lang_score = 0.0
            if lang_values is not None:
                lang_raw = float(lang_values[0, rank].item())
                lang_score = (lang_raw - lang_min) / lang_den
            rerank_score = (score_w * relative_score + rgb_w * rgb_cons +
                            depth_w * depth_cons + motion_w * motion_score + lang_w * lang_score)
            candidates.append({
                'rank': rank,
                'raw_score': raw_score,
                'box': mapped,
                'rgb_cons': rgb_cons,
                'depth_cons': depth_cons,
                'consistency': 0.5 * (rgb_cons + depth_cons),
                'motion_score': motion_score,
                'language_score': lang_score,
                'rerank_score': rerank_score,
            })

        if len(candidates) <= 1:
            self.candidate_rerank_streak = 0
            return best_mapped, best_score_value
        base = candidates[0]
        selected = max(candidates, key=lambda item: item['rerank_score'])
        score_gain = selected['rerank_score'] - base['rerank_score']
        consistency_gain = selected['consistency'] - base['consistency']
        if language_trigger:
            min_gain = float(getattr(test_cfg, "LANGUAGE_CANDIDATE_MIN_GAIN", 0.02))
            min_consistency_gain = float(getattr(test_cfg, "LANGUAGE_CANDIDATE_MIN_CONSISTENCY_GAIN", -0.10))
        elif recovery_trigger:
            min_gain = float(getattr(test_cfg, "CANDIDATE_RERANK_RECOVERY_MIN_GAIN",
                                     getattr(test_cfg, "CANDIDATE_RERANK_MIN_GAIN", 0.03)))
            min_consistency_gain = float(getattr(test_cfg, "CANDIDATE_RERANK_RECOVERY_MIN_CONSISTENCY_GAIN",
                                                 getattr(test_cfg, "CANDIDATE_RERANK_MIN_CONSISTENCY_GAIN", 0.08)))
        else:
            min_gain = float(getattr(test_cfg, "CANDIDATE_RERANK_MIN_GAIN", 0.03))
            min_consistency_gain = float(getattr(test_cfg, "CANDIDATE_RERANK_MIN_CONSISTENCY_GAIN", 0.08))
        min_lang_gain = float(getattr(test_cfg, "LANGUAGE_CANDIDATE_MIN_LANG_GAIN", 0.04))
        language_gain_ok = (not language_trigger or
                            selected['language_score'] - base.get('language_score', 0.0) >= min_lang_gain)
        max_selected_rank = int(getattr(test_cfg, "CANDIDATE_RERANK_MAX_SELECTED_RANK", 0))
        rank_ok = max_selected_rank <= 0 or selected['rank'] <= max_selected_rank
        min_selected_rgb = float(getattr(test_cfg, "CANDIDATE_RERANK_SELECTED_MIN_RGB", 0.0))
        min_selected_depth = float(getattr(test_cfg, "CANDIDATE_RERANK_SELECTED_MIN_DEPTH", 0.0))
        min_selected_consistency = float(getattr(test_cfg, "CANDIDATE_RERANK_SELECTED_MIN_CONSISTENCY", 0.0))
        min_selected_motion = float(getattr(test_cfg, "CANDIDATE_RERANK_SELECTED_MIN_MOTION", 0.0))
        selected_reliable = (
            (min_selected_rgb <= 0.0 or selected['rgb_cons'] >= min_selected_rgb) and
            (min_selected_depth <= 0.0 or selected['depth_cons'] >= min_selected_depth) and
            (min_selected_consistency <= 0.0 or selected['consistency'] >= min_selected_consistency) and
            (min_selected_motion <= 0.0 or selected['motion_score'] >= min_selected_motion)
        )
        if (selected['rank'] > 0 and rank_ok and score_gain >= min_gain and
                consistency_gain >= min_consistency_gain and language_gain_ok and selected_reliable):
            probation_frames = int(getattr(test_cfg, "CANDIDATE_RERANK_PROBATION_FRAMES", 0))
            if probation_frames > 0:
                self.candidate_rerank_streak += 1
                if self.candidate_rerank_streak < probation_frames:
                    self.last_candidate_probation = float(self.candidate_rerank_streak)
                    return best_mapped, best_score_value
            self.last_candidate_rerank = 1.0
            self.last_candidate_rank = float(selected['rank'])
            self.last_candidate_score_gain = float(score_gain)
            self.last_candidate_consistency_gain = float(consistency_gain)
            self.last_candidate_selected_score_ratio = float(selected['raw_score'] / best_ref)
            self.last_candidate_selected_rgb_consistency = float(selected['rgb_cons'])
            self.last_candidate_selected_depth_consistency = float(selected['depth_cons'])
            self.last_candidate_selected_motion_score = float(selected['motion_score'])
            if language_trigger:
                self.last_language_candidate_rerank = 1.0
                self.last_language_candidate_rank = float(selected['rank'])
                self.last_language_candidate_selected = float(selected['language_score'])
            self.last_candidate_selected_iou = self._box_iou_xywh(selected['box'], gt_box)
            return selected['box'], best_score_value
        self.candidate_rerank_streak = 0
        return best_mapped, best_score_value

    def _update_candidate_oracle(self, flat, size_map, offset_map, resize_factor, H, W, gt_box=None,
                                 values=None, indices=None, image=None):
        if gt_box is None:
            return
        topk = min(int(getattr(self.cfg.TEST, "CANDIDATE_DIAG_TOPK", 10)), flat.numel())
        if topk <= 0:
            return
        if values is None or indices is None or values.shape[1] < topk:
            values, indices = torch.topk(flat, k=topk, dim=1)
        best_ref = max(float(values[0, 0].item()), 1e-6)
        oracle_iou = -1.0
        oracle_rank = -1
        oracle_ratio = 0.0
        oracle_box = None
        for rank in range(min(topk, values.shape[1])):
            bbox = self._bbox_from_response_index(indices[0, rank].item(), size_map, offset_map)
            bbox = (bbox * self.params.search_size / resize_factor).tolist()
            mapped = clip_box(self.map_box_back(bbox, resize_factor), H, W, margin=10)
            iou = self._box_iou_xywh(mapped, gt_box)
            if iou > oracle_iou:
                oracle_iou = iou
                oracle_rank = rank
                oracle_ratio = float(values[0, rank].item()) / best_ref
                oracle_box = mapped
        self.last_candidate_oracle_iou = float(oracle_iou)
        self.last_candidate_oracle_rank = float(oracle_rank)
        self.last_candidate_oracle_score_ratio = float(oracle_ratio)
        if oracle_box is not None and image is not None:
            rgb_cons, depth_cons = self._template_consistency_parts(image, oracle_box)
            self.last_candidate_oracle_rgb_consistency = float(rgb_cons)
            self.last_candidate_oracle_depth_consistency = float(depth_cons)
            self.last_candidate_oracle_consistency = float(0.5 * (rgb_cons + depth_cons))
            self.last_candidate_oracle_motion_score = float(self._candidate_motion_score(oracle_box))

    def _stabilize_low_confidence_box(self, pred_box, score, image=None):
        test_cfg = getattr(self.cfg, "TEST", None)
        if not bool(getattr(test_cfg, "LOW_CONF_DAMPING", False)) and not bool(getattr(test_cfg, "TEMPORAL_PRIOR", False)):
            self.last_template_consistency = self._template_consistency(image, pred_box)
            self.last_update_ratio = 1.0
            return pred_box
        score_thr = float(getattr(test_cfg, "DAMPING_SCORE_THR", 0.40))
        lost_score_thr = float(getattr(test_cfg, "LOST_SCORE_THR", 0.25))
        score_ref = score if self.score_ema is None else self.score_ema
        if bool(getattr(test_cfg, "ADAPTIVE_SCORE_DAMPING", False)):
            min_lost_thr = float(getattr(test_cfg, "MIN_LOST_SCORE_THR", 0.06))
            min_damping_thr = float(getattr(test_cfg, "MIN_DAMPING_SCORE_THR", 0.15))
            lost_factor = float(getattr(test_cfg, "LOST_SCORE_EMA_FACTOR", 0.45))
            damping_factor = float(getattr(test_cfg, "DAMPING_SCORE_EMA_FACTOR", 0.85))
            lost_score_thr = max(min_lost_thr, min(lost_score_thr, score_ref * lost_factor))
            score_thr = max(min_damping_thr, min(score_thr, score_ref * damping_factor))
        self.last_effective_lost_thr = lost_score_thr
        self.last_effective_damping_thr = score_thr
        self.last_center_follow = 0.0

        min_update = float(getattr(test_cfg, "DAMPING_MIN_UPDATE", 0.25))
        min_update = min(max(min_update, 0.0), 1.0)
        if lost_score_thr > 0.0 and score < lost_score_thr:
            update_ratio = min(max(float(getattr(test_cfg, "LOST_UPDATE_RATIO", 0.0)), 0.0), 1.0)
            if bool(getattr(test_cfg, "LOW_CONF_CENTER_FOLLOW", False)):
                follow_score_thr = float(getattr(test_cfg, "CENTER_FOLLOW_SCORE_THR", 0.08))
                follow_ratio = min(max(float(getattr(test_cfg, "CENTER_FOLLOW_UPDATE_RATIO", 0.18)), 0.0), 1.0)
                max_peak_distance = float(getattr(test_cfg, "CENTER_FOLLOW_MAX_PEAK_DISTANCE", 0.45))
                max_response_entropy = float(getattr(test_cfg, "CENTER_FOLLOW_MAX_RESPONSE_ENTROPY", 0.88))
                min_lost_frames = int(getattr(test_cfg, "CENTER_FOLLOW_MIN_LOST_FRAMES", 0))
                min_response_entropy = float(getattr(test_cfg, "CENTER_FOLLOW_MIN_RESPONSE_ENTROPY", 0.0))
                max_score_ema = float(getattr(test_cfg, "CENTER_FOLLOW_MAX_SCORE_EMA", 0.0))
                allow_consistency_below = float(getattr(test_cfg, "CENTER_FOLLOW_EMA_ALLOW_CONSISTENCY_BELOW", 0.0))
                allow_rgb_below = float(getattr(test_cfg, "CENTER_FOLLOW_EMA_ALLOW_RGB_BELOW", 0.0))
                allow_depth_below = float(getattr(test_cfg, "CENTER_FOLLOW_EMA_ALLOW_DEPTH_BELOW", 0.0))
                low_consistency_recovery = (
                    (allow_consistency_below > 0.0 and self.last_template_consistency <= allow_consistency_below) or
                    (allow_rgb_below > 0.0 and self.last_template_rgb_consistency <= allow_rgb_below) or
                    (allow_depth_below > 0.0 and self.last_template_depth_consistency <= allow_depth_below)
                )
                score_ema_allowed = (max_score_ema <= 0.0 or self.score_ema is None or
                                     self.score_ema <= max_score_ema or low_consistency_recovery)
                if (score >= follow_score_thr and
                        self.lost_frames >= min_lost_frames and
                        score_ema_allowed and
                        self.last_response_center_distance <= max_peak_distance and
                        min_response_entropy <= self.last_response_entropy <= max_response_entropy):
                    update_ratio = max(update_ratio, follow_ratio)
                    self.last_center_follow = 1.0
            max_freeze = int(getattr(test_cfg, "LOST_MAX_FREEZE_FRAMES", 0))
            if max_freeze > 0 and self.lost_frames >= max_freeze:
                recovery_update = float(getattr(test_cfg, "LOST_RECOVERY_UPDATE_RATIO", update_ratio))
                min_recovery_consistency = float(getattr(test_cfg, "LOST_RECOVERY_MIN_CONSISTENCY", 0.0))
                pred_consistency = self._template_consistency(image, pred_box)
                if pred_consistency >= min_recovery_consistency:
                    update_ratio = max(update_ratio, min(max(recovery_update, 0.0), 1.0))
        elif bool(getattr(test_cfg, "LOW_CONF_DAMPING", False)) and score_thr > 0.0 and score < score_thr:
            update_ratio = min(max(score / score_thr, min_update), 1.0)
        else:
            update_ratio = 1.0

        prev = self.state[1]
        prev_cx, prev_cy = self._box_center(prev)
        pred_cx, pred_cy = self._box_center(pred_box)
        jump = math.hypot(pred_cx - prev_cx, pred_cy - prev_cy)
        jump_ref = max(prev[2], prev[3], 1.0)
        jump_factor = float(getattr(test_cfg, "DAMPING_JUMP_FACTOR", 2.0))
        if jump_factor > 0.0 and jump > jump_factor * jump_ref:
            update_ratio = min(update_ratio, min_update)

        if bool(getattr(test_cfg, "TEMPORAL_PRIOR", False)):
            max_jump_factor = float(getattr(test_cfg, "MAX_CENTER_JUMP_FACTOR", 1.25))
            if max_jump_factor > 0.0 and jump > max_jump_factor * jump_ref:
                update_ratio = min(update_ratio, float(getattr(test_cfg, "VELOCITY_DAMPING", 0.50)))

            max_scale_change = float(getattr(test_cfg, "MAX_SCALE_CHANGE", 1.80))
            if max_scale_change > 1.0:
                scale_w = max(pred_box[2], 1.0) / max(prev[2], 1.0)
                scale_h = max(pred_box[3], 1.0) / max(prev[3], 1.0)
                scale_change = max(scale_w, 1.0 / max(scale_w, 1e-6), scale_h, 1.0 / max(scale_h, 1e-6))
                if scale_change > max_scale_change:
                    update_ratio = min(update_ratio, float(getattr(test_cfg, "VELOCITY_DAMPING", 0.50)))
        if self.last_candidate_rerank > 0.0:
            candidate_update = float(getattr(test_cfg, "CANDIDATE_RERANK_UPDATE_RATIO", 1.0))
            update_ratio = min(update_ratio, min(max(candidate_update, 0.0), 1.0))

        consistency = self._template_consistency(image, pred_box)
        self.last_template_consistency = consistency
        rgb_consistency, depth_consistency = self._template_consistency_parts(image, pred_box, record=True)
        self.last_template_rgb_consistency = rgb_consistency
        self.last_template_depth_consistency = depth_consistency
        consistency_momentum = float(getattr(test_cfg, "CANDIDATE_RERANK_CONSISTENCY_EMA_MOMENTUM", 0.95))
        consistency_momentum = min(max(consistency_momentum, 0.0), 0.999)
        self.template_consistency_ema = (
            consistency_momentum * self.template_consistency_ema + (1.0 - consistency_momentum) * consistency)
        self.template_rgb_consistency_ema = (
            consistency_momentum * self.template_rgb_consistency_ema + (1.0 - consistency_momentum) * rgb_consistency)
        self.template_depth_consistency_ema = (
            consistency_momentum * self.template_depth_consistency_ema + (1.0 - consistency_momentum) * depth_consistency)
        if bool(getattr(test_cfg, "TEMPLATE_CONSISTENCY_GUARD", False)):
            consistency_thr = float(getattr(test_cfg, "CONSISTENCY_SCORE_THR", 0.45))
            consistency_update = float(getattr(test_cfg, "CONSISTENCY_UPDATE_RATIO", 0.35))
            consistency_update = min(max(consistency_update, 0.0), 1.0)
            if self.frame_id > int(getattr(test_cfg, "CONSISTENCY_WARMUP_FRAMES", 5)) and consistency < consistency_thr:
                update_ratio = min(update_ratio, consistency_update)
                hard_thr = float(getattr(test_cfg, "CONSISTENCY_HARD_SCORE_THR", 0.25))
                if hard_thr > 0.0 and consistency < hard_thr:
                    hard_update = float(getattr(test_cfg, "CONSISTENCY_HARD_UPDATE_RATIO", 0.15))
                    update_ratio = min(update_ratio, min(max(hard_update, 0.0), 1.0))

        self.last_depth_guard = 0.0
        if bool(getattr(test_cfg, "DEPTH_CONSISTENCY_GUARD", False)):
            warmup = int(getattr(test_cfg, "DEPTH_GUARD_WARMUP_FRAMES", 5))
            depth_thr = float(getattr(test_cfg, "DEPTH_CONSISTENCY_THR", 0.35))
            score_thr = float(getattr(test_cfg, "DEPTH_GUARD_SCORE_THR", 0.55))
            guard_update = min(max(float(getattr(test_cfg, "DEPTH_GUARD_UPDATE_RATIO", 0.40)), 0.0), 1.0)
            rgb_floor = float(getattr(test_cfg, "DEPTH_GUARD_MIN_RGB_CONSISTENCY", 0.0))
            if (self.frame_id > warmup and depth_consistency < depth_thr and
                    score < score_thr and rgb_consistency >= rgb_floor):
                update_ratio = min(update_ratio, guard_update)
                self.last_depth_guard = 1.0

        self.last_response_guard = 0.0
        if bool(getattr(test_cfg, "RESPONSE_STABILITY_GUARD", False)):
            warmup = int(getattr(test_cfg, "RESPONSE_GUARD_WARMUP_FRAMES", 5))
            entropy_thr = float(getattr(test_cfg, "RESPONSE_ENTROPY_THR", 0.72))
            peak_ratio_thr = float(getattr(test_cfg, "RESPONSE_PEAK_RATIO_THR", 0.92))
            center_distance_thr = float(getattr(test_cfg, "RESPONSE_CENTER_DISTANCE_THR", 0.60))
            guard_update = min(max(float(getattr(test_cfg, "RESPONSE_GUARD_UPDATE_RATIO", 0.45)), 0.0), 1.0)
            ambiguous = (self.last_response_entropy > entropy_thr or
                         self.last_response_peak_ratio > peak_ratio_thr)
            far_peak = self.last_response_center_distance > center_distance_thr
            if self.frame_id > warmup and ambiguous and far_peak:
                update_ratio = min(update_ratio, guard_update)
                self.last_response_guard = 1.0

        if bool(getattr(test_cfg, "CONFIDENT_CONSISTENCY_GUARD", False)):
            warmup = int(getattr(test_cfg, "RESPONSE_GUARD_WARMUP_FRAMES", 5))
            score_thr = float(getattr(test_cfg, "CONFIDENT_GUARD_SCORE_THR", 0.75))
            consistency_thr = float(getattr(test_cfg, "CONFIDENT_GUARD_CONSISTENCY_THR", 0.55))
            depth_thr = float(getattr(test_cfg, "CONFIDENT_GUARD_DEPTH_THR", 0.0))
            rgb_thr = float(getattr(test_cfg, "CONFIDENT_GUARD_RGB_THR", 0.0))
            guard_update = min(max(float(getattr(test_cfg, "CONFIDENT_GUARD_UPDATE_RATIO", 0.35)), 0.0), 1.0)
            low_joint = consistency_thr > 0.0 and consistency < consistency_thr
            low_depth = depth_thr > 0.0 and depth_consistency < depth_thr
            low_rgb = rgb_thr > 0.0 and rgb_consistency < rgb_thr
            if self.frame_id > warmup and score >= score_thr and (low_joint or low_depth or low_rgb):
                update_ratio = min(update_ratio, guard_update)
                hard_thr = float(getattr(test_cfg, "CONFIDENT_GUARD_HARD_CONSISTENCY_THR", 0.35))
                if hard_thr > 0.0 and consistency < hard_thr:
                    hard_update = min(max(float(getattr(test_cfg, "CONFIDENT_GUARD_HARD_UPDATE_RATIO", 0.20)), 0.0), 1.0)
                    update_ratio = min(update_ratio, hard_update)
                self.last_response_guard = 1.0

        self.last_update_ratio = update_ratio
        momentum = float(getattr(test_cfg, "SCORE_EMA_MOMENTUM", 0.95))
        momentum = min(max(momentum, 0.0), 0.999)
        if self.score_ema is None:
            self.score_ema = score
        else:
            self.score_ema = momentum * self.score_ema + (1.0 - momentum) * score
        return [
            prev[i] * (1.0 - update_ratio) + pred_box[i] * update_ratio
            for i in range(4)
        ]

    def _update_temporal_state(self, prev_box, new_box, score):
        test_cfg = getattr(self.cfg, "TEST", None)
        lost_score_thr = float(getattr(test_cfg, "LOST_SCORE_THR", 0.25))
        recovery_score_thr = float(getattr(test_cfg, "RECOVERY_SCORE_THR", 0.35))
        if lost_score_thr > 0.0 and score < lost_score_thr:
            self.lost_frames += 1
            self.stable_frames = 0
            return
        if self.lost_frames > 0 and score < recovery_score_thr:
            self.stable_frames = 0
            return
        self.lost_frames = 0

        prev_cx, prev_cy = self._box_center(prev_box)
        new_cx, new_cy = self._box_center(new_box)
        dx, dy = new_cx - prev_cx, new_cy - prev_cy
        momentum = float(getattr(test_cfg, "VELOCITY_MOMENTUM", 0.80))
        momentum = min(max(momentum, 0.0), 0.99)
        self.velocity[0] = momentum * self.velocity[0] + (1.0 - momentum) * dx
        self.velocity[1] = momentum * self.velocity[1] + (1.0 - momentum) * dy

        jump = math.hypot(dx, dy)
        jump_ref = max(prev_box[2], prev_box[3], 1.0)
        stable_jump = float(getattr(test_cfg, "TEMPLATE_UPDATE_MAX_JUMP_FACTOR", 0.75))
        stable_score = float(getattr(test_cfg, "TEMPLATE_UPDATE_SCORE_THR", 0.82))
        if score >= stable_score and jump <= stable_jump * jump_ref:
            self.stable_frames += 1
        else:
            self.stable_frames = 0

    def _calibrate_output_score(self, score):
        test_cfg = getattr(self.cfg, "TEST", None)
        self.last_raw_score = float(score)
        self.last_score_calibration = 1.0
        if not bool(getattr(test_cfg, "SCORE_CALIBRATION", False)):
            return float(score)
        if self.frame_id <= int(getattr(test_cfg, "SCORE_CALIBRATION_WARMUP_FRAMES", 5)):
            return float(score)

        protect_score = float(getattr(test_cfg, "SCORE_CALIBRATION_PROTECT_SCORE", 0.80))
        protect_cons = float(getattr(test_cfg, "SCORE_CALIBRATION_PROTECT_CONSISTENCY", 0.70))
        protect_rgb = float(getattr(test_cfg, "SCORE_CALIBRATION_PROTECT_RGB", 0.70))
        protect_entropy = float(getattr(test_cfg, "SCORE_CALIBRATION_PROTECT_ENTROPY", 0.45))
        protect_peak = float(getattr(test_cfg, "SCORE_CALIBRATION_PROTECT_PEAK_RATIO", 0.45))
        stable_high_score = (
            score >= protect_score and
            self.last_template_consistency >= protect_cons and
            self.last_template_rgb_consistency >= protect_rgb and
            self.last_response_entropy <= protect_entropy and
            self.last_response_peak_ratio <= protect_peak
        )
        if stable_high_score:
            return float(score)

        penalty = 0.0
        template_thr = float(getattr(test_cfg, "SCORE_CALIBRATION_TEMPLATE_THR", 0.45))
        rgb_thr = float(getattr(test_cfg, "SCORE_CALIBRATION_RGB_THR", 0.45))
        depth_thr = float(getattr(test_cfg, "SCORE_CALIBRATION_DEPTH_THR", 0.20))
        entropy_thr = float(getattr(test_cfg, "SCORE_CALIBRATION_RESPONSE_ENTROPY_THR", 0.65))
        peak_thr = float(getattr(test_cfg, "SCORE_CALIBRATION_PEAK_RATIO_THR", 0.75))
        center_thr = float(getattr(test_cfg, "SCORE_CALIBRATION_CENTER_DISTANCE_THR", 0.35))
        update_thr = float(getattr(test_cfg, "SCORE_CALIBRATION_LOW_UPDATE_THR", 0.40))

        if template_thr > 0.0 and self.last_template_consistency < template_thr:
            penalty += float(getattr(test_cfg, "SCORE_CALIBRATION_TEMPLATE_WEIGHT", 0.25)) * (
                template_thr - self.last_template_consistency) / max(template_thr, 1e-6)
        if rgb_thr > 0.0 and self.last_template_rgb_consistency < rgb_thr:
            penalty += float(getattr(test_cfg, "SCORE_CALIBRATION_RGB_WEIGHT", 0.15)) * (
                rgb_thr - self.last_template_rgb_consistency) / max(rgb_thr, 1e-6)
        if depth_thr > 0.0 and self.last_template_depth_consistency < depth_thr:
            penalty += float(getattr(test_cfg, "SCORE_CALIBRATION_DEPTH_WEIGHT", 0.10)) * (
                depth_thr - self.last_template_depth_consistency) / max(depth_thr, 1e-6)
        if self.last_response_entropy > entropy_thr or self.last_response_peak_ratio > peak_thr:
            entropy_excess = max(self.last_response_entropy - entropy_thr, 0.0) / max(1.0 - entropy_thr, 1e-6)
            peak_excess = max(self.last_response_peak_ratio - peak_thr, 0.0) / max(1.0 - peak_thr, 1e-6)
            center_excess = max(self.last_response_center_distance - center_thr, 0.0) / max(1.0 - center_thr, 1e-6)
            penalty += float(getattr(test_cfg, "SCORE_CALIBRATION_RESPONSE_WEIGHT", 0.20)) * min(
                max(entropy_excess, peak_excess, center_excess), 1.0)
        if update_thr > 0.0 and self.last_update_ratio < update_thr:
            penalty += float(getattr(test_cfg, "SCORE_CALIBRATION_UPDATE_WEIGHT", 0.10)) * (
                update_thr - self.last_update_ratio) / max(update_thr, 1e-6)

        factor = max(float(getattr(test_cfg, "SCORE_CALIBRATION_MIN", 0.25)), min(1.0, 1.0 - penalty))
        self.last_score_calibration = factor
        return float(score) * factor

    def _maybe_update_dynamic_template(self, image, score):
        test_cfg = getattr(self.cfg, "TEST", None)
        self.last_template_update = 0.0
        self.last_template_update_reason = 0.0
        self.last_template_update_gate_score = float(score)
        self.last_template_update_gate_stable = float(self.stable_frames)
        self.last_template_update_gate_consistency = float(self.last_template_consistency)
        self.last_template_update_gate_rgb = float(self.last_template_rgb_consistency)
        self.last_template_update_gate_depth = float(self.last_template_depth_consistency)
        if not bool(getattr(test_cfg, "DYNAMIC_TEMPLATE", False)):
            self.last_template_update_reason = 1.0
            return
        interval = int(getattr(test_cfg, "TEMPLATE_UPDATE_INTERVAL", 100))
        if interval <= 0 or self.frame_id - self.last_update_frame < interval:
            self.last_template_update_reason = 2.0
            return
        if score < float(getattr(test_cfg, "TEMPLATE_UPDATE_SCORE_THR", 0.82)):
            self.last_template_update_reason = 3.0
            return
        if self.stable_frames < int(getattr(test_cfg, "TEMPLATE_UPDATE_MIN_STABLE", 8)):
            self.last_template_update_reason = 4.0
            return
        if bool(getattr(test_cfg, "TEMPLATE_UPDATE_BLOCK_ON_GUARD", True)):
            if self.last_response_guard > 0.0 or self.last_depth_guard > 0.0:
                self.last_template_update_reason = 5.0
                return
        min_consistency = float(getattr(test_cfg, "TEMPLATE_UPDATE_MIN_CONSISTENCY", 0.0))
        if min_consistency > 0.0 and self.last_template_consistency < min_consistency:
            self.last_template_update_reason = 6.0
            return
        min_rgb_consistency = float(getattr(test_cfg, "TEMPLATE_UPDATE_MIN_RGB_CONSISTENCY", 0.0))
        if min_rgb_consistency > 0.0 and self.last_template_rgb_consistency < min_rgb_consistency:
            self.last_template_update_reason = 7.0
            return
        min_depth_consistency = float(getattr(test_cfg, "TEMPLATE_UPDATE_MIN_DEPTH_CONSISTENCY", 0.0))
        if min_depth_consistency > 0.0 and self.last_template_depth_consistency < min_depth_consistency:
            self.last_template_update_reason = 8.0
            return
        self.update_template(image, self.state[1])
        self.last_update_frame = self.frame_id
        self.last_template_update = 1.0
        self.last_template_update_reason = 0.0

    def _current_search_factor(self):
        test_cfg = getattr(self.cfg, "TEST", None)
        base_factor = float(self.params.search_factor)
        self.last_confident_mismatch_search = 0.0
        if not bool(getattr(test_cfg, "ADAPTIVE_SEARCH_FACTOR", False)):
            self.last_search_factor = base_factor
            return base_factor

        if (bool(getattr(test_cfg, "CONFIDENT_MISMATCH_SEARCH", False)) and
                self.frame_id > int(getattr(test_cfg, "CONFIDENT_MISMATCH_WARMUP_FRAMES", 5))):
            score_ref = self.score_ema if self.score_ema is not None else 1.0
            max_rgb = float(getattr(test_cfg, "CONFIDENT_MISMATCH_MAX_RGB_CONSISTENCY", 0.0))
            max_depth = float(getattr(test_cfg, "CONFIDENT_MISMATCH_MAX_DEPTH_CONSISTENCY", 0.0))
            protect = (
                self.last_template_consistency >= float(getattr(test_cfg, "CONFIDENT_MISMATCH_PROTECT_CONSISTENCY", 0.78)) and
                self.last_template_rgb_consistency >= float(getattr(test_cfg, "CONFIDENT_MISMATCH_PROTECT_RGB", 0.78)) and
                self.last_response_entropy <= float(getattr(test_cfg, "CONFIDENT_MISMATCH_PROTECT_ENTROPY", 0.35)) and
                self.last_response_peak_ratio <= float(getattr(test_cfg, "CONFIDENT_MISMATCH_PROTECT_PEAK_RATIO", 0.35))
            )
            low_consistency = (
                self.last_template_consistency <= float(getattr(test_cfg, "CONFIDENT_MISMATCH_MAX_CONSISTENCY", 0.45)) or
                (max_rgb > 0.0 and self.last_template_rgb_consistency <= max_rgb) or
                (max_depth > 0.0 and self.last_template_depth_consistency <= max_depth)
            )
            response_off = (
                self.last_response_center_distance >= float(getattr(test_cfg, "CONFIDENT_MISMATCH_MIN_CENTER_DISTANCE", 0.28)) or
                self.last_response_entropy >= float(getattr(test_cfg, "CONFIDENT_MISMATCH_MIN_ENTROPY", 0.45)) or
                self.last_response_peak_ratio >= float(getattr(test_cfg, "CONFIDENT_MISMATCH_MIN_PEAK_RATIO", 0.45))
            )
            if (not protect and
                    score_ref >= float(getattr(test_cfg, "CONFIDENT_MISMATCH_SCORE_THR", 0.70)) and
                    low_consistency and response_off):
                factor = float(getattr(test_cfg, "CONFIDENT_MISMATCH_SEARCH_FACTOR", base_factor))
                self.last_search_factor = max(
                    base_factor,
                    min(factor, float(getattr(test_cfg, "ADAPTIVE_SEARCH_MAX_FACTOR", factor))))
                self.last_confident_mismatch_search = 1.0
                return self.last_search_factor

        if (bool(getattr(test_cfg, "DEPTH_DOMINANT_SEARCH", False)) and
                self.frame_id > int(getattr(test_cfg, "DEPTH_DOMINANT_WARMUP_FRAMES", 5))):
            depth_consistency = float(self.last_template_depth_consistency)
            rgb_consistency = float(self.last_template_rgb_consistency)
            score_ref = self.score_ema if self.score_ema is not None else 1.0
            min_depth = float(getattr(test_cfg, "DEPTH_DOMINANT_MIN_DEPTH", 0.85))
            max_rgb = float(getattr(test_cfg, "DEPTH_DOMINANT_MAX_RGB", 0.85))
            min_gap = float(getattr(test_cfg, "DEPTH_DOMINANT_MIN_GAP", 0.12))
            min_entropy = float(getattr(test_cfg, "DEPTH_DOMINANT_MIN_ENTROPY", 0.25))
            max_peak = float(getattr(test_cfg, "DEPTH_DOMINANT_MAX_PEAK_RATIO", 0.65))
            min_score = float(getattr(test_cfg, "DEPTH_DOMINANT_MIN_SCORE", 0.0))
            max_score = float(getattr(test_cfg, "DEPTH_DOMINANT_MAX_SCORE", 1.0))
            depth_dominant = (
                depth_consistency >= min_depth and
                rgb_consistency <= max_rgb and
                depth_consistency - rgb_consistency >= min_gap
            )
            response_uncertain = (
                self.last_response_entropy >= min_entropy or
                self.last_response_peak_ratio >= max_peak
            )
            score_allowed = min_score <= score_ref <= max_score
            if depth_dominant and response_uncertain and score_allowed:
                factor = float(getattr(test_cfg, "DEPTH_DOMINANT_SEARCH_FACTOR", base_factor))
                self.last_search_factor = max(base_factor, min(factor, float(getattr(test_cfg, "ADAPTIVE_SEARCH_MAX_FACTOR", factor))))
                return self.last_search_factor

        start_lost = int(getattr(test_cfg, "ADAPTIVE_SEARCH_START_LOST", 1))
        score_thr = float(getattr(test_cfg, "ADAPTIVE_SEARCH_SCORE_THR", 0.0))
        use_score = score_thr > 0.0 and self.score_ema is not None and self.score_ema < score_thr
        if self.lost_frames < start_lost and not use_score:
            self.last_search_factor = base_factor
            return base_factor
        if bool(getattr(test_cfg, "ADAPTIVE_SEARCH_REQUIRE_RESPONSE_GUARD", False)) and self.last_response_guard <= 0.0:
            self.last_search_factor = base_factor
            return base_factor
        max_consistency = float(getattr(test_cfg, "ADAPTIVE_SEARCH_MAX_CONSISTENCY", 0.0))
        if max_consistency > 0.0 and self.last_template_consistency > max_consistency:
            self.last_search_factor = base_factor
            return base_factor
        if bool(getattr(test_cfg, "ADAPTIVE_SEARCH_RELIABILITY_GATE", False)):
            min_depth = float(getattr(test_cfg, "ADAPTIVE_SEARCH_MIN_DEPTH_CONSISTENCY", 0.0))
            max_rgb = float(getattr(test_cfg, "ADAPTIVE_SEARCH_MAX_RGB_CONSISTENCY", 0.0))
            max_template = float(getattr(test_cfg, "ADAPTIVE_SEARCH_MAX_TEMPLATE_CONSISTENCY", 0.0))
            depth_reliable = min_depth > 0.0 and self.last_template_depth_consistency >= min_depth
            rgb_unreliable = max_rgb > 0.0 and self.last_template_rgb_consistency <= max_rgb
            template_unreliable = max_template <= 0.0 or self.last_template_consistency <= max_template
            if not depth_reliable and not (rgb_unreliable and template_unreliable):
                self.last_search_factor = base_factor
                return base_factor

        step = float(getattr(test_cfg, "ADAPTIVE_SEARCH_STEP", 0.5))
        max_factor = float(getattr(test_cfg, "ADAPTIVE_SEARCH_MAX_FACTOR", base_factor))
        lost_count = max(self.lost_frames - start_lost + 1, 1)
        factor = min(max_factor, base_factor + step * lost_count)
        self.last_search_factor = max(base_factor, factor)
        return self.last_search_factor

    @staticmethod
    def _depth_reliability(depth_quality):
        quality = str(depth_quality).lower()
        if any(k in quality for k in ('reliable', 'good', 'clear', 'valid')):
            return 1.0
        if any(k in quality for k in ('poor', 'unreliable', 'missing', 'invalid', 'noisy', 'low')):
            return 0.25
        return 0.6

    def _language_runtime_gate(self):
        test_cfg = getattr(self.cfg, "TEST", None)
        if not bool(getattr(test_cfg, "LANGUAGE_RUNTIME_GATE", False)):
            return 1.0
        if self.frame_id <= int(getattr(test_cfg, "LANGUAGE_GATE_WARMUP_FRAMES", 2)):
            return 1.0

        gate_min = min(max(float(getattr(test_cfg, "LANGUAGE_GATE_MIN", 0.20)), 0.0), 1.0)
        score_ref = self.score_ema if self.score_ema is not None else 1.0
        low_score = score_ref < float(getattr(test_cfg, "LANGUAGE_GATE_LOW_SCORE", 0.45))
        very_low_score = score_ref < float(getattr(test_cfg, "LANGUAGE_GATE_VERY_LOW_SCORE", 0.25))
        ambiguous = (self.last_response_entropy >= float(getattr(test_cfg, "LANGUAGE_GATE_UNSTABLE_ENTROPY", 0.68)) or
                     self.last_response_peak_ratio >= float(getattr(test_cfg, "LANGUAGE_GATE_UNSTABLE_PEAK_RATIO", 0.82)))
        center_jump = self.last_response_center_distance >= float(
            getattr(test_cfg, "LANGUAGE_GATE_UNSTABLE_CENTER_DISTANCE", 0.12))
        low_consistency = self.last_template_consistency <= float(getattr(test_cfg, "LANGUAGE_GATE_LOW_CONSISTENCY", 0.45))
        low_depth_consistency = self.last_template_depth_consistency <= float(
            getattr(test_cfg, "LANGUAGE_GATE_LOW_DEPTH_CONSISTENCY", 0.25))

        gate_mode = str(getattr(test_cfg, "LANGUAGE_GATE_MODE", "legacy")).lower()
        if gate_mode == "lost_only":
            geometry_drift = (
                score_ref >= float(getattr(test_cfg, "LANGUAGE_GATE_HIGH_SCORE", 0.75)) and
                self.last_template_rgb_consistency >= float(getattr(test_cfg, "LANGUAGE_GATE_GEOM_RGB_MIN", 0.68)) and
                self.last_template_consistency >= float(getattr(test_cfg, "LANGUAGE_GATE_GEOM_CONSISTENCY_MIN", 0.65)) and
                self.last_template_depth_consistency <= float(getattr(test_cfg, "LANGUAGE_GATE_GEOM_DEPTH_MAX", 0.30)) and
                self.last_response_entropy <= float(getattr(test_cfg, "LANGUAGE_GATE_GEOM_ENTROPY_MAX", 0.50)) and
                self.last_response_peak_ratio <= float(getattr(test_cfg, "LANGUAGE_GATE_GEOM_PEAK_RATIO_MAX", 0.55))
            )
            low_score_lost = low_score and (ambiguous or center_jump or low_consistency)
            if very_low_score or low_score_lost or geometry_drift:
                return 1.0
            return gate_min

        if low_score or ambiguous or low_consistency or low_depth_consistency:
            return 1.0

        confident = score_ref >= float(getattr(test_cfg, "LANGUAGE_GATE_HIGH_SCORE", 0.75))
        stable_response = (self.last_response_entropy <= float(getattr(test_cfg, "LANGUAGE_GATE_STABLE_ENTROPY", 0.45)) and
                           self.last_response_peak_ratio <= float(getattr(test_cfg, "LANGUAGE_GATE_STABLE_PEAK_RATIO", 0.45)))
        stable_template = (self.last_template_consistency >= float(getattr(test_cfg, "LANGUAGE_GATE_STABLE_CONSISTENCY", 0.78)) and
                           self.last_template_depth_consistency >= float(
                               getattr(test_cfg, "LANGUAGE_GATE_STABLE_DEPTH_CONSISTENCY", 0.50)))
        if confident and stable_response and stable_template:
            return gate_min
        return 1.0

    @staticmethod
    def _first_nonempty(*values):
        for value in values:
            if isinstance(value, str) and value:
                return value
        return ''

    def _build_language_inputs(self):
        category = self.language_info.get('language_category', self.language_info.get('object_class_name', 'object'))
        appearance = self.language_info.get('language_appearance', '')
        description = self.language_info.get('language_description', '')
        depth_relation = self.language_info.get('language_depth_relation', '')
        depth_quality = self.language_info.get('language_depth_quality', '')
        occlusion = self.language_info.get('language_occlusion_state', '')
        distractor = self.language_info.get('language_distractor_relation', '')

        target_parts = []
        if category:
            target_parts.append('target object: {}'.format(category))
        if appearance:
            target_parts.append('appearance: {}'.format(appearance))
        if depth_relation:
            target_parts.append('target depth geometry: {}'.format(depth_relation))
        target_text = self._first_nonempty('; '.join(target_parts), description, 'target object')
        context_parts = ['non-target surrounding background and distractor regions']
        if distractor:
            context_parts.append('distractors/context to suppress: {}'.format(distractor))
        if occlusion:
            context_parts.append('occlusion and surrounding state: {}'.format(occlusion))
        if depth_quality:
            context_parts.append('depth reliability cue: {}'.format(depth_quality))
        context_parts.append('not the tracked target')
        context_text = '; '.join(context_parts)
        runtime_gate = self._language_runtime_gate()
        self.last_language_runtime_gate = runtime_gate
        return [target_text], [context_text], [self._depth_reliability(depth_quality)], [runtime_gate]

    def initialize(self, image, info: dict):
        self.language_info = {k.replace('init_', ''): v for k, v in info.items() if k.startswith('init_language_')}
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image[0], info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = [info['init_bbox'], info['init_bbox']]
        self.velocity = [0.0, 0.0]
        self.stable_frames = 0
        self.last_update_frame = 0
        self.lost_frames = 0
        (self.template_signature,
         self.template_rgb_signature,
         self.template_depth_signature) = self._extract_box_signatures(image[0], info['init_bbox'])
        (self.template_rgb_spatial_signature,
         self.template_depth_spatial_signature) = self._extract_spatial_signatures(image[0], info['init_bbox'])
        self.last_template_consistency = 1.0
        self.last_template_rgb_consistency = 1.0
        self.last_template_depth_consistency = 1.0
        self.template_consistency_ema = 1.0
        self.template_rgb_consistency_ema = 1.0
        self.template_depth_consistency_ema = 1.0
        self.last_template_spatial_rgb_consistency = 1.0
        self.last_template_spatial_depth_consistency = 1.0
        self.last_update_ratio = 1.0
        self.score_ema = None
        self.last_effective_lost_thr = float(getattr(self.cfg.TEST, "LOST_SCORE_THR", 0.25))
        self.last_effective_damping_thr = float(getattr(self.cfg.TEST, "DAMPING_SCORE_THR", 0.40))
        self.last_response_entropy = 0.0
        self.last_response_peak_ratio = 0.0
        self.last_response_center_distance = 0.0
        self.last_response_guard = 0.0
        self.last_depth_guard = 0.0
        self.last_window_penalty = 0.0
        self.last_search_factor = float(self.params.search_factor)
        self.last_confident_mismatch_search = 0.0
        self.last_candidate_rerank = 0.0
        self.last_candidate_rank = 0.0
        self.last_candidate_score_gain = 0.0
        self.last_candidate_consistency_gain = 0.0
        self.last_candidate_selected_score_ratio = 0.0
        self.last_candidate_selected_rgb_consistency = 0.0
        self.last_candidate_selected_depth_consistency = 0.0
        self.last_candidate_selected_motion_score = 0.0
        self.last_candidate_selected_iou = -1.0
        self.last_candidate_oracle_iou = -1.0
        self.last_candidate_oracle_rank = -1.0
        self.last_candidate_oracle_score_ratio = 0.0
        self.last_candidate_oracle_rgb_consistency = 0.0
        self.last_candidate_oracle_depth_consistency = 0.0
        self.last_candidate_oracle_consistency = 0.0
        self.last_candidate_oracle_motion_score = 0.0
        self.candidate_rerank_streak = 0
        self.last_candidate_probation = 0.0
        self.last_candidate_recovery_trigger = 0.0
        self.last_candidate_stable_block = 0.0
        self.last_candidate_ordinary_trigger = 0.0
        self.last_language_candidate_rerank = 0.0
        self.last_language_candidate_rank = 0.0
        self.last_language_candidate_best = 0.0
        self.last_language_candidate_selected = 0.0
        self.last_language_candidate_gain = 0.0
        self.last_template_update = 0.0
        self.last_template_update_reason = 0.0
        self.last_template_update_gate_score = 0.0
        self.last_template_update_gate_stable = 0.0
        self.last_template_update_gate_consistency = 0.0
        self.last_template_update_gate_rgb = 0.0
        self.last_template_update_gate_depth = 0.0
        self.last_language_runtime_gate = 1.0
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def update_template(self, image, bbox):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image[1], bbox, self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        (self.template_signature,
         self.template_rgb_signature,
         self.template_depth_signature) = self._extract_box_signatures(image[1], bbox)
        (self.template_rgb_spatial_signature,
         self.template_depth_spatial_signature) = self._extract_spatial_signatures(image[1], bbox)
        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(bbox, resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

    def track(self, image, info: dict = None):
        H, W, _ = image[0].shape
        self.frame_id += 1
        search_factor = self._current_search_factor()
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image[1], self.state[1], search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        x_patch_arr_last, resize_factor_last, x_amask_arr_last = sample_target(image[0], self.state[0], search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        search_last = self.preprocessor.process(x_patch_arr_last, x_amask_arr_last)
        language_kwargs = {}
        if getattr(getattr(self.cfg.MODEL, "LANGUAGE", None), "USE", False):
            language_target, language_context, language_reliability, language_runtime_gate = self._build_language_inputs()
            language_kwargs = {
                'language_target': language_target,
                'language_context': language_context,
                'language_reliability': language_reliability,
                'language_runtime_gate': language_runtime_gate,
            }
        with torch.no_grad():
            x_dict = search
            x_dict_last = search_last
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=[self.z_dict1.tensors[:,:3,:,:],self.z_dict1.tensors[:,3:,:,:]],
                search=[x_dict.tensors[:,:3,:,:], x_dict.tensors[:,3:,:,:], x_dict_last.tensors[:,:3,:,:], x_dict_last.tensors[:,3:,:,:]],
                ce_template_mask=self.box_mask_z,
                **language_kwargs)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self._apply_score_window(pred_score_map)
        self._update_response_diagnostics(response)
        prev_state = self.state[1]
        mapped_box, max_score = self._select_candidate_box(
            response, out_dict['size_map'], out_dict['offset_map'], resize_factor, image[1], H, W,
            gt_box=info.get('gt_bbox') if info is not None else None,
            language_maps=out_dict)
        mapped_box = self._stabilize_low_confidence_box(mapped_box, max_score, image=image[1])
        self.state[0] = self.state[1]
        self.state[1] = clip_box(mapped_box, H, W, margin=10)
        self._update_temporal_state(prev_state, self.state[1], max_score)
        self._maybe_update_dynamic_template(image, max_score)
        calibrated_score = self._calibrate_output_score(max_score)
        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                # pred_score_map_gaussian = gaussian_filter(pred_score_map.cpu(), sigma=5)
                # self.visdom.register((image[-1][:, :, :3], info['gt_bbox'].tolist(), self.state[-1]), 'Tracking', 1, 'Tracking')
                # self.visdom.register(torch.from_numpy(x_patch_arr[:, :, :3]).permute(2, 0, 1), 'image', 1, 'search_region')
                # self.visdom.register(torch.from_numpy(self.z_patch_arr[:, :, :3]).permute(2, 0, 1), 'image', 1, 'template')
                # self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                # self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')
                #
                # if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                #     removed_indexes_s = out_dict['removed_indexes_s']
                #     removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                #     masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                #     self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')
                if self.frame_id % 5 == 0:
                    vis_attn_maps(pred_score_map.cpu(), x1=self.z_patch_arr[:, :, :3], x2=x_patch_arr[:, :, :3], x3=x_patch_arr[:, :, 3:],
                                  x1_title='TemplateV', x2_title='SearchV', x3_title='SearchI',
                                  save_path='vis_attn_weights/t2sv_vis/%04d_v' % self.frame_id)

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            pred_boxes, _ = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
            pred_boxes = pred_boxes.view(-1, 4)
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state[1],
                    "best_score": calibrated_score,
                    "raw_score": max_score,
                    "score_calibration": self.last_score_calibration,
                    "template_consistency": self.last_template_consistency,
                    "template_rgb_consistency": self.last_template_rgb_consistency,
                    "template_depth_consistency": self.last_template_depth_consistency,
                    "template_consistency_ema": self.template_consistency_ema,
                    "template_rgb_consistency_ema": self.template_rgb_consistency_ema,
                    "template_depth_consistency_ema": self.template_depth_consistency_ema,
                    "template_spatial_rgb_consistency": self.last_template_spatial_rgb_consistency,
                    "template_spatial_depth_consistency": self.last_template_spatial_depth_consistency,
                    "state_update_ratio": self.last_update_ratio,
                    "score_ema": self.score_ema if self.score_ema is not None else max_score,
                    "effective_lost_thr": self.last_effective_lost_thr,
                    "effective_damping_thr": self.last_effective_damping_thr,
                    "response_entropy": self.last_response_entropy,
                    "response_peak_ratio": self.last_response_peak_ratio,
                    "response_center_distance": self.last_response_center_distance,
                    "response_guard": self.last_response_guard,
                    "depth_guard": self.last_depth_guard,
                    "center_follow": self.last_center_follow,
                    "window_penalty": self.last_window_penalty,
                    "search_factor": self.last_search_factor,
                    "confident_mismatch_search": self.last_confident_mismatch_search,
                    "candidate_rerank": self.last_candidate_rerank,
                    "candidate_rank": self.last_candidate_rank,
                    "candidate_score_gain": self.last_candidate_score_gain,
                    "candidate_consistency_gain": self.last_candidate_consistency_gain,
                    "candidate_selected_score_ratio": self.last_candidate_selected_score_ratio,
                    "candidate_selected_rgb_consistency": self.last_candidate_selected_rgb_consistency,
                    "candidate_selected_depth_consistency": self.last_candidate_selected_depth_consistency,
                    "candidate_selected_motion_score": self.last_candidate_selected_motion_score,
                    "candidate_selected_iou": self.last_candidate_selected_iou,
                    "candidate_oracle_iou": self.last_candidate_oracle_iou,
                    "candidate_oracle_rank": self.last_candidate_oracle_rank,
                    "candidate_oracle_score_ratio": self.last_candidate_oracle_score_ratio,
                    "candidate_oracle_rgb_consistency": self.last_candidate_oracle_rgb_consistency,
                    "candidate_oracle_depth_consistency": self.last_candidate_oracle_depth_consistency,
                    "candidate_oracle_consistency": self.last_candidate_oracle_consistency,
                    "candidate_oracle_motion_score": self.last_candidate_oracle_motion_score,
                    "candidate_probation": self.last_candidate_probation,
                    "candidate_recovery_trigger": self.last_candidate_recovery_trigger,
                    "candidate_stable_block": self.last_candidate_stable_block,
                    "candidate_ordinary_trigger": self.last_candidate_ordinary_trigger,
                    "language_candidate_rerank": self.last_language_candidate_rerank,
                    "language_candidate_rank": self.last_language_candidate_rank,
                    "language_candidate_best": self.last_language_candidate_best,
                    "language_candidate_selected": self.last_language_candidate_selected,
                    "language_candidate_gain": self.last_language_candidate_gain,
                    "template_update": self.last_template_update,
                    "template_update_reason": self.last_template_update_reason,
                    "template_update_gate_score": self.last_template_update_gate_score,
                    "template_update_gate_stable": self.last_template_update_gate_stable,
                    "template_update_gate_consistency": self.last_template_update_gate_consistency,
                    "template_update_gate_rgb": self.last_template_update_gate_rgb,
                    "template_update_gate_depth": self.last_template_update_gate_depth,
                    "language_runtime_gate": self.last_language_runtime_gate}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[1][0] + 0.5 * self.state[1][2], self.state[1][1] + 0.5 * self.state[1][3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        curr_state = self.state[1]
        cx_prev = curr_state[0] + 0.5 * curr_state[2]
        cy_prev = curr_state[1] + 0.5 * curr_state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return MPLTTrack
