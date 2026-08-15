"""Fail-closed policy for one bounded RGB-D dynamic-template slot."""

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.test.tracker.temporal_depth_identity import (
    normalized_center_jump,
    read_candidate_depth,
)


@dataclass(frozen=True)
class SafeTemplateDecision:
    checked: bool
    eligible: bool
    replace_dynamic: bool
    drop_dynamic: bool
    rollback_state: bool
    reasons: Tuple[str, ...]
    stable_frames: int
    dynamic_active: bool
    consecutive_state_rollbacks: int
    identity_similarity: Optional[float]
    normalized_center_jump: Optional[float]
    log_depth_change: Optional[float]
    blend_weight: float


def _finite_bbox(bbox):
    try:
        values = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if (len(values) != 4 or
            not all(math.isfinite(value) for value in values) or
            values[2] <= 0.0 or values[3] <= 0.0):
        return None
    return values


def _rgb_descriptor(image, bbox):
    """Return a compact immutable-target color descriptor or ``None``."""
    if (not isinstance(image, np.ndarray) or image.ndim != 3 or
            image.shape[2] < 3 or image.size == 0):
        return None
    bbox = _finite_bbox(bbox)
    if bbox is None:
        return None
    x, y, width, height = bbox
    image_height, image_width = image.shape[:2]
    x0 = min(max(int(math.floor(x)), 0), image_width)
    y0 = min(max(int(math.floor(y)), 0), image_height)
    x1 = min(max(int(math.ceil(x + width)), 0), image_width)
    y1 = min(max(int(math.ceil(y + height)), 0), image_height)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = image[y0:y1, x0:x1, :3]
    if crop.size == 0 or not bool(np.isfinite(crop).all()):
        return None
    crop = np.clip(crop, 0, 255).astype(np.uint8, copy=False)
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        hs_hist = cv2.calcHist(
            [hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]).reshape(-1)
        gray_hist = cv2.calcHist(
            [gray], [0], None, [32], [0, 256]).reshape(-1)
    except cv2.error:
        return None
    hs_total = float(hs_hist.sum())
    gray_total = float(gray_hist.sum())
    if hs_total <= 0.0 or gray_total <= 0.0:
        return None
    descriptor = np.concatenate((
        0.75 * hs_hist / hs_total,
        0.25 * gray_hist / gray_total,
    )).astype(np.float32, copy=False)
    return descriptor


def rgb_identity_similarity(reference_descriptor, image, bbox):
    """Bhattacharyya similarity to the immutable first-frame RGB target."""
    candidate = _rgb_descriptor(image, bbox)
    if (not isinstance(reference_descriptor, np.ndarray) or
            candidate is None or candidate.shape != reference_descriptor.shape or
            not bool(np.isfinite(reference_descriptor).all())):
        return None
    similarity = float(np.sqrt(
        np.maximum(reference_descriptor, 0.0) *
        np.maximum(candidate, 0.0)).sum())
    if not math.isfinite(similarity):
        return None
    return min(max(similarity, 0.0), 1.0)


def nms_response_margin(score_map, selected_index, kernel=5):
    """Return selected-peak margin after suppressing its local neighborhood."""
    if (not torch.is_tensor(score_map) or score_map.ndim != 4 or
            score_map.shape[0] != 1 or score_map.shape[1] != 1 or
            not score_map.is_floating_point() or
            not bool(torch.isfinite(score_map).all())):
        return None
    if (isinstance(kernel, bool) or not isinstance(kernel, int) or
            kernel <= 0 or kernel % 2 == 0):
        return None
    try:
        index = int(torch.as_tensor(selected_index).reshape(-1)[0].item())
    except (TypeError, ValueError, RuntimeError, IndexError):
        return None
    height, width = score_map.shape[-2:]
    if index < 0 or index >= height * width:
        return None
    row, column = divmod(index, width)
    selected = score_map.flatten()[index]
    pooled = F.max_pool2d(
        score_map, kernel_size=kernel, stride=1, padding=kernel // 2)
    peaks = score_map.masked_fill(score_map != pooled, float('-inf')).clone()
    radius = kernel // 2
    peaks[:, :, max(0, row - radius):min(height, row + radius + 1),
          max(0, column - radius):min(width, column + radius + 1)] = (
              float('-inf'))
    competitor = peaks.flatten().max()
    if not bool(torch.isfinite(competitor)):
        competitor = selected.new_zeros(())
    return max(float((selected - competitor).item()), 0.0)


class SafeTemplateUpdatePolicy:
    """Require simultaneous visual, temporal, response, and depth evidence."""

    def __init__(
            self, check_interval=5, min_update_interval=30,
            min_stable_frames=3, min_confidence=0.65,
            min_response_margin=0.10, max_center_jump=0.35,
            min_rgb_identity=0.75, max_log_depth_change=0.08,
            min_depth_valid_ratio=0.50, center_fraction=0.80,
            blend_weight=0.10, max_blend_weight=0.20,
            max_template_age=90,
            hard_conflict_state_rollback=False,
            max_consecutive_state_rollbacks=1):
        integer_values = {
            'CHECK_INTERVAL': check_interval,
            'MIN_UPDATE_INTERVAL': min_update_interval,
            'MIN_STABLE_FRAMES': min_stable_frames,
            'MAX_TEMPLATE_AGE': max_template_age,
        }
        for name, value in integer_values.items():
            if (isinstance(value, bool) or not isinstance(value, Integral) or
                    int(value) <= 0):
                raise ValueError('{} must be a positive integer'.format(name))
        self.check_interval = int(check_interval)
        self.min_update_interval = int(min_update_interval)
        self.min_stable_frames = int(min_stable_frames)
        self.max_template_age = int(max_template_age)

        if not isinstance(hard_conflict_state_rollback, bool):
            raise ValueError(
                'HARD_CONFLICT_STATE_ROLLBACK must be a bool')
        if (isinstance(max_consecutive_state_rollbacks, bool) or
                not isinstance(max_consecutive_state_rollbacks, Integral) or
                int(max_consecutive_state_rollbacks) <= 0):
            raise ValueError(
                'MAX_CONSECUTIVE_STATE_ROLLBACKS must be a positive int')
        self.hard_conflict_state_rollback = hard_conflict_state_rollback
        self.max_consecutive_state_rollbacks = int(
            max_consecutive_state_rollbacks)
        self.min_confidence = float(min_confidence)
        self.min_response_margin = float(min_response_margin)
        self.max_center_jump = float(max_center_jump)
        self.min_rgb_identity = float(min_rgb_identity)
        self.max_log_depth_change = float(max_log_depth_change)
        self.min_depth_valid_ratio = float(min_depth_valid_ratio)
        self.center_fraction = float(center_fraction)
        self.blend_weight = float(blend_weight)
        self.max_blend_weight = float(max_blend_weight)
        scalars = (
            self.min_confidence, self.min_response_margin,
            self.max_center_jump, self.min_rgb_identity,
            self.max_log_depth_change, self.min_depth_valid_ratio,
            self.center_fraction, self.blend_weight,
            self.max_blend_weight)
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError('Safe template thresholds must be finite')
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError('MIN_CONFIDENCE must lie in [0, 1]')
        if self.min_response_margin < 0.0 or self.max_center_jump < 0.0:
            raise ValueError('Response/motion thresholds must be non-negative')
        if not 0.0 <= self.min_rgb_identity <= 1.0:
            raise ValueError('MIN_RGB_IDENTITY must lie in [0, 1]')
        if self.max_log_depth_change < 0.0:
            raise ValueError('MAX_LOG_DEPTH_CHANGE must be non-negative')
        if not 0.0 <= self.min_depth_valid_ratio <= 1.0:
            raise ValueError('MIN_DEPTH_VALID_RATIO must lie in [0, 1]')
        if not 0.0 < self.center_fraction <= 1.0:
            raise ValueError('CENTER_FRACTION must lie in (0, 1]')
        if (not 0.0 < self.blend_weight <= self.max_blend_weight or
                self.max_blend_weight > 1.0):
            raise ValueError(
                'BLEND_WEIGHT must lie in (0, MAX_BLEND_WEIGHT] <= 1')
        self.reset()

    @classmethod
    def from_config(cls, config):
        def value(name):
            if isinstance(config, dict):
                return config[name]
            return getattr(config, name)
        return cls(
            check_interval=value('CHECK_INTERVAL'),
            min_update_interval=value('MIN_UPDATE_INTERVAL'),
            min_stable_frames=value('MIN_STABLE_FRAMES'),
            min_confidence=value('MIN_CONFIDENCE'),
            min_response_margin=value('MIN_RESPONSE_MARGIN'),
            max_center_jump=value('MAX_CENTER_JUMP'),
            min_rgb_identity=value('MIN_RGB_IDENTITY'),
            max_log_depth_change=value('MAX_LOG_DEPTH_CHANGE'),
            min_depth_valid_ratio=value('MIN_DEPTH_VALID_RATIO'),
            center_fraction=value('CENTER_FRACTION'),
            blend_weight=value('BLEND_WEIGHT'),
            max_blend_weight=value('MAX_BLEND_WEIGHT'),
            max_template_age=value('MAX_TEMPLATE_AGE'),
            hard_conflict_state_rollback=value(
                'HARD_CONFLICT_STATE_ROLLBACK'),
            max_consecutive_state_rollbacks=value(
                'MAX_CONSECUTIVE_STATE_ROLLBACKS'))

    def reset(self):
        self.reference_descriptor = None
        self.previous_bbox = None
        self.trusted_depth = None
        self.last_frame_id = 0
        self.last_update_frame = None
        self.pending_update_frame = None
        self.stable_frames = 0
        self.dynamic_active = False
        self.consecutive_state_rollbacks = 0

    def initialize(self, image, bbox, depth_path):
        """Freeze first-frame RGB/depth evidence; return whether it is usable."""
        self.reset()
        self.reference_descriptor = _rgb_descriptor(image, bbox)
        bbox_values = _finite_bbox(bbox)
        observation = read_candidate_depth(
            depth_path, bbox, center_fraction=self.center_fraction)
        if (self.reference_descriptor is None or bbox_values is None or
                observation.median is None or
                observation.valid_ratio < self.min_depth_valid_ratio):
            self.reference_descriptor = None
            return False
        self.previous_bbox = list(bbox_values)
        self.trusted_depth = float(observation.median)
        return True

    def _decision(
            self, checked, eligible, replace_dynamic, drop_dynamic, reasons,
            identity=None, jump=None, depth_change=None,
            rollback_state=False):
        return SafeTemplateDecision(
            checked=bool(checked), eligible=bool(eligible),
            replace_dynamic=bool(replace_dynamic),
            drop_dynamic=bool(drop_dynamic), reasons=tuple(reasons),
            rollback_state=bool(rollback_state),
            stable_frames=int(self.stable_frames),
            dynamic_active=bool(self.dynamic_active),
            consecutive_state_rollbacks=int(
                self.consecutive_state_rollbacks),
            identity_similarity=identity,
            normalized_center_jump=jump,
            log_depth_change=depth_change,
            blend_weight=(self.blend_weight if self.dynamic_active else 0.0))

    def observe(
            self, frame_id, image, bbox, confidence, response_margin,
            depth_path, temporal_rejected=False):
        """Evaluate one frame.  Replacement still requires ``commit``."""
        if self.reference_descriptor is None or self.trusted_depth is None:
            return self._decision(
                False, False, False, False, ('anchor_unavailable',))
        if (isinstance(frame_id, bool) or not isinstance(frame_id, Integral) or
                int(frame_id) <= self.last_frame_id):
            return self._decision(
                False, False, False, False, ('malformed_frame_id',))
        frame_id = int(frame_id)
        self.last_frame_id = frame_id
        bbox_values = _finite_bbox(bbox)
        try:
            confidence = float(confidence)
            response_margin = float(response_margin)
        except (TypeError, ValueError, OverflowError):
            return self._decision(
                False, False, False, False, ('malformed_evidence',))
        if (bbox_values is None or not math.isfinite(confidence) or
                not math.isfinite(response_margin) or
                not 0.0 <= confidence <= 1.0 or response_margin < 0.0 or
                not isinstance(temporal_rejected, bool)):
            return self._decision(
                False, False, False, False, ('malformed_evidence',))

        drop_dynamic = False
        if (self.dynamic_active and self.last_update_frame is not None and
                frame_id - self.last_update_frame > self.max_template_age):
            self.dynamic_active = False
            drop_dynamic = True

        identity = rgb_identity_similarity(
            self.reference_descriptor, image, bbox_values)
        jump = normalized_center_jump(self.previous_bbox, bbox_values)
        observation = read_candidate_depth(
            depth_path, bbox_values, center_fraction=self.center_fraction)
        depth_valid = bool(
            observation.median is not None and
            observation.valid_ratio >= self.min_depth_valid_ratio)
        depth_change = None
        if depth_valid:
            depth_change = abs(math.log(
                float(observation.median) / self.trusted_depth))

        reasons = []
        if confidence < self.min_confidence:
            reasons.append('low_confidence')
        if response_margin < self.min_response_margin:
            reasons.append('small_response_margin')
        if jump > self.max_center_jump:
            reasons.append('large_center_jump')
        if identity is None:
            reasons.append('malformed_rgb_identity')
        elif identity < self.min_rgb_identity:
            reasons.append('low_static_rgb_identity')
        if not depth_valid:
            reasons.append('missing_or_unreliable_depth')
        elif depth_change > self.max_log_depth_change:
            reasons.append('large_depth_change')
        if temporal_rejected:
            reasons.append('temporal_identity_rejected')

        hard_conflict = any(reason in reasons for reason in (
            'large_center_jump', 'low_static_rgb_identity',
            'large_depth_change', 'temporal_identity_rejected'))
        rollback_state = False
        if hard_conflict:
            self.pending_update_frame = None
            self.stable_frames = 0
            if self.dynamic_active:
                self.dynamic_active = False
                drop_dynamic = True
            if (self.hard_conflict_state_rollback and
                    self.consecutive_state_rollbacks <
                    self.max_consecutive_state_rollbacks):
                # Keep bbox/depth references aligned with the state that the
                # tracker will restore.  The bounded budget prevents a real
                # fast-moving target from being frozen indefinitely.
                rollback_state = True
                self.consecutive_state_rollbacks += 1
            else:
                # Existing safe-v1 remains fail-open.  A v3 rollback budget
                # exhaustion also rebases the accepted recursive state, and
                # the budget stays exhausted until a non-conflict frame.
                self.previous_bbox = list(bbox_values)
                if self.hard_conflict_state_rollback and depth_valid:
                    self.trusted_depth = float(observation.median)
        elif reasons:
            self.previous_bbox = list(bbox_values)
            self.stable_frames = 0
            self.consecutive_state_rollbacks = 0
        else:
            self.previous_bbox = list(bbox_values)
            self.stable_frames += 1
            self.trusted_depth = float(observation.median)
            self.consecutive_state_rollbacks = 0

        checked = frame_id % self.check_interval == 0
        if reasons:
            return self._decision(
                checked, False, False, drop_dynamic, reasons,
                identity, jump, depth_change,
                rollback_state=rollback_state)
        if not checked:
            return self._decision(
                False, True, False, drop_dynamic, ('not_check_frame',),
                identity, jump, depth_change)
        if self.pending_update_frame is not None:
            return self._decision(
                True, True, False, drop_dynamic, ('update_pending',),
                identity, jump, depth_change)
        if self.stable_frames < self.min_stable_frames:
            return self._decision(
                True, True, False, drop_dynamic,
                ('stable_streak_incomplete',), identity, jump, depth_change)
        if (self.last_update_frame is not None and
                frame_id - self.last_update_frame < self.min_update_interval):
            return self._decision(
                True, True, False, drop_dynamic,
                ('minimum_update_interval',), identity, jump, depth_change)

        self.pending_update_frame = frame_id
        self.stable_frames = 0
        return self._decision(
            True, True, True, drop_dynamic, (),
            identity, jump, depth_change)

    def commit(self, frame_id):
        if int(frame_id) != self.pending_update_frame:
            raise ValueError('Template commit does not match the pending frame')
        self.pending_update_frame = None
        self.last_update_frame = int(frame_id)
        self.dynamic_active = True
        return self.blend_weight

    def synchronized_commit(self, frame_id):
        """Commit a same-frame template authorized by a stricter peer policy."""
        if (isinstance(frame_id, bool) or not isinstance(frame_id, Integral) or
                int(frame_id) != self.last_frame_id or
                self.reference_descriptor is None or self.trusted_depth is None):
            raise ValueError(
                'Synchronized template commit requires current-frame evidence')
        self.pending_update_frame = None
        self.last_update_frame = int(frame_id)
        self.stable_frames = 0
        self.dynamic_active = True
        return self.blend_weight

    def cancel(self, frame_id):
        if int(frame_id) == self.pending_update_frame:
            self.pending_update_frame = None


__all__ = [
    'SafeTemplateDecision', 'SafeTemplateUpdatePolicy',
    'nms_response_margin', 'rgb_identity_similarity',
]
