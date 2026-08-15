"""Online RGB-D identity rejection without ground-truth information.

The gate reads the raw depth image rather than the colorized depth channels
seen by the network.  A candidate is rejected only when motion, depth and
score all provide strong and simultaneous evidence of an identity switch.
Unavailable or unreliable depth therefore leaves the visual tracker's output
unchanged for that frame.
"""

from dataclasses import dataclass
import math
from numbers import Integral
import os
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthObservation:
    """Robust raw-depth summary for the central part of one candidate box."""

    median: Optional[float]
    valid_ratio: float
    valid_count: int
    sample_count: int


@dataclass(frozen=True)
class TemporalDepthDecision:
    """One gate decision and its auditable, non-GT diagnostics."""

    bbox: list
    score: float
    rejected: bool
    depth: Optional[float]
    depth_valid_ratio: float
    normalized_center_jump: float
    log_depth_change: Optional[float]
    score_ratio: float


def _finite_bbox(bbox: Iterable[float]) -> Optional[Tuple[float, float, float, float]]:
    try:
        values = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return None
    if values[2] <= 0.0 or values[3] <= 0.0:
        return None
    return values


def read_depth_image(depth_path):
    """Read one scalar raw-depth image, returning ``None`` on any failure."""
    if depth_path is None or isinstance(depth_path, bool):
        return None
    try:
        path = os.fspath(depth_path)
    except TypeError:
        return None
    if not path:
        return None
    try:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except (cv2.error, OSError, TypeError):
        return None
    if depth is None:
        return None
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2 or depth.size == 0:
        return None
    return depth


def observe_candidate_depth(depth, bbox, center_fraction=0.8):
    """Summarize one candidate from an already-loaded raw-depth image."""
    unavailable = DepthObservation(
        median=None, valid_ratio=0.0, valid_count=0, sample_count=0)
    if not isinstance(depth, np.ndarray) or depth.ndim != 2 or depth.size == 0:
        return unavailable
    bbox_values = _finite_bbox(bbox)
    if bbox_values is None:
        return unavailable

    try:
        fraction = float(center_fraction)
    except (TypeError, ValueError):
        return unavailable
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        return unavailable

    x, y, width, height = bbox_values
    inset = 0.5 * (1.0 - fraction)
    x0 = int(math.floor(x + inset * width))
    y0 = int(math.floor(y + inset * height))
    x1 = int(math.ceil(x + (1.0 - inset) * width))
    y1 = int(math.ceil(y + (1.0 - inset) * height))
    image_height, image_width = depth.shape
    x0 = min(max(x0, 0), image_width)
    y0 = min(max(y0, 0), image_height)
    x1 = min(max(x1, 0), image_width)
    y1 = min(max(y1, 0), image_height)
    if x1 <= x0 or y1 <= y0:
        return unavailable

    crop = depth[y0:y1, x0:x1]
    sample_count = int(crop.size)
    if sample_count == 0:
        return unavailable
    valid = np.isfinite(crop) & (crop > 0)
    valid_count = int(np.count_nonzero(valid))
    valid_ratio = float(valid_count / sample_count)
    if valid_count == 0:
        return DepthObservation(
            median=None, valid_ratio=valid_ratio,
            valid_count=valid_count, sample_count=sample_count)
    median = float(np.median(crop[valid].astype(np.float64, copy=False)))
    if not math.isfinite(median) or median <= 0.0:
        median = None
    return DepthObservation(
        median=median, valid_ratio=valid_ratio,
        valid_count=valid_count, sample_count=sample_count)


def read_candidate_depth(depth_path, bbox, center_fraction=0.8):
    """Read and summarize the centered raw depth within ``bbox``.

    Invalid paths, non-scalar depth images, invalid boxes and empty crops all
    return an unavailable observation.  This is intentional: missing depth is
    a strict fallback condition, never evidence for rejection.
    """
    return observe_candidate_depth(
        read_depth_image(depth_path), bbox,
        center_fraction=center_fraction)


def normalized_center_jump(previous_bbox, candidate_bbox, eps=1.0e-6):
    """Center displacement normalized by the trusted box's geometric scale."""

    previous = _finite_bbox(previous_bbox)
    candidate = _finite_bbox(candidate_bbox)
    if previous is None or candidate is None:
        return 0.0
    previous_cx = previous[0] + 0.5 * previous[2]
    previous_cy = previous[1] + 0.5 * previous[3]
    candidate_cx = candidate[0] + 0.5 * candidate[2]
    candidate_cy = candidate[1] + 0.5 * candidate[3]
    scale = math.sqrt(previous[2] * previous[3])
    return float(math.hypot(candidate_cx - previous_cx,
                            candidate_cy - previous_cy) / max(scale, eps))


class TemporalDepthIdentityGate:
    """Conservative temporal identity switch rejector for RGB-D tracking."""

    def __init__(self, center_fraction=0.8, min_valid_ratio=0.5,
                 center_jump_thr=1.5, log_depth_thr=0.12,
                 score_ratio_thr=1.5, reject_score_scale=0.05,
                 eps=1.0e-6, max_consecutive_rejections=1,
                 distractor_aware_use=False,
                 distractor_center_jump_thr=1.2,
                 distractor_log_depth_thr=0.06,
                 distractor_min_score=0.4,
                 distractor_max_score_ratio=1.0,
                 distractor_recovery_center_jump_thr=0.8,
                 distractor_recovery_log_depth_thr=0.08,
                 distractor_max_quarantine_frames=3,
                 distractor_cumulative_anchor_use=False,
                 distractor_anchor_arm_center_jump_thr=0.8):
        if (isinstance(max_consecutive_rejections, bool) or
                not isinstance(max_consecutive_rejections, Integral)):
            raise ValueError('MAX_CONSECUTIVE_REJECTIONS must be a non-negative int')
        if (isinstance(distractor_max_quarantine_frames, bool) or
                not isinstance(distractor_max_quarantine_frames, Integral)):
            raise ValueError(
                'DISTRACTOR_MAX_QUARANTINE_FRAMES must be a non-negative int')
        self.center_fraction = float(center_fraction)
        self.min_valid_ratio = float(min_valid_ratio)
        self.center_jump_thr = float(center_jump_thr)
        self.log_depth_thr = float(log_depth_thr)
        self.score_ratio_thr = float(score_ratio_thr)
        self.reject_score_scale = float(reject_score_scale)
        self.eps = float(eps)
        self.max_consecutive_rejections = int(max_consecutive_rejections)
        self.distractor_aware_use = bool(distractor_aware_use)
        self.distractor_center_jump_thr = float(
            distractor_center_jump_thr)
        self.distractor_log_depth_thr = float(distractor_log_depth_thr)
        self.distractor_min_score = float(distractor_min_score)
        self.distractor_max_score_ratio = float(
            distractor_max_score_ratio)
        self.distractor_recovery_center_jump_thr = float(
            distractor_recovery_center_jump_thr)
        self.distractor_recovery_log_depth_thr = float(
            distractor_recovery_log_depth_thr)
        self.distractor_max_quarantine_frames = int(
            distractor_max_quarantine_frames)
        self.distractor_cumulative_anchor_use = bool(
            distractor_cumulative_anchor_use)
        self.distractor_anchor_arm_center_jump_thr = float(
            distractor_anchor_arm_center_jump_thr)
        self._validate_parameters()
        self.trusted_bbox = None
        self.trusted_depth = None
        self.last_accepted_score = None
        self.consecutive_rejections = 0
        self.similar_distractors = False
        self.distractor_quarantine = False
        self.distractor_quarantine_frames = 0
        self.distractor_anchor_bbox = None
        self.distractor_anchor_depth = None
        self.distractor_anchor_score = None

    def _validate_parameters(self):
        values = (
            self.center_fraction, self.min_valid_ratio,
            self.center_jump_thr, self.log_depth_thr,
            self.score_ratio_thr, self.reject_score_scale, self.eps,
            self.distractor_center_jump_thr,
            self.distractor_log_depth_thr,
            self.distractor_min_score,
            self.distractor_max_score_ratio,
            self.distractor_recovery_center_jump_thr,
            self.distractor_recovery_log_depth_thr,
            self.distractor_anchor_arm_center_jump_thr)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Temporal depth identity parameters must be finite')
        if not 0.0 < self.center_fraction <= 1.0:
            raise ValueError('CENTER_FRACTION must be in (0, 1]')
        if not 0.0 <= self.min_valid_ratio <= 1.0:
            raise ValueError('MIN_VALID_RATIO must be in [0, 1]')
        if min(self.center_jump_thr, self.log_depth_thr,
               self.score_ratio_thr, self.reject_score_scale,
               self.distractor_center_jump_thr,
               self.distractor_log_depth_thr,
               self.distractor_min_score,
               self.distractor_max_score_ratio,
               self.distractor_recovery_center_jump_thr,
               self.distractor_recovery_log_depth_thr,
               self.distractor_anchor_arm_center_jump_thr) < 0.0:
            raise ValueError('Temporal depth identity thresholds must be non-negative')
        if (self.distractor_anchor_arm_center_jump_thr >
                self.distractor_center_jump_thr):
            raise ValueError(
                'DISTRACTOR_ANCHOR_ARM_CENTER_JUMP_THR must not exceed '
                'DISTRACTOR_CENTER_JUMP_THR')
        if self.eps <= 0.0:
            raise ValueError('EPS must be positive')
        if self.max_consecutive_rejections < 0:
            raise ValueError('MAX_CONSECUTIVE_REJECTIONS must be a non-negative int')
        if self.distractor_max_quarantine_frames < 0:
            raise ValueError(
                'DISTRACTOR_MAX_QUARANTINE_FRAMES must be a non-negative int')

    @classmethod
    def from_config(cls, config):
        """Construct from ``TEST.TEMPORAL_DEPTH_IDENTITY``."""

        def value(name, default=None):
            if isinstance(config, dict):
                return config.get(name, default)
            return getattr(config, name, default)

        return cls(
            center_fraction=value('CENTER_FRACTION'),
            min_valid_ratio=value('MIN_VALID_RATIO'),
            center_jump_thr=value('CENTER_JUMP_THR'),
            log_depth_thr=value('LOG_DEPTH_THR'),
            score_ratio_thr=value('SCORE_RATIO_THR'),
            reject_score_scale=value('REJECT_SCORE_SCALE'),
            eps=value('EPS'),
            max_consecutive_rejections=value('MAX_CONSECUTIVE_REJECTIONS'),
            distractor_aware_use=value('DISTRACTOR_AWARE_USE', False),
            distractor_center_jump_thr=value(
                'DISTRACTOR_CENTER_JUMP_THR', 1.2),
            distractor_log_depth_thr=value(
                'DISTRACTOR_LOG_DEPTH_THR', 0.06),
            distractor_min_score=value('DISTRACTOR_MIN_SCORE', 0.4),
            distractor_max_score_ratio=value(
                'DISTRACTOR_MAX_SCORE_RATIO', 1.0),
            distractor_recovery_center_jump_thr=value(
               'DISTRACTOR_RECOVERY_CENTER_JUMP_THR', 0.8),
            distractor_recovery_log_depth_thr=value(
                'DISTRACTOR_RECOVERY_LOG_DEPTH_THR', 0.08),
            distractor_max_quarantine_frames=value(
                'DISTRACTOR_MAX_QUARANTINE_FRAMES', 3),
            distractor_cumulative_anchor_use=value(
                'DISTRACTOR_CUMULATIVE_ANCHOR_USE', False),
            distractor_anchor_arm_center_jump_thr=value(
                'DISTRACTOR_ANCHOR_ARM_CENTER_JUMP_THR', 0.8))

    def _clear_distractor_anchor(self):
        self.distractor_anchor_bbox = None
        self.distractor_anchor_depth = None
        self.distractor_anchor_score = None

    def _set_distractor_anchor(self, bbox, depth, score):
        """Retain exactly one pre-jump reference for the following frame."""
        self._clear_distractor_anchor()
        if (_finite_bbox(bbox) is None or depth is None or
                not math.isfinite(float(depth)) or float(depth) <= 0.0 or
                not math.isfinite(float(score))):
            return
        self.distractor_anchor_bbox = list(bbox)
        self.distractor_anchor_depth = float(depth)
        self.distractor_anchor_score = float(score)

    def initialize(self, depth_path, bbox, score=1.0,
                   distractor_relation=None):
        """Initialize trusted state and depth from the raw first frame."""

        bbox_values = _finite_bbox(bbox)
        if bbox_values is None:
            raise ValueError('Initial target bbox must be finite with positive size')
        score = float(score)
        if not math.isfinite(score):
            raise ValueError('Initial target score must be finite')
        self.trusted_bbox = list(bbox)
        self.last_accepted_score = score
        self.consecutive_rejections = 0
        relation = str(distractor_relation or '').strip().casefold().replace(
            '-', '_').replace(' ', '_')
        self.similar_distractors = relation in {
            'multiple_similar', 'nearby_similar'}
        self.distractor_quarantine = False
        self.distractor_quarantine_frames = 0
        self._clear_distractor_anchor()
        observation = read_candidate_depth(
            depth_path, bbox, center_fraction=self.center_fraction)
        self.trusted_depth = (
            observation.median
            if observation.median is not None and
            observation.valid_ratio >= self.min_valid_ratio else None)
        return observation

    def _accept(self, candidate_bbox, raw_score, observation, jump,
                log_depth_change, score_ratio, next_distractor_anchor=None):
        """Atomically refresh trusted state after an accepted candidate."""
        self.consecutive_rejections = 0
        self.distractor_quarantine = False
        self.distractor_quarantine_frames = 0
        self._clear_distractor_anchor()
        self.trusted_bbox = list(candidate_bbox)
        self.last_accepted_score = raw_score
        self.trusted_depth = (
            observation.median
            if observation.median is not None and
            observation.valid_ratio >= self.min_valid_ratio else None)
        if next_distractor_anchor is not None:
            self._set_distractor_anchor(*next_distractor_anchor)
        return TemporalDepthDecision(
            bbox=list(candidate_bbox), score=raw_score, rejected=False,
            depth=observation.median,
            depth_valid_ratio=observation.valid_ratio,
            normalized_center_jump=jump,
            log_depth_change=log_depth_change,
            score_ratio=score_ratio)

    def evaluate(self, depth_path, candidate_bbox, raw_score):
        """Reject an identity switch or atomically accept all new references."""

        if self.trusted_bbox is None or self.last_accepted_score is None:
            observation = self.initialize(
                depth_path, candidate_bbox, score=raw_score)
            return TemporalDepthDecision(
                bbox=list(candidate_bbox), score=float(raw_score), rejected=False,
                depth=self.trusted_depth,
                depth_valid_ratio=observation.valid_ratio,
                normalized_center_jump=0.0, log_depth_change=None,
                score_ratio=1.0)

        candidate_values = _finite_bbox(candidate_bbox)
        if candidate_values is None:
            # Rejection requires all four registered signals.  An invalid box
            # cannot supply a motion signal, so this optional gate must fail
            # open and leave the base tracker's output untouched.
            score = float(raw_score)
            self.consecutive_rejections = 0
            self.distractor_quarantine = False
            self.distractor_quarantine_frames = 0
            self._clear_distractor_anchor()
            return TemporalDepthDecision(
                bbox=list(candidate_bbox), score=score,
                rejected=False, depth=None, depth_valid_ratio=0.0,
                normalized_center_jump=0.0, log_depth_change=None,
                score_ratio=score / max(self.last_accepted_score, self.eps))

        raw_score = float(raw_score)
        prior_bbox = list(self.trusted_bbox)
        prior_depth = self.trusted_depth
        prior_score = self.last_accepted_score
        jump = normalized_center_jump(
            self.trusted_bbox, candidate_bbox, eps=self.eps)
        observation = read_candidate_depth(
            depth_path, candidate_bbox, center_fraction=self.center_fraction)
        score_ratio = raw_score / max(self.last_accepted_score, self.eps)
        log_depth_change = None
        if observation.median is not None and self.trusted_depth is not None:
            log_depth_change = abs(math.log(
                observation.median / self.trusted_depth))

        # Raw depth is a necessary part of both the trigger and the recovery
        # evidence.  A missing or unreliable depth image must therefore fail
        # open, including when a prior frame entered distractor quarantine.
        if (observation.median is None or self.trusted_depth is None or
                observation.valid_ratio < self.min_valid_ratio):
            return self._accept(
                candidate_bbox, raw_score, observation, jump,
                log_depth_change, score_ratio)

        cumulative_anchor_event = False
        if (self.distractor_cumulative_anchor_use and
                self.distractor_aware_use and self.similar_distractors and
                not self.distractor_quarantine and
                self.distractor_anchor_bbox is not None and
                self.distractor_anchor_depth is not None and
                self.distractor_anchor_score is not None):
            anchor_jump = normalized_center_jump(
                self.distractor_anchor_bbox, candidate_bbox, eps=self.eps)
            anchor_log_depth_change = abs(math.log(
                observation.median / self.distractor_anchor_depth))
            anchor_score_ratio = (
                raw_score / max(self.distractor_anchor_score, self.eps))
            cumulative_low_score_event = bool(
                anchor_jump > self.distractor_center_jump_thr and
                anchor_log_depth_change > self.distractor_log_depth_thr and
                raw_score >= self.distractor_min_score and
                anchor_score_ratio <= self.distractor_max_score_ratio)
            cumulative_high_score_event = bool(
                anchor_jump > self.center_jump_thr and
                anchor_log_depth_change > self.log_depth_thr and
                anchor_score_ratio > self.score_ratio_thr)
            cumulative_anchor_event = bool(
                cumulative_low_score_event or cumulative_high_score_event)
            if cumulative_anchor_event:
                # Restore the pre-jump reference before entering the existing
                # bounded quarantine.  The candidate is then judged against
                # exactly the same registered motion/depth/score thresholds;
                # only their temporal baseline spans two adjacent steps.
                self.trusted_bbox = list(self.distractor_anchor_bbox)
                self.trusted_depth = self.distractor_anchor_depth
                self.last_accepted_score = self.distractor_anchor_score
                jump = anchor_jump
                log_depth_change = anchor_log_depth_change
                score_ratio = anchor_score_ratio
        # An anchor is deliberately valid for one following observation only.
        self._clear_distractor_anchor()

        high_score_rejection_event = bool(
            jump > self.center_jump_thr and
            log_depth_change is not None and
            log_depth_change > self.log_depth_thr and
            score_ratio > self.score_ratio_thr and
            observation.valid_ratio >= self.min_valid_ratio)
        low_score_distractor_swap_event = bool(
            self.distractor_aware_use and
            self.similar_distractors and
            not self.distractor_quarantine and
            jump > self.distractor_center_jump_thr and
            log_depth_change is not None and
            log_depth_change > self.distractor_log_depth_thr and
            raw_score >= self.distractor_min_score and
            score_ratio <= self.distractor_max_score_ratio and
            observation.valid_ratio >= self.min_valid_ratio)
        # A visually similar object can also produce a sharp confidence rise.
        # Reuse the stricter three-signal signature as a quarantine trigger, but
        # only for sequences whose reviewed first frame declares similar
        # distractors.  The generic rejection budget remains independent.
        high_score_distractor_swap_event = bool(
            self.distractor_aware_use and
            self.similar_distractors and
            not self.distractor_quarantine and
            high_score_rejection_event)
        distractor_swap_event = bool(
            low_score_distractor_swap_event or
            high_score_distractor_swap_event or
            cumulative_anchor_event)
        if distractor_swap_event:
            self.distractor_quarantine = True
            self.distractor_quarantine_frames = 0

        quarantine_recovered = bool(
            self.distractor_quarantine and
            jump <= self.distractor_recovery_center_jump_thr and
            log_depth_change is not None and
            log_depth_change <= self.distractor_recovery_log_depth_thr and
            observation.valid_ratio >= self.min_valid_ratio)
        if quarantine_recovered:
            self.distractor_quarantine = False
            self.distractor_quarantine_frames = 0

        if self.distractor_quarantine and not quarantine_recovered:
            if (self.distractor_quarantine_frames <
                    self.distractor_max_quarantine_frames):
                self.distractor_quarantine_frames += 1
                # This state is deliberately separate from the generic
                # rejection budget: its purpose is to wait briefly for a
                # same-identity candidate to return after a switch to a
                # visually similar distractor.
                self.consecutive_rejections = 0
                return TemporalDepthDecision(
                    bbox=list(self.trusted_bbox),
                    score=min(raw_score, self.last_accepted_score) *
                    self.reject_score_scale,
                    rejected=True, depth=observation.median,
                    depth_valid_ratio=observation.valid_ratio,
                    normalized_center_jump=jump,
                    log_depth_change=log_depth_change,
                    score_ratio=score_ratio)
            # A quarantine may not stall a tracker indefinitely.  Once its
            # independent bounded budget is exhausted, accept and rebuild all
            # references exactly as the generic gate's fail-open path does.
            return self._accept(
                candidate_bbox, raw_score, observation, jump,
                log_depth_change, score_ratio)

        if (high_score_rejection_event and
                self.consecutive_rejections <
                self.max_consecutive_rejections):
            self.consecutive_rejections += 1
            return TemporalDepthDecision(
                bbox=list(self.trusted_bbox),
                score=min(raw_score, self.last_accepted_score) *
                self.reject_score_scale,
                rejected=True, depth=observation.median,
                depth_valid_ratio=observation.valid_ratio,
                normalized_center_jump=jump,
                log_depth_change=log_depth_change,
                score_ratio=score_ratio)

        # Every accept, including the fail-open event after the generic
        # rejection budget is exhausted, atomically rebuilds every reference.
        next_distractor_anchor = None
        if (self.distractor_cumulative_anchor_use and
                self.distractor_aware_use and self.similar_distractors and
                not self.distractor_quarantine and
                jump >= self.distractor_anchor_arm_center_jump_thr and
                jump <= self.distractor_center_jump_thr and
                raw_score < self.distractor_min_score and
                observation.valid_ratio >= self.min_valid_ratio and
                prior_depth is not None):
            next_distractor_anchor = (
                prior_bbox, prior_depth, prior_score)
        return self._accept(
            candidate_bbox, raw_score, observation, jump,
            log_depth_change, score_ratio,
            next_distractor_anchor=next_distractor_anchor)


__all__ = [
    'DepthObservation', 'TemporalDepthDecision',
    'TemporalDepthIdentityGate', 'normalized_center_jump',
    'observe_candidate_depth', 'read_candidate_depth', 'read_depth_image',
]
