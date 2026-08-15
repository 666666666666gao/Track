"""Runtime-only SUTrack rollback gate with exact training feature parity."""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


ARTIFACT_SCHEMA = 'sutrack-state-gate-artifact/v1'
TRAINING_SCHEMA = 'sutrack-state-gate-training/v1'
TEMPORAL_FEATURE_SCHEMA = 'current-delta1-mean2/v1'
HARD_REASONS = (
    'large_center_jump',
    'low_static_rgb_identity',
    'large_depth_change',
    'temporal_identity_rejected',
)
BASE_FEATURE_NAMES = (
    'confidence',
    'response_margin',
    'identity_similarity',
    'identity_missing',
    'center_jump',
    'log_depth_change',
    'depth_change_missing',
    'log_area_ratio',
    'log_aspect_ratio',
    'dynamic_active',
    'checked',
    'stable_frames_log1p',
    'low_confidence_reason',
    'small_response_margin_reason',
    'large_center_jump_reason',
    'low_identity_reason',
    'large_depth_change_reason',
    'missing_depth_reason',
)
FEATURE_NAMES = tuple(
    ['current__' + name for name in BASE_FEATURE_NAMES] +
    ['delta1__' + name for name in BASE_FEATURE_NAMES] +
    ['mean2__' + name for name in BASE_FEATURE_NAMES])


@dataclass(frozen=True)
class StateGateDecision:
    checked: bool
    hard_conflict: bool
    rollback_state: bool
    probability: Optional[float]
    threshold: float
    reasons: Tuple[str, ...]
    cooldown_remaining: int


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha(value, name):
    value = str(value).strip().lower()
    if len(value) != 64:
        raise ValueError('{} must contain 64 hex characters'.format(name))
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError('{} is not hexadecimal'.format(name)) from error
    return value


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError('{} is not an object'.format(path))
    return value


def _finite_bbox(values):
    try:
        bbox = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if (len(bbox) != 4 or not all(math.isfinite(value) for value in bbox) or
            bbox[2] <= 0.0 or bbox[3] <= 0.0):
        return None
    return bbox


def _optional_scalar(value):
    if value is None:
        return 0.0, 1.0
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(scalar):
        return None
    return scalar, 0.0


def base_features(evidence):
    """Mirror the post-inference analyzer without reading any GT."""
    if not isinstance(evidence, dict):
        return None
    prior = _finite_bbox(evidence.get('prior_bbox'))
    candidate = _finite_bbox(evidence.get('candidate_bbox'))
    reasons_value = evidence.get('reasons')
    if (prior is None or candidate is None or
            not isinstance(reasons_value, (list, tuple)) or
            not all(isinstance(reason, str) for reason in reasons_value)):
        return None
    reasons = set(reasons_value)
    identity = _optional_scalar(evidence.get('identity_similarity'))
    depth_change = _optional_scalar(evidence.get('log_depth_change'))
    if identity is None or depth_change is None:
        return None
    identity_value, identity_missing = identity
    depth_value, depth_missing = depth_change
    try:
        confidence = float(evidence['confidence'])
        response_margin = float(evidence['response_margin'])
        center_jump = float(evidence['normalized_center_jump'])
        stable_frames = int(evidence['stable_frames'])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (not all(math.isfinite(value) for value in (
            confidence, response_margin, center_jump)) or
            stable_frames < 0):
        return None
    area_ratio = ((candidate[2] * candidate[3]) /
                  (prior[2] * prior[3]))
    aspect_ratio = ((candidate[2] / candidate[3]) /
                    (prior[2] / prior[3]))
    values = np.asarray((
        confidence,
        response_margin,
        identity_value,
        identity_missing,
        center_jump,
        depth_value,
        depth_missing,
        math.log(max(area_ratio, 1.0e-12)),
        math.log(max(aspect_ratio, 1.0e-12)),
        float(bool(evidence.get('dynamic_active'))),
        float(bool(evidence.get('checked'))),
        math.log1p(stable_frames),
        float('low_confidence' in reasons),
        float('small_response_margin' in reasons),
        float('large_center_jump' in reasons),
        float('low_static_rgb_identity' in reasons),
        float('large_depth_change' in reasons),
        float('missing_or_unreliable_depth' in reasons),
    ), dtype=np.float64)
    if values.shape != (len(BASE_FEATURE_NAMES),) or not np.isfinite(values).all():
        return None
    return values


class SUTRACKStateGate:
    """Load one audited linear artifact and make bounded online decisions."""

    def __init__(self, artifact_path, artifact_sha256,
                 training_result_path, training_result_sha256):
        self.artifact_path = Path(os.path.abspath(os.fspath(artifact_path)))
        self.training_result_path = Path(os.path.abspath(
            os.fspath(training_result_path)))
        expected_artifact_sha = _validate_sha(
            artifact_sha256, 'artifact SHA256')
        expected_training_sha = _validate_sha(
            training_result_sha256, 'training-result SHA256')
        if sha256_file(self.artifact_path) != expected_artifact_sha:
            raise ValueError('state-gate artifact SHA mismatch')
        if sha256_file(self.training_result_path) != expected_training_sha:
            raise ValueError('state-gate training-result SHA mismatch')
        result = _load_json(self.training_result_path)
        if (result.get('schema') != TRAINING_SCHEMA or
                result.get('complete') is not True or
                result.get('decision') != 'ready_for_recursive_audit' or
                result.get('ready_for_recursive_audit') is not True or
                result.get('all_seeds_oof_passed') is not True or
                result.get('immediate_audit_evaluated') is not True or
                result.get('immediate_audit_policies_evaluated') != 1 or
                result.get('immediate_audit_passed') is not True or
                result.get('seed_selection_used_audit') is not False or
                result.get('backbone_frozen') is not True or
                result.get('future_frame_text_used') is not False or
                result.get('public_evaluation') is not False):
            raise ValueError('state-gate training result is not eligible')
        deployment_seed = int(result['deployment_seed'])
        artifact_record = next((record for record in result['artifacts']
                                if int(record['seed']) == deployment_seed),
                               None)
        if (not isinstance(artifact_record, dict) or
                Path(artifact_record['path']).resolve() !=
                self.artifact_path.resolve() or
                artifact_record.get('sha256') != expected_artifact_sha or
                int(artifact_record.get('bytes', -1)) !=
                self.artifact_path.stat().st_size):
            raise ValueError('deployment artifact is not bound by training result')
        artifact = torch.load(str(self.artifact_path), map_location='cpu')
        if (not isinstance(artifact, dict) or
                artifact.get('schema') != ARTIFACT_SCHEMA or
                int(artifact.get('seed', -1)) != deployment_seed or
                artifact.get('eligible_from_oof') is not True or
                artifact.get('base_feature_names') != list(BASE_FEATURE_NAMES) or
                artifact.get('feature_names') != list(FEATURE_NAMES) or
                artifact.get('temporal_feature_schema') !=
                TEMPORAL_FEATURE_SCHEMA or
                artifact.get('backbone_frozen') is not True or
                artifact.get('future_frame_text_used') is not False or
                artifact.get('public_evaluation') is not False or
                artifact.get('maximum_consecutive_gate_rollbacks') != 1):
            raise ValueError('state-gate artifact contract failed')
        weight = torch.as_tensor(artifact.get('weight')).detach().cpu().numpy()
        mean = torch.as_tensor(artifact.get('mean')).detach().cpu().numpy()
        std = torch.as_tensor(artifact.get('std')).detach().cpu().numpy()
        try:
            bias = float(artifact['bias'])
            threshold = float(artifact['threshold'])
            cooldown = int(artifact['cooldown_frames_after_rollback'])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError('malformed state-gate scalars') from error
        expected_shape = (len(FEATURE_NAMES),)
        if (weight.shape != expected_shape or mean.shape != expected_shape or
                std.shape != expected_shape or
                not np.isfinite(weight).all() or
                not np.isfinite(mean).all() or
                not np.isfinite(std).all() or np.any(std <= 0.0) or
                not math.isfinite(bias) or not math.isfinite(threshold) or
                not 0.0 <= threshold <= 1.0 or cooldown < 0):
            raise ValueError('invalid state-gate numeric payload')
        self.weight = weight.astype(np.float64, copy=True)
        self.mean = mean.astype(np.float64, copy=True)
        self.std = std.astype(np.float64, copy=True)
        self.bias = bias
        self.threshold = threshold
        self.cooldown_frames_after_rollback = cooldown
        self.artifact_sha256 = expected_artifact_sha
        self.training_result_sha256 = expected_training_sha
        self.reset()

    def reset(self):
        self.history = []
        self.cooldown_remaining = 0

    def _decision(self, checked, hard_conflict, rollback, probability,
                  reasons):
        return StateGateDecision(
            checked=bool(checked), hard_conflict=bool(hard_conflict),
            rollback_state=bool(rollback), probability=probability,
            threshold=float(self.threshold), reasons=tuple(reasons),
            cooldown_remaining=int(self.cooldown_remaining))

    def observe(self, evidence):
        current = base_features(evidence)
        if current is None:
            self.history = []
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
            return self._decision(
                False, False, False, None, ('malformed_online_evidence',))
        previous = self.history[-1] if self.history else current
        mean_two = (np.mean(self.history[-2:], axis=0)
                    if self.history else current)
        vector = np.concatenate((current, current - previous, mean_two))
        self.history.append(current)
        if len(self.history) > 2:
            self.history.pop(0)
        normalized = (vector - self.mean) / self.std
        logit = float(normalized.dot(self.weight) + self.bias)
        if not math.isfinite(logit):
            return self._decision(
                False, False, False, None, ('non_finite_gate_logit',))
        if logit >= 0.0:
            probability = 1.0 / (1.0 + math.exp(-logit))
        else:
            exp_logit = math.exp(logit)
            probability = exp_logit / (1.0 + exp_logit)
        reasons = tuple(evidence.get('reasons', ()))
        hard_conflict = any(reason in reasons for reason in HARD_REASONS)
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return self._decision(
                True, hard_conflict, False, probability,
                ('rollback_cooldown',))
        rollback = bool(hard_conflict and probability >= self.threshold)
        if rollback:
            self.cooldown_remaining = self.cooldown_frames_after_rollback
            decision_reasons = ('learned_high_precision_rollback',)
        elif not hard_conflict:
            decision_reasons = ('no_hard_conflict',)
        else:
            decision_reasons = ('below_learned_threshold',)
        return self._decision(
            True, hard_conflict, rollback, probability, decision_reasons)


__all__ = [
    'ARTIFACT_SCHEMA',
    'BASE_FEATURE_NAMES',
    'FEATURE_NAMES',
    'HARD_REASONS',
    'SUTRACKStateGate',
    'StateGateDecision',
    'TEMPORAL_FEATURE_SCHEMA',
    'base_features',
    'sha256_file',
]
