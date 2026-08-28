"""Fail-closed protected/tentative transaction for recursive tracker state.

This module is deliberately independent from ``SUTRACK.track``. It owns
deep-cloned protected and tentative snapshots, evaluates two consecutive
future-frame observations, and returns exactly one complete snapshot on
promote or rollback. Nothing here mutates a live tracker until the caller
explicitly installs the resolved snapshot.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import torch


_SCALARS = (str, bytes, bool, int, float, type(None))
_REQUIRED_CONFIRM_FRAMES = 2


def _clone_tree(value):
    """Deep-clone the bounded tensor/scalar tree accepted by a snapshot."""
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError('object-dtype arrays are not supported')
        return value.copy()
    if isinstance(value, _SCALARS):
        return value
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if isinstance(value, Mapping):
        output = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError('snapshot mapping keys must be strings')
            output[key] = _clone_tree(item)
        return output
    raise TypeError(
        'unsupported recursive snapshot value: {}'.format(type(value)))


def _finite_bbox(values):
    if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != 4:
        return None
    if any(isinstance(value, (bool, np.bool_)) or
           not isinstance(value, Real) for value in values):
        return None
    bbox = tuple(float(value) for value in values)
    if (not all(math.isfinite(value) for value in bbox) or
            bbox[2] <= 0.0 or bbox[3] <= 0.0):
        return None
    return bbox


def _positive_integer(name, value):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, Integral) or int(value) <= 0):
        raise ValueError('{} must be a positive integer'.format(name))
    return int(value)


def _bounded(name, value, low=0.0, high=1.0):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, Real)):
        raise ValueError('{} must be a real number'.format(name))
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError('{} must lie in [{}, {}]'.format(name, low, high))
    return value


def _identity_anchor(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('identity_anchor must be a non-empty string')
    return value.strip()


@dataclass(frozen=True)
class TrackerRecursiveSnapshot:
    """One atomic unit of bbox, templates, annotations, and auxiliary state.

    ``template_annotations`` intentionally need not have the same length as
    ``templates``. SUTrack initializes multiple template slots with one
    broadcast annotation; a later dynamic-template state can use one
    annotation per slot. The transaction validates each annotation layout
    independently while keeping the number of template slots stable.
    """

    state: Tuple[float, float, float, float]
    templates: Tuple[Any, ...]
    template_annotations: Tuple[Any, ...]
    auxiliary: Mapping[str, Any]

    @classmethod
    def capture(cls, state, templates, template_annotations, auxiliary=None):
        bbox = _finite_bbox(state)
        if bbox is None:
            raise ValueError('recursive snapshot bbox is malformed')
        if not isinstance(templates, (list, tuple)) or not templates:
            raise ValueError('recursive snapshot templates are empty')
        if (not isinstance(template_annotations, (list, tuple)) or
                not template_annotations):
            raise ValueError('recursive snapshot annotations are empty')
        auxiliary = {} if auxiliary is None else auxiliary
        if not isinstance(auxiliary, Mapping):
            raise ValueError('recursive snapshot auxiliary state is malformed')
        return cls(
            state=bbox,
            templates=tuple(_clone_tree(list(templates))),
            template_annotations=tuple(
                _clone_tree(list(template_annotations))),
            auxiliary=_clone_tree(dict(auxiliary)),
        )

    def clone(self):
        return TrackerRecursiveSnapshot.capture(
            self.state, self.templates, self.template_annotations,
            self.auxiliary)

    def materialize(self):
        """Return a fresh mutable tree suitable for one atomic installation."""
        return {
            'state': list(self.state),
            'template_list': list(_clone_tree(self.templates)),
            'template_anno_list': list(
                _clone_tree(self.template_annotations)),
            'auxiliary': _clone_tree(self.auxiliary),
        }


@dataclass(frozen=True)
class BranchEvidence:
    """Online-only branch evidence; every score uses higher-is-better units."""

    confidence: float
    response_margin: float
    identity_similarity: float
    depth_consistency: float
    temporal_continuity: float
    identity_anchor: str
    hard_conflict: bool = False

    def __post_init__(self):
        object.__setattr__(
            self, 'confidence', _bounded('confidence', self.confidence))
        if (isinstance(self.response_margin, (bool, np.bool_)) or
                not isinstance(self.response_margin, Real)):
            raise ValueError('response_margin must be a real number')
        margin = float(self.response_margin)
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError('response_margin must be finite and non-negative')
        object.__setattr__(self, 'response_margin', margin)
        object.__setattr__(
            self, 'identity_similarity', _bounded(
                'identity_similarity', self.identity_similarity))
        object.__setattr__(
            self, 'depth_consistency', _bounded(
                'depth_consistency', self.depth_consistency))
        object.__setattr__(
            self, 'temporal_continuity', _bounded(
                'temporal_continuity', self.temporal_continuity))
        object.__setattr__(
            self, 'identity_anchor', _identity_anchor(self.identity_anchor))
        if not isinstance(self.hard_conflict, bool):
            raise ValueError('hard_conflict must be a bool')


@dataclass(frozen=True)
class TransactionDecision:
    event_id: int
    frame_id: int
    action: str
    reasons: Tuple[str, ...]
    age: int
    consecutive_confirmations: int
    protected_utility: Optional[float]
    tentative_utility: Optional[float]
    resolved_snapshot: Optional[TrackerRecursiveSnapshot]


class ProtectedTentativeTemplateTransaction:
    """Validate a tentative recursive branch before one atomic state change."""

    def __init__(
            self, confirm_frames=2, max_shadow_frames=2,
            min_confidence=0.65, min_response_margin=0.10,
            min_identity=0.75, min_depth_consistency=0.70,
            min_temporal_continuity=0.70,
            max_confidence_deficit=0.05, max_margin_deficit=0.05,
            min_utility_advantage=0.02,
            confidence_weight=0.30, margin_weight=0.20,
            identity_weight=0.25, depth_weight=0.10,
            temporal_weight=0.15):
        confirm_frames = _positive_integer('confirm_frames', confirm_frames)
        if confirm_frames != _REQUIRED_CONFIRM_FRAMES:
            raise ValueError('confirm_frames must be exactly two')
        self.confirm_frames = _REQUIRED_CONFIRM_FRAMES
        self.max_shadow_frames = _positive_integer(
            'max_shadow_frames', max_shadow_frames)
        if self.max_shadow_frames != _REQUIRED_CONFIRM_FRAMES:
            raise ValueError('max_shadow_frames must be exactly two')
        self.min_confidence = _bounded(
            'min_confidence', min_confidence)
        self.min_response_margin = _bounded(
            'min_response_margin', min_response_margin)
        self.min_identity = _bounded('min_identity', min_identity)
        self.min_depth_consistency = _bounded(
            'min_depth_consistency', min_depth_consistency)
        self.min_temporal_continuity = _bounded(
            'min_temporal_continuity', min_temporal_continuity)
        self.max_confidence_deficit = _bounded(
            'max_confidence_deficit', max_confidence_deficit)
        self.max_margin_deficit = _bounded(
            'max_margin_deficit', max_margin_deficit)
        self.min_utility_advantage = _bounded(
            'min_utility_advantage', min_utility_advantage)
        weights = (
            confidence_weight, margin_weight, identity_weight,
            depth_weight, temporal_weight)
        weights = tuple(
            _bounded('evidence_weight', value) for value in weights)
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError('evidence weights must sum to one')
        self.weights = weights
        self._event_counter = 0
        self._last_frame_id = 0
        self._clear_active()

    @property
    def active(self):
        return self._protected is not None

    @property
    def event_count(self):
        return self._event_counter

    @property
    def last_frame_id(self):
        return self._last_frame_id

    def active_snapshots(self):
        """Return isolated branch snapshots for the next shadow inference."""
        if not self.active:
            raise RuntimeError('no active template transaction')
        protected = self._protected.clone()
        tentative = self._tentative.clone()
        return protected, tentative

    def _clear_active(self):
        self._protected = None
        self._tentative = None
        self._event_id = None
        self._event_frame = None
        self._identity_anchor = None
        self._age = 0
        self._confirmations = 0

    @staticmethod
    def _validate_snapshot_layout(name, snapshot):
        annotation_count = len(snapshot.template_annotations)
        template_count = len(snapshot.templates)
        if annotation_count not in (1, template_count):
            raise ValueError(
                '{} annotation slots must be one or match templates'.format(
                    name))

    @classmethod
    def _validate_pair(cls, protected_snapshot, tentative_snapshot):
        if not isinstance(protected_snapshot, TrackerRecursiveSnapshot):
            raise TypeError('protected snapshot has the wrong type')
        if not isinstance(tentative_snapshot, TrackerRecursiveSnapshot):
            raise TypeError('tentative snapshot has the wrong type')
        cls._validate_snapshot_layout('protected', protected_snapshot)
        cls._validate_snapshot_layout('tentative', tentative_snapshot)
        if (len(protected_snapshot.templates) !=
                len(tentative_snapshot.templates)):
            raise ValueError('protected/tentative template slots differ')

    def _validate_active_pair(self, protected_snapshot, tentative_snapshot):
        self._validate_pair(protected_snapshot, tentative_snapshot)
        if len(protected_snapshot.templates) != len(self._protected.templates):
            raise ValueError('protected template slot count changed')
        if len(tentative_snapshot.templates) != len(self._tentative.templates):
            raise ValueError('tentative template slot count changed')

    def _utility(self, evidence):
        values = (
            evidence.confidence,
            min(evidence.response_margin, 1.0),
            evidence.identity_similarity,
            evidence.depth_consistency,
            evidence.temporal_continuity,
        )
        return sum(weight * value for weight, value in zip(
            self.weights, values))

    def begin(
            self, frame_id, protected_snapshot, tentative_snapshot,
            identity_anchor):
        frame_id = _positive_integer('frame_id', frame_id)
        if self.active:
            raise RuntimeError('a template transaction is already active')
        if frame_id <= self._last_frame_id:
            raise ValueError('transaction frame_id must be strictly increasing')
        self._validate_pair(protected_snapshot, tentative_snapshot)
        identity_anchor = _identity_anchor(identity_anchor)

        protected_clone = protected_snapshot.clone()
        tentative_clone = tentative_snapshot.clone()
        next_event_id = self._event_counter + 1

        self._event_counter = next_event_id
        self._event_id = next_event_id
        self._event_frame = frame_id
        self._last_frame_id = frame_id
        self._identity_anchor = identity_anchor
        self._protected = protected_clone
        self._tentative = tentative_clone
        self._age = 0
        self._confirmations = 0
        return TransactionDecision(
            event_id=self._event_id, frame_id=frame_id, action='hold',
            reasons=('transaction_started',), age=0,
            consecutive_confirmations=0, protected_utility=None,
            tentative_utility=None, resolved_snapshot=None)

    def _finish(
            self, frame_id, action, reasons, age, confirmations,
            resolved_snapshot, protected_utility=None,
            tentative_utility=None):
        if action not in ('promote', 'rollback'):
            raise ValueError('transaction resolution action is invalid')
        decision = TransactionDecision(
            event_id=int(self._event_id), frame_id=int(frame_id),
            action=action, reasons=tuple(reasons), age=int(age),
            consecutive_confirmations=int(confirmations),
            protected_utility=protected_utility,
            tentative_utility=tentative_utility,
            resolved_snapshot=resolved_snapshot)
        self._last_frame_id = int(frame_id)
        self._clear_active()
        return decision

    def observe(
            self, frame_id, protected_snapshot, tentative_snapshot,
            protected_evidence, tentative_evidence):
        if not self.active:
            raise RuntimeError('no active template transaction')
        frame_id = _positive_integer('frame_id', frame_id)
        if frame_id <= self._last_frame_id:
            raise ValueError('transaction frame_id must be strictly increasing')
        self._validate_active_pair(protected_snapshot, tentative_snapshot)
        if not isinstance(protected_evidence, BranchEvidence):
            raise TypeError('protected evidence has the wrong type')
        if not isinstance(tentative_evidence, BranchEvidence):
            raise TypeError('tentative evidence has the wrong type')
        if (protected_evidence.identity_anchor != self._identity_anchor or
                tentative_evidence.identity_anchor != self._identity_anchor):
            raise ValueError('branch evidence identity anchor changed')

        protected_clone = protected_snapshot.clone()
        tentative_clone = tentative_snapshot.clone()
        age = frame_id - int(self._event_frame)
        protected_utility = self._utility(protected_evidence)
        tentative_utility = self._utility(tentative_evidence)

        if frame_id != self._last_frame_id + 1:
            return self._finish(
                frame_id, 'rollback', ('nonconsecutive_shadow_frame',), age, 0,
                protected_clone, protected_utility, tentative_utility)

        if protected_evidence.hard_conflict:
            return self._finish(
                frame_id, 'rollback', ('protected_hard_conflict',), age, 0,
                protected_clone, protected_utility, tentative_utility)
        if tentative_evidence.hard_conflict:
            return self._finish(
                frame_id, 'rollback', ('tentative_hard_conflict',), age, 0,
                protected_clone, protected_utility, tentative_utility)

        reasons = []
        if tentative_evidence.confidence < self.min_confidence:
            reasons.append('tentative_confidence_below_minimum')
        if tentative_evidence.response_margin < self.min_response_margin:
            reasons.append('tentative_margin_below_minimum')
        if tentative_evidence.identity_similarity < self.min_identity:
            reasons.append('tentative_identity_below_minimum')
        if tentative_evidence.depth_consistency < self.min_depth_consistency:
            reasons.append('tentative_depth_below_minimum')
        if (tentative_evidence.temporal_continuity <
                self.min_temporal_continuity):
            reasons.append('tentative_temporal_below_minimum')
        if (tentative_evidence.confidence + self.max_confidence_deficit <
                protected_evidence.confidence):
            reasons.append('tentative_confidence_deficit')
        if (tentative_evidence.response_margin + self.max_margin_deficit <
                protected_evidence.response_margin):
            reasons.append('tentative_margin_deficit')
        if (tentative_utility <
                protected_utility + self.min_utility_advantage):
            reasons.append('tentative_utility_not_better')

        confirmations = 0 if reasons else self._confirmations + 1
        if not reasons:
            reasons.append('tentative_confirmation')
        if confirmations >= self.confirm_frames:
            return self._finish(
                frame_id, 'promote', ('confirmed_future_advantage',),
                age, confirmations, tentative_clone,
                protected_utility, tentative_utility)
        if age >= self.max_shadow_frames:
            return self._finish(
                frame_id, 'rollback',
                tuple(reasons) + ('shadow_horizon_expired',),
                age, confirmations, protected_clone,
                protected_utility, tentative_utility)

        self._last_frame_id = frame_id
        self._age = age
        self._confirmations = confirmations
        self._protected = protected_clone
        self._tentative = tentative_clone
        return TransactionDecision(
            event_id=int(self._event_id), frame_id=frame_id, action='hold',
            reasons=tuple(reasons), age=age,
            consecutive_confirmations=confirmations,
            protected_utility=protected_utility,
            tentative_utility=tentative_utility,
            resolved_snapshot=None)

    def cancel(self, frame_id, reason='external_cancel'):
        if not self.active:
            raise RuntimeError('no active template transaction')
        frame_id = _positive_integer('frame_id', frame_id)
        if frame_id <= self._last_frame_id:
            raise ValueError('cancel frame_id must be strictly increasing')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('cancel reason must be a non-empty string')
        protected_clone = self._protected.clone()
        age = frame_id - int(self._event_frame)
        return self._finish(
            frame_id, 'rollback', (reason.strip(),), age,
            self._confirmations, protected_clone)

    def rollback_current(self, frame_id, protected_snapshot, reason):
        """Fail closed to a validated current-frame protected snapshot."""
        if not self.active:
            raise RuntimeError('no active template transaction')
        frame_id = _positive_integer('frame_id', frame_id)
        if frame_id <= self._last_frame_id:
            raise ValueError('rollback frame_id must be strictly increasing')
        if not isinstance(protected_snapshot, TrackerRecursiveSnapshot):
            raise TypeError('protected snapshot has the wrong type')
        self._validate_snapshot_layout('protected', protected_snapshot)
        if len(protected_snapshot.templates) != len(self._protected.templates):
            raise ValueError('protected template slot count changed')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('rollback reason must be a non-empty string')
        protected_clone = protected_snapshot.clone()
        age = frame_id - int(self._event_frame)
        return self._finish(
            frame_id, 'rollback', (reason.strip(),), age, 0,
            protected_clone)


__all__ = [
    'BranchEvidence',
    'ProtectedTentativeTemplateTransaction',
    'TrackerRecursiveSnapshot',
    'TransactionDecision',
]
