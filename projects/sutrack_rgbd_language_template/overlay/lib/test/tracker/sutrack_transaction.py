"""SUTrack with protected/tentative dynamic-template transactions.

For a template-only candidate, the public branch accepts the already-gated
current bbox but keeps the old template; the tentative branch differs only by
the new template.  For a state-conflict candidate, the public branch keeps the
last trusted bbox and the tentative branch holds the disputed bbox.  Both
branches are rolled forward on the next two frames. Only two consecutive,
identity-anchored advantages promote the complete tentative state; every
conflict or timeout installs the complete protected state.
"""

import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re

import torch

from lib.test.tracker.protected_tentative_transaction import (
    BranchEvidence,
    ProtectedTentativeTemplateTransaction,
    TrackerRecursiveSnapshot,
)
from lib.test.tracker.safe_template_update import (
    SafeTemplateUpdatePolicy,
    nms_response_margin,
)
from lib.test.tracker.sutrack import SUTRACK
from lib.test.tracker.temporal_depth_identity import read_candidate_depth
from lib.test.tracker.utils import sample_target, transform_image_to_crop
from lib.utils.box_ops import clip_box


class SUTRACKProtectedTransaction(SUTRACK):
    """Delay every safe-v1 template write until a two-frame shadow verdict."""

    def __init__(self, params, dataset_name):
        super().__init__(params, dataset_name)
        if not self.safe_template_update:
            raise ValueError(
                'Protected transaction requires SAFE_TEMPLATE_UPDATE.USE=True')
        if self.num_template != 2:
            raise ValueError(
                'Protected transaction currently requires exactly two slots')
        if self.safe_template_policy.hard_conflict_state_rollback:
            raise ValueError(
                'Legacy state rollback must be off for atomic transactions')
        settings = getattr(
            params, 'protected_tentative_transaction', None)
        if not isinstance(settings, dict):
            raise ValueError('Missing protected transaction parameter mapping')
        allowed = {
            'confirm_frames', 'max_shadow_frames', 'min_identity',
            'min_confidence', 'min_response_margin',
            'min_depth_consistency', 'min_temporal_continuity',
            'max_confidence_deficit', 'max_margin_deficit',
            'min_utility_advantage', 'confidence_weight', 'margin_weight',
            'identity_weight', 'depth_weight', 'temporal_weight',
        }
        if set(settings) != allowed:
            raise ValueError('Protected transaction parameter keys differ')
        self.transaction_settings = copy.deepcopy(settings)
        self.template_transaction = self._new_transaction()
        self.identity_anchor = None
        self.transaction_events_started = 0
        self.transaction_trace_root = str(getattr(
            params, 'transaction_trace_root', '') or '')
        self.transaction_trace_path = None

    def _new_transaction(self):
        return ProtectedTentativeTemplateTransaction(
            **copy.deepcopy(self.transaction_settings))

    def initialize(self, image, info):
        output = super().initialize(image, info)
        self.template_transaction = self._new_transaction()
        self.transaction_events_started = 0
        anchor_index = info.get('anchor_index')
        identity_payload = '|'.join((
            str(self.sequence_name),
            str(anchor_index),
            ','.join('{:.8f}'.format(float(value))
                     for value in info['init_bbox']),
            str(info.get('init_nlp', '')),
        ))
        self.identity_anchor = hashlib.sha256(
            identity_payload.encode('utf-8')).hexdigest()
        self.transaction_trace_path = None
        if self.transaction_trace_root:
            if (isinstance(anchor_index, bool) or
                    not isinstance(anchor_index, int) or anchor_index < 0):
                raise ValueError(
                    'Transaction trace requires a non-negative anchor index')
            safe_sequence = re.sub(
                r'[^A-Za-z0-9_.-]+', '_', str(self.sequence_name))
            trace_root = Path(self.transaction_trace_root).resolve()
            trace_root.mkdir(parents=True, exist_ok=True)
            self.transaction_trace_path = (
                trace_root /
                '{}__anchor-{:06d}.jsonl'.format(
                    safe_sequence, anchor_index))
            with open(
                    self.transaction_trace_path, 'w',
                    encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps({
                    'type': 'initialize',
                    'sequence_name': str(self.sequence_name),
                    'anchor_index': int(anchor_index),
                    'identity_anchor': self.identity_anchor,
                    'init_bbox': [float(value)
                                  for value in info['init_bbox']],
                    'checkpoint': str(self.params.checkpoint),
                }, sort_keys=True) + '\n')
        return output

    def _write_transaction_trace(self, payload):
        if self.transaction_trace_path is None:
            return
        record = dict(payload)
        record['type'] = 'transaction_frame'
        record['frame_id'] = int(self.frame_id)
        with open(
                self.transaction_trace_path, 'a',
                encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(record, sort_keys=True) + '\n')

    def _infer(
            self, image, anchor_bbox, templates, template_annos,
            text_src=None, task_index_batch=None):
        height, width = image.shape[:2]
        patch, resize_factor = sample_target(
            image, anchor_bbox, self.params.search_factor,
            output_sz=self.params.search_size)
        search = self.preprocessor.process(patch)
        if self.multi_modal_vision and search.size(1) == 3:
            search = torch.cat((search, search), axis=1)
        if text_src is None:
            text_src = self.text_src
        if task_index_batch is None:
            task_index_batch = self.task_index_batch
        with torch.no_grad():
            encoded = self.network.forward_encoder(
                templates, [search], template_annos, text_src,
                task_index_batch)
            output = self.network.forward_decoder(feature=encoded)
        score_map = output['score_map']
        response = (self.output_window * score_map
                    if self.cfg.TEST.WINDOW else score_map)
        if 'size_map' in output:
            boxes, confidence = self.network.decoder.cal_bbox(
                response, output['size_map'], output['offset_map'],
                return_score=True)
        else:
            boxes, confidence = self.network.decoder.cal_bbox(
                response, output['offset_map'], return_score=True)
        prediction = (
            boxes.view(-1, 4).mean(dim=0) *
            self.params.search_size / resize_factor).tolist()
        cx_prev = anchor_bbox[0] + 0.5 * anchor_bbox[2]
        cy_prev = anchor_bbox[1] + 0.5 * anchor_bbox[3]
        cx, cy, box_width, box_height = prediction
        half_side = 0.5 * self.params.search_size / resize_factor
        bbox = clip_box([
            cx + cx_prev - half_side - 0.5 * box_width,
            cy + cy_prev - half_side - 0.5 * box_height,
            box_width,
            box_height,
        ], height, width, margin=10)
        confidence_value = float(
            confidence.detach().reshape(-1).max().item())
        selected_index = int(response.detach().reshape(-1).argmax().item())
        margin = nms_response_margin(
            response.detach(), selected_index,
            kernel=self.safe_template_nms_kernel)
        if (margin is None or not all(math.isfinite(value) for value in
                                      (confidence_value, margin, *bbox))):
            raise ValueError('Non-finite protected transaction inference')
        return list(bbox), confidence, confidence_value, float(margin)

    def _make_template(self, image, bbox):
        patch, resize_factor = sample_target(
            image, bbox, self.params.template_factor,
            output_sz=self.params.template_size)
        template = self.preprocessor.process(patch)
        if self.multi_modal_vision and template.size(1) == 3:
            template = torch.cat((template, template), axis=1)
        annotation = transform_image_to_crop(
            torch.tensor(bbox), torch.tensor(bbox), resize_factor,
            torch.Tensor([self.params.template_size,
                          self.params.template_size]),
            normalize=True)
        annotation = annotation.to(template.device).unsqueeze(0)
        if self.safe_template_apply_tensor_blend:
            weight = float(self.safe_template_policy.blend_weight)
            template = torch.lerp(self.static_template, template, weight)
            annotation = torch.lerp(
                self.static_template_anno, annotation, weight)
        return template, annotation

    @staticmethod
    def _append_dynamic(templates, annos, template, annotation, limit):
        destination_templates = list(templates)
        destination_templates.append(template)
        if len(destination_templates) > limit:
            destination_templates.pop(1)
        destination_annos = list(annos)
        destination_annos.append(annotation)
        if len(destination_annos) > limit:
            destination_annos.pop(1)
        return destination_templates, destination_annos

    @staticmethod
    def _policy_state(policy):
        return copy.deepcopy(vars(policy))

    def _policy_from_state(self, state):
        if not isinstance(state, dict):
            raise ValueError('Safe policy snapshot is malformed')
        policy = SafeTemplateUpdatePolicy.from_config(
            self.cfg.TEST.SAFE_TEMPLATE_UPDATE)
        if set(state) != set(vars(policy)):
            raise ValueError('Safe policy snapshot keys differ')
        for key, value in state.items():
            setattr(policy, key, copy.deepcopy(value))
        return policy

    def _capture_snapshot(
            self, state, templates, annos, policy,
            text_src=None, task_index_batch=None):
        if text_src is None:
            text_src = self.text_src
        if task_index_batch is None:
            task_index_batch = self.task_index_batch
        return TrackerRecursiveSnapshot.capture(
            state, templates, annos, {
                'safe_policy_state': self._policy_state(policy),
                'text_src': text_src,
                'task_index_batch': task_index_batch,
            })

    def _install_snapshot(self, snapshot):
        materialized = snapshot.materialize()
        auxiliary = materialized['auxiliary']
        if set(auxiliary) != {
                'safe_policy_state', 'text_src', 'task_index_batch'}:
            raise ValueError('Recursive snapshot auxiliary keys differ')
        policy = self._policy_from_state(auxiliary['safe_policy_state'])
        self.state = materialized['state']
        self.template_list = materialized['template_list']
        self.template_anno_list = materialized['template_anno_list']
        self.text_src = auxiliary['text_src']
        self.task_index_batch = auxiliary['task_index_batch']
        self.safe_template_policy = policy

    @staticmethod
    def _bounded_continuity(change, accepted_change):
        if change is None or not math.isfinite(float(change)):
            return 0.0
        accepted_change = float(accepted_change)
        if accepted_change <= 0.0:
            return 1.0 if float(change) == 0.0 else 0.0
        return max(0.0, 1.0 - min(
            float(change) / (2.0 * accepted_change), 1.0))

    def _branch_evidence(self, policy, decision, confidence, margin):
        identity = decision.identity_similarity
        identity = 0.0 if identity is None else float(identity)
        depth_consistency = self._bounded_continuity(
            decision.log_depth_change, policy.max_log_depth_change)
        temporal_continuity = self._bounded_continuity(
            decision.normalized_center_jump, policy.max_center_jump)
        hard_conflict = self._is_hard_conflict(decision)
        return BranchEvidence(
            confidence=float(confidence),
            response_margin=float(margin),
            identity_similarity=identity,
            depth_consistency=depth_consistency,
            temporal_continuity=temporal_continuity,
            identity_anchor=self.identity_anchor,
            hard_conflict=hard_conflict)

    @staticmethod
    def _is_hard_conflict(decision):
        return any(reason in decision.reasons for reason in (
            'large_center_jump', 'low_static_rgb_identity',
            'large_depth_change', 'temporal_identity_rejected',
            'missing_or_unreliable_depth', 'malformed_rgb_identity',
            'malformed_evidence', 'malformed_frame_id',
        ))

    def _advance_branch(
            self, snapshot, image, depth_path, prediction):
        materialized = snapshot.materialize()
        auxiliary = materialized['auxiliary']
        prior_policy = self._policy_from_state(
            auxiliary['safe_policy_state'])
        candidate_policy = copy.deepcopy(prior_policy)
        bbox, _, confidence, margin = prediction
        decision = candidate_policy.observe(
            self.frame_id, image, bbox, confidence, margin, depth_path)
        templates = materialized['template_list']
        annos = materialized['template_anno_list']
        if decision.drop_dynamic:
            templates = [self.static_template] * self.num_template
            annos = [self.static_template_anno]
        if decision.replace_dynamic:
            candidate_policy.cancel(self.frame_id)
        hard_conflict = self._is_hard_conflict(decision)
        if hard_conflict:
            selected_state = materialized['state']
            selected_policy = prior_policy
            selected_policy.pending_update_frame = None
            selected_policy.stable_frames = 0
            if decision.drop_dynamic:
                selected_policy.dynamic_active = False
        else:
            selected_state = bbox
            selected_policy = candidate_policy
        evidence = self._branch_evidence(
            candidate_policy, decision, confidence, margin)
        updated = self._capture_snapshot(
            selected_state, templates, annos, selected_policy,
            text_src=auxiliary['text_src'],
            task_index_batch=auxiliary['task_index_batch'])
        return updated, evidence, decision

    @staticmethod
    def _decision_trace(decision):
        return {
            'event_id': int(decision.event_id),
            'frame_id': int(decision.frame_id),
            'action': str(decision.action),
            'reasons': list(decision.reasons),
            'age': int(decision.age),
            'consecutive_confirmations': int(
                decision.consecutive_confirmations),
            'protected_utility': decision.protected_utility,
            'tentative_utility': decision.tentative_utility,
        }

    def _track_active_transaction(self, image, depth_path):
        protected_previous, tentative_previous = (
            self.template_transaction.active_snapshots())
        try:
            protected_prediction = self._infer(
                image, protected_previous.state,
                protected_previous.templates,
                protected_previous.template_annotations,
                text_src=protected_previous.auxiliary['text_src'],
                task_index_batch=protected_previous.auxiliary[
                    'task_index_batch'])
            protected_current, protected_evidence, protected_policy_decision = (
                self._advance_branch(
                    protected_previous, image, depth_path,
                    protected_prediction))
        except Exception as error:
            if (isinstance(error, RuntimeError) and
                    'out of memory' in str(error).lower()):
                raise
            transaction_decision = self.template_transaction.cancel(
                self.frame_id, 'protected_branch_error')
            self._install_snapshot(transaction_decision.resolved_snapshot)
            output = {
                'target_bbox': self.state,
                'best_score': 0.0,
                'protected_transaction': {
                    'active_before': True,
                    'selected_branch': 'stored_protected',
                    'decision': self._decision_trace(transaction_decision),
                    'recoverable_error': '{}: {}'.format(
                        type(error).__name__, str(error)[:300]),
                },
            }
            self._write_transaction_trace(output['protected_transaction'])
            return output

        try:
            tentative_prediction = self._infer(
                image, tentative_previous.state,
                tentative_previous.templates,
                tentative_previous.template_annotations,
                text_src=tentative_previous.auxiliary['text_src'],
                task_index_batch=tentative_previous.auxiliary[
                    'task_index_batch'])
            tentative_current, tentative_evidence, tentative_policy_decision = (
                self._advance_branch(
                    tentative_previous, image, depth_path,
                    tentative_prediction))
            transaction_decision = self.template_transaction.observe(
                self.frame_id, protected_current, tentative_current,
                protected_evidence, tentative_evidence)
        except Exception as error:
            if (isinstance(error, RuntimeError) and
                    'out of memory' in str(error).lower()):
                raise
            transaction_decision = self.template_transaction.rollback_current(
                self.frame_id, protected_current,
                'tentative_branch_error')
            self._install_snapshot(transaction_decision.resolved_snapshot)
            output = {
                'target_bbox': self.state,
                'best_score': protected_prediction[1],
                'protected_transaction': {
                    'active_before': True,
                    'selected_branch': 'protected',
                    'decision': self._decision_trace(transaction_decision),
                    'protected_bbox': list(protected_current.state),
                    'protected_evidence': asdict(protected_evidence),
                    'protected_policy_decision': asdict(
                        protected_policy_decision),
                    'recoverable_error': '{}: {}'.format(
                        type(error).__name__, str(error)[:300]),
                },
            }
            self._write_transaction_trace(output['protected_transaction'])
            return output
        if transaction_decision.action == 'promote':
            selected_confidence = tentative_prediction[1]
            selected_branch = 'tentative'
            self._install_snapshot(transaction_decision.resolved_snapshot)
        elif transaction_decision.action == 'rollback':
            selected_confidence = protected_prediction[1]
            selected_branch = 'protected'
            self._install_snapshot(transaction_decision.resolved_snapshot)
        else:
            selected_confidence = protected_prediction[1]
            selected_branch = 'protected'
            self._install_snapshot(protected_current)
        output = {
            'target_bbox': self.state,
            'best_score': selected_confidence,
            'protected_transaction': {
                'active_before': True,
                'selected_branch': selected_branch,
                'decision': self._decision_trace(transaction_decision),
                'protected_bbox': list(protected_current.state),
                'tentative_bbox': list(tentative_current.state),
                'protected_evidence': asdict(protected_evidence),
                'tentative_evidence': asdict(tentative_evidence),
                'protected_policy_decision': asdict(
                    protected_policy_decision),
                'tentative_policy_decision': asdict(
                    tentative_policy_decision),
            },
        }
        self._write_transaction_trace(output['protected_transaction'])
        return output

    def track(self, image, info=None):
        if self.debug:
            raise ValueError('Protected transaction tracker requires debug=0')
        self.frame_id += 1
        depth_path = None if info is None else info.get('depth_path')
        if self.template_transaction.active:
            return self._track_active_transaction(image, depth_path)

        prior_state = list(self.state)
        prior_templates = list(self.template_list)
        prior_annos = list(self.template_anno_list)
        prior_policy = copy.deepcopy(self.safe_template_policy)
        writer_decision = None
        transaction_decision = None
        event_kind = None
        confidence_tensor = 0.0
        try:
            prediction = self._infer(
                image, prior_state, self.template_list,
                self.template_anno_list)
            bbox, confidence_tensor, confidence, margin = prediction
            candidate_policy = copy.deepcopy(prior_policy)
            writer_decision = candidate_policy.observe(
                self.frame_id, image, bbox, confidence, margin, depth_path)
            if writer_decision.rollback_state:
                raise ValueError(
                    'Legacy state rollback is forbidden in transaction tracker')
            candidate_templates = list(prior_templates)
            candidate_annos = list(prior_annos)
            if writer_decision.drop_dynamic:
                candidate_templates = [
                    self.static_template] * self.num_template
                candidate_annos = [self.static_template_anno]

            hard_conflict = self._is_hard_conflict(writer_decision)
            protected_state = prior_state
            protected_templates = prior_templates
            protected_annos = prior_annos
            protected_policy = prior_policy
            if writer_decision.replace_dynamic:
                if hard_conflict:
                    raise ValueError(
                        'Template replacement cannot coincide with conflict')
                # The current bbox already passed every safe-v1 gate.  Keep it
                # in both branches so this transaction isolates only the
                # causal effect of writing the dynamic template.  The
                # protected policy advances through the current observation
                # but cancels the pending write.
                protected_state = list(bbox)
                protected_templates = list(candidate_templates)
                protected_annos = list(candidate_annos)
                protected_policy = copy.deepcopy(candidate_policy)
                protected_policy.cancel(self.frame_id)
                candidate_template, candidate_annotation = self._make_template(
                    image, bbox)
                candidate_templates, candidate_annos = self._append_dynamic(
                    candidate_templates, candidate_annos,
                    candidate_template, candidate_annotation,
                    self.num_template)
                candidate_policy.commit(self.frame_id)
                event_kind = 'template_candidate'
            elif hard_conflict:
                candidate_policy.pending_update_frame = None
                event_kind = 'state_conflict_candidate'

            if event_kind is not None:
                protected_snapshot = self._capture_snapshot(
                    protected_state, protected_templates, protected_annos,
                    protected_policy)
                tentative_snapshot = self._capture_snapshot(
                    bbox, candidate_templates, candidate_annos,
                    candidate_policy)
                transaction_decision = self.template_transaction.begin(
                    self.frame_id, protected_snapshot, tentative_snapshot,
                    self.identity_anchor)
                self.transaction_events_started += 1
                self._install_snapshot(protected_snapshot)
            else:
                self.state = list(bbox)
                self.template_list = candidate_templates
                self.template_anno_list = candidate_annos
                self.safe_template_policy = candidate_policy
        except Exception as error:
            if (isinstance(error, RuntimeError) and
                    'out of memory' in str(error).lower()):
                raise
            self.template_transaction = self._new_transaction()
            self.state = prior_state
            self.template_list = prior_templates
            self.template_anno_list = prior_annos
            self.safe_template_policy = prior_policy
            output = {
                'target_bbox': self.state,
                'best_score': 0.0,
                'protected_transaction': {
                    'active_before': False,
                    'selected_branch': 'stored_prior',
                    'decision': None,
                    'writer_decision': (
                        None if writer_decision is None else
                        asdict(writer_decision)),
                    'event_kind': 'creation_error',
                    'prior_bbox': prior_state,
                    'protected_bbox': list(self.state),
                    'events_started': int(self.transaction_events_started),
                    'recoverable_error': '{}: {}'.format(
                        type(error).__name__, str(error)[:300]),
                },
            }
            self._write_transaction_trace(output['protected_transaction'])
            return output

        output = {
            'target_bbox': self.state,
            'best_score': (
                0.0 if event_kind == 'state_conflict_candidate' else
                confidence_tensor),
            'protected_transaction': {
                'active_before': False,
                'selected_branch': 'protected',
                'decision': (None if transaction_decision is None else
                             self._decision_trace(transaction_decision)),
                'writer_decision': asdict(writer_decision),
                'event_kind': event_kind,
                'prior_bbox': prior_state,
                'protected_bbox': list(self.state),
                'events_started': int(self.transaction_events_started),
            },
        }
        if event_kind is not None:
            self._write_transaction_trace(output['protected_transaction'])
        return output


def get_tracker_class():
    return SUTRACKProtectedTransaction
