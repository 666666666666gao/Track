"""SUTrack Train-only same-frame expanded-search recovery experiment."""

import math

import torch

from lib.test.tracker.safe_template_update import (
    nms_response_margin,
    rgb_identity_similarity,
)
from lib.test.tracker.sutrack import SUTRACK
from lib.test.tracker.temporal_depth_identity import (
    normalized_center_jump,
    read_candidate_depth,
)
from lib.test.tracker.utils import sample_target
from lib.utils.box_ops import clip_box


class SUTRACKRecoverySearch(SUTRACK):
    """Run at most one strictly verified expanded search on a conflict frame."""

    def __init__(self, params, dataset_name):
        super().__init__(params, dataset_name)
        self.recovery_search_use = bool(params.recovery_search_use)
        self.recovery_search_factor = float(params.recovery_search_factor)
        self.recovery_search_max_consecutive = int(
            params.recovery_search_max_consecutive)
        self.recovery_search_cooldown_frames = int(
            params.recovery_search_cooldown_frames)
        if (not math.isfinite(self.recovery_search_factor) or
                self.recovery_search_factor <= float(self.params.search_factor) or
                self.recovery_search_max_consecutive <= 0 or
                self.recovery_search_cooldown_frames < 0):
            raise ValueError('Malformed expanded-search recovery parameters')
        if self.recovery_search_use and not self.safe_template_update:
            raise ValueError('Expanded-search recovery requires safe RGB-D evidence')
        self._recovery_search_consecutive = 0
        self._recovery_search_cooldown = 0

    def initialize(self, image, info):
        result = super().initialize(image, info)
        self._recovery_search_consecutive = 0
        self._recovery_search_cooldown = 0
        return result

    def _infer_from_anchor(self, image, anchor_bbox, search_factor):
        height, width = image.shape[:2]
        patch, resize_factor = sample_target(
            image, anchor_bbox, search_factor,
            output_sz=self.params.search_size)
        search = self.preprocessor.process(patch)
        if self.multi_modal_vision and search.size(1) == 3:
            search = torch.cat((search, search), axis=1)
        with torch.no_grad():
            encoded = self.network.forward_encoder(
                self.template_list, [search], self.template_anno_list,
                self.text_src, self.task_index_batch)
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
        mapped = [
            cx + cx_prev - half_side - 0.5 * box_width,
            cy + cy_prev - half_side - 0.5 * box_height,
            box_width,
            box_height,
        ]
        bbox = clip_box(mapped, height, width, margin=10)
        confidence_value = float(
            confidence.detach().reshape(-1).max().item())
        selected_index = int(
            response.detach().reshape(-1).argmax().item())
        margin = float(nms_response_margin(
            response.detach(), selected_index,
            kernel=self.safe_template_nms_kernel))
        if not all(math.isfinite(value) for value in (
                confidence_value, margin, *bbox)):
            raise ValueError('Non-finite expanded-search inference')
        return list(bbox), confidence_value, margin

    def _passive_evidence(
            self, image, prior_bbox, candidate_bbox, confidence,
            response_margin, depth_path):
        policy = self.safe_template_policy
        identity = rgb_identity_similarity(
            policy.reference_descriptor, image, candidate_bbox)
        jump = normalized_center_jump(prior_bbox, candidate_bbox)
        observation = read_candidate_depth(
            depth_path, candidate_bbox,
            center_fraction=policy.center_fraction)
        depth_valid = bool(
            observation.median is not None and
            observation.valid_ratio >= policy.min_depth_valid_ratio and
            policy.trusted_depth is not None and
            float(policy.trusted_depth) > 0.0)
        depth_change = None
        if depth_valid:
            depth_change = abs(math.log(
                float(observation.median) / float(policy.trusted_depth)))
        hard_conflict = bool(
            jump > policy.max_center_jump or
            (identity is not None and identity < policy.min_rgb_identity) or
            (depth_change is not None and
             depth_change > policy.max_log_depth_change))
        # Recovery intentionally does not reject a large center displacement;
        # that is the condition the expanded search is designed to repair.
        strict_recovery = bool(
            identity is not None and identity >= policy.min_rgb_identity and
            depth_valid and depth_change <= policy.max_log_depth_change and
            confidence >= policy.min_confidence and
            response_margin >= policy.min_response_margin)
        quality = None
        if identity is not None and depth_change is not None:
            depth_quality = math.exp(
                -depth_change / max(policy.max_log_depth_change, 1.0e-6))
            quality = float(
                0.35 * confidence +
                0.15 * min(max(response_margin, 0.0), 1.0) +
                0.35 * identity +
                0.15 * depth_quality)
        return {
            'identity_similarity': identity,
            'normalized_center_jump': float(jump),
            'depth_valid': depth_valid,
            'log_depth_change': depth_change,
            'confidence': float(confidence),
            'response_margin': float(response_margin),
            'hard_conflict': hard_conflict,
            'strict_recovery': strict_recovery,
            'quality': quality,
        }

    def track(self, image, info=None):
        if not self.recovery_search_use:
            output = super().track(image, info)
            evidence = output.get('online_state_evidence')
            if not isinstance(evidence, dict):
                raise ValueError(
                    'source-identical OFF path is missing online evidence')
            baseline_bbox = evidence.get('candidate_bbox')
            if (not isinstance(baseline_bbox, (list, tuple)) or
                    len(baseline_bbox) != 4):
                raise ValueError(
                    'source-identical OFF path is missing candidate bbox')
            output['recovery_search_evidence'] = {
                'enabled': False,
                'factor': float(self.recovery_search_factor),
                'second_pass': False,
                'recovery_selected': False,
                'baseline_bbox': list(baseline_bbox),
                'baseline_evidence': None,
                'recovery_bbox': None,
                'recovery_evidence': None,
                'cooldown_remaining': 0,
            }
            return output

        self.frame_id += 1
        prior_state = list(self.state)
        depth_path = None if info is None else info.get('depth_path')
        baseline_bbox, baseline_confidence, baseline_margin = (
            self._infer_from_anchor(
                image, prior_state, float(self.params.search_factor)))
        baseline_evidence = self._passive_evidence(
            image, prior_state, baseline_bbox, baseline_confidence,
            baseline_margin, depth_path)

        cooldown_blocked = self._recovery_search_cooldown > 0
        if cooldown_blocked:
            self._recovery_search_cooldown -= 1
        second_pass = False
        recovery_bbox = None
        recovery_confidence = None
        recovery_margin = None
        recovery_evidence = None
        choose_recovery = False
        if (self.recovery_search_use and
                baseline_evidence['hard_conflict'] and
                not cooldown_blocked):
            second_pass = True
            recovery_bbox, recovery_confidence, recovery_margin = (
                self._infer_from_anchor(
                    image, prior_state, self.recovery_search_factor))
            recovery_evidence = self._passive_evidence(
                image, prior_state, recovery_bbox, recovery_confidence,
                recovery_margin, depth_path)
            choose_recovery = bool(recovery_evidence['strict_recovery'])
            self._recovery_search_consecutive += 1
            if (self._recovery_search_consecutive >=
                    self.recovery_search_max_consecutive):
                self._recovery_search_consecutive = 0
                self._recovery_search_cooldown = (
                    self.recovery_search_cooldown_frames)
        elif not baseline_evidence['hard_conflict']:
            self._recovery_search_consecutive = 0

        if choose_recovery:
            candidate_state = list(recovery_bbox)
            confidence = recovery_confidence
            response_margin = recovery_margin
        else:
            candidate_state = list(baseline_bbox)
            confidence = baseline_confidence
            response_margin = baseline_margin
        self.state = list(candidate_state)

        decision = self.safe_template_policy.observe(
            self.frame_id, image, self.state, confidence,
            response_margin, depth_path)
        if decision.drop_dynamic:
            self._drop_dynamic_template()
        if decision.rollback_state:
            self.state = prior_state
        if decision.replace_dynamic:
            self._replace_dynamic_template(image)
            self.safe_template_policy.commit(self.frame_id)

        online_state_evidence = {
            'frame_id': int(self.frame_id),
            'prior_bbox': [float(value) for value in prior_state],
            'candidate_bbox': [float(value) for value in candidate_state],
            'confidence': float(confidence),
            'response_margin': float(response_margin),
            'identity_similarity': decision.identity_similarity,
            'normalized_center_jump': decision.normalized_center_jump,
            'log_depth_change': decision.log_depth_change,
            'checked': bool(decision.checked),
            'eligible': bool(decision.eligible),
            'dynamic_active': bool(decision.dynamic_active),
            'stable_frames': int(decision.stable_frames),
            'consecutive_state_rollbacks': int(
                decision.consecutive_state_rollbacks),
            'reasons': list(decision.reasons),
        }
        recovery_search_evidence = {
            'enabled': bool(self.recovery_search_use),
            'factor': float(self.recovery_search_factor),
            'second_pass': second_pass,
            'recovery_selected': choose_recovery,
            'baseline_bbox': [float(value) for value in baseline_bbox],
            'baseline_evidence': baseline_evidence,
            'recovery_bbox': (None if recovery_bbox is None else
                              [float(value) for value in recovery_bbox]),
            'recovery_evidence': recovery_evidence,
            'cooldown_remaining': int(self._recovery_search_cooldown),
        }
        return {
            'target_bbox': self.state,
            'best_score': confidence,
            'safe_template_decision': decision,
            'online_state_evidence': online_state_evidence,
            'recovery_search_evidence': recovery_search_evidence,
        }


def get_tracker_class():
    return SUTRACKRecoverySearch
