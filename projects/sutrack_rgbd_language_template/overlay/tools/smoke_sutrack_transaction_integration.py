#!/usr/bin/env python3
"""CPU-only structural smoke for the isolated low22 transaction tracker."""

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.test.tracker.protected_tentative_transaction import (
    ProtectedTentativeTemplateTransaction,
)
from lib.test.tracker.safe_template_update import (
    SafeTemplateDecision,
    SafeTemplateUpdatePolicy,
)
from lib.test.tracker.sutrack import SUTRACK
from lib.test.tracker.sutrack_transaction import (
    SUTRACKProtectedTransaction,
)
from tools.finalize_vot_transaction_low22 import count_confirmed_failures


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-{}'.format(os.getpid()))
    with open(temporary, 'x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_yaml(path):
    with open(path, 'r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()

    candidate_yaml = (
        REPO_ROOT / 'experiments' / 'sutrack' /
        'sutrack_l384_rgbd_anchor_identity_template_transaction_low22.yaml')
    baseline_yaml = (
        REPO_ROOT / 'experiments' / 'sutrack' /
        'sutrack_l384_rgbd_anchor_identity_low22.yaml')
    tracker_source = (
        REPO_ROOT / 'lib' / 'test' / 'tracker' /
        'sutrack_transaction.py')
    parameter_source = (
        REPO_ROOT / 'lib' / 'test' / 'parameter' /
        'sutrack_transaction.py')
    vot_source = (
        REPO_ROOT / 'lib' / 'test' / 'vot' /
        'sutrack_l384_rgbd_anchor_identity_template_transaction_low22.py')
    vot_adapter_source = (
        REPO_ROOT / 'lib' / 'test' / 'vot' /
        'sutrack_transaction_class.py')
    controller_source = (
        REPO_ROOT / 'lib' / 'test' / 'tracker' /
        'protected_tentative_transaction.py')
    prepare_source = (
        REPO_ROOT / 'tools' / 'prepare_vot_transaction_low22.py')
    launcher_source = (
        REPO_ROOT / 'tools' / 'launch_vot_transaction_low22.sh')
    finalizer_source = (
        REPO_ROOT / 'tools' / 'finalize_vot_transaction_low22.py')
    diagnostics_source = (
        REPO_ROOT / 'tools' /
        'finalize_vot_transaction_low22_diagnostics.py')
    gpu_smoke_source = (
        REPO_ROOT / 'tools' / 'smoke_sutrack_transaction_gpu.py')
    parity_smoke_source = (
        REPO_ROOT / 'tools' /
        'smoke_sutrack_template_transaction_parity.py')
    gpu_smoke_text = gpu_smoke_source.read_text(encoding='utf-8')
    launcher_text = launcher_source.read_text(encoding='utf-8')
    diagnostics_text = diagnostics_source.read_text(encoding='utf-8')

    transaction_run_root = Path(
        '/root/autodl-tmp/sutrack_template_transaction_low22_v2/run').resolve()
    expected_trace_root = transaction_run_root / 'transaction_traces'
    os.environ['SUTRACK_TRANSACTION_TRACE_ROOT'] = str(expected_trace_root)
    parameter_module = importlib.import_module(
        'lib.test.parameter.sutrack_transaction')
    params = parameter_module.parameters(
        'sutrack_l384_rgbd_anchor_identity_template_transaction_low22')
    settings = params.protected_tentative_transaction
    controller = ProtectedTentativeTemplateTransaction(**settings)
    checks = {
        'candidate_yaml_semantically_matches_low22_baseline': (
            read_yaml(candidate_yaml) == read_yaml(baseline_yaml)),
        'same_depthtrack_checkpoint_exists': Path(params.checkpoint).is_file(),
        'formal_trace_root_is_bound_inside_low22_run': bool(
            Path(params.transaction_trace_root).resolve() ==
            expected_trace_root and
            expected_trace_root.parent == transaction_run_root),
        'anchor_language_enabled': bool(
            params.cfg.TEST.RGBD_LANGUAGE.USE and
            params.cfg.TEST.RGBD_LANGUAGE.ANCHOR_SPECIFIC and
            params.cfg.TEST.RGBD_LANGUAGE.EXPECTED_RECORD_COUNT == 303),
        'safe_v1_writer_enabled': bool(
            params.cfg.TEST.SAFE_TEMPLATE_UPDATE.USE),
        'legacy_bbox_rollback_disabled': bool(
            not params.cfg.TEST.SAFE_TEMPLATE_UPDATE.
            HARD_CONFLICT_STATE_ROLLBACK),
        'two_frame_transaction_constructed': bool(
            controller.confirm_frames == 2 and
            controller.max_shadow_frames == 2 and
            settings['min_utility_advantage'] > 0.0),
        'tracker_is_distinct_subclass': bool(
            issubclass(SUTRACKProtectedTransaction, SUTRACK) and
            SUTRACKProtectedTransaction.track is not SUTRACK.track),
        'launcher_enforces_low22_gate_and_trace_binding': bool(
            'if [[ ! -f "$FULL_RESULT" ]]' in
            launcher_text and
            'SUTRACK_TRANSACTION_TRACE_ROOT="$TRACE_ROOT"' in
            launcher_text),
        'launcher_runs_metric_blind_gpu_smoke_before_low22_prepare': bool(
            'tools/smoke_sutrack_transaction_gpu.py' in launcher_text and
            'tools/smoke_sutrack_template_transaction_parity.py' in
            launcher_text and
            'SUTRACK_TRANSACTION_SMOKE_GPU' in launcher_text and
            launcher_text.index(
                'tools/smoke_sutrack_transaction_gpu.py') <
            launcher_text.index(
                'tools/smoke_sutrack_template_transaction_parity.py') <
            launcher_text.index('tools/prepare_vot_transaction_low22.py')),
        'gpu_smoke_matches_vot_runtime_flags': bool(
            'params.visualization = False' in gpu_smoke_text and
            'params.debug = False' in gpu_smoke_text),
        'gpu_smoke_uses_formal_depthtrack_runtime_path': bool(
            "SUTRACKProtectedTransaction(params, 'depthtrack')" in
            gpu_smoke_text),
        'launcher_restarts_diagnostics_after_completed_gate': bool(
            'gate_complete=false' in launcher_text and
            'if [[ "$gate_complete" == true ]]' in launcher_text and
            launcher_text.index(
                'tools/finalize_vot_transaction_low22_diagnostics.py') <
            launcher_text.index('if [[ "$gate_complete" == true ]]')),
        'finalizer_freezes_machine_readable_low22_gate': bool(
            "'eao_strictly_improved'" in
            finalizer_source.read_text(encoding='utf-8') and
            "'rob_strictly_improved'" in
            finalizer_source.read_text(encoding='utf-8') and
            "'acc_within_minus_0_10_pp'" in
            finalizer_source.read_text(encoding='utf-8') and
            "'confirmed_failures_not_increased'" in
            finalizer_source.read_text(encoding='utf-8') and
            "'automatic_full127_launch': False" in
            finalizer_source.read_text(encoding='utf-8')),
        'diagnostics_bind_sequence_metrics_and_transaction_traces': bool(
            "'candidate_minus_reference_percent'" in diagnostics_text and
            "'transaction_trace_summary'" in diagnostics_text and
            "'automatic_full127_launch': False" in diagnostics_text and
            'EXPECTED_SEQUENCES = 22' in diagnostics_text and
            'EXPECTED_ANCHORS = 303' in diagnostics_text),
    }

    probe = object.__new__(SUTRACKProtectedTransaction)
    probe.cfg = params.cfg
    policy = SafeTemplateUpdatePolicy.from_config(
        params.cfg.TEST.SAFE_TEMPLATE_UPDATE)
    policy.reference_descriptor = None
    policy_state = probe._policy_state(policy)
    restored = probe._policy_from_state(policy_state)
    checks['safe_policy_state_roundtrip'] = bool(
        set(vars(restored)) == set(vars(policy)) and
        restored.check_interval == policy.check_interval and
        restored.max_template_age == policy.max_template_age)

    static_template = torch.ones((1, 2), dtype=torch.float16)
    candidate_template = torch.full((1, 2), 2.0, dtype=torch.float16)
    static_annotation = torch.ones((1, 4), dtype=torch.float16)
    candidate_annotation = torch.full((1, 4), 2.0, dtype=torch.float16)
    templates, annos = probe._append_dynamic(
        [static_template, static_template], [static_annotation],
        candidate_template, candidate_annotation, 2)
    checks['real_sutrack_annotation_transition_supported'] = bool(
        len(templates) == 2 and len(annos) == 2 and
        torch.equal(templates[1], candidate_template) and
        torch.equal(annos[1], candidate_annotation))
    checks['continuity_transform_is_bounded'] = bool(
        probe._bounded_continuity(0.0, 0.08) == 1.0 and
        probe._bounded_continuity(0.08, 0.08) == 0.5 and
        probe._bounded_continuity(None, 0.08) == 0.0)

    probe.params = params
    probe.debug = 0
    probe.frame_id = 0
    probe.num_template = 2
    probe.state = [10.0, 10.0, 20.0, 20.0]
    probe.static_template = torch.ones((1, 2), dtype=torch.float32)
    probe.static_template_anno = torch.ones((1, 4), dtype=torch.float32)
    probe.template_list = [probe.static_template, probe.static_template]
    probe.template_anno_list = [probe.static_template_anno]
    probe.text_src = torch.ones((1, 3), dtype=torch.float32)
    probe.task_index_batch = None
    probe.safe_template_policy = SafeTemplateUpdatePolicy.from_config(
        params.cfg.TEST.SAFE_TEMPLATE_UPDATE)
    probe.identity_anchor = 'immutable-smoke-anchor'
    probe.transaction_settings = dict(settings)
    probe.template_transaction = ProtectedTentativeTemplateTransaction(
        **settings)
    probe.transaction_events_started = 0
    probe.transaction_trace_path = None
    candidate_bbox = [30.0, 30.0, 20.0, 20.0]
    probe._infer = lambda *unused_args, **unused_kwargs: (
        candidate_bbox, torch.tensor([0.90]), 0.90, 0.30)
    probe._make_template = lambda unused_image, unused_bbox: (
        torch.full((1, 2), 2.0), torch.full((1, 4), 2.0))

    def scripted_observe(
            policy_instance, frame_id, unused_image, bbox,
            unused_confidence, unused_margin, unused_depth):
        policy_instance.last_frame_id = int(frame_id)
        policy_instance.pending_update_frame = int(frame_id)
        policy_instance.previous_bbox = list(bbox)
        return SafeTemplateDecision(
            checked=True, eligible=True, replace_dynamic=True,
            drop_dynamic=False, rollback_state=False, reasons=(),
            stable_frames=0, dynamic_active=False,
            consecutive_state_rollbacks=0, identity_similarity=0.90,
            normalized_center_jump=0.10, log_depth_change=0.01,
            blend_weight=0.0)

    with patch.object(
            SafeTemplateUpdatePolicy, 'observe', new=scripted_observe):
        protected_output = probe.track(np.zeros((8, 8, 6)), {})
    protected_snapshot, tentative_snapshot = (
        probe.template_transaction.active_snapshots())
    checks['template_candidate_keeps_current_bbox_and_old_template'] = bool(
        protected_output['target_bbox'] == candidate_bbox and
        torch.equal(protected_output['best_score'], torch.tensor([0.90])) and
        protected_snapshot.state == tuple(candidate_bbox) and
        torch.equal(protected_snapshot.templates[1], probe.static_template) and
        tentative_snapshot.state == (30.0, 30.0, 20.0, 20.0) and
        torch.equal(tentative_snapshot.templates[1], torch.full((1, 2), 2.0)) and
        protected_snapshot.auxiliary['safe_policy_state']['last_frame_id'] == 1 and
        protected_snapshot.auxiliary[
            'safe_policy_state']['pending_update_frame'] is None and
        not protected_snapshot.auxiliary[
            'safe_policy_state']['dynamic_active'] and
        tentative_snapshot.auxiliary['safe_policy_state']['last_frame_id'] == 1 and
        tentative_snapshot.auxiliary['safe_policy_state']['dynamic_active'])

    active_anchor_types = []

    def list_only_active_infer(
            unused_image, anchor_bbox, unused_templates, unused_annos,
            **unused_kwargs):
        active_anchor_types.append(type(anchor_bbox))
        if not isinstance(anchor_bbox, list):
            raise TypeError('active snapshot bbox was not materialized')
        return candidate_bbox, torch.tensor([0.90]), 0.90, 0.30

    probe._infer = list_only_active_infer
    with patch.object(
            SafeTemplateUpdatePolicy, 'observe', new=scripted_observe):
        active_output = probe.track(np.zeros((8, 8, 6)), {})
    checks['active_branches_materialize_snapshot_bbox_as_list'] = bool(
        active_anchor_types == [list, list] and
        active_output['protected_transaction'].get(
            'recoverable_error') is None)

    probe.template_transaction = ProtectedTentativeTemplateTransaction(
        **settings)
    probe.state = [10.0, 10.0, 20.0, 20.0]
    probe.template_list = [probe.static_template, probe.static_template]
    probe.template_anno_list = [probe.static_template_anno]
    probe.safe_template_policy = SafeTemplateUpdatePolicy.from_config(
        params.cfg.TEST.SAFE_TEMPLATE_UPDATE)

    def scripted_conflict(
            policy_instance, frame_id, unused_image, bbox,
            unused_confidence, unused_margin, unused_depth):
        policy_instance.last_frame_id = int(frame_id)
        policy_instance.pending_update_frame = None
        policy_instance.previous_bbox = list(bbox)
        return SafeTemplateDecision(
            checked=True, eligible=False, replace_dynamic=False,
            drop_dynamic=False, rollback_state=False,
            reasons=('large_center_jump',), stable_frames=0,
            dynamic_active=False, consecutive_state_rollbacks=0,
            identity_similarity=0.90, normalized_center_jump=1.0,
            log_depth_change=0.01, blend_weight=0.0)

    with patch.object(
            SafeTemplateUpdatePolicy, 'observe', new=scripted_conflict):
        conflict_output = probe.track(np.zeros((8, 8, 6)), {})
    checks['state_conflict_preserves_direct_baseline_bbox_semantics'] = bool(
        conflict_output['target_bbox'] == candidate_bbox and
        torch.equal(conflict_output['best_score'], torch.tensor([0.90])) and
        probe.state == candidate_bbox and
        not probe.template_transaction.active and
        conflict_output['protected_transaction']['event_kind'] is None and
        probe.safe_template_policy.last_frame_id == probe.frame_id and
        probe.safe_template_policy.previous_bbox == candidate_bbox)

    conflict_snapshot = probe._capture_snapshot(
        [10.0, 10.0, 20.0, 20.0],
        [probe.static_template, probe.static_template],
        [probe.static_template_anno],
        SafeTemplateUpdatePolicy.from_config(
            params.cfg.TEST.SAFE_TEMPLATE_UPDATE))
    with patch.object(
            SafeTemplateUpdatePolicy, 'observe', new=scripted_conflict):
        advanced_conflict, conflict_evidence, unused_decision = (
            probe._advance_branch(
                conflict_snapshot, np.zeros((8, 8, 6)), None,
                (candidate_bbox, torch.tensor([0.90]), 0.90, 0.30)))
    checks['active_shadow_conflict_also_accepts_current_bbox'] = bool(
        advanced_conflict.state == tuple(candidate_bbox) and
        conflict_evidence.hard_conflict and
        advanced_conflict.auxiliary[
            'safe_policy_state']['previous_bbox'] == candidate_bbox)

    probe.template_transaction = ProtectedTentativeTemplateTransaction(
        **settings)
    probe.state = [10.0, 10.0, 20.0, 20.0]
    probe.template_list = [probe.static_template, probe.static_template]
    probe.template_anno_list = [probe.static_template_anno]
    probe.safe_template_policy = SafeTemplateUpdatePolicy.from_config(
        params.cfg.TEST.SAFE_TEMPLATE_UPDATE)

    def malformed_template(unused_image, unused_bbox):
        raise ValueError('malformed candidate template')

    probe._make_template = malformed_template
    with patch.object(
            SafeTemplateUpdatePolicy, 'observe', new=scripted_observe):
        recovered_output = probe.track(np.zeros((8, 8, 6)), {})
    checks['transaction_creation_error_restores_complete_prior'] = bool(
        recovered_output['target_bbox'] == [10.0, 10.0, 20.0, 20.0] and
        recovered_output['best_score'] == 0.0 and
        not probe.template_transaction.active and
        recovered_output['protected_transaction']['event_kind'] ==
        'creation_error' and
        'ValueError' in recovered_output[
            'protected_transaction']['recoverable_error'])
    with tempfile.TemporaryDirectory() as trace_directory:
        probe.transaction_trace_path = (
            Path(trace_directory) / 'sequence__anchor-000001.jsonl')
        probe._write_transaction_trace({
            'decision': {'action': 'hold'},
            'selected_branch': 'protected',
        })
        trace_record = json.loads(
            probe.transaction_trace_path.read_text(encoding='utf-8'))
    checks['trajectory_transaction_trace_is_persisted'] = bool(
        trace_record['type'] == 'transaction_frame' and
        trace_record['frame_id'] == probe.frame_id and
        trace_record['decision']['action'] == 'hold')
    probe.transaction_trace_path = None

    public_tracker = (
        REPO_ROOT / 'lib' / 'test' / 'tracker' / 'sutrack.py')
    public_vot = (
        REPO_ROOT / 'lib' / 'test' / 'vot' /
        'sutrack_l384_rgbd_anchor_identity_all127.py')
    public_yaml = (
        REPO_ROOT / 'experiments' / 'sutrack' /
        'sutrack_l384_rgbd_anchor_identity_all127.yaml')
    public_text = '\n'.join(
        path.read_text(encoding='utf-8')
        for path in (public_tracker, public_vot, public_yaml))
    checks['current_full127_path_is_unmodified_and_unwired'] = (
        'sutrack_transaction' not in public_text and
        'ProtectedTentativeTemplateTransaction' not in public_text)

    source_snapshot = Path(
        '/root/autodl-tmp/sutrack_vot_all127_anchor_identity_v3/run/'
        'finalizer_source_snapshot.json')
    expected_snapshot_sha = (
        '6084aae897dbdbdc0621d6e6fa9f4ccc0b9f10726b4ce2f32e5b3734a6daa2c6')
    snapshot_payload = json.loads(source_snapshot.read_text(encoding='utf-8'))
    repo_entries = [
        value for value in snapshot_payload['sources'].values()
        if value['path'].startswith(str(REPO_ROOT) + '/')
    ]
    checks['current_full127_frozen_repo_sources_match_exact_sha'] = bool(
        sha256_file(source_snapshot) == expected_snapshot_sha and
        all(sha256_file(Path(item['path'])) == item['sha256']
            for item in repo_entries))
    checkpoint_entry = snapshot_payload['sources']['checkpoint']
    checks['depthtrack_checkpoint_binding_is_exact'] = bool(
        Path(params.checkpoint).resolve() ==
        Path(checkpoint_entry['path']).resolve() and
        checkpoint_entry['sha256'] ==
        '2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4' and
        sha256_file(Path(params.checkpoint)) == checkpoint_entry['sha256'] and
        Path(params.checkpoint).stat().st_size == checkpoint_entry['size'])

    baseline_failures, baseline_failure_rows, baseline_failure_settings = (
        count_confirmed_failures(
            Path('/root/autodl-tmp/sutrack_vot_low22_anchor_identity_v1/'
                 'run/master'),
            'sutrack_l384_rgbd_anchor_identity_low22'))
    checks['official_failure_counter_reproduces_frozen_195'] = bool(
        baseline_failures == 195 and
        sum(row['anchors'] for row in baseline_failure_rows.values()) == 303 and
        baseline_failure_settings['grace'] == 10 and
        math.isclose(
            baseline_failure_settings['threshold'], 0.1,
            rel_tol=0.0, abs_tol=1.0e-12))

    transaction_processes = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = (process / 'cmdline').read_bytes().replace(b'\0', b' ')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b'sutrack_l384_rgbd_anchor_identity_template_transaction_low22' in command:
            transaction_processes.append(int(process.name))
    checks['transaction_low22_has_not_started'] = bool(
        not transaction_processes and not transaction_run_root.exists())

    required_sources = (
        tracker_source, parameter_source, vot_source, vot_adapter_source,
        candidate_yaml, controller_source, prepare_source, launcher_source,
        finalizer_source, diagnostics_source, gpu_smoke_source,
        parity_smoke_source)
    checks['all_isolated_sources_exist'] = all(
        path.is_file() for path in required_sources)

    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise AssertionError(
            'transaction integration smoke failed: {}'.format(failed))

    payload = {
        'schema': 'sutrack_template_transaction_low22_structural_smoke_v2',
        'status': 'passed',
        'gpu_inference_exercised': False,
        'public_full127_path_changed': False,
        'low22_vot_started': not checks['transaction_low22_has_not_started'],
        'transaction_process_ids': transaction_processes,
        'checks': checks,
        'sources': {
            str(path): sha256_file(path) for path in required_sources
        },
        'smoke_script': {
            'path': str(Path(__file__).resolve()),
            'sha256': sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(args.output_json.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
