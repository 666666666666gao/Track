#!/usr/bin/env python3
"""CPU structural checks for baseline-first counterfactual template veto."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.test.parameter.sutrack_template_veto_transaction import parameters
from lib.test.tracker.protected_tentative_transaction import (
    BranchEvidence,
    ProtectedTentativeTemplateTransaction,
    TrackerRecursiveSnapshot,
)
from lib.test.tracker.sutrack_template_veto_transaction import (
    SUTRACKBaselineFirstTemplateVeto,
    get_tracker_class,
)
from lib.test.tracker.sutrack_transaction import SUTRACKProtectedTransaction


YAML_NAME = 'sutrack_l384_rgbd_anchor_identity_template_veto_low22'


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(value):
    return TrackerRecursiveSnapshot.capture(
        [10.0, 12.0, 20.0, 18.0],
        [torch.tensor([value]), torch.tensor([value])],
        [torch.tensor([value])],
        {'role': 'old' if value == 1.0 else 'new'})


def evidence(confidence, margin, identity, depth, temporal):
    return BranchEvidence(
        confidence=confidence,
        response_margin=margin,
        identity_similarity=identity,
        depth_consistency=depth,
        temporal_continuity=temporal,
        identity_anchor='frozen-anchor',
        hard_conflict=False)


def atomic_json(path, payload):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-{}'.format(os.getpid()))
    with open(temporary, 'x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()

    candidate_yaml = (
        REPO_ROOT / 'experiments' / 'sutrack' /
        'sutrack_l384_rgbd_anchor_identity_template_veto_low22.yaml')
    baseline_yaml = (
        REPO_ROOT / 'experiments' / 'sutrack' /
        'sutrack_l384_rgbd_anchor_identity_low22.yaml')
    with candidate_yaml.open('r', encoding='utf-8') as stream:
        candidate_config = yaml.safe_load(stream)
    with baseline_yaml.open('r', encoding='utf-8') as stream:
        baseline_config = yaml.safe_load(stream)

    trace_root = Path(
        '/root/autodl-tmp/sutrack_template_veto_low22_v3/'
        'structural_trace_contract').resolve()
    os.environ['SUTRACK_TRANSACTION_TRACE_ROOT'] = str(trace_root)
    params = parameters(YAML_NAME)
    settings = dict(params.protected_tentative_transaction)

    old = snapshot(1.0)
    new = snapshot(2.0)
    base_protected, base_tentative = (
        SUTRACKProtectedTransaction.
        _orient_template_transaction_snapshots(old, new))
    veto_protected, veto_tentative = (
        SUTRACKBaselineFirstTemplateVeto.
        _orient_template_transaction_snapshots(old, new))

    controller = ProtectedTentativeTemplateTransaction(**settings)
    controller.begin(1, veto_protected, veto_tentative, 'frozen-anchor')
    weak_new = evidence(0.70, 0.20, 0.80, 0.80, 0.80)
    strong_old = evidence(0.90, 0.40, 0.90, 0.90, 0.90)
    first = controller.observe(
        2, veto_protected, veto_tentative, weak_new, strong_old)
    second = controller.observe(
        3, veto_protected, veto_tentative, weak_new, strong_old)

    rollback_controller = ProtectedTentativeTemplateTransaction(**settings)
    rollback_controller.begin(
        1, veto_protected, veto_tentative, 'frozen-anchor')
    rollback_controller.observe(
        2, veto_protected, veto_tentative, strong_old, weak_new)
    rollback = rollback_controller.observe(
        3, veto_protected, veto_tentative, strong_old, weak_new)

    tracker_source = (
        REPO_ROOT / 'lib/test/tracker/sutrack_template_veto_transaction.py')
    wrapper_source = (
        REPO_ROOT / 'lib/test/vot/'
        'sutrack_l384_rgbd_anchor_identity_template_veto_low22.py')
    launcher_source = (
        REPO_ROOT / 'tools/launch_vot_template_veto_low22.sh')
    tracker_text = tracker_source.read_text(encoding='utf-8')
    wrapper_text = wrapper_source.read_text(encoding='utf-8')
    launcher_text = launcher_source.read_text(encoding='utf-8')

    checks = {
        'candidate_yaml_semantically_matches_frozen_identity_baseline': (
            candidate_config == baseline_config),
        'tracker_loader_returns_distinct_veto_subclass': (
            get_tracker_class() is SUTRACKBaselineFirstTemplateVeto and
            issubclass(
                SUTRACKBaselineFirstTemplateVeto,
                SUTRACKProtectedTransaction)),
        'historical_tracker_orientation_is_unchanged': (
            base_protected.auxiliary['role'] == 'old' and
            base_tentative.auxiliary['role'] == 'new'),
        'veto_public_protected_branch_is_direct_baseline_new_template': (
            veto_protected.auxiliary['role'] == 'new'),
        'veto_tentative_branch_is_counterfactual_old_template': (
            veto_tentative.auxiliary['role'] == 'old'),
        'two_future_advantages_promote_old_template_veto': (
            first.action == 'hold' and second.action == 'promote' and
            second.resolved_snapshot.auxiliary['role'] == 'old'),
        'failed_veto_keeps_direct_baseline_new_template': (
            rollback.action == 'rollback' and
            rollback.resolved_snapshot.auxiliary['role'] == 'new'),
        'trace_declares_branch_semantics': (
            'direct_baseline_new_template' in tracker_text and
            'counterfactual_old_template' in tracker_text and
            'veto_template_update' in tracker_text),
        'vot_wrapper_binds_only_veto_tracker_and_low22_yaml': (
            "'sutrack_template_veto_transaction'" in wrapper_text and
            "'sutrack_l384_rgbd_anchor_identity_template_veto_low22'" in
            wrapper_text),
        'launcher_is_low22_only_and_has_no_full127_execution': (
            'sutrack_template_veto_low22_v3/run' in launcher_text and
            'run_vot_failure_family_shards.py' in launcher_text and
            'full127_authorized' not in launcher_text and
            'launch_vot_full127' not in launcher_text),
        'frozen_two_frame_controller_contract': (
            settings['confirm_frames'] == 2 and
            settings['max_shadow_frames'] == 2),
    }
    failed = sorted(name for name, value in checks.items() if not value)
    if failed:
        raise AssertionError('Template-veto structural checks failed: {}'.format(
            failed))
    payload = {
        'schema': 'sutrack_template_veto_low22_structural_smoke_v1',
        'status': 'passed',
        'check_count': len(checks),
        'checks': checks,
        'metric_computed': False,
        'future_ground_truth_read': False,
        'source_sha256': {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in (tracker_source, wrapper_source, launcher_source)
        },
    }
    atomic_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
