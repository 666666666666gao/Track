#!/usr/bin/env python3
"""Explain low22 transaction results with exact sequence and trace diagnostics."""

import argparse
from collections import Counter
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


SCHEMA = 'sutrack_transaction_low22_diagnostics_v1'
GATE_SCHEMA = 'sutrack_transaction_low22_gate_v1'
BASELINE_TRACKER = 'sutrack_l384_rgbd_anchor_identity_low22'
CANDIDATE_TRACKER = (
    'sutrack_l384_rgbd_anchor_identity_template_transaction_low22')
EXPECTED_SEQUENCES = 22
EXPECTED_ANCHORS = 303
EXPECTED_CHECKPOINT = (
    '2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4')


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-{}'.format(os.getpid()))
    with open(temporary, 'x', encoding='utf-8', newline='\n') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_text(
        path,
        json.dumps(
            value, ensure_ascii=False, indent=2, sort_keys=True,
            allow_nan=False) + '\n')


def read_json(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('{} must contain an object'.format(path))
    return value


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError('Unable to load {}'.format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_gate(gate):
    expected = {
        'schema': GATE_SCHEMA,
        'status': 'complete',
        'tracker': CANDIDATE_TRACKER,
        'toolkit': '0.7.1',
        'sequence_count': EXPECTED_SEQUENCES,
        'anchor_count': EXPECTED_ANCHORS,
        'automatic_full127_launch': False,
    }
    for key, value in expected.items():
        if gate.get(key) != value:
            raise ValueError(
                'gate {} mismatch: {} != {}'.format(
                    key, gate.get(key), value))
    if gate.get('full127_authorized') != gate.get('gate_passed'):
        raise ValueError('gate authorization differs from gate outcome')


def checkpoint_from_gate(gate):
    record = gate.get('source_snapshot', {})
    path = Path(record.get('path', '')).resolve()
    if not path.is_file() or sha256_file(path) != record.get('sha256'):
        raise ValueError('gate source snapshot binding differs')
    snapshot = read_json(path)
    checkpoint = snapshot.get('sources', {}).get('checkpoint', {})
    checkpoint_sha = checkpoint.get('sha256')
    if checkpoint_sha != EXPECTED_CHECKPOINT:
        raise ValueError('transaction checkpoint SHA differs')
    return checkpoint_sha


def create_analysis_view(root, gate, baseline_report_sha, checkpoint_sha):
    master = root / 'master'
    for name in ('config.yaml', 'trackers.ini', 'sequences', 'results'):
        if not (master / name).exists():
            raise FileNotFoundError(master / name)
    merge = Path(gate['merge_result']['path']).resolve()
    if sha256_file(merge) != gate['merge_result']['sha256']:
        raise ValueError('gate merge artifact SHA mismatch')
    view = root / 'transaction_diagnostics_workspace_view'
    manifest = {
        'schema': 'sutrack_transaction_low22_analysis_adapter_v1',
        'protocol': 'official_vot2022_rgbd_multistart',
        'tracker_id': CANDIDATE_TRACKER,
        'checkpoint': {'sha256': checkpoint_sha},
        'source_master_workspace': str(master.resolve()),
        'source_gate_result_sha256': sha256_file(
            root / 'low22_gate_result.json'),
        'source_merge_result_sha256': gate['merge_result']['sha256'],
        'baseline_report_sha256': baseline_report_sha,
        'sequence_count': EXPECTED_SEQUENCES,
        'anchor_count': EXPECTED_ANCHORS,
        'purpose': 'read_only_transaction_sequence_and_trace_diagnostics',
    }
    if view.exists():
        if read_json(view / 'manifest.json') != manifest:
            raise ValueError('existing transaction analysis view differs')
        for name in ('config.yaml', 'trackers.ini', 'sequences', 'results'):
            link = view / name
            if not link.is_symlink() or link.resolve() != (master / name).resolve():
                raise ValueError('analysis view binding changed: {}'.format(name))
        return view

    temporary = view.with_name(view.name + '.tmp-{}'.format(os.getpid()))
    temporary.mkdir(parents=True)
    try:
        for name in ('config.yaml', 'trackers.ini', 'sequences', 'results'):
            (temporary / name).symlink_to(
                (master / name).resolve(),
                target_is_directory=(master / name).is_dir())
        atomic_json(temporary / 'manifest.json', manifest)
        os.replace(temporary, view)
    except BaseException:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
        raise
    return view


def assert_metrics(label, measured, expected):
    mapping = {'eao': 'eao', 'accuracy': 'accuracy', 'robustness': 'robustness'}
    for measured_key, expected_key in mapping.items():
        if not math.isclose(
                float(measured[measured_key]), float(expected[expected_key]),
                rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                '{} {} mismatch: {} != {}'.format(
                    label, measured_key, measured[measured_key],
                    expected[expected_key]))


def sequence_comparison(args, baseline_report, gate, candidate_view):
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.sequence_tools_root.resolve()))
    sys.path.insert(0, str(repo_root))
    analyzer_path = (
        args.sequence_tools_root /
        'tools/analyze_votrgbd2022_sequence_gaps.py')
    analyzer = load_module(analyzer_path, 'transaction_low22_sequence_gaps')
    names = sorted(baseline_report['candidate']['sequences'])
    if len(names) != EXPECTED_SEQUENCES:
        raise ValueError('baseline report does not contain frozen low22')
    payload = analyzer.analyze_pair(
        args.baseline_workspace.resolve(), candidate_view,
        reference_tracker_id=BASELINE_TRACKER,
        candidate_tracker_id=CANDIDATE_TRACKER,
        top_n=EXPECTED_SEQUENCES, sequence_names=names)
    comparison = payload['comparison']
    baseline_expected = {
        'eao': baseline_report['candidate']['aggregate']['eao'],
        'accuracy': baseline_report['candidate']['aggregate']['accuracy'],
        'robustness': baseline_report['candidate']['aggregate']['robustness'],
    }
    candidate_metrics = gate['candidate']['metrics_fraction']
    candidate_expected = {
        'eao': candidate_metrics['eao'],
        'accuracy': candidate_metrics['acc'],
        'robustness': candidate_metrics['rob'],
    }
    assert_metrics('reference', comparison['reference'], baseline_expected)
    assert_metrics('candidate', comparison['candidate'], candidate_expected)
    if sorted(row['sequence'] for row in comparison['per_sequence']) != names:
        raise ValueError('per-sequence comparison coverage differs')
    compact = {
        'reference': comparison['reference'],
        'candidate': comparison['candidate'],
        'candidate_minus_reference_percent': (
            comparison['candidate_minus_reference_percent']),
        'sequence_contribution_sums_percent': (
            comparison['sequence_contribution_sums_percent']),
        'per_sequence': comparison['per_sequence'],
    }
    return compact, analyzer_path


def new_event_counter():
    return Counter({
        'events_started': 0,
        'template_candidates': 0,
        'state_conflict_candidates': 0,
        'promotes': 0,
        'rollbacks': 0,
        'unresolved_at_trajectory_end': 0,
        'creation_errors': 0,
        'recoverable_errors': 0,
        'transaction_frames': 0,
    })


def summarize_traces(gate, expected_sequences):
    trace_root = Path(gate['transaction_traces']['root']).resolve()
    paths = sorted(trace_root.glob('*.jsonl'))
    if len(paths) != EXPECTED_ANCHORS:
        raise ValueError('transaction trace file count differs')
    totals = new_event_counter()
    by_sequence = {name: new_event_counter() for name in expected_sequences}
    rollback_reasons = Counter()
    start_kinds = Counter()
    trajectories_without_events = 0

    for path in paths:
        lines = [line for line in path.read_text(encoding='utf-8').splitlines()
                 if line.strip()]
        if not lines:
            raise ValueError('empty trace {}'.format(path))
        records = [json.loads(line) for line in lines]
        initialize = records[0]
        if initialize.get('type') != 'initialize':
            raise ValueError('trace lacks initialize record: {}'.format(path))
        sequence = initialize.get('sequence_name')
        if sequence not in by_sequence:
            raise ValueError('trace sequence is outside low22: {}'.format(sequence))
        sequence_counter = by_sequence[sequence]
        events = {}
        for record in records[1:]:
            if record.get('type') != 'transaction_frame':
                raise ValueError('unexpected trace record type in {}'.format(path))
            totals['transaction_frames'] += 1
            sequence_counter['transaction_frames'] += 1
            if record.get('recoverable_error'):
                totals['recoverable_errors'] += 1
                sequence_counter['recoverable_errors'] += 1
            if record.get('event_kind') == 'creation_error':
                totals['creation_errors'] += 1
                sequence_counter['creation_errors'] += 1
            decision = record.get('decision')
            if decision is None:
                continue
            event_id = int(decision['event_id'])
            action = decision.get('action')
            reasons = tuple(decision.get('reasons', []))
            if 'transaction_started' in reasons:
                if event_id in events:
                    raise ValueError('duplicate transaction event start')
                kind = record.get('event_kind')
                if kind not in ('template_candidate', 'state_conflict_candidate'):
                    raise ValueError('transaction start kind is malformed')
                events[event_id] = {'kind': kind, 'resolution': None}
                totals['events_started'] += 1
                sequence_counter['events_started'] += 1
                start_kinds[kind] += 1
                counter_key = (
                    'template_candidates' if kind == 'template_candidate' else
                    'state_conflict_candidates')
                totals[counter_key] += 1
                sequence_counter[counter_key] += 1
            if action in ('promote', 'rollback'):
                if event_id not in events:
                    raise ValueError('transaction resolution without start')
                if events[event_id]['resolution'] is not None:
                    raise ValueError('transaction event resolved more than once')
                events[event_id]['resolution'] = action
                counter_key = 'promotes' if action == 'promote' else 'rollbacks'
                totals[counter_key] += 1
                sequence_counter[counter_key] += 1
                if action == 'rollback':
                    rollback_reasons.update(reasons)
        unresolved = sum(
            event['resolution'] is None for event in events.values())
        totals['unresolved_at_trajectory_end'] += unresolved
        sequence_counter['unresolved_at_trajectory_end'] += unresolved
        if not events:
            trajectories_without_events += 1

    if totals['events_started'] != (
            totals['promotes'] + totals['rollbacks'] +
            totals['unresolved_at_trajectory_end']):
        raise ValueError('transaction event accounting does not balance')
    if sum(counter['events_started'] for counter in by_sequence.values()) != (
            totals['events_started']):
        raise ValueError('per-sequence event accounting does not balance')
    return {
        'trace_root': str(trace_root),
        'trace_file_count': len(paths),
        'trajectories_without_events': trajectories_without_events,
        'totals': dict(totals),
        'start_kinds': dict(sorted(start_kinds.items())),
        'rollback_reasons': dict(sorted(rollback_reasons.items())),
        'by_sequence': {
            key: dict(value) for key, value in sorted(by_sequence.items())},
    }


def render_markdown(payload):
    comparison = payload['comparison']
    delta = comparison['candidate_minus_reference_percent']
    gate = payload['gate']
    trace = payload['transaction_trace_summary']
    totals = trace['totals']
    failures_old = payload['failures']['reference_by_sequence']
    failures_new = payload['failures']['candidate_by_sequence']
    lines = [
        '# 保护—暂存模板事务 low22 诊断', '',
        '## 聚合结果', '',
        '| 指标 | anchor身份文本基线 | 模板事务 | Δpp | 门控 |',
        '|---|---:|---:|---:|---|',
        '| EAO | {:.6f} | {:.6f} | {:+.6f} | {} |'.format(
            100 * comparison['reference']['eao'],
            100 * comparison['candidate']['eao'], delta['eao'],
            gate['gate_checks']['eao_strictly_improved']),
        '| ACC | {:.6f} | {:.6f} | {:+.6f} | {} |'.format(
            100 * comparison['reference']['accuracy'],
            100 * comparison['candidate']['accuracy'], delta['accuracy'],
            gate['gate_checks']['acc_within_minus_0_10_pp']),
        '| ROB | {:.6f} | {:.6f} | {:+.6f} | {} |'.format(
            100 * comparison['reference']['robustness'],
            100 * comparison['candidate']['robustness'], delta['robustness'],
            gate['gate_checks']['rob_strictly_improved']), '',
        '- gate_passed={}；full127_authorized={}；automatic_full127_launch=false。'.format(
            gate['gate_passed'], gate['full127_authorized']),
        '- 确认失败anchor：{}→{}（{:+d}）。'.format(
            gate['baseline']['confirmed_failures'],
            gate['candidate']['confirmed_failures'], gate['failure_delta']), '',
        '## 事务行为', '',
        '- 启动{}次：模板候选{}次，状态冲突候选{}次。'.format(
            totals['events_started'], totals['template_candidates'],
            totals['state_conflict_candidates']),
        '- promote {}次，rollback {}次，轨迹结束时未决{}次。'.format(
            totals['promotes'], totals['rollbacks'],
            totals['unresolved_at_trajectory_end']),
        '- 可恢复异常{}次，创建异常{}次；无事务trajectory {}条。'.format(
            totals['recoverable_errors'], totals['creation_errors'],
            trace['trajectories_without_events']), '',
        '## 逐序列', '',
        '| 序列 | EAO全局贡献Δpp | ACCΔpp | ROBΔpp | 失败旧→新 | 启动 | promote | rollback | 未决 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in sorted(comparison['per_sequence'], key=lambda item: item['sequence']):
        name = row['sequence']
        events = trace['by_sequence'][name]
        lines.append(
            '| {} | {:+.6f} | {:+.6f} | {:+.6f} | {}→{} | {} | {} | {} | {} |'.format(
                name, row['eao_global_contribution_delta_percent'],
                row['accuracy_delta_percent'], row['robustness_delta_percent'],
                failures_old[name], failures_new[name],
                events['events_started'], events['promotes'],
                events['rollbacks'], events['unresolved_at_trajectory_end']))
    lines.extend([
        '', '## 解释约束', '',
        '- 逐序列EAO列是对low22聚合EAO差值的精确可加贡献，不是singleton EAO。',
        '- trace只记录公开在线证据和事务动作；GT只在评测后离线计算指标。',
        '- 即使门控通过，也不会自动启动full127，必须人工复核本报告。',
    ])
    return '\n'.join(lines) + '\n'


def run(args):
    root = args.root.resolve()
    gate_path = root / 'low22_gate_result.json'
    status_path = root / 'transaction_diagnostics_status.json'
    while not gate_path.is_file():
        progress = None
        finalizer_status = root / 'transaction_finalizer_status.json'
        if finalizer_status.is_file():
            progress = read_json(finalizer_status).get('completed_anchors')
        atomic_json(status_path, {
            'schema': SCHEMA,
            'stage': 'waiting_for_low22_gate_result',
            'completed_anchors': progress,
            'updated_at': utc_now(),
        })
        time.sleep(args.poll_seconds)

    gate = read_json(gate_path)
    validate_gate(gate)
    baseline_report = read_json(args.baseline_report)
    baseline_report_sha = sha256_file(args.baseline_report)
    expected_sequences = sorted(baseline_report['candidate']['sequences'])
    checkpoint_sha = checkpoint_from_gate(gate)
    candidate_view = create_analysis_view(
        root, gate, baseline_report_sha, checkpoint_sha)
    atomic_json(status_path, {
        'schema': SCHEMA, 'stage': 'sequence_comparison',
        'completed_anchors': EXPECTED_ANCHORS, 'updated_at': utc_now()})
    comparison, analyzer_path = sequence_comparison(
        args, baseline_report, gate, candidate_view)
    atomic_json(status_path, {
        'schema': SCHEMA, 'stage': 'transaction_trace_analysis',
        'completed_anchors': EXPECTED_ANCHORS, 'updated_at': utc_now()})
    trace = summarize_traces(gate, expected_sequences)

    baseline_failures = {
        name: int(row['failed_anchors'])
        for name, row in baseline_report['candidate']['sequences'].items()}
    candidate_failures = {
        name: int(row['confirmed_failures'])
        for name, row in gate['per_sequence_failures'].items()}
    if sorted(baseline_failures) != expected_sequences or sorted(
            candidate_failures) != expected_sequences:
        raise ValueError('failure sequence coverage differs')
    if sum(baseline_failures.values()) != gate['baseline']['confirmed_failures']:
        raise ValueError('baseline per-sequence failures do not sum')
    if sum(candidate_failures.values()) != gate['candidate']['confirmed_failures']:
        raise ValueError('candidate per-sequence failures do not sum')

    payload = {
        'schema': SCHEMA,
        'status': 'complete',
        'generated_at': utc_now(),
        'gate': {
            'gate_passed': gate['gate_passed'],
            'full127_authorized': gate['full127_authorized'],
            'automatic_full127_launch': gate['automatic_full127_launch'],
            'gate_checks': gate['gate_checks'],
            'baseline': gate['baseline'],
            'candidate': gate['candidate'],
            'failure_delta': gate['failure_delta'],
        },
        'comparison': comparison,
        'failures': {
            'reference_by_sequence': baseline_failures,
            'candidate_by_sequence': candidate_failures,
        },
        'transaction_trace_summary': trace,
        'artifacts': {
            'gate_result': {
                'path': str(gate_path), 'sha256': sha256_file(gate_path)},
            'baseline_report': {
                'path': str(args.baseline_report.resolve()),
                'sha256': baseline_report_sha},
            'candidate_analysis_view': str(candidate_view),
        },
        'provenance': {
            'sequence_analyzer': {
                'path': str(analyzer_path.resolve()),
                'sha256': sha256_file(analyzer_path)},
            'script': {
                'path': str(Path(__file__).resolve()),
                'sha256': sha256_file(Path(__file__).resolve())},
        },
    }
    output_json = root / 'low22_transaction_diagnostics.json'
    output_md = root / 'low22_transaction_diagnostics.md'
    atomic_json(output_json, payload)
    atomic_text(output_md, render_markdown(payload))
    atomic_json(status_path, {
        'schema': SCHEMA,
        'stage': 'complete',
        'completed_anchors': EXPECTED_ANCHORS,
        'output_json': str(output_json),
        'output_json_sha256': sha256_file(output_json),
        'output_markdown': str(output_md),
        'output_markdown_sha256': sha256_file(output_md),
        'updated_at': utc_now(),
    })
    print(json.dumps({
        'status': 'complete',
        'gate_passed': gate['gate_passed'],
        'metric_delta_percent': comparison[
            'candidate_minus_reference_percent'],
        'failure_delta': gate['failure_delta'],
        'events': trace['totals'],
        'output_json_sha256': sha256_file(output_json),
    }, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=120.0)
    parser.add_argument(
        '--baseline-report', type=Path,
        default=Path(
            '/root/autodl-tmp/sutrack_vot_low22_anchor_identity_v1/'
            'LOW22_REPORT.json'))
    parser.add_argument(
        '--baseline-workspace', type=Path,
        default=Path(
            '/root/autodl-tmp/sutrack_vot_low22_anchor_identity_v1/'
            'analysis_workspace_view'))
    parser.add_argument(
        '--sequence-tools-root', type=Path,
        default=Path('/home/SRTrack_RGBD_L'))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError('poll seconds must be positive')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_stream = open(
        root / 'transaction_diagnostics.lock', 'a+', encoding='utf-8')
    try:
        fcntl.flock(
            lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(
            'another transaction diagnostics process owns the lock') from error
    try:
        run(args)
    except Exception as error:
        atomic_json(root / 'transaction_diagnostics_status.json', {
            'schema': SCHEMA,
            'stage': 'failed',
            'error_type': type(error).__name__,
            'error': str(error),
            'updated_at': utc_now(),
        })
        raise


if __name__ == '__main__':
    main()
