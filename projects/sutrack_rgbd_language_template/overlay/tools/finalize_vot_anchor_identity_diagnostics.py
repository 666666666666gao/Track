#!/usr/bin/env python3
"""Produce full-127 sequence and failure diagnostics after VOT finalization."""

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


SCHEMA = 'sutrack_vot_anchor_identity_full_diagnostics_v1'
EXPECTED_TRACKER = 'sutrack_l384_rgbd_anchor_identity_all127'
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
        path, json.dumps(
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


def validate_terminal(candidate, baseline):
    expected_candidate = {
        'status': 'complete',
        'tracker': EXPECTED_TRACKER,
        'toolkit': '0.7.1',
        'sequence_count': 127,
        'anchor_count': 1765,
    }
    for key, expected in expected_candidate.items():
        if candidate.get(key) != expected:
            raise ValueError(
                'candidate {} mismatch: {} != {}'.format(
                    key, candidate.get(key), expected))
    if (baseline.get('status') != 'complete' or
            baseline.get('toolkit') != '0.7.1' or
            baseline.get('sequence_count') != 127 or
            baseline.get('anchor_count') != 1765):
        raise ValueError('baseline terminal result contract mismatch')
    baseline_metrics = baseline['metrics_fraction']
    frozen = candidate.get('comparison_reference_percent', {})
    for metric in ('eao', 'acc', 'rob'):
        expected = 100.0 * float(baseline_metrics[metric])
        if not math.isclose(
                float(frozen.get(metric, math.nan)), expected,
                rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                'candidate comparison does not bind baseline {}'.format(
                    metric))


def validate_source_bindings(root, candidate, baseline_result):
    snapshot_path = root / 'finalizer_source_snapshot.json'
    expected_snapshot = candidate.get('source_snapshot', {})
    if (Path(expected_snapshot.get('path', '')).resolve() !=
            snapshot_path.resolve()):
        raise ValueError('candidate source snapshot path mismatch')
    if sha256_file(snapshot_path) != expected_snapshot.get('sha256'):
        raise ValueError('candidate source snapshot SHA mismatch')
    snapshot = read_json(snapshot_path)
    baseline_result = baseline_result.resolve()
    matching = [
        record for record in snapshot.get('sources', {}).values()
        if Path(record.get('path', '')).resolve() == baseline_result]
    if len(matching) != 1:
        raise ValueError('baseline terminal artifact is not uniquely frozen')
    if matching[0].get('sha256') != sha256_file(baseline_result):
        raise ValueError('frozen baseline terminal SHA mismatch')
    return snapshot


def create_analysis_view(root, full_result, checkpoint_sha):
    master = root / 'master'
    for name in ('config.yaml', 'trackers.ini', 'sequences', 'results'):
        if not (master / name).exists():
            raise FileNotFoundError(master / name)
    view = root.parent / 'analysis_workspace_view'
    manifest = {
        'schema': 'sutrack_vot_anchor_identity_analysis_adapter_v1',
        'protocol': 'official_vot2022_rgbd_multistart',
        'tracker_id': EXPECTED_TRACKER,
        'checkpoint': {'sha256': checkpoint_sha},
        'source_master_workspace': str(master.resolve()),
        'source_shard_manifest_sha256': full_result['shard_manifest_sha256'],
        'source_merge_result_sha256': full_result['merge_result']['sha256'],
        'sequence_count': 127,
        'anchor_count': 1765,
        'purpose': 'read_only_full127_sequence_and_failure_diagnostics',
    }
    if view.exists():
        existing = read_json(view / 'manifest.json')
        if existing != manifest:
            raise ValueError('existing analysis view manifest differs')
        for name in ('config.yaml', 'trackers.ini', 'sequences', 'results'):
            if not (view / name).is_symlink():
                raise ValueError('analysis view binding is not a symlink')
            if (view / name).resolve() != (master / name).resolve():
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


def aggregate_payload(module, reference, candidate, sequence_names):
    value = module.analyze_pair(
        reference, candidate,
        reference_tracker_id='sutrack_l384_rgbd_language_safe_template',
        candidate_tracker_id=EXPECTED_TRACKER,
        top_n=127, sequence_names=sequence_names)
    return value


def assert_metric_match(label, measured, expected):
    mapping = {'accuracy': 'acc', 'robustness': 'rob', 'eao': 'eao'}
    for metric, terminal_key in mapping.items():
        if not math.isclose(
                float(measured[metric]), float(expected[terminal_key]),
                rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                '{} {} mismatch: {} != {}'.format(
                    label, metric, measured[metric], expected[terminal_key]))


def failed_counts(payload):
    rows = payload['runs']
    counts = Counter(
        row['sequence'] for row in rows if bool(row['failed']))
    return {
        'total': sum(counts.values()),
        'by_sequence': dict(sorted(counts.items())),
        'runs': len(rows),
    }


def compact_comparison(value):
    comparison = value['comparison']
    return {
        'reference': comparison['reference'],
        'candidate': comparison['candidate'],
        'candidate_minus_reference_percent': (
            comparison['candidate_minus_reference_percent']),
        'sequence_contribution_sums_percent': (
            comparison['sequence_contribution_sums_percent']),
        'per_sequence': comparison['per_sequence'],
    }


def render_markdown(result):
    full = result['subsets']['full127']
    low = result['subsets']['low22']
    nonlow = result['subsets']['nonlow105']
    failures = result['failures']

    def table_row(name, row):
        reference = row['reference']
        candidate = row['candidate']
        delta = row['candidate_minus_reference_percent']
        return (
            '| {} | {:.6f}→{:.6f} ({:+.6f}) | '
            '{:.6f}→{:.6f} ({:+.6f}) | '
            '{:.6f}→{:.6f} ({:+.6f}) |'.format(
                name,
                100 * reference['eao'], 100 * candidate['eao'], delta['eao'],
                100 * reference['accuracy'],
                100 * candidate['accuracy'], delta['accuracy'],
                100 * reference['robustness'],
                100 * candidate['robustness'], delta['robustness']))

    rows = full['per_sequence']
    best = sorted(
        rows,
        key=lambda row: (
            -row['eao_global_contribution_delta_percent'], row['sequence']))[:15]
    worst = sorted(
        rows,
        key=lambda row: (
            row['eao_global_contribution_delta_percent'], row['sequence']))[:15]
    positive = sum(
        row['eao_global_contribution_delta_percent'] > 0 for row in rows)
    negative = sum(
        row['eao_global_contribution_delta_percent'] < 0 for row in rows)
    unchanged = len(rows) - positive - negative
    lines = [
        '# SUTrack VOT-RGBD2022 anchor身份文本 full-127诊断', '',
        '## 聚合比较', '',
        '| 集合 | EAO（旧→新，Δpp） | ACC（旧→新，Δpp） | ROB（旧→新，Δpp） |',
        '|---|---:|---:|---:|',
        table_row('全127', full),
        table_row('冻结低22', low),
        table_row('其余105', nonlow), '',
        '- EAO全局贡献：{}条改善、{}条退化、{}条不变。'.format(
            positive, negative, unchanged),
        '- 确认失败anchors：{}→{}（{:+d}）。'.format(
            failures['reference']['total'], failures['candidate']['total'],
            failures['candidate']['total'] - failures['reference']['total']),
        '', '## EAO贡献改善最大的15条', '',
        '| 序列 | EAO全局贡献Δpp | ACCΔpp | ROBΔpp | 失败anchor旧→新 |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in best:
        name = row['sequence']
        lines.append('| {} | {:+.6f} | {:+.6f} | {:+.6f} | {}→{} |'.format(
            name, row['eao_global_contribution_delta_percent'],
            row['accuracy_delta_percent'], row['robustness_delta_percent'],
            failures['reference']['by_sequence'].get(name, 0),
            failures['candidate']['by_sequence'].get(name, 0)))
    lines.extend(['', '## EAO贡献退化最大的15条', '',
                  '| 序列 | EAO全局贡献Δpp | ACCΔpp | ROBΔpp | 失败anchor旧→新 |',
                  '|---|---:|---:|---:|---:|'])
    for row in worst:
        name = row['sequence']
        lines.append('| {} | {:+.6f} | {:+.6f} | {:+.6f} | {}→{} |'.format(
            name, row['eao_global_contribution_delta_percent'],
            row['accuracy_delta_percent'], row['robustness_delta_percent'],
            failures['reference']['by_sequence'].get(name, 0),
            failures['candidate']['by_sequence'].get(name, 0)))
    lines.extend([
        '', '## 解释约束', '',
        '- EAO贡献是对全局EAO差值的精确可加分解，不是短序列singleton EAO。',
        '- 失败原因由离线GT诊断获得，只用于分析，未反馈到推理或文本生成。',
        '- 是否保留新注释必须同时看全127总分、低22/非低105分解和失败anchor变化。',
    ])
    return '\n'.join(lines) + '\n'


def run(args):
    root = args.root.resolve()
    result_path = root / 'full_result.json'
    status_path = root / 'diagnostics_status.json'
    while not result_path.is_file():
        progress = None
        finalizer_status = root / 'finalizer_status.json'
        if finalizer_status.is_file():
            progress = read_json(finalizer_status).get('completed_anchors')
        atomic_json(status_path, {
            'schema': SCHEMA,
            'stage': 'waiting_for_full_result',
            'completed_anchors': progress,
            'updated_at': utc_now(),
        })
        time.sleep(args.poll_seconds)

    candidate_terminal = read_json(result_path)
    baseline_terminal = read_json(args.baseline_result)
    validate_terminal(candidate_terminal, baseline_terminal)
    source_snapshot = validate_source_bindings(
        root, candidate_terminal, args.baseline_result)
    checkpoint = source_snapshot['sources']['checkpoint']['sha256']
    if checkpoint != EXPECTED_CHECKPOINT:
        raise ValueError('candidate checkpoint changed')
    view = create_analysis_view(root, candidate_terminal, checkpoint)

    repo_root = Path(__file__).resolve().parents[1]
    # Keep SUTrack's ``lib`` ahead of the legacy tools root.  The latter is
    # used only for generic offline VOT analyzers and must never shadow the
    # candidate tracker's modules if the toolkit resolves the registry.
    sys.path.insert(0, str(args.sequence_tools_root.resolve()))
    sys.path.insert(0, str(repo_root))
    gap_module = load_module(
        args.sequence_tools_root / 'tools/analyze_votrgbd2022_sequence_gaps.py',
        'vot_sequence_gap_analysis')
    failure_module = load_module(
        args.failure_analyzer.resolve(), 'vot_failure_precursors')
    low_report = read_json(args.low22_report)
    low_names = sorted(low_report['baseline']['sequences'])
    all_names = sorted(candidate_terminal.get('sequence_names', []))
    if not all_names:
        all_names = sorted(read_json(
            root / 'shard_manifest.json')['sequences'])
    nonlow_names = sorted(set(all_names) - set(low_names))
    if len(low_names) != 22 or len(nonlow_names) != 105:
        raise ValueError('low/non-low partition mismatch')

    atomic_json(status_path, {
        'schema': SCHEMA, 'stage': 'sequence_comparison',
        'completed_anchors': 1765, 'updated_at': utc_now()})
    full = aggregate_payload(
        gap_module, args.baseline_workspace, view, None)
    low = aggregate_payload(
        gap_module, args.baseline_workspace, view, low_names)
    nonlow = aggregate_payload(
        gap_module, args.baseline_workspace, view, nonlow_names)
    assert_metric_match(
        'reference', full['comparison']['reference'],
        baseline_terminal['metrics_fraction'])
    assert_metric_match(
        'candidate', full['comparison']['candidate'],
        candidate_terminal['metrics_fraction'])

    atomic_json(status_path, {
        'schema': SCHEMA, 'stage': 'failure_analysis',
        'completed_anchors': 1765, 'updated_at': utc_now()})
    reference_failures = failure_module.analyze_workspace(
        args.baseline_workspace,
        tracker_id='sutrack_l384_rgbd_language_safe_template')
    candidate_failures = failure_module.analyze_workspace(
        view, tracker_id=EXPECTED_TRACKER)
    reference_failure_path = root / 'baseline_full127_failure_precursors.json'
    candidate_failure_path = root / 'candidate_full127_failure_precursors.json'
    atomic_json(reference_failure_path, reference_failures)
    atomic_json(candidate_failure_path, candidate_failures)

    payload = {
        'schema': SCHEMA,
        'status': 'complete',
        'generated_at': utc_now(),
        'candidate_full_result': {
            'path': str(result_path), 'sha256': sha256_file(result_path)},
        'baseline_full_result': {
            'path': str(args.baseline_result.resolve()),
            'sha256': sha256_file(args.baseline_result)},
        'candidate_analysis_view': str(view),
        'subsets': {
            'full127': compact_comparison(full),
            'low22': compact_comparison(low),
            'nonlow105': compact_comparison(nonlow),
        },
        'failures': {
            'reference': failed_counts(reference_failures),
            'candidate': failed_counts(candidate_failures),
            'reference_artifact': {
                'path': str(reference_failure_path),
                'sha256': sha256_file(reference_failure_path)},
            'candidate_artifact': {
                'path': str(candidate_failure_path),
                'sha256': sha256_file(candidate_failure_path)},
        },
        'provenance': {
            'sequence_analyzer': {
                'path': str((args.sequence_tools_root / 'tools/analyze_votrgbd2022_sequence_gaps.py').resolve()),
                'sha256': sha256_file(
                    args.sequence_tools_root / 'tools/analyze_votrgbd2022_sequence_gaps.py')},
            'failure_analyzer': {
                'path': str(args.failure_analyzer.resolve()),
                'sha256': sha256_file(args.failure_analyzer)},
            'script': {
                'path': str(Path(__file__).resolve()),
                'sha256': sha256_file(Path(__file__).resolve())},
        },
    }
    output_json = root / 'full127_sequence_and_failure_diagnostics.json'
    output_md = root / 'full127_sequence_and_failure_diagnostics.md'
    atomic_json(output_json, payload)
    atomic_text(output_md, render_markdown(payload))
    atomic_json(status_path, {
        'schema': SCHEMA,
        'stage': 'complete',
        'completed_anchors': 1765,
        'output_json': str(output_json),
        'output_json_sha256': sha256_file(output_json),
        'output_markdown': str(output_md),
        'output_markdown_sha256': sha256_file(output_md),
        'updated_at': utc_now(),
    })
    print(json.dumps({
        'status': 'complete',
        'metrics': payload['subsets']['full127'][
            'candidate_minus_reference_percent'],
        'failure_delta': (
            payload['failures']['candidate']['total'] -
            payload['failures']['reference']['total']),
        'output_json_sha256': sha256_file(output_json),
    }, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=120.0)
    parser.add_argument(
        '--baseline-result', type=Path,
        default=Path('/root/autodl-tmp/sutrack_rgbd_language_safe_template_vot_full127_v1/full_result.json'))
    parser.add_argument(
        '--baseline-workspace', type=Path,
        default=Path('/root/autodl-tmp/sutrack_rgbd_language_safe_template_vot_full127_v1/analysis_workspace_view'))
    parser.add_argument(
        '--low22-report', type=Path,
        default=Path('/root/autodl-tmp/sutrack_vot_low22_anchor_identity_v1/LOW22_REPORT.json'))
    parser.add_argument(
        '--sequence-tools-root', type=Path,
        default=Path('/home/SRTrack_RGBD_L'))
    parser.add_argument(
        '--failure-analyzer', type=Path,
        default=Path('/root/autodl-tmp/sutrack_vot_low22_anchor_identity_v1/tools/analyze_votrgbd2022_failure_precursors.py'))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError('poll seconds must be positive')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_stream = open(root / 'diagnostics.lock', 'a+', encoding='utf-8')
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError('another diagnostics process owns the lock') from error
    try:
        run(args)
    except Exception as error:
        atomic_json(root / 'diagnostics_status.json', {
            'schema': SCHEMA,
            'stage': 'failed',
            'error_type': type(error).__name__,
            'error': str(error),
            'updated_at': utc_now(),
        })
        raise


if __name__ == '__main__':
    main()
