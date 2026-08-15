#!/usr/bin/env python3
"""Wait for a sharded VOT run, validate it, analyze it, and seal results."""

import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


SUFFIXES = ('.bin', '_confidence.value', '_time.value')
SNAPSHOT_SCHEMA = 'sutrack_vot_full_source_snapshot_v1'
RESULT_SCHEMA = 'sutrack_vot_full_terminal_result_v1'
STATUS_SCHEMA = 'sutrack_vot_full_finalizer_status_v1'
DOC_RECEIPT_SCHEMA = 'sutrack_vot_full_doc_update_v1'
DOC_BEGIN = '<!-- SUTRACK_FULL127_RESULT_BEGIN -->'
DOC_END = '<!-- SUTRACK_FULL127_RESULT_END -->'

OLD_METRICS = {
    'eao': 0.729089559737049,
    'acc': 0.8253586769952865,
    'rob': 0.879880710788393,
}
TARGETS = {'eao': 0.779, 'acc': 0.821, 'rob': 0.937}
OFFICIAL_SUTRACK_REPORTED = {'eao': 0.766, 'acc': 0.835, 'rob': 0.922}

IMPLEMENTATION_FILES = (
    'lib/config/sutrack/config.py',
    'lib/models/sutrack/encoder.py',
    'lib/test/evaluation/local.py',
    'lib/test/parameter/sutrack.py',
    'lib/test/tracker/rgbd_frame.py',
    'lib/test/tracker/rgbd_language_manifest.py',
    'lib/test/tracker/safe_template_update.py',
    'lib/test/tracker/temporal_depth_identity.py',
    'lib/test/tracker/sutrack.py',
    'lib/test/vot/sutrack_class.py',
    'lib/test/vot/vot.py',
    'lib/test/vot/sutrack_l384_rgbd_language_safe_template.py',
    'tools/create_vot_failure_family_shards.py',
    'tools/run_vot_failure_family_shards.py',
    'tools/seed_vot_shards_from_master.py',
    'tools/finalize_vot_full127.py',
)

DEFAULT_DOCS = (
    '/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_SAFE_TEMPLATE_PORT_ZH.md',
    '/home/SRTrack_RGBD_L/docs/final_language_model_architecture_and_inference_zh.md',
    '/home/SRTrack_RGBD_L/refine-logs/EXPERIMENT_TRACKER.md',
)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-{}'.format(os.getpid()))
    with open(temporary, 'x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path, payload):
    atomic_text(
        path, json.dumps(payload, indent=2, sort_keys=True) + '\n')


def load_json(path):
    with open(path, 'r', encoding='utf-8') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError('{} must contain a JSON object'.format(path))
    return value


def require_file(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError('Missing non-empty file: {}'.format(path))
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument(
        '--repo-root', type=Path, default=Path('/home/SUTrack_RGBD_L'))
    parser.add_argument(
        '--python', default='/root/miniconda3/envs/mplt/bin/python')
    parser.add_argument('--poll-seconds', type=float, default=60.0)
    parser.add_argument('--analysis-name', default='full127_analysis')
    parser.add_argument('--expected-anchor-count', type=int, default=1765)
    parser.add_argument('--expected-sequence-count', type=int, default=127)
    parser.add_argument(
        '--expected-tracker',
        default='sutrack_l384_rgbd_language_safe_template')
    parser.add_argument('--expected-toolkit', default='0.7.1')
    parser.add_argument('--expected-manifest-sha256', default='')
    parser.add_argument(
        '--checkpoint', type=Path,
        default=Path('/root/autodl-tmp/sutrack_assets/weights/SUTRACK_ep0180_l384.pth.tar'))
    parser.add_argument(
        '--clip-checkpoint', type=Path,
        default=Path('/root/autodl-tmp/sutrack_assets/weights/ViT-L-14.pt'))
    parser.add_argument(
        '--language-manifest', type=Path,
        default=Path('/home/OSTrack_RGBD_L_dataset_modified/annotations_cleaned/votrgbd2022_language.jsonl'))
    parser.add_argument(
        '--configuration', type=Path,
        default=Path('/home/SUTrack_RGBD_L/experiments/sutrack/sutrack_l384_rgbd_language_safe_template.yaml'))
    parser.add_argument('--update-docs', action='store_true')
    parser.add_argument('--doc', action='append', type=Path, default=[])
    parser.add_argument('--validate-only', action='store_true')
    return parser.parse_args()


def load_manifest(args):
    root = args.root.resolve()
    manifest_path = require_file(root / 'shard_manifest.json')
    manifest_sha = sha256_file(manifest_path)
    if (args.expected_manifest_sha256 and
            manifest_sha != args.expected_manifest_sha256):
        raise ValueError(
            'Manifest SHA mismatch: {} != {}'.format(
                manifest_sha, args.expected_manifest_sha256))
    manifest = load_json(manifest_path)
    if manifest.get('tracker') != args.expected_tracker:
        raise ValueError('Unexpected tracker in shard manifest')
    if manifest.get('total_anchor_count') != args.expected_anchor_count:
        raise ValueError('Unexpected anchor count in shard manifest')
    sequences = manifest.get('sequences')
    if (not isinstance(sequences, list) or
            len(sequences) != args.expected_sequence_count or
            len(sequences) != len(set(sequences))):
        raise ValueError('Invalid sequence coverage in shard manifest')
    shards = manifest.get('shards')
    if not isinstance(shards, list) or not shards:
        raise ValueError('Shard manifest has no shards')
    trajectories = []
    for shard in shards:
        shard_root = Path(shard['root']).resolve()
        if root not in shard_root.parents:
            raise ValueError('Shard root escapes run root: {}'.format(shard_root))
        expected = shard.get('expected_trajectories')
        if (not isinstance(expected, list) or
                len(expected) != shard.get('anchor_count')):
            raise ValueError('Invalid shard trajectory list')
        trajectories.extend(expected)
        for name, key in (
                ('config.yaml', 'config_sha256'),
                ('trackers.ini', 'trackers_sha256'),
                ('sequences/list.txt', 'list_sha256')):
            path = require_file(shard_root / name)
            if sha256_file(path) != shard[key]:
                raise ValueError('Shard source changed: {}'.format(path))
    if (len(trajectories) != args.expected_anchor_count or
            len(trajectories) != len(set(trajectories))):
        raise ValueError('Shard trajectories are not an exact unique cover')
    return root, manifest_path, manifest_sha, manifest, trajectories


def completed_count(manifest):
    tracker = manifest['tracker']
    complete = 0
    for shard in manifest['shards']:
        shard_root = Path(shard['root'])
        for trajectory in shard['expected_trajectories']:
            sequence = trajectory.rsplit('_', 1)[0]
            result_root = (
                shard_root / 'results' / tracker / 'baseline' / sequence)
            paths = [
                result_root / (trajectory + suffix) for suffix in SUFFIXES]
            if all(path.is_file() and path.stat().st_size > 0
                   for path in paths):
                complete += 1
    return complete


def source_records(args):
    repo_root = args.repo_root.resolve()
    records = {}
    for relative in IMPLEMENTATION_FILES:
        path = require_file(repo_root / relative)
        records['implementation/' + relative] = {
            'path': str(path.resolve()),
            'size': path.stat().st_size,
            'sha256': sha256_file(path),
        }
    external = {
        'checkpoint': args.checkpoint,
        'clip_checkpoint': args.clip_checkpoint,
        'language_manifest': args.language_manifest,
        'configuration': args.configuration,
    }
    for name, raw_path in external.items():
        path = require_file(raw_path)
        records[name] = {
            'path': str(path.resolve()),
            'size': path.stat().st_size,
            'sha256': sha256_file(path),
        }
    return records


def ensure_source_snapshot(args, root, manifest_sha):
    snapshot_path = root / 'finalizer_source_snapshot.json'
    current = source_records(args)
    if snapshot_path.exists():
        snapshot = load_json(snapshot_path)
        if (snapshot.get('schema') != SNAPSHOT_SCHEMA or
                snapshot.get('manifest_sha256') != manifest_sha or
                snapshot.get('sources') != current):
            raise ValueError('Existing source snapshot does not match runtime')
    else:
        snapshot = {
            'schema': SNAPSHOT_SCHEMA,
            'created_at': utc_now(),
            'manifest_sha256': manifest_sha,
            'repo_root': str(args.repo_root.resolve()),
            'sources': current,
        }
        atomic_json(snapshot_path, snapshot)
    return snapshot_path, snapshot


def validate_source_snapshot(snapshot_path):
    snapshot = load_json(snapshot_path)
    for name, record in snapshot['sources'].items():
        path = require_file(record['path'])
        if (path.stat().st_size != record['size'] or
                sha256_file(path) != record['sha256']):
            raise ValueError('Frozen source changed: {}'.format(name))
    return snapshot


def validate_merge(root, manifest_path, manifest_sha, manifest, trajectories):
    merge_path = require_file(root / 'merge_result.json')
    merge = load_json(merge_path)
    tracker = manifest['tracker']
    master = (root / 'master').resolve()
    if (merge.get('schema') != 'sutrack_vot_failure_family_anchor_merge_v1' or
            merge.get('status') != 'complete' or
            merge.get('tracker') != tracker or
            Path(merge.get('master_workspace', '')).resolve() != master or
            Path(merge.get('source_manifest', '')).resolve() != manifest_path.resolve() or
            merge.get('source_manifest_sha256') != manifest_sha or
            merge.get('anchor_count') != len(trajectories) or
            merge.get('result_file_count') != len(trajectories) * len(SUFFIXES)):
        raise ValueError('Merge receipt contract mismatch')
    if sha256_file(manifest_path) != manifest_sha:
        raise ValueError('Manifest changed after merge')
    result_sha = merge.get('result_sha256')
    if not isinstance(result_sha, dict):
        raise ValueError('Merge receipt lacks result SHA map')
    expected_paths = {}
    for trajectory in trajectories:
        sequence = trajectory.rsplit('_', 1)[0]
        for suffix in SUFFIXES:
            relative = str(
                Path('results') / tracker / 'baseline' / sequence /
                (trajectory + suffix))
            expected_paths[relative] = master / relative
    if set(result_sha) != set(expected_paths):
        raise ValueError('Merged result path coverage mismatch')
    for relative, path in expected_paths.items():
        path = require_file(path)
        if sha256_file(path) != result_sha[relative]:
            raise ValueError('Merged result SHA mismatch: {}'.format(relative))
    expected_config = (
        'registry:\n- ./trackers.ini\nsequences: ./sequences\n'
        'stack: vot2022/rgbd\n')
    if (master / 'config.yaml').read_text(encoding='utf-8') != expected_config:
        raise ValueError('Master workspace config mismatch')
    shard_zero = Path(manifest['shards'][0]['root'])
    if sha256_file(master / 'trackers.ini') != sha256_file(
            shard_zero / 'trackers.ini'):
        raise ValueError('Master tracker registry mismatch')
    listed = (master / 'sequences/list.txt').read_text(
        encoding='utf-8').splitlines()
    if listed != manifest['sequences']:
        raise ValueError('Master sequence order mismatch')
    source_root = Path(manifest['source_sequences_root']).resolve()
    for sequence in manifest['sequences']:
        if (master / 'sequences' / sequence).resolve() != (source_root / sequence):
            raise ValueError('Master sequence binding mismatch: {}'.format(sequence))
    return merge_path, merge, master


def run_analysis(args, root, master, tracker):
    analysis_path = master / 'analysis' / (args.analysis_name + '.json')
    log_path = root / 'finalizer_analysis.log'
    if not analysis_path.exists():
        command = [
            args.python, '-m', 'vot', 'analysis',
            '--workspace', str(master), '--format', 'json',
            '--name', args.analysis_name, tracker,
        ]
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(args.repo_root.resolve())
        with open(log_path, 'ab', buffering=0) as log:
            process = subprocess.run(
                command, cwd=str(args.repo_root.resolve()), env=environment,
                stdout=log, stderr=subprocess.STDOUT, check=False)
        if process.returncode != 0:
            raise RuntimeError(
                'VOT analysis failed with code {}'.format(process.returncode))
    return require_file(analysis_path)


def parse_analysis(args, analysis_path, manifest):
    analysis = load_json(analysis_path)
    if analysis.get('toolkit') != args.expected_toolkit:
        raise ValueError('Unexpected VOT toolkit version')
    if set(analysis.get('sequences', {})) != set(manifest['sequences']):
        raise ValueError('Analysis sequence coverage mismatch')
    if args.expected_tracker not in analysis.get('trackers', {}):
        raise ValueError('Analysis tracker mismatch')
    try:
        results = analysis['results']['baseline']['results']
        eao = float(results[0][0][0])
        acc = float(results[2][0][0])
        rob = float(results[2][0][1])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError('Unexpected VOT analysis result shape') from error
    metrics = {'eao': eao, 'acc': acc, 'rob': rob}
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0
               for value in metrics.values()):
        raise ValueError('VOT metrics are invalid')
    return analysis, metrics


def build_result(args, root, manifest_sha, manifest, merge_path,
                 analysis_path, metrics, snapshot_path):
    preseed_path = root / 'preseed_receipt.json'
    preseed = None
    if preseed_path.exists():
        preseed = {
            'path': str(preseed_path),
            'sha256': sha256_file(preseed_path),
        }
    checks = {
        name: metrics[name] >= TARGETS[name] for name in metrics}
    result = {
        'schema': RESULT_SCHEMA,
        'status': 'complete',
        'generated_at': utc_now(),
        'tracker': manifest['tracker'],
        'toolkit': args.expected_toolkit,
        'sequence_count': len(manifest['sequences']),
        'anchor_count': manifest['total_anchor_count'],
        'metrics_fraction': metrics,
        'metrics_percent': {
            name: value * 100.0 for name, value in metrics.items()},
        'historical_formal_percent': {
            name: value * 100.0 for name, value in OLD_METRICS.items()},
        'comparison_reference': {
            'name': 'SRTrack historical formal full-127 reference',
            'is_sutrack_baseline': False,
            'server_measured': True,
        },
        'official_sutrack_reported_percent': {
            name: value * 100.0
            for name, value in OFFICIAL_SUTRACK_REPORTED.items()},
        'official_sutrack_baseline_rerun': False,
        'delta_vs_historical_pp': {
            name: (metrics[name] - OLD_METRICS[name]) * 100.0
            for name in metrics},
        'targets_percent': {
            name: value * 100.0 for name, value in TARGETS.items()},
        'target_checks': checks,
        'all_targets_met': all(checks.values()),
        'all_historical_metrics_improved': all(
            metrics[name] >= OLD_METRICS[name] for name in metrics),
        'shard_manifest_sha256': manifest_sha,
        'merge_result': {
            'path': str(merge_path), 'sha256': sha256_file(merge_path)},
        'analysis': {
            'path': str(analysis_path), 'sha256': sha256_file(analysis_path)},
        'source_snapshot': {
            'path': str(snapshot_path), 'sha256': sha256_file(snapshot_path)},
        'preseed_receipt': preseed,
    }
    return result


def write_or_validate_result(path, result):
    if path.exists():
        existing = load_json(path)
        stable_keys = set(result) - {'generated_at'}
        if any(existing.get(key) != result.get(key) for key in stable_keys):
            raise ValueError('Existing terminal result does not match analysis')
        return existing
    atomic_json(path, result)
    return result


def percent(value):
    return '{:.6f}'.format(value * 100.0)


def signed_pp(value):
    return '{:+.6f}'.format(value * 100.0)


def doc_heading(path):
    name = Path(path).name
    if name == 'final_language_model_architecture_and_inference_zh.md':
        return '### 24.94 SUTrack full-127 自动终态'
    if name == 'EXPERIMENT_TRACKER.md':
        return '## SUTrack full-127 terminal result'
    return '## 9. full-127 正式结果'


def render_doc_block(path, result):
    metrics = result['metrics_fraction']
    deltas = {
        name: metrics[name] - OLD_METRICS[name] for name in metrics}
    target_text = '、'.join(
        '{}={}'.format(name.upper(), '通过' if passed else '未通过')
        for name, passed in result['target_checks'].items())
    outcome = (
        '三项目标全部达到。' if result['all_targets_met']
        else '至少一项目标未达到，不能写成目标已完成。')
    return '\n'.join((
        doc_heading(path),
        '',
        '无人值守 finalizer 已在 VOT toolkit `{}` 下验证 127 序列、1,765 anchors 和全部结果 SHA，'.format(
            result['toolkit']),
        '随后生成正式 full-127 汇总。{} 检查：{}。'.format(outcome, target_text),
        '',
        '| 结果 | EAO | ACC | ROB |',
        '|---|---:|---:|---:|',
        '| SRTrack 历史正式参考（非 SUTrack baseline） | {} | {} | {} |'.format(
            percent(OLD_METRICS['eao']), percent(OLD_METRICS['acc']),
            percent(OLD_METRICS['rob'])),
        '| SUTrack 官方论文报告（未在本服务器复测） | {} | {} | {} |'.format(
            percent(OFFICIAL_SUTRACK_REPORTED['eao']),
            percent(OFFICIAL_SUTRACK_REPORTED['acc']),
            percent(OFFICIAL_SUTRACK_REPORTED['rob'])),
        '| SUTrack-L384 + 结构化语言 + safe-v1（本服务器实测） | **{}** | **{}** | **{}** |'.format(
            percent(metrics['eao']), percent(metrics['acc']),
            percent(metrics['rob'])),
        '| 相对 SRTrack 历史正式参考变化（pp） | {} | {} | {} |'.format(
            signed_pp(deltas['eao']), signed_pp(deltas['acc']),
            signed_pp(deltas['rob'])),
        '| 目标 | 77.900000 | 82.100000 | 93.700000 |',
        '',
        '权威结果：`{}`；analysis SHA256 `{}`；merge SHA256 `{}`。'.format(
            str(Path(result['analysis']['path']).parents[2] / 'full_result.json'),
            result['analysis']['sha256'], result['merge_result']['sha256']),
        'SUTrack 官方 baseline 按要求未重跑，因此这里不声称“创新相对 SUTrack baseline 的 full-127 增益”；',
        '只有 SUTrack+创新的绝对实测指标，以及相对 SRTrack 历史正式参考的变化。',
        '该 full-127 只更新 VOT 证据；DepthTrack/CDTB 未在 SUTrack 移植上重测，原有已达标正式数字保持不变。',
    ))


def replace_doc_block(content, body):
    if content.count(DOC_BEGIN) != 1 or content.count(DOC_END) != 1:
        raise ValueError('Document must contain exactly one final-result marker pair')
    begin = content.index(DOC_BEGIN)
    end = content.index(DOC_END, begin)
    if end <= begin:
        raise ValueError('Document result markers are out of order')
    end += len(DOC_END)
    replacement = DOC_BEGIN + '\n' + body.rstrip() + '\n' + DOC_END
    return content[:begin] + replacement + content[end:]


def update_docs(root, docs, result, result_path):
    receipt_path = root / 'doc_update_receipt.json'
    if receipt_path.exists():
        receipt = load_json(receipt_path)
        if (receipt.get('schema') != DOC_RECEIPT_SCHEMA or
                receipt.get('full_result_sha256') != sha256_file(result_path)):
            raise ValueError('Existing doc update receipt is invalid')
        for record in receipt['documents']:
            if sha256_file(record['path']) != record['after_sha256']:
                raise ValueError('Document changed after terminal update')
        return receipt
    prepared = []
    for path in docs:
        path = require_file(path)
        content = path.read_text(encoding='utf-8')
        updated = replace_doc_block(content, render_doc_block(path, result))
        prepared.append({
            'path': path,
            'before_sha256': sha256_file(path),
            'updated': updated,
        })
    records = []
    for item in prepared:
        atomic_text(item['path'], item['updated'])
        records.append({
            'path': str(item['path']),
            'before_sha256': item['before_sha256'],
            'after_sha256': sha256_file(item['path']),
        })
    receipt = {
        'schema': DOC_RECEIPT_SCHEMA,
        'updated_at': utc_now(),
        'full_result_sha256': sha256_file(result_path),
        'documents': records,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def write_status(path, stage, complete, total, extra=None):
    payload = {
        'schema': STATUS_SCHEMA,
        'updated_at': utc_now(),
        'stage': stage,
        'completed_anchors': complete,
        'total_anchors': total,
    }
    if extra:
        payload.update(extra)
    atomic_json(path, payload)


def main():
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError('--poll-seconds must be positive')
    root, manifest_path, manifest_sha, manifest, trajectories = load_manifest(args)

    if args.validate_only:
        merge_path, _, master = validate_merge(
            root, manifest_path, manifest_sha, manifest, trajectories)
        analysis_path = require_file(
            master / 'analysis' / (args.analysis_name + '.json'))
        _, metrics = parse_analysis(args, analysis_path, manifest)
        print(json.dumps({
            'status': 'validated',
            'manifest_sha256': manifest_sha,
            'merge_sha256': sha256_file(merge_path),
            'analysis_sha256': sha256_file(analysis_path),
            'metrics_percent': {
                name: value * 100.0 for name, value in metrics.items()},
        }, indent=2, sort_keys=True))
        return 0

    lock_path = root / 'finalizer.lock'
    lock_stream = open(lock_path, 'a+', encoding='utf-8')
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError('Another finalizer owns the run lock') from error

    snapshot_path, _ = ensure_source_snapshot(args, root, manifest_sha)
    status_path = root / 'finalizer_status.json'
    last_complete = None
    while not (root / 'merge_result.json').exists():
        complete = completed_count(manifest)
        if complete != last_complete:
            write_status(
                status_path, 'waiting_for_merge', complete,
                manifest['total_anchor_count'],
                {'manifest_sha256': manifest_sha})
            print('WAIT {}/{}'.format(
                complete, manifest['total_anchor_count']), flush=True)
            last_complete = complete
        time.sleep(args.poll_seconds)

    write_status(
        status_path, 'validating_merge', manifest['total_anchor_count'],
        manifest['total_anchor_count'])
    validate_source_snapshot(snapshot_path)
    merge_path, _, master = validate_merge(
        root, manifest_path, manifest_sha, manifest, trajectories)
    write_status(
        status_path, 'running_analysis', manifest['total_anchor_count'],
        manifest['total_anchor_count'])
    analysis_path = run_analysis(args, root, master, manifest['tracker'])
    _, metrics = parse_analysis(args, analysis_path, manifest)
    validate_source_snapshot(snapshot_path)
    result = build_result(
        args, root, manifest_sha, manifest, merge_path,
        analysis_path, metrics, snapshot_path)
    result_path = root / 'full_result.json'
    result = write_or_validate_result(result_path, result)
    doc_receipt = None
    if args.update_docs:
        docs = args.doc if args.doc else [Path(path) for path in DEFAULT_DOCS]
        doc_receipt = update_docs(root, docs, result, result_path)
    write_status(
        status_path, 'complete', manifest['total_anchor_count'],
        manifest['total_anchor_count'], {
            'full_result': str(result_path),
            'full_result_sha256': sha256_file(result_path),
            'all_targets_met': result['all_targets_met'],
            'doc_update_receipt': (
                str(root / 'doc_update_receipt.json')
                if doc_receipt is not None else None),
        })
    print(json.dumps({
        'status': 'complete',
        'metrics_percent': result['metrics_percent'],
        'all_targets_met': result['all_targets_met'],
        'full_result_sha256': sha256_file(result_path),
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        try:
            parsed = parse_args()
            root = parsed.root.resolve()
            if root.is_dir():
                write_status(
                    root / 'finalizer_status.json', 'failed', 0,
                    parsed.expected_anchor_count,
                    {'error_type': type(error).__name__, 'error': str(error)})
        except Exception:
            pass
        raise
