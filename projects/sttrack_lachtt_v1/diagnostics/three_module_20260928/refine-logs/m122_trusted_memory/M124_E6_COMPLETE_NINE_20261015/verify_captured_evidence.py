"""Read-only standard-library audit of supplied M124-E6 captured evidence."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import ast
import csv
import hashlib
import io
import json
import math

ROOT = Path(r'C:\Users\gb\.codex_track_publish_m29_20260902')
ART = ROOT / '.aris/m122_full_causal_20261007'
OUT = ART / 'M124_E6_FULL_SUITE_AUDIT_20261015'
REMOTE = '/root/autodl-tmp/m124_ABC_joint_epoch20_20261011'
EVAL = REMOTE + '/epochs/epoch-06/evaluation'
BUNDLE_PATH = ART / 'M124_E6_terminal_audit_artifacts_20261015.json'
COMPACT_PATH = ART / 'M124_best_VOT_seal_completion_compact_observation_20261015.json'
RECEIPT_PATH = ART / 'M124_E6_terminal_audit_artifacts_20261015_receipt.json'
SCRIPT_NAMES = ['run_m124_epochwise_suite.py', 'run_m124_epochwise_official.py',
                'run_m123_joint_vot_shards.py', 'analyze_m124_epochwise_vot.py',
                'collect_m124_epochwise_results.py', 'm124_epochwise_official_runtime.py',
                'full_dense_tracker.py', 'dense_target_decoder.py',
                'm123_joint_full_tracker.py', 'train_m124_joint_epoch.py']
LOCAL_INPUTS = [BUNDLE_PATH, RECEIPT_PATH, COMPACT_PATH, OUT / 'DRAFT_NINE_METRIC_REPORT.md'] + [
    ROOT / 'projects/sttrack_lachtt_v1/diagnostics/three_module_20260928' / n for n in SCRIPT_NAMES]
payload = json.loads(BUNDLE_PATH.read_text(encoding='utf-8'))
files = payload['files']
compact = json.loads(COMPACT_PATH.read_text(encoding='utf-8'))
receipt = json.loads(RECEIPT_PATH.read_text(encoding='utf-8'))
counts = Counter()
failures = []
hashes = {}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(category, condition, evidence):
    counts[category] += 1
    if not condition:
        failures.append(dict(category=category, evidence=evidence))


def parsed(path):
    return json.loads(files[path]['text'])


def agrees_hash(path, expected, category='cross_file_digest'):
    check(category, path in files and files[path]['sha256'] == expected, path)


for path in LOCAL_INPUTS:
    hashes[str(path)] = 'sha256:' + digest(path.read_bytes())
hashes[str(OUT / 'AUDIT_REQUEST.txt')] = 'sha256:' + digest((OUT / 'AUDIT_REQUEST.txt').read_bytes())
for path, entry in files.items():
    data = entry['text'].encode('utf-8')
    actual = digest(data)
    hashes[path] = 'sha256:' + actual
    check('captured_sha256', actual == entry['sha256'], path)
    check('captured_bytes', len(data) == entry['bytes'], path)
check('transport', receipt['raw_sha256'] == digest(BUNDLE_PATH.read_bytes()), str(RECEIPT_PATH))
check('transport', receipt['raw_bytes'] == BUNDLE_PATH.stat().st_size, str(RECEIPT_PATH))
check('transport', receipt['files'] == len(files) == 7288 and receipt['returncode'] == 0, str(RECEIPT_PATH))
compact_count = 0
compact_duplicate_count = 0
for name, entry in compact['files'].items():
    if entry.get('scope') == 'file count only':
        check('compact_file_count', int(entry['text']) == (50 if '/depthtrack_test/' in name else 80), name)
        continue
    compact_count += 1
    path = REMOTE + '/' + name
    data = entry['text'].encode('utf-8')
    check('compact_sha256', digest(data) == entry['sha256'], path)
    check('compact_bytes', len(data) == entry['bytes'], path)
    if path in files:
        compact_duplicate_count += 1
        check('capture_consistency', entry == files[path], path)
    else:
        hashes[path] = 'sha256:' + digest(data)

bundle = parsed(EVAL + '/bundle.json')
execution = parsed(EVAL + '/vot/execution.json')
for container in [bundle, execution]:
    for path, value in container['source_sha256'].items():
        agrees_hash(path, value, 'frozen_source_digest')
for name in SCRIPT_NAMES:
    local = ROOT / 'projects/sttrack_lachtt_v1/diagnostics/three_module_20260928' / name
    agrees_hash(REMOTE + '/code/' + name, digest(local.read_bytes()), 'local_frozen_source_identity')
    ast.parse(local.read_text(encoding='utf-8'))
    counts['required_script_AST'] += 1

best = parsed(REMOTE + '/best.json')
history = parsed(REMOTE + '/epoch_history.json')
selection = parsed(EVAL + '/selection.json')
results = parsed(EVAL + '/all_results.json')
vot = parsed(EVAL + '/vot/result.json')
manifest = parsed(EVAL + '/vot/run/shard_manifest.json')
merge = parsed(EVAL + '/vot/run/merge_result.json')
analysis_path = EVAL + '/vot/run/master/analysis/ABC_joint_epoch06_full127.json'
analysis = parsed(analysis_path)
terminal = parsed(REMOTE + '/result.json')
training_state = parsed(REMOTE + '/current_training_state.json')
final = best['final_sha256']
check('same_final', payload['actual_selected_weight_sha256'] == final, 'captured selected weight digest')
for label, record in [('bundle', bundle), ('selection', selection), ('vot', vot)] + [
        (r['dataset'], r) for r in results['results']]:
    check('same_final', record['final_sha256'] == final, label)
check('same_final', best['epoch'] == 6 and len(selection['models']) == 1, 'best/selection')
agrees_hash(EVAL + '/bundle.json', selection['bundle_sha256'])
agrees_hash(EVAL + '/selection.json', results['selection_sha256'])
agrees_hash(EVAL + '/vot/run/merge_result.json', vot['merge_sha256'])
agrees_hash(EVAL + '/vot/run/shard_manifest.json', merge['source_manifest_sha256'])
agrees_hash(analysis_path, vot['analysis_sha256'])
for dataset, plan_ref in selection['plans'].items():
    agrees_hash(plan_ref['path'], plan_ref['sha256'])
    plan = parsed(plan_ref['path'])
    check('bindings', plan['bank_sha256'] == selection['bank_sha256'] and plan['binding_sha256'] == selection['binding_sha256'], dataset)
    agrees_hash(plan['bundle_path'], plan['bundle_sha256'])
check('bindings', vot['bank_sha256'] == selection['bank_sha256'] and vot['binding_sha256'] == selection['binding_sha256'], 'vot')

ope_metadata = {(x['dataset'], x['sequence']): x for x in payload['ope_GT_file_metadata']}
ope_scope = {}
for dataset, expected_n, expected_frames in [('depthtrack_test', 50, 76373), ('cdtb', 80, 101956)]:
    plan = parsed(EVAL + '/' + dataset + '/plan.json')
    cases = parsed(plan['cases_path'])
    predictions = parsed(EVAL + '/' + dataset + '/predictions/receipt.json')
    metrics_path = EVAL + '/' + dataset + '/predictions/metrics.json'
    measured = parsed(metrics_path)
    values = measured['metrics']
    row = next(r for r in results['results'] if r['dataset'] == dataset)
    agrees_hash(plan['cases_path'], plan['cases_sha256'])
    agrees_hash(plan['metric_source'], plan['metric_source_sha256'])
    agrees_hash(EVAL + '/' + dataset + '/plan.json', measured['plan_sha256'])
    agrees_hash(EVAL + '/' + dataset + '/predictions/receipt.json', measured['receipt_sha256'])
    agrees_hash(metrics_path, row['result_sha256'])
    check('OPE_extent', len(cases) == len({r['sequence'] for r in cases}) == len(predictions['sequences']) == values['sequences'] == expected_n, dataset)
    check('OPE_extent', sum(r['frames'] for r in cases) == predictions['frames'] == values['frames'] == expected_frames, dataset)
    check('OPE_receipt', predictions['final_sha256'] == measured['final_sha256'] == final and predictions['bank_sha256'] == selection['bank_sha256'], dataset)
    check('OPE_receipt', predictions['optimizer_steps'] == 0 and predictions['frozen_before_after_exact'] and not predictions['subsequent_GT_opened'] and not predictions['text_updated_online'], dataset)
    for case, pred in zip(cases, predictions['sequences']):
        check('OPE_sequence_binding', case['sequence'] == pred['sequence'] and case['frames'] == pred['frames'], dataset + '/' + case['sequence'])
        meta = ope_metadata[(dataset, case['sequence'])]
        check('OPE_GT_metadata', meta['sha256'] == case['gt_sha256'] and meta['path'] == plan['dataset_root'] + '/' + case['sequence'] + '/groundtruth.txt', meta['path'])
        for field in ['bbox_sha256', 'confidence_sha256']:
            check('OPE_prediction_digest_format', len(pred[field]) == 64 and all(x in '0123456789abcdef' for x in pred[field]), pred['sequence'] + '/' + field)
        if meta['path'] in files:
            agrees_hash(meta['path'], meta['sha256'], 'OPE_GT_sample')
            truth = [[float(x) for x in line.replace(',', ' ').split()] for line in files[meta['path']]['text'].splitlines()]
            check('OPE_GT_sample', len(truth) == case['frames'] and truth[0] == case['init_bbox'], meta['path'])
    for name in ['precision', 'recall', 'f_score']:
        check('metric_consistency', values[name + '_percent'] == values[name] * 100 == row['metrics_percent'][name], dataset + '/' + name)
    check('metric_arithmetic', abs(2 * values['precision'] * values['recall'] / (values['precision'] + values['recall']) - values['f_score']) < 1e-14, dataset + '/harmonic mean')
    ope_scope[dataset] = dict(sequences=expected_n, frames=expected_frames, thresholds=values['threshold'], metrics_percent=row['metrics_percent'])

previous = history[0]['previous_F_percent']
streak = 0
best_score = None
best_epoch = None
for row in history:
    current = [row['metrics'][dataset]['metrics']['f_score_percent'] for dataset in ['depthtrack_test', 'cdtb']]
    dual = all(now < before for now, before in zip(current, previous))
    streak = streak + 1 if dual else 0
    score = sum(current) / 2
    replace = best_score is None or score > best_score
    check('selection_recomputed', row['previous_F_percent'] == previous and row['F_percent'] == current and row['dual_decline'] == dual and row['consecutive_dual_declines'] == streak and row['stop'] == (streak == 2) and row['score'] == score and row['replace_best'] == replace, 'epoch ' + str(row['epoch']))
    if replace:
        best_epoch, best_score = row['epoch'], score
    previous = current
check('selection_recomputed', best_epoch == best['epoch'] == 6 and best_score == best['score'], 'selected epoch')
check('selection_recomputed', len(history) == terminal['epochs_completed'] == training_state['epoch_completed'] == 8 and streak == 2 and terminal['early_stopped'], 'terminal epoch')
for epoch in ['7', '8']:
    training = compact['training_reports'][epoch]['summary']
    summary = compact['sequence_log_summaries'][epoch]
    check('training_summary_arithmetic', training['sequence_runs'] == int(epoch) * 152 and training['track_calls'] == int(epoch) * 219802, epoch)
    check('training_summary_arithmetic', training['optimizer_steps'] == training['C_optimizer_steps'] == int(epoch) * 6807, epoch)
    check('training_summary_arithmetic', summary['saved_records_current_epoch'] == 152 and summary['saved_track_calls_current_epoch'] == 219802, epoch)
    check('training_summary_binding', training['final_sha256'] == history[int(epoch)-1]['final_sha256'] and training['seed'] == 2027 and training['legacy_handoff_RNG_exact'] is False, epoch)

anchors = {}
vot_gt_metadata = {r['sequence']: r for r in payload['VOT_GT_file_metadata']}
vot_gt_copies = 0
for sequence, info in manifest['source'].items():
    path = info['root'] + '/anchor.value'
    agrees_hash(path, info['anchor_sha256'], 'VOT_original_anchor_digest')
    values = [float(x) for x in files[path]['text'].splitlines()]
    check('VOT_original_extent', len(values) == info['frame_count'] == analysis['sequences'][sequence]['length'], sequence)
    for index, value in enumerate(values):
        if value != 0:
            direction = 'forward' if value > 0 else 'backward'
            length = len(values)-index if value > 0 else index+1
            anchors[(sequence, index, direction)] = length
    for shard in manifest['shards']:
        gtpath = shard['root'] + '/sequences/' + sequence + '/groundtruth.txt'
        if gtpath in files:
            vot_gt_copies += 1
            agrees_hash(gtpath, vot_gt_metadata[sequence]['sha256'], 'VOT_GT_copy_digest')
            check('VOT_GT_copy_extent', len(files[gtpath]['text'].splitlines()) == info['frame_count'] and files[gtpath]['bytes'] == vot_gt_metadata[sequence]['bytes'], gtpath)
    check('VOT_GT_coverage', any(s['root'] + '/sequences/' + sequence + '/groundtruth.txt' in files for s in manifest['shards']), sequence)
shard_anchors = []
trajectories = []
for shard in manifest['shards']:
    for filename, field in [('config.yaml', 'config_sha256'), ('trackers.ini', 'trackers_sha256'), ('sequences/list.txt', 'list_sha256')]:
        agrees_hash(shard['root'] + '/' + filename, shard[field], 'VOT_shard_configuration')
    expected = []
    for row in shard['anchors']:
        key = (row['sequence'], row['index'], row['direction'])
        shard_anchors.append(key)
        expected.append(row['sequence'] + '_' + format(row['index'], '08d'))
        check('VOT_shard_anchor', row['estimated_frames'] == anchors[key], str(key))
        path = shard['root'] + '/sequences/' + row['sequence'] + '/anchor.value'
        values = files[path]['text'].splitlines()
        check('VOT_shard_anchor', float(values[row['index']]) == row['value'], path + ':' + str(row['index']+1))
    check('VOT_shard_extent', expected == shard['expected_trajectories'] and len(expected) == shard['anchor_count'], str(shard['index']))
    check('VOT_shard_extent', sum(r['estimated_frames'] for r in shard['anchors']) == shard['estimated_frames'], str(shard['index']))
    for sequence in files[shard['root'] + '/sequences/list.txt']['text'].splitlines():
        path = shard['root'] + '/sequences/' + sequence + '/anchor.value'
        actual = {(sequence, index, 'forward' if float(value) > 0 else 'backward')
                  for index, value in enumerate(files[path]['text'].splitlines()) if float(value) != 0}
        assigned = {(r['sequence'], r['index'], r['direction']) for r in shard['anchors'] if r['sequence'] == sequence}
        check('VOT_shard_anchor_exact_cover', actual == assigned, path)
    trajectories.extend(expected)
check('VOT_exact_cover', len(anchors) == len(shard_anchors) == len(set(shard_anchors)) == 1765 and set(shard_anchors) == set(anchors), 'original vs shards')
check('VOT_exact_cover', sum(anchors.values()) == manifest['total_estimated_frames'] == 1327004, 'proxy frame slots')
check('VOT_exact_cover', set(manifest['sequences']) == set(manifest['source']) == set(analysis['sequences']) == set(vot['per_sequence_failures']) and len(manifest['sequences']) == 127, 'sequences')
expected_files = {'results/' + manifest['tracker'] + '/baseline/' + t.rsplit('_', 1)[0] + '/' + t + suffix for t in trajectories for suffix in ['.bin', '_confidence.value', '_time.value']}
check('VOT_merge_inventory', set(merge['result_sha256']) == expected_files and len(expected_files) == merge['result_file_count'] == 5295, 'all merged output keys')
for path, value in merge['result_sha256'].items():
    check('VOT_merge_digest_format', len(value) == 64 and all(x in '0123456789abcdef' for x in value), path)
outcomes = vot['failure_outcomes']
outcome_keys = set()
per = defaultdict(lambda: [0, 0, 0, 0])
for name, row in outcomes.items():
    key = (row['sequence'], row['anchor'], row['direction'])
    outcome_keys.add(key)
    check('VOT_failure_outcomes', row['anchor_key'] == name and row['run_length'] == anchors[key] and 0 <= row['progress'] <= row['run_length'] and row['failed'] == (row['progress'] < row['run_length']), name)
    p = per[row['sequence']]
    p[0] += 1
    p[1] += row['failed']
    p[2] += row['progress']
    p[3] += row['run_length']
check('VOT_failure_cover', len(outcomes) == 1765 and outcome_keys == set(anchors), 'all outcomes')
for sequence, values in per.items():
    check('VOT_failure_count', dict(anchors=values[0], confirmed_failures=values[1]) == vot['per_sequence_failures'][sequence], sequence)
check('VOT_failure_count', sum(v[1] for v in per.values()) == vot['confirmed_failures'] == 231, 'confirmed failures')
ar = analysis['results']['baseline']['results'][2][0]
total_frames = sum(r['length'] for r in analysis['sequences'].values())
weighted_robustness = sum((values[2] / values[3]) * analysis['sequences'][seq]['length'] for seq, values in per.items()) / total_frames
check('VOT_derived_aggregation', abs(weighted_robustness - ar[1]) < 1e-14 and sum(v[2] for v in per.values()) == ar[3] and total_frames == ar[4], 'ROB from sealed failure progress and source lengths')
params = analysis['results']['baseline']['parameters']['analyses'][0]
curve = analysis['results']['baseline']['results'][1][0][0]
selected_curve = curve[params['low']:params['high']+1]
eao = sum(selected_curve)/len(selected_curve)
check('VOT_derived_aggregation', abs(eao - analysis['results']['baseline']['results'][0][0][0]) < 1e-14, 'EAO from sealed curve')
for name, value in [('EAO', analysis['results']['baseline']['results'][0][0][0]), ('ACC', ar[0]), ('ROB', ar[1])]:
    check('metric_consistency', value * 100 == vot['metrics_percent'][name], 'VOT/' + name)

metrics = results['gates'][0]['metrics_percent']
gates = dict(depthtrack_P=metrics['depthtrack_test']['precision'] >= 65.2,
             depthtrack_R=metrics['depthtrack_test']['recall'] >= 64.9,
             depthtrack_F=metrics['depthtrack_test']['f_score'] >= 65.1,
             cdtb_P=metrics['cdtb']['precision'] >= 72.9,
             cdtb_R=metrics['cdtb']['recall'] >= 75.6,
             cdtb_F=metrics['cdtb']['f_score'] >= 74.2,
             vot_EAO=metrics['vot']['EAO'] > 77.9,
             vot_ACC=metrics['vot']['ACC'] > 82.1,
             vot_ROB=metrics['vot']['ROB'] > 93.7)
check('gate_arithmetic', gates == results['gates'][0]['passed'] and sum(gates.values()) == 5 and results['any_joint_pass'] is False, 'nine thresholds')
csvrow = next(csv.DictReader(io.StringIO(files[EVAL + '/metrics.csv']['text'])))
for prefix, dataset, names in [('DepthTrack', 'depthtrack_test', [('P','precision'),('R','recall'),('F','f_score')]), ('CDTB', 'cdtb', [('P','precision'),('R','recall'),('F','f_score')]), ('VOT', 'vot', [('EAO','EAO'),('ACC','ACC'),('ROB','ROB')])]:
    for key, metric in names:
        check('metric_csv', float(csvrow[prefix + '_' + key]) == metrics[dataset][metric], prefix + '/' + key)
draft = (OUT / 'DRAFT_NINE_METRIC_REPORT.md').read_text(encoding='utf-8')
for dataset, values in metrics.items():
    check('draft_numbers', ('| ' + dataset + ' | ' + ' | '.join(format(v, '.6f') for v in values.values()) + ' |') in draft, dataset)

for path, entry in files.items():
    if path.startswith(REMOTE + '/') and path.endswith('.exit'):
        check('terminal_exit', entry['text'].strip() == '0', path)
for stage in terminal['stages']:
    check('terminal_stage', stage['exit_code'] == 0 and stage['started_at'] < stage['finished_at'], stage['stage'])
check('terminal_status', len(terminal['stages']) == 53 and terminal['status'] == 'complete_M124_epochwise_training_best_and_full_VOT', 'controller result')
check('terminal_status', parsed(REMOTE + '/controller.supervisor.terminal.json')['exit_code'] == 0, 'supervisor')
launches = parsed(EVAL + '/vot/run/launch.json')
for launch, shard in zip(launches, manifest['shards']):
    check('terminal_shards', launch['exit'] == 0 and launch['completed_anchors'] == shard['anchor_count'] and launch['gpu'] == shard['gpu'], str(launch['index']))

summary = dict(generated_at=datetime.now(timezone.utc).isoformat(), verifier='standard-library local CPU; input code not executed',
               status='PASS' if not failures else 'FAIL', checks=dict(counts), total_checks=sum(counts.values()), failures=failures,
               supplied_primary_inputs=len(LOCAL_INPUTS), captured_files=len(files), captured_bytes=sum(v['bytes'] for v in files.values()),
               compact_exact_files=compact_count, compact_duplicate_files=compact_duplicate_count,
               bundle_source_files=len(bundle['source_sha256']), execution_source_files=len(execution['source_sha256']),
               audited_input_count=len(hashes), metrics_percent=metrics, gates=gates,
               ope_scope=ope_scope, selected_epoch=best_epoch, epochs_completed=len(history),
               seed=2027, vot_unique_sequence_frames=total_frames, vot_anchors=len(anchors), vot_proxy_frame_slots=sum(anchors.values()),
               vot_direction_counts=dict(Counter(key[2] for key in anchors)), vot_GT_copies=vot_gt_copies,
               vot_confirmed_failures=sum(v[1] for v in per.values()), vot_robustness_reaggregated=weighted_robustness,
               vot_eao_reaggregated=eao, vot_eao_curve_length=len(curve), vot_eao_slice_length=len(selected_curve),
               raw_OPE_predictions_recomputed=False, raw_VOT_trajectory_metrics_recomputed=False,
               live_tensor_or_runtime_attestation=False, original_weight_bytes_rehashed_locally=False,
               baseline_reference_primary_result_available=False)
(OUT / 'deterministic_checks.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
(OUT / 'audited_input_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
