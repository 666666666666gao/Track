"""Bounded CPU-only audit of the exact supplied artifacts; never imports tracker code."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import numpy as np

BASE = Path(r'C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007')
SOURCE = BASE.parent.parent / 'projects/sttrack_lachtt_v1/diagnostics/three_module_20260928'
INPUT = BASE / 'M125_R2_RESULT_INPUTS_20261015'
OUT = BASE / 'M125_R2_RESULT_AUDIT_20261015'
REMOTE = '/root/autodl-tmp/m125_selected_E6_split_replay_R2_20261015'
ARCHIVE = BASE / 'M125_R2_complete_replay_primary_inputs_retry1_20261015.json'
SUMMARY = BASE / 'M125_R2_complete_box_coverage_summary_20261015.json'
PLAN = BASE / 'M125_SPLIT_REPLAY_SOURCE_20261015/PLAN.json'
manifest = json.loads((INPUT / 'INPUTS.json').read_bytes())
local = {r['remote_path']: Path(r['path']) for r in manifest}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
load = lambda p: json.loads(Path(p).read_bytes())
read_remote = lambda p: load(local[p])
checks = []

def check(name, condition, detail=None):
    checks.append(dict(name=name, pass_=bool(condition), detail=detail))

def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')

paths = [Path(r['path']) for r in manifest] + [INPUT / 'INPUTS.json', ARCHIVE, SUMMARY,
    SOURCE / 'summarize_m125_selected_E6_observation.py', SOURCE / 'full_dense_tracker.py', PLAN]
inventory = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in paths]
save('input_inventory.json', inventory)
archive = load(ARCHIVE)
summary = load(SUMMARY)
plan = load(PLAN)
config = read_remote(REMOTE + '/config.json')
launch = read_remote(REMOTE + '/actual_launch.json')
launches = read_remote(REMOTE + '/launches.json')
terminal = read_remote(REMOTE + '/result.json')
for r in manifest:
    check('manifest_sha:' + Path(r['path']).name, sha(r['path']) == r['sha256'])
check('archive_entry_count', len(archive['files']) == 48)
for remote, r in archive['files'].items():
    raw = r['text'].encode('utf-8')
    check('archive_raw_sha:' + remote, hashlib.sha256(raw).hexdigest() == r['sha256'])
    check('archive_raw_bytes:' + remote, len(raw) == r['bytes'])
    check('archive_extracted_exact:' + remote, local[remote].read_bytes() == raw)
check('summary_archive_sha', summary['archive_sha256'] == sha(ARCHIVE))
check('summary_source_sha', summary['source_sha256'] == sha(SOURCE / 'summarize_m125_selected_E6_observation.py'))
check('embedded_config_plan', config['plan'] == plan)
check('launch_config_sha', launch['config_sha256'] == sha(local[REMOTE + '/config.json']))
check('launch_driver_sha', launch['driver_sha256'] == sha(local[REMOTE + '/driver.py']))
check('terminal_launches', terminal['launches'] == launches)
check('terminal_status', terminal['status'] == 'complete_M125_E6_split_readonly_panel')
check('controller_exit_sentinel', local[REMOTE + '/controller.exit'].read_text().strip() == '0')
for name, digest in plan['sources'].items():
    p = next(p for remote,p in local.items() if remote.endswith('/'+name))
    check('plan_source_sha:' + name, sha(p) == digest)

sources = [p for p in paths if p.suffix == '.py']
source_ast = {}
for p in sources:
    tree = ast.parse(p.read_text(encoding='utf-8-sig'))
    source_ast[p.name] = tree
    check('AST_parse:' + p.name, isinstance(tree, ast.Module))
search_sizes = [node.value.value for node in ast.walk(source_ast['full_dense_tracker.py'])
    if isinstance(node, ast.keyword) and node.arg == 'search_size']
check('one_source_search_size', search_sizes == [256])
search_size = search_sizes[0]

# Probe only the two supplied interpreters; subprocesses preserve actual import failures.
probe_code = "import sys,importlib.util,json; print(json.dumps({'python':sys.version,'executable':sys.executable,'modules':{m: (None if importlib.util.find_spec(m) is None else importlib.util.find_spec(m).origin) for m in ['numpy','cv2','vot','numba']}}))"
probe_results = []
for interpreter in [sys.executable, r'C:\Users\gb\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe']:
    command = [interpreter, '-c', probe_code]
    p = subprocess.run(command, capture_output=True, text=True)
    probe_results.append(dict(command=command, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr))
metric_path = INPUT / 'locked_depthtrack_pr.py'
commands = [
    [sys.executable, '-c', "import importlib.util; s=importlib.util.spec_from_file_location('audit_locked_metric'," + repr(str(metric_path)) + "); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)"],
    [sys.executable, '-c', 'from vot.region import Rectangle,RegionType,Special,calculate_overlaps'],
]
for command in commands:
    p = subprocess.run(command, capture_output=True, text=True)
    probe_results.append(dict(command=command, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr))
save('dependency_probes.json', probe_results)
official_available = all(importlib.util.find_spec(m) is not None for m in ['cv2','vot'])
check('official_recompute_not_substituted', not official_available, 'Recorded boundary: no cv2/vot; no alternative raster or continuous IoU is used.')

sequences = []
states = []
total_actions = total_snapshots = total_rows = 0
metric_sha = sha(metric_path)
first_stage = None
for record in terminal['results']:
    name = record['sequence']
    prefix = REMOTE + '/' + name
    stage = read_remote(prefix + '/prediction_stage.json')
    prediction = read_remote(prefix + '/replayed_predictions.json')
    result = read_remote(prefix + '/result.json')
    seq_launch = next(r for r in launches if r['sequence'] == name)
    if first_stage is None:
        first_stage = stage
    for key in ['plan','bundle','selected_best','plan_sha256','terminal_suite_result_sha256','source_sha256']:
        check(name + ':shared_stage:' + key, stage[key] == first_stage[key])
    for kind in ['NN','CPU']:
        check(name + ':' + kind + '_exit', local[REMOTE + '/' + name + '.' + kind + '.exit'].read_text().strip() == '0' and seq_launch[kind + '_exit'] == 0)
        lines = local[REMOTE + '/' + name + '.' + kind + '.log'].read_text().splitlines()
        log = json.loads(lines[-1])
        expected = stage if kind == 'NN' else result
        check(name + ':' + kind + '_log', all(expected[k] == v for k,v in log.items()))
    args = seq_launch['args']
    check(name + ':launch_frames', [int(x) for x in args[args.index('--frames')+1:args.index('--suite')]] == plan['panel'][name] == stage['frames'])
    check(name + ':launch_plan', args[args.index('--plan')+1] == stage['args']['plan'] == plan['plan'])
    check(name + ':launch_sequence', args[args.index('--sequence')+1] == stage['args']['sequence'] == name)
    check(name + ':launch_suite', args[args.index('--suite')+1] == stage['args']['suite'] == plan['suite'])
    check(name + ':launch_output', args[args.index('--output')+1] == stage['args']['output'] == prefix)
    check(name + ':execution_order', seq_launch['started_at'] <= seq_launch['NN_finished_at'] <= seq_launch['CPU_finished_at'])
    check(name + ':result_sha', record['sha256'] == sha(local[prefix + '/result.json']))
    check(name + ':prediction_sha', stage['prediction_sha256'] == result['replayed_prediction_sha256'] == sha(local[prefix + '/replayed_predictions.json']))
    check(name + ':source_sha', stage['source_sha256'] == result['source_sha256']['prediction'] == plan['sources']['replay_m125_selected_E6_observation.py'])
    check(name + ':analysis_sha', result['source_sha256']['analysis'] == plan['sources']['analyze_m125_selected_E6_observation.py'])
    check(name + ':epoch_checkpoint', stage['selected_epoch'] == result['selected_epoch'] == stage['selected_best']['epoch'] == 6 and stage['bundle']['final_sha256'] == result['original_final_sha256'] == stage['selected_best']['final_sha256'] == plan['selected_final_sha256'])
    check(name + ':metric_sha', stage['plan']['metric_source_sha256'] == result['metric_source_sha256'] == metric_sha)
    source_hashes = stage['bundle']['source_sha256']
    dense_hashes = [v for k,v in source_hashes.items() if k.endswith('/full_dense_tracker.py')]
    check(name + ':full_dense_source_sha', len(dense_hashes)>0 and all(v == sha(SOURCE / 'full_dense_tracker.py') for v in dense_hashes))
    check(name + ':sealed_flags', not prediction['current_GT_opened'] and not stage['current_GT_opened'] and prediction['optimizer_steps'] == stage['optimizer_steps'] == result['optimizer_steps'] == 0 and not prediction['alternatives_submitted'] and prediction['state_after_alternatives_unchanged'])
    gt_path = local[stage['plan']['dataset_root'] + '/' + name + '/groundtruth.txt']
    saved_box_path = local[stage['plan']['output'] + '/' + name + '.txt']
    saved_score_path = local[stage['plan']['output'] + '/' + name + '_all_scores.txt']
    gt = np.loadtxt(gt_path, delimiter=',', dtype=np.float64).reshape(-1,4)
    saved_boxes = np.loadtxt(saved_box_path, delimiter='\t', dtype=np.float64).reshape(-1,4)
    saved_scores = np.loadtxt(saved_score_path, dtype=np.float64).reshape(-1)
    check(name + ':GT_sha', sha(gt_path) == stage['case']['gt_sha256'] == result['GT_sha256'])
    check(name + ':saved_box_sha', sha(saved_box_path) == result['official_prediction_sha256'])
    check(name + ':full_dataset_rows', len(gt) == len(saved_boxes) == len(saved_scores) == stage['case']['frames'])
    boxes = np.asarray(prediction['boxes'], dtype=np.float64)
    scores = np.asarray(prediction['scores'], dtype=np.float64)
    n = len(boxes)
    actions = prediction['actions']
    snapshots = prediction['sampled_candidates']
    check(name + ':prefix_counts', n == max(plan['panel'][name]) + 1 == len(scores) == len(actions)+1 == stage['tracker_calls']+1 == result['prefix_frames'] == record['tracker_calls']+1)
    check(name + ':sampled_counts', len(snapshots) == len(stage['frames']) == len(result['diagnostics']) == stage['extra_Empty_decoder_calls'] == result['candidate_states'] == record['candidate_states'])
    check(name + ':initial_GT_only', np.array_equal(boxes[0],gt[0]) and np.array_equal(boxes[0],stage['case']['init_bbox']) and scores[0] == 1.0)
    check(name + ':prefix_finite', np.isfinite(boxes).all() and np.isfinite(scores).all())
    box_diff = np.abs(boxes - saved_boxes[:n])
    score_diff = np.abs(scores - saved_scores[:n])
    check(name + ':prefix_bbox_rounding', bool((box_diff <= 5.01e-7).all()))
    check(name + ':prefix_score_rounding', bool((score_diff <= 5.01e-7).all()))
    check(name + ':prefix_diff_report', float(box_diff.max()) == result['bbox_max_abs_difference_from_saved'] and float(score_diff.max()) == result['score_max_abs_difference_from_saved'])
    decimal_bbox_equal = np.array([[float(format(x,'.6f')) for x in b] for b in boxes]) == saved_boxes[:n]
    decimal_score_equal = np.asarray([float(format(x,'.6f')) for x in scores]) == saved_scores[:n]
    crop_mismatches = []
    resize_mismatches = []
    for i, action in enumerate(actions,1):
        check(name + ':action_chain:' + str(i), action['frame']==i and action['previous_bbox'] == prediction['boxes'][i-1] and action['bbox'] == prediction['boxes'][i] and action['best_score'] == prediction['scores'][i])
        x,y,w,h = action['previous_bbox']
        side = math.ceil(math.sqrt(w*h)*4.)
        crop = [float(round(x + .5*w - .5*side)), float(round(y + .5*h - .5*side)), float(side)]
        if crop != action['crop_origin']:
            crop_mismatches.append(i)
        if not math.isclose(action['resize_factor'] * action['crop_origin'][2], search_size, rel_tol=0., abs_tol=1e-12):
            resize_mismatches.append(i)
    check(name + ':crop_arithmetic_from_prior', not crop_mismatches, crop_mismatches)
    check(name + ':resize_actual_source_size', not resize_mismatches, resize_mismatches)
    check(name + ':sampled_frame_ids', [s['frame'] for s in snapshots] == stage['frames'] == [d['frame'] for d in result['diagnostics']])
    for snapshot, diagnostic in zip(snapshots,result['diagnostics']):
        frame = snapshot['frame']
        label = name + ':' + str(frame)
        action = actions[frame-1]
        check(label + ':snapshot_action', snapshot['action'] == action and snapshot['crop_origin'] == action['crop_origin'])
        check(label + ':image_shape', snapshot['image_shape'] == stage['image_shape'])
        check(label + ':candidate_shapes', all(np.asarray(snapshot[k]).shape == (256,4) for k in ['native_boxes_float32','refined_boxes']))
        vectors = ['full_response','native_response','visual_response','full_response_without_quality','quality_probability','semantic_delta','phrase_cosine']
        check(label + ':vector_shapes', all(np.asarray(snapshot[k]).shape == (256,) for k in vectors))
        check(label + ':candidate_finite', all(np.isfinite(snapshot[k]).all() for k in vectors + ['native_boxes_float32','refined_boxes']))
        check(label + ':native_float32', np.array_equal(np.asarray(snapshot['native_boxes_float32']),np.asarray(snapshot['native_boxes_float32'],dtype=np.float32).astype(np.float64)))
        for field, vector in [('full_selected','full_response'),('native_selected','native_response'),('visual_selected','visual_response'),('without_quality_selected','full_response_without_quality')]:
            check(label + ':argmax:' + field, snapshot[field] == int(np.argmax(snapshot[vector])))
        selected = snapshot['full_selected']
        check(label + ':submitted_selected', selected == action['selected'] and snapshot['refined_boxes'][selected] == action['bbox'] and snapshot['full_response'][selected] == action['best_score'] and snapshot['quality_probability'][selected] == action['selected_quality'])
        quality = np.asarray(snapshot['quality_probability'])
        raw_division = np.asarray(snapshot['full_response']) / quality
        division_error = float(np.max(np.abs(raw_division - np.asarray(snapshot['full_response_without_quality']))))
        check(label + ':quality_positive', ((quality>0)&(quality<=1)).all())
        check(label + ':quality_removal_algebra', np.allclose(raw_division, snapshot['full_response_without_quality'], rtol=2e-7, atol=0.), division_error)
        native_iou = np.asarray(diagnostic['native_iou_by_candidate'])
        refined_iou = np.asarray(diagnostic['refined_iou_by_candidate'])
        check(label + ':archived_IoU_vectors', native_iou.shape == refined_iou.shape == (256,) and np.isfinite(native_iou).all() and np.isfinite(refined_iou).all() and ((native_iou>=0)&(native_iou<=1)).all() and ((refined_iou>=0)&(refined_iou<=1)).all())
        best = int(refined_iou.argmax())
        ranking = np.argsort(-np.asarray(snapshot['full_response']),kind='stable')
        recomputed = dict(native_candidate_best_IoU=float(native_iou.max()), refined_candidate_best_IoU=float(refined_iou.max()),
            native_candidate_IoU_ge_05=int((native_iou>=.5).sum()), refined_candidate_IoU_ge_05=int((refined_iou>=.5).sum()),
            native_score_native_geometry_IoU=float(native_iou[snapshot['native_selected']]),
            native_score_refined_geometry_IoU=float(refined_iou[snapshot['native_selected']]),
            full_score_native_geometry_IoU=float(native_iou[selected]), full_score_refined_geometry_IoU=float(refined_iou[selected]),
            Empty_score_refined_geometry_IoU=float(refined_iou[snapshot['visual_selected']]),
            without_quality_score_refined_geometry_IoU=float(refined_iou[snapshot['without_quality_selected']]),
            best_refined_candidate_score_rank=int(np.flatnonzero(ranking==best)[0])+1,
            selected_semantic_delta=snapshot['semantic_delta'][selected], best_refined_candidate_semantic_delta=snapshot['semantic_delta'][best],
            selected_quality=snapshot['quality_probability'][selected], best_refined_candidate_quality=snapshot['quality_probability'][best])
        check(label + ':archived_IoU_arithmetic', all(diagnostic[k] == v for k,v in recomputed.items()))
        check(label + ':selected_IoU_consistency', diagnostic['replayed_submitted_IoU'] == float(refined_iou[selected]))
        truth = gt[frame]
        valid = bool(np.isfinite(truth).all() and truth[2]>0 and truth[3]>0)
        check(label + ':independent_GT_validity', diagnostic['GT_valid'] == valid)
        row = dict(sequence=name, frame=frame, frame_index_convention='zero_based; corresponding dataset image stem is frame+1',
            GT_valid=valid, image_shape=snapshot['image_shape'], crop_origin=snapshot['crop_origin'], source_search_size=search_size,
            recorded_refined_best_IoU=diagnostic['refined_candidate_best_IoU'], recorded_refined_count_ge_05=diagnostic['refined_candidate_IoU_ge_05'],
            official_overlap_recomputed=False, quality_removal_max_abs_error=division_error,
            full_selected=selected, Empty_selected=snapshot['visual_selected'], without_quality_selected=snapshot['without_quality_selected'])
        reported = next(r for r in summary['states'] if r['sequence']==name and r['frame']==frame)
        check(label + ':summary_GT_validity', reported['GT_valid'] == valid)
        if valid:
            x,y,w,h = truth.tolist()
            im_h, im_w = snapshot['image_shape']
            cx,cy,side = snapshot['crop_origin']
            # Independently clip GT to image, then clip that rectangle to the search crop.
            left, top, right, bottom = max(0.,x), max(0.,y), min(float(im_w),x+w), min(float(im_h),y+h)
            image_area = max(0.,right-left)*max(0.,bottom-top)
            crop_area = max(0.,min(right,cx+side)-max(left,cx))*max(0.,min(bottom,cy+side)-max(top,cy))
            center = bool(cx <= x+.5*w < cx+side and cy <= y+.5*h < cy+side)
            row.update(GT_box=truth.tolist(), GT_box_area_in_image=image_area, GT_box_area_in_image_and_crop=crop_area,
                GT_box_crop_coverage_fraction=crop_area/image_area, GT_width_in_search=w*search_size/side, GT_height_in_search=h*search_size/side,
                GT_center_in_crop=center)
            check(label + ':center', center == diagnostic['GT_center_in_crop'] == reported['GT_center_in_crop'])
            mapping = {'GT_box_area_in_image':'GT_box_area_in_image', 'GT_box_area_in_image_and_crop':'GT_box_area_in_image_and_crop',
                'GT_box_crop_coverage_fraction':'GT_box_crop_coverage_fraction', 'GT_width_in_search':'GT_width_in_256_search', 'GT_height_in_search':'GT_height_in_256_search'}
            check(label + ':independent_coverage_summary', all(math.isclose(row[k],reported[v],rel_tol=1e-12,abs_tol=1e-12) for k,v in mapping.items()) and reported['GT_box'] == truth.tolist())
        else:
            row.update(GT_box=None, GT_center_in_crop=None, GT_box_crop_coverage_fraction=None, GT_width_in_search=None, GT_height_in_search=None)
            check(label + ':invalid_excluded', diagnostic['GT_center_in_crop'] is None and 'GT_box_crop_coverage_fraction' not in reported)
        copies = {'selected_IoU':'replayed_submitted_IoU','native_best_IoU':'native_candidate_best_IoU','refined_best_IoU':'refined_candidate_best_IoU',
            'native_count_ge_05':'native_candidate_IoU_ge_05','refined_count_ge_05':'refined_candidate_IoU_ge_05',
            'Empty_selected_IoU':'Empty_score_refined_geometry_IoU','without_quality_selected_IoU':'without_quality_score_refined_geometry_IoU','selected_quality':'selected_quality'}
        check(label + ':summary_recorded_metric_copy', all(reported[k] == diagnostic[v] for k,v in copies.items()))
        states.append(row)
    sequences.append(dict(sequence=name, full_dataset_rows=len(gt), prefix_rows=n, tracker_step_calls=len(actions), sampled_candidate_states=len(snapshots),
        bbox_max_abs_difference=float(box_diff.max()), confidence_max_abs_difference=float(score_diff.max()),
        bbox_within_half_6decimal_unit=bool((box_diff<=5.01e-7).all()),confidence_within_half_6decimal_unit=bool((score_diff<=5.01e-7).all()),
        formatted_6decimal_bbox_mismatched_components=int((~decimal_bbox_equal).sum()), formatted_6decimal_confidence_mismatches=int((~decimal_score_equal).sum()),
        crop_arithmetic_mismatches=crop_mismatches, resize_mismatches=resize_mismatches,
        actions_with_C_prediction=sum(a['C_prediction'] is not None for a in actions), template_writes=sum(a['template_write'] for a in actions)))
    total_actions += len(actions)
    total_snapshots += len(snapshots)
    total_rows += n
check('total_tracker_calls', total_actions == terminal['tracker_calls'] == plan['tracker_calls'] == summary['counts']['tracker_calls'] == 2894)
check('total_candidate_states', total_snapshots == terminal['candidate_states'] == plan['candidate_states'] == summary['counts']['candidate_states'] == 25)
check('total_prefix_rows', total_rows == summary['counts']['prefix_rows'] == 2898)
valid_states = [r for r in states if r['GT_valid']]
counts = dict(sequences=len(sequences), tracker_step_calls=total_actions, initialized_prefix_rows=total_rows, candidate_states=total_snapshots,
    candidate_boxes_each_geometry=total_snapshots*256, initializations=len(sequences), extra_Empty_decoder_calls=total_snapshots,
    GT_valid_states=len(valid_states),GT_invalid_states=len(states)-len(valid_states),
    valid_states_full_continuous_coverage=sum(r['GT_box_crop_coverage_fraction']==1 for r in valid_states),
    valid_states_zero_continuous_coverage=sum(r['GT_box_crop_coverage_fraction']==0 for r in valid_states),
    valid_states_without_refined_candidate_ge_05_from_recorded_IoUs=sum(r['recorded_refined_count_ge_05']==0 for r in valid_states))
check('summary_counts_from_primary', summary['counts']['GT_valid_states']==counts['GT_valid_states'] and summary['counts']['GT_invalid_states']==counts['GT_invalid_states'] and summary['counts']['valid_states_without_refined_candidate_ge_05']==counts['valid_states_without_refined_candidate_ge_05_from_recorded_IoUs'])
unchanged = [dict(path=r['path'],same_sha256=sha(r['path'])==r['sha256']) for r in inventory]
check('all_inputs_unchanged', all(r['same_sha256'] for r in unchanged))
save('final_input_integrity.json',unchanged)
save('independent_coverage.json',dict(definition='continuous area(GT intersect image intersect crop) / area(GT intersect image); no pixel visibility, occlusion or semantic identity inference',
    official_overlap_recomputed=False, counts=counts, sequences=sequences, states=states))
result = dict(generated_at=datetime.now(timezone.utc).isoformat(), interpreter=sys.executable,python=sys.version,numpy=np.__version__,
    input_count=len(inventory),check_count=len(checks),passed=sum(r['pass_'] for r in checks),failed=sum(not r['pass_'] for r in checks),
    counts=counts,official_overlap_verification='UNVERIFIED_DEPENDENCIES_UNAVAILABLE',checks=checks)
save('deterministic_checks.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
print(json.dumps({'failed_checks':[r for r in checks if not r['pass_']],'sequences':sequences},indent=2))
