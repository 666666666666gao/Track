from pathlib import Path
import ast, hashlib, json
D=Path(r'C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928')
source=(D/'replay_currentC_cdtb_onsets.py').read_text(encoding='utf-8')
source=source.replace('Read-only CurrentC own-history replay','Read-only selected-best M124 own-history replay')
source=source.replace('from m122_current_C_official_runtime import checked_plan, OfficialDenseTracker','from m124_epochwise_official_runtime import checked_plan, OfficialDenseTracker')
start=source.index('FRAMES = {')
end=source.index('\n\n\ndef write',start)
source=source[:start]+source[end+2:]
source=source.replace("parser.add_argument('--sequence', choices=sorted(FRAMES), required=True)","parser.add_argument('--sequence', required=True)\n    parser.add_argument('--frames', type=int, nargs='+', required=True)\n    parser.add_argument('--suite', type=Path, required=True)")
source=source.replace("plan, bundle = checked_plan(args.plan)\n    assert plan['dataset'] == 'cdtb'", """suite_result_path = args.suite / 'result.json'
    suite = json.loads(suite_result_path.read_text())
    assert suite['status'] == 'complete_M124_epochwise_training_best_and_full_VOT'
    assert suite['formal_nine_metrics'] and suite['automatic_retries'] == 0
    assert int((args.suite / 'controller.exit').read_text()) == 0
    selected_best = json.loads((args.suite / 'best.json').read_text())
    assert suite['best'] == selected_best
    plan, bundle = checked_plan(args.plan)
    assert plan['dataset'] == 'cdtb'
    assert bundle['final_sha256'] == selected_best['final_sha256']
    assert args.plan.resolve() == (Path(selected_best['evaluation']) / 'cdtb/plan.json').resolve()
    complete = json.loads((Path(selected_best['evaluation']) / 'all_results.json').read_text())
    assert complete['status'] == 'three_M124_selected_epoch_full_evaluations_complete'
    assert len(complete['results']) == 3
    assert all(row['final_sha256'] == selected_best['final_sha256'] for row in complete['results'])""")
source=source.replace("frames = FRAMES[args.sequence]", "frames = args.frames\n    assert frames == sorted(set(frames)) and min(frames) >= 1")
source=source.replace('locked_CurrentC_onset_metric','locked_M124_selected_best_onset_metric')
source=source.replace('complete_readonly_CurrentC_CDTB_onset_replay','complete_readonly_M124_selected_best_CDTB_onset_replay')
source=source.replace("prefix_frames=len(boxes), tracker_calls=len(boxes)-1,", "prefix_frames=len(boxes), tracker_calls=len(boxes)-1,\n        selected_epoch=selected_best['epoch'], terminal_suite_result_sha256=sha(suite_result_path),\n        terminal_suite_root=str(args.suite), required_full_training_and_VOT_terminal=True,\n        frame_sampling_scope='Caller-specified posthoc diagnostic panel; never chooses a tracking candidate or resets state.',")
source=source.replace("causal_scope='alternatives are same replay state only;", "causal_scope='Read-only score/geometry counterfactuals on selected-best own history; alternatives are same replay state only;")
anchor = "    receipt = json.loads((Path(plan['output']) / 'receipt.json').read_text())\n"
assert source.count(anchor) == 1
source = source.replace(anchor, anchor + "    assert receipt['plan_sha256'] == sha(args.plan)\n    assert receipt['final_sha256'] == bundle['final_sha256']\n    assert receipt['bank_sha256'] == plan['bank_sha256'] and receipt['bundle_sha256'] == plan['bundle_sha256']\n")
ast.parse(source, feature_version=(3,8))
assert 'FRAMES' not in source and 'm122_current_C_official_runtime' not in source
assert source.count("suite['status'] == 'complete_M124_epochwise_training_best_and_full_VOT'")==1
assert source.index("suite['status']") < source.index('actor = OfficialDenseTracker')
path=D/'replay_m124_selected_best_cdtb_onsets.py'
assert not path.exists()
path.write_bytes(source.encode('utf-8'))
print(json.dumps(dict(status='prepared_source_only_M124_selected_best_CDTB_replay',path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),NN_calls=0,GPU_queries=0,server_queries=0,training_source_changed=False,review_completed=False)))
