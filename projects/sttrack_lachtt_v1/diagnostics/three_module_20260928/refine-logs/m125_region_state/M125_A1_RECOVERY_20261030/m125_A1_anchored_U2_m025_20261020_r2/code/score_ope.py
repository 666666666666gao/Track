import importlib.util, json
from pathlib import Path
spec = importlib.util.spec_from_file_location('m', '/home/SRTrack_RGBD_L/lib/test/analysis/depthtrack_pr.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
C = Path('/root/autodl-tmp/m125_CAD_same_R0_uniform_profiles_20261016/profiles/U2')
for d in ['depthtrack_test', 'cdtb']:
    plan = json.loads((C / d / 'plan.json').read_text()); cases = json.loads(Path(plan['cases_path']).read_text())
    v = m.evaluate_depthtrack_results(plan['dataset_root'], Path(d), resolution=100, sequence_names=[c['sequence'] for c in cases])
    Path(d, 'metrics.json').write_text(json.dumps(dict(metrics=v, profile='U2', margin=.25, radius=1), indent=2) + '\n')
    print(d, v['precision_percent'], v['recall_percent'], v['f_score_percent'], v['sequences'], v['frames'], flush=True)
