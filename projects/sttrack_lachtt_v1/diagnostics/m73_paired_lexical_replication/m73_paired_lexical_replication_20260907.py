"""Two prospectively fixed, paired category/empty full causal training replications."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

B = Path('/root/autodl-tmp')
PARENT = B / 'sttrack_m67_supervised_semantic_support_20260907'
ROOT = B / 'sttrack_m73_paired_lexical_replication_20260907'
PYTHON = B / 'envs/sttrack/bin/python'
SEEDS = [2027, 2028]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def write(path, data): Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
def now(): return datetime.now(timezone.utc).isoformat()


def change(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def prepare():
    import torch
    assert not ROOT.exists()
    assert sha(PARENT / 'training_spec.json') == '2027b1893343e5e1520c447bd82a5c45468542cecaab0f59b3ec72a345e48b2e'
    assert sha(PARENT / 'recursive_result.json') == '9f2cfe2457a967fdfa2ba9beae737b6a1575e7224c4029ee438472e547db4d07'
    content = B / 'sttrack_m72_m67_control_content_20260907/result.json'
    assert sha(content) == '8774a10626d9dd7127d1cd91f8f87107370415fcc842c514a59a08a74cf73326'
    assert read(content)['descriptive_criteria_pass']
    previous = read(PARENT / 'training_spec.json'); recursion = read(PARENT / 'recursive_spec.json')
    assert sha(PARENT / 'train_causal.py') == previous['training_script_sha256']
    assert sha(PARENT / 'run_recursive.py') == recursion['runner_sha256']
    assert sha(PARENT / 'causal_training.py') == previous['causal_script_sha256']
    integration = read(PARENT / 'integration.json')
    for n, h in integration['source_sha256'].items(): assert sha(PARENT / 'code' / n) == h
    ROOT.mkdir(); (ROOT / 'banks').mkdir()
    bank_info = {}
    for split in ['fit', 'development']:
        source = PARENT / ('text_' + split + '.pt')
        assert sha(source) == previous['text_' + split + '_sha256']
        cat = torch.load(source, map_location='cpu'); empty = dict(cat)
        empty['tokens'] = cat['tokens'].clone()
        empty['tokens'][cat['mask']] = cat['empty'].to(cat['tokens'].dtype)
        empty['lexical_policy'] = 'CLIP empty vector in every valid slot; preserve mask, five-slot structure and padding.'
        p = ROOT / 'banks' / (split + '_empty.pt'); torch.save(empty, p)
        assert torch.equal(empty['mask'], cat['mask'])
        assert torch.equal(empty['tokens'][~cat['mask']], cat['tokens'][~cat['mask']])
        assert torch.equal(empty['tokens'][cat['mask']], cat['empty'].to(cat['tokens'].dtype).expand(int(cat['mask'].sum()), -1))
        assert bool(cat['mask'][:, 0].all())
        changed = (empty['tokens'][:, 0] != cat['tokens'][:, 0]).any(dim=-1)
        assert bool(changed.all())
        bank_info[split] = {
            'category': dict(path=str(source), sha256=sha(source)),
            'empty': dict(path=str(p), sha256=sha(p)),
            'sequences': len(cat['sequences']), 'changed_category_slots': int(changed.sum()),
        }
    adapter_path = PARENT / 'code/lib/models/sttrack/semantic_spatial_adapter.py'
    module_spec = importlib.util.spec_from_file_location('m73_initial_adapter', str(adapter_path))
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    preparations = {}
    for seed in SEEDS:
        root = ROOT / ('seed' + str(seed)); root.mkdir()
        for n, h in integration['source_sha256'].items():
            dst = root / 'code' / n; dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PARENT / 'code' / n, dst); assert sha(dst) == h
        for n in ['integration.json', 'data_inventory.json', 'causal_training.py', 'support_loss.py', 'recursive_metric.py']:
            shutil.copyfile(PARENT / n, root / n)
        train = (PARENT / 'train_causal.py').read_text().replace(str(PARENT), str(root))
        train = train.replace("'control'", "'empty'").replace("'support'", "'category'")
        train = change(train, "assert sha(root / 'text_fit.pt') == spec['text_fit_sha256']",
            "assert sha(spec['banks']['fit'][args.arm]['path']) == spec['banks']['fit'][args.arm]['sha256']")
        train = change(train, "torch.load(root / 'text_fit.pt', map_location='cpu')",
            "torch.load(spec['banks']['fit'][args.arm]['path'], map_location='cpu')")
        train = change(train, "root = Path('" + str(root) + "')",
            "root = Path('" + str(root) + "')\n    assert (root / 'frozen.json').is_file()")
        train = change(train, "    spec = json.loads(spec_path.read_text())",
            "    spec = json.loads(spec_path.read_text())\n    assert sha(spec_path) == json.loads((root / 'frozen.json').read_text())['training_spec_sha256']")
        train = train.replace('Matched Null-architecture training; original loss versus explicit box-level support supervision.',
            'Matched category versus empty-content causal training; identical Null architecture and zero auxiliary loss.')
        (root / 'train_causal.py').write_text(train)
        runner = (PARENT / 'run_recursive.py').read_text().replace(str(PARENT), str(root))
        runner = runner.replace("'support'", "'category'").replace("'control'", "'empty'")
        runner = change(runner, "    assert sha(ROOT/'text_development.pt') == training['text_development_sha256']",
            "    for arm in ['category', 'empty']:\n        assert sha(training['banks']['development'][arm]['path']) == training['banks']['development'][arm]['sha256']")
        runner = change(runner, "torch.load(ROOT/'text_development.pt', map_location='cpu')",
            "torch.load(training['banks']['development'][arm]['path'], map_location='cpu')")
        runner = runner.replace('support_pooled_', 'category_pooled_')
        runner = change(runner, "    spec = json.loads((ROOT/'recursive_spec.json').read_text())",
            "    spec = json.loads((ROOT/'recursive_spec.json').read_text())\n    assert sha(ROOT/'recursive_spec.json') == json.loads((ROOT/'frozen.json').read_text())['recursive_spec_sha256']")
        (root / 'run_recursive.py').write_text(runner)
        queue = (PARENT / 'run_m67.sh').read_text().replace(str(PARENT), str(root))
        queue = queue.replace('run_train control', 'run_train empty').replace('run_train support', 'run_train category')
        queue = queue.replace('run_eval control', 'run_eval empty').replace('run_eval support', 'run_eval category')
        (root / 'run_pair.sh').write_text(queue)
        (root / 'native_parity').mkdir()
        torch.manual_seed(seed); adapter = module.SemanticSpatialAdapter(null_support=True)
        assert sum(p.numel() for p in adapter.parameters()) == 289154
        assert torch.count_nonzero(adapter.delta[-1].weight) == 0 and torch.count_nonzero(adapter.delta[-1].bias) == 0
        initial = dict(architecture='semantic_spatial_support_v1', model=adapter.state_dict(),
            null_support=True, use_text=True, support_loss_weight=0., seed=seed,
            base_checkpoint_sha256=previous['native_checkpoint_sha256'], status='paired_zero_initialization')
        p = root / 'native_parity/category_zero.pth'; torch.save(initial, p)
        shutil.copyfile(p, root / 'native_parity/empty_zero.pth')
        keep = ['supervision_coordinate_convention', 'epochs', 'architecture', 'learned_parameters', 'visual_base_frozen_eval',
            'native_checkpoint', 'native_checkpoint_sha256', 'integration_sha256', 'causal_script_sha256', 'inventory_sha256',
            'dataset_root', 'development_sequences', 'sequence_order', 'total_training_image_frames', 'total_training_track_calls',
            'maximum_optimizer_steps', 'optimizer', 'learning_rate', 'weight_decay', 'gradient_accumulation_frames', 'gradient_clip',
            'loss', 'invalid_gt', 'target_centre_outside_crop', 'state_protocol', 'backprop_through_time_or_discrete_crops',
            'native_update_interval', 'native_update_threshold', 'caption_protocol', 'frozen_native_development_aggregate',
            'native_result_sha256', 'checkpoint_retention', 'optimizer_stop_conditions', 'support_loss_sha256', 'protected_prior_control_sequences']
        spec = {k: previous[k] for k in keep}
        gates = {k.replace('support_pooled_', 'category_pooled_'): v for k, v in previous['promotion_gates'].items()}
        spec.update(status='prepared_not_training_authorized', revision='m73_paired_lexical_replication_v1', seed=seed,
            primary_arm='category', control_arm='empty', support_loss_weights={'category': 0., 'empty': 0.},
            banks=bank_info, training_script_sha256=sha(root / 'train_causal.py'), run_queue_sha256=sha(root / 'run_pair.sh'),
            initial_checkpoint_sha256={a: sha(root / 'native_parity' / (a + '_zero.pth')) for a in ['category', 'empty']},
            promotion_gates=gates, parent_training_spec_sha256=sha(PARENT / 'training_spec.json'),
            hypothesis='Category-conditioned training improves complete causal recursion versus identical empty-content training across predeclared initializations.',
            architecture_control='Both use the exact M67 Control architecture, frozen visual base, no auxiliary loss, identical within-seed initialization and frame order. Only valid-slot text content differs.',
            selection_protocol='Fixed seed2027 category final checkpoint is the only proposed candidate; seed2028 is replication. No best-seed or early-checkpoint selection.',
            text_control='Both use_text=True and five original slots/mask/padding. Category uses original slot0 plus empty attributes. Empty replaces every valid slot with the same CLIP empty embedding. No new caption or embedding.',
            monitoring='Check initial coverage about 240 seconds after launch; later near estimated completion. Preserve errors and do not restart on observation timeout.',
            runtime_estimate_seconds_per_arm=13700, independent_review_pass=False,
            scope_limitations=['Two prospectively fixed seeds on reused Train development22 are not an unbiased benchmark.',
                'GPU arithmetic is not bitwise deterministic; paired initial tensors and data order are checked, not assumed trajectory identity after optimization.',
                'Empty text still uses the same trainable adapter. Native STTrack remains an additional reference.',
                'Original M65/M67 gates and diagnostic-only M72 result remain unchanged.'],
            after_training='Complete both seeds even if the first performance gate fails; runtime errors stop the queue. Both seed gates must pass before fixed-new-head lexical checks. No automatic public evaluation.',
            final_three_dataset_metrics_exist=False)
        write(root / 'prepared_training_spec.json', spec)
        rs = dict(status='prepared_not_training_authorized', runner_sha256=sha(root / 'run_recursive.py'),
            queue_sha256=sha(root / 'run_pair.sh'), metric_sha256=recursion['metric_sha256'],
            native_result_path=recursion['native_result_path'], variants=['empty', 'category'], cases=recursion['cases'],
            fixed_final_heads_only=True, full_prediction_families_sealed_before_metric_GT=True, poll_interval_seconds=240,
            public_evaluation=False)
        write(root / 'prepared_recursive_spec.json', rs)
        subprocess.run(['bash', '-n', str(root / 'run_pair.sh')], check=True)
        preparations[str(seed)] = dict(training_spec_sha256=sha(root / 'prepared_training_spec.json'),
            recursive_spec_sha256=sha(root / 'prepared_recursive_spec.json'), initial_sha256=sha(p))
    assert preparations['2027']['initial_sha256'] != preparations['2028']['initial_sha256']
    shell = '''#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_m73_paired_lexical_replication_20260907
python=/root/autodl-tmp/envs/sttrack/bin/python
cd "$root" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$python" -u /root/autodl-tmp/m73_paired_lexical_replication_20260907.py eligible > eligibility.log 2>&1
status=$?; printf '%s\\n' "$status" > eligibility.exit
if [ "$status" -ne 0 ]; then printf '1\\n' > controller.exit; exit 1; fi
for seed in 2027 2028; do
    printf '%s\\n' "$seed" > current_seed.txt
    bash "seed$seed/run_pair.sh" > "seed$seed/queue.log" 2>&1
    status=$?; printf '%s\\n' "$status" > "seed$seed.exit"
    if [ "$status" -ne 0 ]; then printf '1\\n' > controller.exit; exit 1; fi
done
"$python" -u /root/autodl-tmp/m73_paired_lexical_replication_20260907.py summarize > analysis.log 2>&1
status=$?; printf '%s\\n' "$status" > analysis.exit
printf '%s\\n' "$status" > controller.exit
exit "$status"
'''
    (ROOT / 'run_m73.sh').write_text(shell); subprocess.run(['bash', '-n', str(ROOT / 'run_m73.sh')], check=True)
    spec = dict(status='prepared_before_causal_checks', source_sha256=sha(__file__), observed_utc=now(), seeds=SEEDS,
        candidate_seed=2027, candidate_arm='category', compared_arms=['category', 'empty'], preparations=preparations,
        parent_M72_result_sha256=sha(content), banks=bank_info, seed_order=SEEDS,
        total_training_track_calls=4 * 186694, total_development_track_calls=4 * 33108,
        paired_training_only_variable='valid-slot lexical content; distinct seeds change adapter initialization, not data order',
        run_queue_sha256=sha(ROOT / 'run_m73.sh'), check_script_sha256=sha(B / 'check_m73_causal_20260907.py'),
        criteria='Both seeds individually pass inherited eleven development comparisons, then fixed-head category/empty/swapped controls. Never select the better seed.',
        public_evaluation_allowed=False, independent_model_review_pass=False)
    write(ROOT / 'spec.json', spec)
    print(json.dumps(dict(root=str(ROOT), spec_sha256=sha(ROOT / 'spec.json'), preparations=preparations)))


def freeze():
    s = read(ROOT / 'spec.json'); assert s['source_sha256'] == sha(__file__)
    assert not (ROOT / 'frozen.json').exists()
    proofs = {}
    for seed in SEEDS:
        r = ROOT / ('seed' + str(seed)); proof = read(r / 'causal_check.json')
        assert sha(r / 'prepared_training_spec.json') == s['preparations'][str(seed)]['training_spec_sha256']
        assert sha(r / 'prepared_recursive_spec.json') == s['preparations'][str(seed)]['recursive_spec_sha256']
        assert (r / 'causal_check.exit').read_text().strip() == '0'
        assert proof['status'] == 'completed_M73_native_parity_and_causal_smoke'
        assert proof['checker_sha256'] == s['check_script_sha256']
        assert proof['prepared_spec_sha256'] == sha(r / 'prepared_training_spec.json')
        spec = read(r / 'prepared_training_spec.json')
        assert proof['zero_residual_public_state_exact'] and proof['base_frozen_all_arms']
        assert proof['same_output_original_loss_exact'] and proof['formal_optimizer_steps'] == 0
        assert all(v['optimizer_steps'] == 3 and v['frames'] == 96 for v in proof['arms'].values())
        assert sha(r / 'train_causal.py') == spec['training_script_sha256']
        assert sha(r / 'run_pair.sh') == spec['run_queue_sha256']
        assert all(sha(r / 'native_parity' / (arm + '_zero.pth')) == spec['initial_checkpoint_sha256'][arm] for arm in ['category', 'empty'])
        spec.update(status='frozen_before_formal_training', causal_check_sha256=sha(r / 'causal_check.json'), observed_utc=now())
        write(r / 'training_spec.json', spec)
        rs = read(r / 'prepared_recursive_spec.json'); rs.update(status='frozen_before_formal_training', training_spec_sha256=sha(r / 'training_spec.json'))
        write(r / 'recursive_spec.json', rs)
        proof_binding = dict(training_spec_sha256=sha(r / 'training_spec.json'), recursive_spec_sha256=sha(r / 'recursive_spec.json'),
            check_sha256=sha(r / 'causal_check.json'), seed=seed)
        write(r / 'frozen.json', proof_binding); proofs[str(seed)] = proof_binding
    write(ROOT / 'frozen.json', dict(spec_sha256=sha(ROOT / 'spec.json'), seeds=proofs, observed_utc=now(),
        independently_reviewed=False, public_evaluation_allowed=False))
    print(json.dumps(read(ROOT / 'frozen.json')))


def eligible():
    s = read(ROOT / 'spec.json'); f = read(ROOT / 'frozen.json')
    assert s['source_sha256'] == sha(__file__) and f['spec_sha256'] == sha(ROOT / 'spec.json')
    assert sha(ROOT / 'run_m73.sh') == s['run_queue_sha256']
    for seed in SEEDS:
        r = ROOT / ('seed' + str(seed)); assert not (r / 'training').exists()
        assert read(r / 'frozen.json') == f['seeds'][str(seed)]
        for p, h in [('training_spec.json', 'training_spec_sha256'), ('recursive_spec.json', 'recursive_spec_sha256')]: assert sha(r / p) == f['seeds'][str(seed)][h]
        t = read(r / 'training_spec.json'); assert sha(t['native_checkpoint']) == t['native_checkpoint_sha256']
        for n, h in read(r / 'integration.json')['source_sha256'].items(): assert sha(r / 'code' / n) == h
        for split in ['fit', 'development']:
            for arm in ['category', 'empty']: assert sha(s['banks'][split][arm]['path']) == s['banks'][split][arm]['sha256']
    used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).decode().splitlines()
    assert len(used) == 2 and all(int(x) < 500 for x in used)
    assert shutil.disk_usage(str(B)).free > 1_000_000_000
    print(json.dumps(dict(status='frozen_training_eligible', gpu_memory_MiB=used, free_bytes=shutil.disk_usage(str(B)).free)))


def summarize():
    spec = read(ROOT / 'spec.json'); frozen = read(ROOT / 'frozen.json'); results = {}
    for seed in SEEDS:
        r = ROOT / ('seed' + str(seed)); assert (r / 'controller.exit').read_text().strip() == '0'
        j = read(r / 'recursive_result.json')
        assert j['training_spec_sha256'] == frozen['seeds'][str(seed)]['training_spec_sha256']
        assert j['recursive_spec_sha256'] == frozen['seeds'][str(seed)]['recursive_spec_sha256']
        results[str(seed)] = dict(result_sha256=sha(r / 'recursive_result.json'), aggregates=j['aggregates'], gates=j['gates'], primary_pass=j['primary_pass'])
    passed = all(v['primary_pass'] for v in results.values())
    result = dict(status='completed_two_seed_paired_lexical_replication', observed_utc=now(), spec_sha256=sha(ROOT / 'spec.json'),
        source_sha256=sha(__file__), results=results, both_seed_development_gates_pass=passed,
        candidate_seed=2027, candidate_head=str(ROOT / 'seed2027/training/category/final.pth'),
        fixed_head_content_preparation_allowed=passed, public_evaluation_allowed=False,
        independent_model_review_pass=False, goal_completed=False)
    write(ROOT / 'result.json', result); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['prepare', 'freeze', 'eligible', 'summarize'])
    {'prepare': prepare, 'freeze': freeze, 'eligible': eligible, 'summarize': summarize}[p.parse_args().action]()
