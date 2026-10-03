"""One preparation, two GPU sanity jobs, then two fixed-budget M111 finals."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from analyze_train_states import sha


def phase(root, mode, bank, labels, manifest, category_bank):
    children = []
    for gpu, weight in enumerate([0, 1]):
        arm = 'weight' + str(weight)
        args = [sys.executable, '-u', str(Path(__file__).with_name('train_m111_explicit_category_evidence.py')),
                '--cache', '/root/autodl-tmp/sttrack_m90_train_states_20260928',
                '--contexts', '/root/autodl-tmp/sttrack_m98_train_contexts_20260928',
                '--origins', '/root/autodl-tmp/sttrack_m95_initial_origins_20260928',
                '--bank', str(bank), '--labels', str(labels), '--manifest', str(manifest),
                '--category-bank', str(category_bank),
                '--parent', '/root/autodl-tmp/sttrack_m101_ab_native_pair_20260928/train_weight1/final.pt',
                '--parent-result', '/root/autodl-tmp/sttrack_m101_ab_native_pair_20260928/train_weight1/result.json',
                '--output', str(root / (mode + '_' + arm)), '--evidence-weight', str(weight), '--mode', mode]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        with (root / (mode + '_' + arm + '.log')).open('w') as f:
            child = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, env=env)
        receipt = dict(pid=child.pid, gpu=gpu, evidence_weight=weight, arm=arm,
                       mode=mode, args=args, started=time.time())
        children.append((child, receipt)); print(json.dumps(receipt), flush=True)
    (root / (mode + '_launch.json')).write_text(json.dumps([r for _, r in children], indent=2) + '\n')
    for child, receipt in children:
        receipt.update(exit=child.wait(), finished=time.time())
        (root / (mode + '_' + receipt['arm'] + '.exit')).write_text(str(receipt['exit']) + '\n')
    assert all(r['exit'] == 0 for _, r in children), 'Inspect preserved logs; no retry.'
    return [r for _, r in children]


def main():
    p = argparse.ArgumentParser()
    for name in ['output', 'labels', 'manifest']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False); started = time.time()
    bank = Path('/root/autodl-tmp/sttrack_m108_frozen_semantic_pair_20261002/text.pt')
    assert sha(bank) == '8ff9887ceae2a26f36867824a3a04941fb076fc15b0e93575538f5374633de10'
    assert sha(a.labels) == '95144cc4c47987fb1960e2d8f63cb9fa41531a83c6249ea7950d0f54845887a7'
    manifest = json.loads(a.manifest.read_text())
    assert manifest['labels_sha256'] == sha(a.labels) and not manifest['human_confirmed']
    category_bank = a.output / 'category_bank.pt'
    args = [sys.executable, '-u', str(Path(__file__).with_name('prepare_m111_category_bank.py')),
            '--manifest', str(a.manifest), '--output', str(category_bank)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with (a.output / 'prepare.log').open('w') as f:
        prep_exit = subprocess.call(args, stdout=f, stderr=subprocess.STDOUT, env=env)
    (a.output / 'prepare.exit').write_text(str(prep_exit) + '\n')
    assert prep_exit == 0, 'Inspect preserved encoding log; no retry.'
    sanity = phase(a.output, 'sanity', bank, a.labels, a.manifest, category_bank)
    for r in sanity:
        x = json.loads((a.output / ('sanity_' + r['arm']) / 'result.json').read_text())
        assert x['status'] == 'complete_M111_gpu_sanity' and x['optimizer_steps'] == 3 and not x['checkpoint_saved']
    trained = phase(a.output, 'train', bank, a.labels, a.manifest, category_bank)
    results = {}
    for r in trained:
        x = json.loads((a.output / ('train_' + r['arm']) / 'result.json').read_text())
        assert x['status'] == 'complete_M111_explicit_category_arm' and x['optimizer_steps'] == 480
        results[r['arm']] = {k: x[k] for k in ['fit', 'content_conditions', 'fixed_state_checks',
                                              'final_sha256', 'empty_all3039_scores_quality_exact',
                                              'initial_category_readout', 'final_category_readout',
                                              'weak_vs_own_empty', 'category_example_draw_counts']}
    assert results['weight0']['category_example_draw_counts'] == results['weight1']['category_example_draw_counts']
    result = dict(status='complete_M111_explicit_category_pair', source_sha256=sha(__file__),
                  labels_sha256=sha(a.labels), manifest_sha256=sha(a.manifest),
                  bank_sha256=sha(bank), category_bank_sha256=sha(category_bank), human_confirmed=False,
                  started=started, finished=time.time(), elapsed_seconds=time.time() - started,
                  sanity_launches=sanity, train_launches=trained, arms=results,
                  no_automatic_promotion=True, no_recursive_or_public_evaluation=True,
                  no_claim_of_current_candidate_visibility=True)
    (a.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    (a.output / 'driver.exit').write_text('0\n')
    print(json.dumps(dict(status=result['status'], elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
