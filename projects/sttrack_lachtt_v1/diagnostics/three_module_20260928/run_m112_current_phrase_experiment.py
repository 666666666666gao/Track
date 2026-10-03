"""Two-card sanity/reference acceptance, followed by one fixed M112 training."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def launch(root, mode, gpu, manifest, labels):
    script = Path(__file__).with_name('train_m112_current_phrase_evidence.py')
    args = [sys.executable, '-u', str(script), '--mode', mode,
            '--cache', '/root/autodl-tmp/sttrack_m90_train_states_20260928',
            '--contexts', '/root/autodl-tmp/sttrack_m98_train_contexts_20260928',
            '--origins', '/root/autodl-tmp/sttrack_m95_initial_origins_20260928',
            '--bank', '/root/autodl-tmp/sttrack_m108_frozen_semantic_pair_20261002/text.pt',
            '--labels', str(labels), '--manifest', str(manifest),
            '--parent', '/root/autodl-tmp/sttrack_m101_ab_native_pair_20260928/train_weight1/final.pt',
            '--parent-result', '/root/autodl-tmp/sttrack_m101_ab_native_pair_20260928/train_weight1/result.json',
            '--control', '/root/autodl-tmp/sttrack_m111_category_pair_20261003/train_weight0/final.pt',
            '--control-result', '/root/autodl-tmp/sttrack_m111_category_pair_20261003/train_weight0/result.json',
            '--output', str(root / mode)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with (root / (mode + '.log')).open('w') as log:
        child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
    receipt = dict(mode=mode, gpu=gpu, pid=child.pid, args=args, started=time.time())
    (root / (mode + '_launch.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)
    return child, receipt


def finish(root, child, receipt):
    receipt.update(exit=child.wait(), finished=time.time())
    (root / (receipt['mode'] + '.exit')).write_text(str(receipt['exit']) + '\n')
    (root / (receipt['mode'] + '_launch.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    assert receipt['exit'] == 0, 'Inspect preserved log; no retry.'
    result = json.loads((root / receipt['mode'] / 'result.json').read_text())
    assert result['empty_all3039_scores_quality_exact'] and result['frozen_parameters_buffers_exact']
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ('output', 'manifest', 'labels'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(exist_ok=False); started = time.time()
    sanity_child, sanity_launch = launch(a.output, 'sanity', 0, a.manifest, a.labels)
    reference_child, reference_launch = launch(a.output, 'reference', 1, a.manifest, a.labels)
    # Join both before accepting either gate; no sibling is restarted on failure.
    codes = [sanity_child.wait(), reference_child.wait()]
    for code, launch_record in zip(codes, (sanity_launch, reference_launch)):
        launch_record.update(exit=code, finished=time.time())
        (a.output / (launch_record['mode'] + '.exit')).write_text(str(code) + '\n')
        (a.output / (launch_record['mode'] + '_launch.json')).write_text(json.dumps(launch_record, indent=2) + '\n')
    assert codes == [0, 0], 'Inspect preserved sanity/reference logs; no training or retry.'
    sanity = json.loads((a.output / 'sanity' / 'result.json').read_text())
    reference = json.loads((a.output / 'reference' / 'result.json').read_text())
    assert sanity['status'] == 'complete_M112_gpu_sanity' and sanity['optimizer_steps'] == 3
    assert reference['status'] == 'complete_M112_readonly_reference' and reference['optimizer_steps'] == 0
    assert reference['all_stored_control_selection_rows_reproduced']
    assert all(r['empty_all3039_scores_quality_exact'] and r['frozen_parameters_buffers_exact'] for r in (sanity, reference))
    train_child, train_launch = launch(a.output, 'train', 0, a.manifest, a.labels)
    trained = finish(a.output, train_child, train_launch)
    assert trained['status'] == 'complete_M112_current_phrase_training' and trained['optimizer_steps'] == 480
    result = dict(status='complete_M112_current_phrase_experiment', started=started, finished=time.time(),
                  elapsed_seconds=time.time() - started, launches=[sanity_launch, reference_launch, train_launch],
                  reused_control_not_retrained=True, reference_optimizer_steps=0,
                  semantic_development_examples=0, no_automatic_promotion=True,
                  no_recursive_or_public_evaluation=True,
                  reference=dict(phrase=reference['final_phrase_readout'], content=reference['content_conditions']),
                  trained=dict(phrase=trained['final_phrase_readout'], content=trained['content_conditions'],
                               weak_vs_own_empty=trained['weak_vs_own_empty'], final_sha256=trained['final_sha256']))
    (a.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    (a.output / 'driver.exit').write_text('0\n')
    print(json.dumps(dict(status=result['status'], elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
