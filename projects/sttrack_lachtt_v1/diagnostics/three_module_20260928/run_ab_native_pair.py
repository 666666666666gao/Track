"""One M101 two-GPU fixed-budget pair, sanity first, no retries or promotion."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch


def read(path):
    return json.loads(path.read_text())


def run_phase(root, mode):
    children = []
    for weight in (0, 1):
        stem = f'{mode}_weight{weight}'
        args = [sys.executable, '-u', str(Path(__file__).with_name('train_ab_visual_control.py')),
                '--cache', '/root/autodl-tmp/sttrack_m90_train_states_20260928',
                '--contexts', '/root/autodl-tmp/sttrack_m98_train_contexts_20260928',
                '--origins', '/root/autodl-tmp/sttrack_m95_initial_origins_20260928',
                '--empty-bank', '/root/autodl-tmp/sttrack_full152_paired_20260925/text_full152.pt',
                '--output', str(root/stem), '--mode', mode, '--device', 'cuda',
                '--native-preservation-weight', str(weight)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(weight), OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        with (root/f'{stem}.log').open('w') as log:
            child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
        receipt = dict(pid=child.pid, args=args, gpu=weight, weight=weight,
                       started=time.time(), stem=stem)
        (root/f'{stem}_launch.json').write_text(json.dumps(receipt, indent=2)+'\n')
        children.append((child, receipt))
    for child, receipt in children:
        code = child.wait()
        (root/f"{receipt['stem']}.exit").write_text(str(code)+'\n')
        receipt.update(exit=code, finished=time.time())
    assert all(receipt['exit'] == 0 for _, receipt in children), 'Read the preserved child logs; no retry.'
    return [receipt for _, receipt in children]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(exist_ok=False)
    started = time.time()
    sanity_receipts = run_phase(root, 'gpu-sanity')
    for weight in (0, 1):
        sanity = read(root/f'gpu-sanity_weight{weight}/result.json')
        assert sanity['status'] == 'complete_visual_control_sanity_only'
        assert sanity['optimizer_steps'] == 2 and not sanity['development_evaluated']
        assert not sanity['checkpoint_saved'] and sanity['native_preservation_weight'] == weight
    final_receipts = run_phase(root, 'train')
    results = {str(weight):read(root/f'train_weight{weight}/result.json') for weight in (0, 1)}
    for weight, result in results.items():
        assert result['status'] == 'complete_visual_control'
        assert result['optimizer_steps'] == 480 and result['epochs'] == 12
        assert result['native_preservation_weight'] == int(weight)
    original = Path('/root/autodl-tmp/sttrack_m99_ab_visual_control_20260928/corrected_run/final_train')
    old_result = read(original/'result.json')
    old_state = torch.load(original/'final.pt', map_location='cpu')
    new_state = torch.load(root/'train_weight0/final.pt', map_location='cpu')
    assert old_state.keys() == new_state.keys()
    differences = [name for name in old_state if not torch.equal(old_state[name], new_state[name])]
    verification = dict(final_tensor_differences=differences,
                        development_rows_equal=results['0']['development_rows'] == old_result['development_rows'],
                        fit_summary_equal=results['0']['fit'] == old_result['fit'],
                        development_summary_equal=results['0']['development'] == old_result['development'])
    exact = not differences and all(verification[key] for key in
                                   ('development_rows_equal', 'fit_summary_equal', 'development_summary_equal'))
    report = dict(status='complete_fixed_state_preservation_pair', elapsed_seconds=time.time()-started,
                  sanity_launches=sanity_receipts, final_launches=final_receipts,
                  zero_weight_reproduction_exact=exact, reproduction=verification,
                  arms={weight:{key:result[key] for key in ('native_preservation_weight', 'final_weights_sha256',
                                                           'optimizer_steps', 'fit', 'development', 'fixed_state_capacity_checks')}
                        for weight,result in results.items()},
                  no_semantic_identity_labels=True, no_tracker_state_or_public_evaluation=True,
                  no_automatic_promotion=True)
    (root/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
