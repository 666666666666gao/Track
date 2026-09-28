"""M102 two disjoint GPU shards with exact-binding sanity first; no retries."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def phase(root, mode):
    children = []
    for shard in (0, 1):
        stem = f'{mode}_shard{shard}'
        args = [sys.executable, '-u', str(Path(__file__).with_name('audit_single_candidate_vlm.py')),
                '--output', str(root/f'{stem}.json'), '--mode', mode, '--shard', str(shard)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), OMP_NUM_THREADS='4',
                   MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
        with (root/f'{stem}.log').open('w') as log:
            child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
        receipt = dict(pid=child.pid, args=args, gpu=shard, started=time.time(), stem=stem)
        (root/f'{stem}_launch.json').write_text(json.dumps(receipt, indent=2)+'\n')
        children.append((child, receipt))
    for child, receipt in children:
        receipt.update(exit=child.wait(), finished=time.time())
        (root/f"{receipt['stem']}.exit").write_text(str(receipt['exit'])+'\n')
    assert all(receipt['exit'] == 0 for _,receipt in children), 'Read preserved logs; no retry.'
    return [receipt for _,receipt in children]


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output; root.mkdir(exist_ok=False)
    started = time.time()
    sanity_launches = phase(root, 'sanity')
    for shard in (0, 1):
        result = read(root/f'sanity_shard{shard}.json')
        assert result['status'] == 'complete_single_candidate_sanity_only'
        assert result['summary']['events'] == 1 and result['shard'] == shard
        assert all(result['rows'][0]['exact_image_pair_swap'].values())
    full_launches = phase(root, 'full')
    results = [read(root/f'full_shard{shard}.json') for shard in (0, 1)]
    for shard,result in enumerate(results):
        assert result['status'] == 'complete_single_candidate_identity_readout_shard'
        assert result['summary']['events'] == 12 and result['shard'] == shard
    rows = sorted([row for result in results for row in result['rows']], key=lambda row:row['audit_id'])
    assert len(rows) == 24 and len({row['audit_id'] for row in rows}) == 24
    summary = {key:sum(result['summary'][key] for result in results) for key in
               ('events', 'initial_first_correct', 'current_first_correct', 'both_correct',
                'rank_consistent', 'rank_changed')}
    for order in ('initial_first', 'current_first'):
        for kind in ('ranks', 'token_choices'):
            counts = Counter()
            for result in results:
                counts.update(result['summary'][order+'_'+kind])
            summary[order+'_'+kind] = dict(counts)
    report = dict(status='complete_read_only_single_candidate_identity_probe', seed=2027,
                  summary=summary, rows=rows, sanity_launches=sanity_launches, full_launches=full_launches,
                  screening_gate=dict(both_order_correct_at_least20=summary['both_correct'] >= 20,
                                      rank_changes_at_most2=summary['rank_changed'] <= 2),
                  token_ids=results[0]['token_ids'], elapsed_seconds=time.time()-started,
                  no_teacher_label_or_tracker_state=True, no_automatic_semantic_training=True,
                  independent_human_review_still_required=True, no_public_benchmark_or_checkpoint=True,
                  protocol_changes_not_a_single_factor_causal_ablation=True)
    assert results[0]['token_ids'] == results[1]['token_ids']
    (root/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
