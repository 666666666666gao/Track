"""Deterministic summaries of M109 stored batches, not an independent GPU replay."""
import argparse,json,math,statistics
from pathlib import Path


def stats(values):
    values=[x for x in values if x is not None]
    assert all(math.isfinite(x) for x in values)
    return dict(count=len(values),mean=statistics.mean(values),median=statistics.median(values),
                minimum=min(values),maximum=max(values)) if values else dict(count=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=json.loads((a.root/'result.json').read_text());assert root['status']=='complete_M109_two_gpu_readonly_diagnostic'
    assert (a.root/'controller.exit').read_text().strip()=='0'
    summary={}
    for arm in ['generic','weak_text']:
        r=root['arms'][arm]
        assert r==json.loads((a.root/arm/'result.json').read_text())
        assert (a.root/(arm+'.exit')).read_text().strip()=='0'
        assert r['model_state_exact'] and r['no_optimizer_or_training'] and r['no_development_model_evaluation']
        assert sum(r['pairing'][k] for k in ['preferred_higher_iou','preferred_lower_iou','equal_iou'])==11
        summary[arm]=dict(pairing=r['pairing'],snapshots={})
        for snap,rows in r['batches'].items():
            assert len(rows)==40 and [row['batch'] for row in rows]==list(range(40))
            assert sum(row['count'] for row in rows)==2544 and sum(row['weak_pairs'] for row in rows)==11
            assert all(math.isfinite(x) for row in rows for x in row['losses']+row['gradient_norms'])
            paired=[row for row in rows if row['weak_pairs']]
            summary[arm]['snapshots'][snap]=dict(total_batches=40,paired_batches=len(paired),
                paired_localization_norm=stats([row['gradient_norms'][0] for row in paired]),
                paired_preservation_norm=stats([row['gradient_norms'][1] for row in paired]),
                paired_weak_norm=stats([row['gradient_norms'][2] for row in paired]),
                weak_to_baseline_norm=stats([row['weak_to_baseline_norm'] for row in paired]),
                weak_vs_baseline_cosine=stats([row['weak_vs_baseline_cosine'] for row in paired]),
                weak_vs_localization_cosine=stats([row['weak_vs_localization_cosine'] for row in paired]),
                weak_vs_preservation_cosine=stats([row['weak_vs_preservation_cosine'] for row in paired]),
                opposing_weak_vs_baseline=sum(row['weak_vs_baseline_cosine'] is not None and row['weak_vs_baseline_cosine']<0 for row in paired),
                zero_preservation_batches=sum(row['gradient_norms'][1]==0 for row in rows),
                all_localization_vs_preservation_cosine=stats([row['localization_vs_preservation_cosine'] for row in rows]))
    result=dict(status='M109_stored_batch_summary_pass',arms=summary,
                scope='Fit-only initial/final raw gradient coordinates; Adam updates and historical intermediate gradients not reconstructed.')
    a.output.write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(result))


if __name__=='__main__':main()
