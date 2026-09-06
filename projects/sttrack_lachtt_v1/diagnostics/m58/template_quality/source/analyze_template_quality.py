"""Posthoc quality of M58's unchanged default update condition; no threshold fitting."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

PARENT=Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v2_20260906')
AUDIT=Path('/root/autodl-tmp/sttrack_m58_completion_audit_v3_20260906')
M59=Path('/root/autodl-tmp/sttrack_m59_fixed_head_content_diagnostic_20260906')
OUT=Path('/root/autodl-tmp/sttrack_m58_template_quality_20260906')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quality(mask, valid, iou):
    return dict(total=int(mask.sum()),invalid_gt=int((mask&~valid).sum()),
        correct_iou_ge_05=int((mask&valid&(iou>=.5)).sum()),
        partial_iou_01_to_05=int((mask&valid&(iou>.1)&(iou<.5)).sum()),
        low_iou_le_01=int((mask&valid&(iou<=.1)).sum()))


def main():
    assert sha(PARENT/'recursive_result.json')=='54fd8445297e1eef761fa36e8a17def01afa027f71371984998f08229f6f5ee9'
    result=json.loads((PARENT/'recursive_result.json').read_text())
    assert not result['primary_pass']
    spec=json.loads((PARENT/'recursive_spec.json').read_text())
    training=json.loads((PARENT/'training_spec.json').read_text())
    assert training['native_update_interval']==50 and training['native_update_threshold']==.75
    assert sha('/root/autodl-tmp/audit_m58_completion_v3_20260906.py')=='5181b20f456e3150d714528d229d0fecc53ce1b9310d61b55f7e4ecda14fdae6'
    sys.path.insert(0,'/root/autodl-tmp')
    from audit_m58_completion_v3_20260906 import values
    names=[c['sequence'] for c in spec['cases']]
    rows={'native':defaultdict(list),'text':{},'visual':{}}
    baseline=json.loads((AUDIT/'audit_result.json').read_text())['baseline_trace_sha256']
    for path,digest in baseline.items():
        data=Path(path).read_bytes();assert hashlib.sha256(data).hexdigest()==digest
        for row in json.loads(data)['rows']:
            if row['sequence'] in names:
                rows['native'][row['sequence']].append(dict(frame=row['frame_index'],bbox=row['public_bbox'],score=row['public_score']))
    for arm in ['text','visual']:
        receipt_path=PARENT/(arm+'_recursive_receipt.json')
        assert sha(receipt_path)==result['receipts'][arm]
        receipt=json.loads(receipt_path.read_text())
        assert receipt['total_frames']==33130 and receipt['status']=='complete'
        for item in receipt['sequences']:
            path=PARENT/'recursive'/arm/(item['sequence']+'.json')
            assert sha(path)==item['sha256']
            rows[arm][item['sequence']]=json.loads(path.read_text())['rows']
    # Every compared trajectory family is sealed before reading subsequent GT.
    per={a:{} for a in rows}
    pair_per=[]
    for case in spec['cases']:
        name=case['sequence']
        path=Path(training['dataset_root'])/name/'groundtruth.txt'
        assert sha(path)==case['gt_sha256']
        gt=np.loadtxt(path,delimiter=',').reshape(-1,4)
        assert len(gt)==case['frames']
        masks={}
        for arm in rows:
            current=sorted(rows[arm][name],key=lambda r:r['frame'])
            assert [r['frame'] for r in current]==list(range(case['frames']))
            score=np.asarray([np.nan if r['frame']==0 else r['score'] for r in current])
            assert np.isfinite(score[1:]).all()
            valid,iou=values(np.asarray([r['bbox'] for r in current]),gt)
            checks=np.arange(len(gt))%50==0;checks[0]=False
            writes=checks&(score>.75)
            skips=checks&~writes
            all_after=np.arange(len(gt))>0
            per[arm][name]=dict(checks=quality(checks,valid,iou),writes=quality(writes,valid,iou),
                skipped=quality(skips,valid,iou),
                correct_frame_count=int((valid&(iou>=.5)).sum()),
                correct_frame_score_median=float(np.median(score[valid&(iou>=.5)])),
                confidence_above_default=quality(all_after&(score>.75),valid,iou))
            masks[arm]=dict(writes=writes,skips=skips,valid=valid,iou=iou)
        for comparison in ['native','visual']:
            text=masks['text'];other=masks[comparison]
            at_other_write_only=other['writes']&text['skips']
            both_correct=at_other_write_only&text['valid']&(text['iou']>=.5)&(other['iou']>=.5)
            text_correct_other_low=at_other_write_only&text['valid']&(text['iou']>=.5)&(other['iou']<=.1)
            pair_per.append(dict(sequence=name,comparison=comparison,
                counterpart_writes_text_skips=int(at_other_write_only.sum()),
                both_current_boxes_correct=int(both_correct.sum()),
                text_current_correct_counterpart_current_low=int(text_correct_other_low.sum()),
                diagnostic_only_no_future_template_utility_known=True))
    aggregates={}
    for arm,seqs in per.items():
        aggregates[arm]={group:{key:sum(s[group][key] for s in seqs.values()) for key in next(iter(seqs.values()))[group]}
            for group in ['checks','writes','skipped','confidence_above_default']}
    previous=json.loads((M59/'m58_template_update_reconstruction.json').read_text())
    for arm in aggregates:
        assert aggregates[arm]['writes']['total']==previous['aggregates'][arm]['reconstructed_template_writes']
        for group,stats in aggregates[arm].items():
            assert sum(v for k,v in stats.items() if k!='total')==stats['total']
    comparisons={}
    for arm in ['native','visual']:
        group=[r for r in pair_per if r['comparison']==arm]
        comparisons[arm]={key:sum(r[key] for r in group) for key in ['counterpart_writes_text_skips','both_current_boxes_correct','text_current_correct_counterpart_current_low']}
    OUT.mkdir()
    final=dict(status='completed_posthoc_template_condition_quality',observed_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(__file__),parent_result_sha256=sha(PARENT/'recursive_result.json'),
        reconstruction_sha256=sha(M59/'m58_template_update_reconstruction.json'),
        scope='One seed, reused Train development22. Current-box quality at default update times; no causal claim about future template utility.',
        thresholds=dict(correct_iou=.5,low_iou=.1,update_score=.75,interval=50),
        aggregates=aggregates,comparisons=comparisons,per_sequence=per,paired_per_sequence=pair_per,
        new_model_calls=0,new_training_steps=0,threshold_fitting=False,M59_predictions_or_GT_analyzed=False,
        independent_review_pass=False)
    (OUT/'result.json').write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in final.items() if k not in ['per_sequence','paired_per_sequence']},indent=2))


if __name__=='__main__':
    main()
