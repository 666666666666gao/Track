"""Reconstruct the inherited template update condition from all sealed M58 scores."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

PARENT=Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v2_20260906')
AUDIT=Path('/root/autodl-tmp/sttrack_m58_completion_audit_v3_20260906')
OUT=Path('/root/autodl-tmp/sttrack_m59_fixed_head_content_diagnostic_20260906')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert sha(PARENT/'recursive_result.json')=='54fd8445297e1eef761fa36e8a17def01afa027f71371984998f08229f6f5ee9'
    audit=json.loads((AUDIT/'audit_result.json').read_text())
    spec=json.loads((PARENT/'recursive_spec.json').read_text())
    training=json.loads((PARENT/'training_spec.json').read_text())
    assert training['native_update_interval']==50 and training['native_update_threshold']==.75
    integration=json.loads((PARENT/'integration.json').read_text())
    for name in ['lib/test/tracker/sttrack.py','lib/test/tracker/sttrack_semantic.py']:
        assert sha(PARENT/'code'/name)==integration['source_sha256'][name]
    names=[c['sequence'] for c in spec['cases']]
    scores={'native':{n:[] for n in names},'text':{},'visual':{}}
    for path,digest in audit['baseline_trace_sha256'].items():
        data=Path(path).read_bytes();assert hashlib.sha256(data).hexdigest()==digest
        for row in json.loads(data)['rows']:
            if row['sequence'] in names:
                scores['native'][row['sequence']].append((row['frame_index'],row['public_score']))
    for arm in ['text','visual']:
        receipt_path=PARENT/(arm+'_recursive_receipt.json')
        receipt=json.loads(receipt_path.read_text())
        parent=json.loads((PARENT/'recursive_result.json').read_text())
        assert sha(receipt_path)==parent['receipts'][arm]
        for item in receipt['sequences']:
            path=PARENT/'recursive'/arm/(item['sequence']+'.json')
            data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==item['sha256']
            scores[arm][item['sequence']]=[(r['frame'],r['score']) for r in json.loads(data)['rows']]
    out={}
    for arm,seqs in scores.items():
        per={};all_scores=[];checks=0
        for name,rows in seqs.items():
            rows.sort(key=lambda r:r[0])
            case=next(c for c in spec['cases'] if c['sequence']==name)
            assert [i for i,s in rows]==list(range(case['frames']))
            values=np.asarray([s for i,s in rows if i>0]);assert np.isfinite(values).all()
            points=[(i,s) for i,s in rows if i>0 and i%50==0]
            selected=[i for i,s in points if s>.75]
            per[name]=dict(scheduled_checks=len(points),reconstructed_template_writes=len(selected),
                write_frames=selected,score_median=float(np.median(values)))
            checks+=len(points);all_scores.extend(values.tolist())
        out[arm]=dict(scheduled_checks=checks,reconstructed_template_writes=sum(v['reconstructed_template_writes'] for v in per.values()),
            score_median=float(np.median(all_scores)),score_q25=float(np.quantile(all_scores,.25)),
            score_q75=float(np.quantile(all_scores,.75)),per_sequence=per)
    result=dict(scope='Posthoc reconstruction from sealed scores and inherited STTrack condition frame_id%50==0 and raw Hann maximum>0.75. No new model/GT calls, no threshold changes; not a causal intervention.',
        parent_result_sha256=audit['parent_result_sha256'],auditor_sha256=sha(__file__),
        observed_utc=datetime.now(timezone.utc).isoformat(),
        aggregates={k:{n:v for n,v in x.items() if n!='per_sequence'} for k,x in out.items()},
        per_sequence={k:x['per_sequence'] for k,x in out.items()})
    previous=json.loads((AUDIT/'template_update_reconstruction.json').read_text())
    assert result['aggregates']==previous['aggregates'] and result['per_sequence']==previous['per_sequence']
    (OUT/'m58_template_update_reconstruction.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['aggregates'],indent=2))


if __name__=='__main__':
    main()
