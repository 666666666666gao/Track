"""Count multiple IoU-valid box hypotheses in existing fitting inputs only."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import torch


source=Path('/root/autodl-tmp/sttrack_m44_candidate_set_v1_20260905')
output=Path('/root/autodl-tmp/sttrack_m47_correspondence_audit_v1_20260905')
spec=json.loads((source/'spec.json').read_text());repo=Path(spec['repository']);sys.path.insert(0,str(repo))
from tools.train_sttrack_m44 import sha,check_binding
from tools.train_sttrack_m42 import overlaps

torch.set_num_threads(1);check_binding(source,spec)
assert sha(source/'training_labels.json')==spec['labels_sha256']
labels=json.loads((source/'training_labels.json').read_text())
cases={r['sequence'] for r in json.loads((source/'inference_inputs.json').read_text()) if r['split']=='fit'}
assert len(cases)==63
receipts=[x for shard in [0,1] for x in json.loads((source/f'shard{shard}_receipt.json').read_text())['sequences'] if x['sequence'] in cases]
assert len(receipts)==63
rows=[];histograms={side:Counter() for side in ['current','previous']};sequence_stats={}
for receipt in sorted(receipts,key=lambda r:r['sequence']):
    path=source/'features'/(receipt['sequence']+'.pt');assert sha(path)==receipt['feature_sha256'];data=torch.load(path,map_location='cpu')
    assert data['fold'] in spec['fit_folds']
    seqrows=[]
    for i,record in enumerate(data['records']):
        label=labels[record['key']];assert label['sequence'] in cases and label['fold'] in spec['fit_folds']
        counts={};valid_indices={}
        for side,boxkey,publickey in [('current','current_boxes','public_bbox'),('previous','previous_boxes','previous_public_bbox')]:
            if label[side] is None:values=torch.zeros(10)
            else:
                gt=torch.tensor(label[side]);values=overlaps(data[boxkey][i],gt);values[0]=overlaps(data[publickey][i],gt)
            indices=(values>=.5).nonzero().flatten().tolist();counts[side]=len(indices);valid_indices[side]=indices;histograms[side][len(indices)]+=1
        row=dict(key=record['key'],valid_counts=counts,valid_indices=valid_indices);rows.append(row);seqrows.append(row)
    sequence_stats[data['sequence']]=dict(events=len(seqrows),current_multiple=sum(r['valid_counts']['current']>1 for r in seqrows),previous_multiple=sum(r['valid_counts']['previous']>1 for r in seqrows))
assert len(rows)==1511
both=[r for r in rows if r['valid_counts']['current'] and r['valid_counts']['previous']]
summary=dict(events=len(rows),sequences=63,current_nonempty=sum(r['valid_counts']['current']>0 for r in rows),
    current_multiple=sum(r['valid_counts']['current']>1 for r in rows),previous_nonempty=sum(r['valid_counts']['previous']>0 for r in rows),
    previous_multiple=sum(r['valid_counts']['previous']>1 for r in rows),both_nonempty=len(both),
    active_forward_rows_with_other_valid_previous_boxes=sum(r['valid_counts']['previous']>1 for r in both),
    active_reverse_rows_with_other_valid_current_boxes=sum(r['valid_counts']['current']>1 for r in both),
    current_extra_iou_valid_boxes=sum(max(0,r['valid_counts']['current']-1) for r in rows),
    current_multiple_sequences=sum(v['current_multiple']>0 for v in sequence_stats.values()))
result=dict(status='complete',summary=summary,histograms={k:dict(sorted(v.items())) for k,v in histograms.items()},sequence_stats=sequence_stats,
    source_spec_sha256=sha(source/'spec.json'),labels_sha256=sha(source/'training_labels.json'),audit_source_sha256=sha(__file__),
    scope='Existing63fitting sequences only, IoU>=.5 diagnostic definition. No new model inference/training or development/public performance.',
    interpretation='The10grid-NMS peaks are box hypotheses, not10verified physical instances. Multiple IoU-valid boxes do not make a single action label erroneous. This audits whether auxiliary correspondence should distinguish valid boxes of the target from a unique committed action.',
    rows=rows)
output.mkdir(exist_ok=True)
(output/'valid_candidate_multiplicity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status='complete',summary=summary,histograms=result['histograms'],audit_source_sha256=sha(__file__)),indent=2))
