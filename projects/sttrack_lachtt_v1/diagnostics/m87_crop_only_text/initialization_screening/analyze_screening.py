from pathlib import Path
from collections import Counter
import csv, hashlib, json

R=Path(__file__).resolve().parent
W=R.parent
OLD=W.parent/'sttrack_m86_initialization_audit_20260921'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sealed=read(R/'sealed_new_labels.json')
assert sha(R/'sealed_new_labels.json')=='14f6a8d146de3bc4af6200828f7698dc410b163d5dacdb08f2bb72a29bb13c91'
assert sealed['status']=='sealed_before_joining_M86_verdicts'
assert sha(W/'prepared_evidence/captions/records.jsonl')==sealed['caption_records_sha256']
old=read(OLD/'reviewed_register.json')
new=[json.loads(x) for x in (W/'prepared_evidence/captions/records.jsonl').read_text(encoding='utf-8').splitlines()]
labels=sealed['labels']
assert len(old)==len(new)==len(labels)==152
states=['supported','conflicting','uncertain']
rows=[]
for a,b,c in zip(old,new,labels):
    assert a['sequence']==b['sequence'] and a['audit_id']==c['audit_id']
    assert a['image_sha256']==b['image_sha256']
    assert a['generator_target_xyxy']==b['target_xyxy']
    assert a['category_support'] in states and c['verdict'] in states
    rows.append(dict(audit_id=c['audit_id'],sequence=b['sequence'],split=b['split'],
        old_category=a['parsed']['category'],new_category=b['category'],
        category_changed=a['parsed']['category']!=b['category'],
        old_screening=a['category_support'],new_screening=c['verdict'],
        new_visual_reason=c['reason'],old_visual_reason=a['visual_evidence'],
        image_sha256=b['image_sha256'],crop_sha256=a['crop_sha256']))
counts=lambda rs,key:{s:sum(r[key]==s for r in rs) for s in states}
matrix=lambda rs:{a:{b:sum(r['old_screening']==a and r['new_screening']==b for r in rs) for b in states} for a in states}
summary=dict(status='completed_initialization_only_assistant_screening',rows=len(rows),
    labels_sha256=sha(R/'sealed_new_labels.json'),screening_plan_sha256=sha(R/'SCREENING_PLAN.md'),
    caption_records_sha256=sealed['caption_records_sha256'],old_register_sha256=sha(OLD/'reviewed_register.json'),
    old_counts=counts(rows,'old_screening'),new_counts=counts(rows,'new_screening'),
    by_split={split:dict(rows=sum(r['split']==split for r in rows),old=counts([r for r in rows if r['split']==split],'old_screening'),new=counts([r for r in rows if r['split']==split],'new_screening')) for split in sorted({r['split'] for r in rows})},
    changed_category_rows=sum(r['category_changed'] for r in rows),
    transitions_all=matrix(rows),transitions_changed=matrix([r for r in rows if r['category_changed']]),
    unchanged_category_judgment_disagreements=[r for r in rows if not r['category_changed'] and r['old_screening']!=r['new_screening']],
    object_category_rows=[r['audit_id'] for r in rows if r['new_category'].strip().lower()=='object'],
    new_conflicts=[r['audit_id'] for r in rows if r['new_screening']=='conflicting'],
    limits=['assistant screening, not independent blind human truth',
            'same first RGB observations only; no subsequent frame or tracking metric used',
            'judgments are not training labels and do not modify frozen M87',
            'cross-pass disagreements for identical strings are measurement limitations, not generator changes',
            'crop-only images and category-only prompt changed together',
            'counts are not empirical caption accuracy or tracking performance'])
(R/'paired_register.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(R/'screening_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (R/'paired_register.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(summary,ensure_ascii=True,indent=2))
