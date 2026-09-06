"""Export complete audited M63 evidence without raw predictions or labels."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,shutil,tarfile

BASE=Path('/root/autodl-tmp')
ROOT=BASE/'sttrack_m63_location_write_factorial_20260907'
OUT=ROOT/'completed_publication_evidence'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert sha(ROOT/'result.json')=='e0c437b8f08631161afbe212a529a06adc78e827d0b55d07160544e34d86a863'
    result=json.loads((ROOT/'result.json').read_text())
    audit=json.loads((ROOT/'completed_evidence_audit.json').read_text())
    assert audit['result_sha256']==sha(ROOT/'result.json')
    assert audit['auditor_sha256']==sha(BASE/'m63_evidence_audit_20260907.py')
    exits=['full_'+m+'.exit' for m in ['CC','SS','CS','SC']]+['analysis.exit','job.exit','completed_evidence_audit.exit']
    for n in exits:assert (ROOT/n).read_text().strip()=='0'
    summaries={};images=calls=0
    for mode in ['CC','SS','CS','SC']:
        receipt=json.loads((ROOT/('full_'+mode)/'receipt.json').read_text())
        assert sha(ROOT/('full_'+mode)/'receipt.json')==result['receipts'][mode]
        images+=receipt['frames'];calls+=receipt['frames']-len(receipt['sequences'])
    assert images==132520 and calls==132432
    for a,b in [('CS','CC'),('SC','SS')]:
        rows=[]
        for seq,x in result['per_sequence'][a].items():
            y=result['per_sequence'][b][seq]
            rows.append(dict(sequence=seq,delta_mean_iou=x['mean_iou']-y['mean_iou'],delta_low_frames=x['low_iou_frames']-y['low_iou_frames'],delta_H10=x['failure_episodes']-y['failure_episodes']))
        summaries[a+'_minus_'+b]=dict(better=sum(r['delta_mean_iou']>1e-12 for r in rows),worse=sum(r['delta_mean_iou']< -1e-12 for r in rows),equal=sum(abs(r['delta_mean_iou'])<=1e-12 for r in rows),per_sequence=rows)
    OUT.mkdir()
    names=['result.json','completed_evidence_audit.json','completed_evidence_audit.log','analysis.log']+exits
    for n in names:shutil.copyfile(ROOT/n,OUT/n)
    for mode in ['CC','SS','CS','SC']:shutil.copyfile(ROOT/('full_'+mode)/'receipt.json',OUT/('full_'+mode+'_receipt.json'))
    for p in [BASE/'m63_location_write_factorial_20260907.py',BASE/'m63_evidence_audit_20260907.py',Path(__file__)]:shutil.copyfile(p,OUT/p.name)
    summary=dict(status='completed_audited_factorial_export',observed_utc=datetime.now(timezone.utc).isoformat(),
        result_sha256=sha(ROOT/'result.json'),audit_sha256=sha(ROOT/'completed_evidence_audit.json'),
        source_sha256=sha(BASE/'m63_location_write_factorial_20260907.py'),full_image_frames=images,full_track_calls=calls,
        new_training_steps=0,new_captions=0,condition_changes=summaries,
        mean_iou_interaction_CC_minus_CS_minus_SC_plus_SS=result['aggregates']['CC']['mean_iou']-result['aggregates']['CS']['mean_iou']-result['aggregates']['SC']['mean_iou']+result['aggregates']['SS']['mean_iou'],
        public_evaluation_allowed=False,independent_model_review_pass=False,
        next='Keep the separately frozen M64 category protocol unchanged while its complete low22 evaluation runs. No donor-category deployment or threshold retuning. Use full results to decide subsequent Train-only learned modifications if needed.')
    (OUT/'completion_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'evidence_manifest.json').write_text(json.dumps([dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir())],indent=2)+'\n')
    archive=ROOT/'completed_publication_evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in sorted(OUT.iterdir()):t.add(p,arcname=p.name)
    print(json.dumps(dict(archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,result_sha256=summary['result_sha256'],audit_sha256=summary['audit_sha256'],full_image_frames=images,full_track_calls=calls),indent=2))


if __name__=='__main__':main()
