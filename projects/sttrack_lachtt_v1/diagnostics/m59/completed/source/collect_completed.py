"""Verify and export complete M59 results; never analyze partially completed controls."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import numpy as np

ROOT=Path('/root/autodl-tmp/sttrack_m59_fixed_head_content_diagnostic_20260906')
PARENT=Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v2_20260906')
OUT=Path('/root/autodl-tmp/sttrack_m59_completed_export_20260906')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    for name in ['worker0.exit','worker1.exit','analysis.exit','finalization.exit']:
        assert (ROOT/name).read_text().strip()=='0',name
    assert json.loads((ROOT/'finalization_result.json').read_text())['status']=='completed'
    sys.path.insert(0,str(ROOT))
    from run_controls import parent_ready
    spec,training,parent,training_results=parent_ready()
    result=json.loads((ROOT/'result.json').read_text())
    assert result['status']=='completed_train_only_same_head_content_diagnosis'
    assert result['content_spec_sha256']==sha(ROOT/'spec.json')
    assert result['bundle_sha256']==sha(ROOT/'bundle.json')
    assert result['parent_result_sha256']==sha(PARENT/'recursive_result.json')
    assert result['head_sha256']==training_results['text']['final_checkpoint_sha256']
    assert result['parent_promotion_pass'] is False and result['public_evaluation_allowed'] is False
    assert not parent['primary_pass']
    names=['original']+list(spec['controls'])
    template_counts={}
    per_sequence_templates={}
    total_frames={}
    first_score_differences={}
    original_scores={}
    for name in names:
        original=name=='original'
        receipt_path=PARENT/'text_recursive_receipt.json' if original else ROOT/(name+'_receipt.json')
        assert sha(receipt_path)==result['receipt_sha256'][name]
        receipt=json.loads(receipt_path.read_text())
        assert receipt['status']=='complete'
        selected=[c['sequence'] for c in spec['cases']] if original else spec['controls'][name]['sequences']
        assert [r['sequence'] for r in receipt['sequences']]==selected
        if original:
            assert receipt['head_sha256']==result['head_sha256']
        else:
            assert receipt['bundle_sha256']==result['bundle_sha256']
            assert receipt['bank_sha256']==spec['bank_sha256'][name]
            assert receipt['optimizer_steps']==0 and receipt['online_text_updates'] is False
        frames=0;per_sequence_templates[name]={}
        if not original:first_score_differences[name]={}
        for item in receipt['sequences']:
            folder=PARENT/'recursive/text' if original else ROOT/'predictions'/name
            path=folder/(item['sequence']+'.json')
            assert sha(path)==item['sha256']
            data=json.loads(path.read_text())
            rows=data['rows']
            assert data['sequence']==item['sequence']
            case=next(c for c in spec['cases'] if c['sequence']==item['sequence'])
            assert len(rows)==item['frames']==case['frames']
            assert [r['frame'] for r in rows]==list(range(case['frames']))
            assert rows[0]['bbox']==case['init_bbox']
            boxes=np.asarray([r['bbox'] for r in rows])
            scores=np.asarray([r['score'] for r in rows[1:]])
            assert np.isfinite(boxes).all() and (boxes[:,2:]>0).all() and np.isfinite(scores).all()
            frame_ids=np.arange(1,len(rows))
            writes=frame_ids[(frame_ids%50==0)&(scores>.75)]
            per_sequence_templates[name][item['sequence']]=dict(checks=int((frame_ids%50==0).sum()),
                reconstructed_writes=len(writes),write_frames=writes.tolist(),score_median=float(np.median(scores)))
            if original:
                original_scores[item['sequence']]=scores
            else:
                differences=np.flatnonzero(np.abs(scores-original_scores[item['sequence']])>1e-6)+1
                first_score_differences[name][item['sequence']]=dict(changed_score_frames=len(differences),
                    first_score_difference_frame=int(differences[0]) if len(differences) else None)
            frames+=len(rows)
        total_frames[name]=frames
        if not original:assert frames==spec['controls'][name]['frames']
        metrics=result['per_sequence'][name]
        assert set(metrics)==set(selected)
        current=result['aggregates'][name]
        for key in ['valid_frames','iou_sum','low_iou_frames','failure_episodes']:
            assert abs(current[key]-sum(r[key] for r in metrics.values()))<1e-8
        assert abs(current['mean_iou']-current['iou_sum']/current['valid_frames'])<1e-12
        assert abs(current['macro_sequence_mean_iou']-np.mean([r['mean_iou'] for r in metrics.values()]))<1e-12
        template_counts[name]={key:sum(r[key] for r in per_sequence_templates[name].values()) for key in ['checks','reconstructed_writes']}
    assert total_frames==dict(original=33130,empty=33130,category=33130,mismatch=33130,attributes=20392)
    for key in ['valid_frames','iou_sum','low_iou_frames','failure_episodes','mean_iou','macro_sequence_mean_iou']:
        assert abs(result['aggregates']['original'][key]-parent['aggregates']['text'][key])<1e-8
    selected=spec['controls']['attributes']['sequences']
    template_counts['original_for_attributes']={key:sum(per_sequence_templates['original'][n][key] for n in selected) for key in ['checks','reconstructed_writes']}
    for name,comp in result['comparisons'].items():
        reference='original_for_attributes' if name=='attributes' else 'original'
        a=result['aggregates'][reference];b=result['aggregates'][name]
        assert abs(comp['original_minus_control_mean_iou']-(a['mean_iou']-b['mean_iou']))<1e-12
        assert comp['original_minus_control_low_frames']==a['low_iou_frames']-b['low_iou_frames']
        assert comp['original_minus_control_H10']==a['failure_episodes']-b['failure_episodes']
    OUT.mkdir()
    audit=dict(status='complete_same_head_content_artifacts_verified',observed_utc=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=sha(__file__),result_sha256=sha(ROOT/'result.json'),content_spec_sha256=sha(ROOT/'spec.json'),
        head_sha256=result['head_sha256'],bundle_sha256=result['bundle_sha256'],receipt_sha256=result['receipt_sha256'],
        frames=total_frames,new_control_frames=119782,new_control_track_calls=119702,
        complete_control_trajectories=80,reference_trajectories=22,
        all_prediction_sha256_and_structure_verified=True,aggregate_arithmetic_verified=True,
        template_counts=template_counts,per_sequence_template_counts=per_sequence_templates,
        first_score_differences=first_score_differences,
        source_and_bank_hashes_verified=True,subsequent_gt_opened_by_this_auditor=False,
        new_optimizer_steps=0,new_caption_calls=0,parent_promotion_pass=False,public_evaluation_allowed=False,
        independent_review_pass=False,
        scope='Full M59 data after all workers and original analyzer finish; original_for_attributes uses the same14. Update counts reconstructed from inherited runtime conditions; direct semantic and recursive effects are not separated.')
    write(OUT/'audit_result.json',audit)
    files={name:ROOT/name for name in ['result.json','worker0.exit','worker1.exit','worker0.log','worker1.log','analysis.exit','analysis.log','finalization.exit','finalization.log','finalization_result.json','launch.json','prefix_parity_result.json']}
    for name in spec['controls']:files[name+'_receipt.json']=ROOT/(name+'_receipt.json')
    files['original_receipt.json']=PARENT/'text_recursive_receipt.json'
    files['audit_result.json']=OUT/'audit_result.json'
    files['collect_completed.py']=Path(__file__)
    snapshot=OUT/'evidence';snapshot.mkdir()
    for name,path in files.items():shutil.copyfile(path,snapshot/name)
    manifest=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(snapshot.iterdir()) if p.is_file()]
    write(snapshot/'evidence_manifest.json',manifest)
    archive=OUT/'evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for path in sorted(snapshot.iterdir()):tar.add(path,arcname=path.name)
    print(json.dumps(dict(status=audit['status'],archive=str(archive),bytes=archive.stat().st_size,
        archive_sha256=sha(archive),audit_sha256=sha(OUT/'audit_result.json'),
        aggregates=result['aggregates'],comparisons=result['comparisons'],template_counts=template_counts),indent=2))


if __name__=='__main__':
    main()
