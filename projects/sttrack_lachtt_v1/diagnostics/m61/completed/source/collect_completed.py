"""Recompute M61 event evidence and export completed M61/M62 receipts."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tarfile
import numpy as np
import torch

BASE = Path('/root/autodl-tmp')
M61 = BASE / 'sttrack_m61_fixed_state_language_20260907'
M62 = BASE / 'sttrack_m62_learned_entry_parity_20260907'
OUT = BASE / 'sttrack_m61_m62_completed_export_20260907'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    p.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def iou(box, gt):
    x, y, w, h = map(float, box); gx, gy, gw, gh = map(float, gt)
    intersection = max(0., min(x+w, gx+gw)-max(x, gx)) * max(0., min(y+h, gy+gh)-max(y, gy))
    return intersection / (w*h + gw*gh - intersection)


def main():
    assert sha(M61/'result.json') == 'dafe3d3fd3d1667e2702fff179345025efe5c741c586b93c520b818174bfaa6a'
    assert sha(M62/'result.json') == '731bf649920f91087f242442f384b6611c4b7a7d42186aaedb1dc4bb67325d22'
    for root, exits in [(M61, ['collect_original','collect_category','analysis','job']), (M62,['ope','trax','verification','job'])]:
        for name in exits: assert (root/(name+'.exit')).read_text().strip() == '0'
    source = load_module('m61_collection', BASE/'m61_fixed_state_language_20260907.py')
    _, spec, training, _ = source.checked()
    sys.path.insert(0, str(BASE/'sttrack_m58_semantic_spatial_v2_20260906/code'))
    from lib.test.utils.hann import hann2d
    window = hann2d(torch.tensor([16,16]).long(), centered=True).reshape(-1).numpy()
    result = json.loads((M61/'result.json').read_text())
    paired = json.loads((M61/'paired_event_analysis.json').read_text())
    gt = {}
    for c in spec['cases']:
        p = Path(training['dataset_root'])/c['sequence']/'groundtruth.txt'
        assert sha(p) == c['gt_sha256']
        gt[c['sequence']] = np.loadtxt(p, delimiter=',').reshape(-1,4)
    totals = {}; max_score_error = 0.; max_iou_error = 0.; max_parity_bbox = 0.; max_parity_score = 0.
    maps_checked = 0
    for ref in spec['references']:
        folder=M61/('collect_'+ref); receipt=json.loads((folder/'receipt.json').read_text())
        assert sha(folder/'receipt.json') == result['receipt_sha256'][ref]
        assert receipt['frames']==33130 and receipt['events']==654 and len(receipt['rows'])==22
        assert receipt['all_cloned_inputs_unchanged'] and not receipt['diagnostic_outputs_committed']
        linked={(r['sequence'],r['frame']):r for r in paired[ref]}
        sums={name:dict(events=0, valid=0, iou_sum=0., correct=0, low=0, peak_changes=0, write_eligible=0, flips=0, same_peak_flips=0) for name in spec['conditions']}
        for row in receipt['rows']:
            name=row['sequence']; p=folder/(name+'.json'); q=folder/(name+'.npz')
            assert sha(p)==row['event_sha256'] and sha(q)==row['maps_sha256']
            events=json.loads(p.read_text())['events']
            assert [r['frame'] for r in events] == list(range(50,row['frames'],50))
            max_parity_bbox=max(max_parity_bbox,row['max_bbox_error']); max_parity_score=max(max_parity_score,row['max_score_error'])
            with np.load(q) as maps:
                for key,ch in [('score_map',1),('size_map',2),('offset_map',2)]:
                    assert maps[key].shape==(len(events),5,ch,16,16) and np.isfinite(maps[key]).all()
                for j,event in enumerate(events):
                    target=gt[name][event['frame']]
                    valid=bool(np.isfinite(target).all() and (target[2:]>0).all())
                    link=linked[(name,event['frame'])]; assert link['valid_GT']==valid
                    reference=event['conditions'][ref]
                    assert reference['write_eligible']==event['actual_template_write']
                    for k,condition in enumerate(spec['conditions']):
                        r=event['conditions'][condition]; a=sums[condition]
                        response=maps['score_map'][j,k].reshape(-1)*window
                        peak=int(response.argmax()); score=float(response[peak]); fixed=float(response[event['reference_peak']])
                        assert peak==r['peak'] and abs(score-r['score'])<1e-7 and abs(fixed-r['score_at_reference_peak'])<1e-7
                        max_score_error=max(max_score_error,abs(score-r['score']),abs(fixed-r['score_at_reference_peak']))
                        assert (score>.75)==r['write_eligible']
                        a['events']+=1; a['valid']+=valid; a['peak_changes']+=peak!=reference['peak']; a['write_eligible']+=score>.75
                        flip=(score>.75)!=reference['write_eligible']; a['flips']+=flip; a['same_peak_flips']+=flip and peak==reference['peak']
                        if valid:
                            overlap=iou(r['bbox'],target); fixed_iou=iou(r['fixed_reference_peak_bbox'],target)
                            linked_row=link['conditions'][condition]
                            max_iou_error=max(max_iou_error,abs(overlap-linked_row['iou']),abs(fixed_iou-linked_row['fixed_peak_iou']))
                            assert abs(overlap-linked_row['iou'])<1e-12 and abs(fixed_iou-linked_row['fixed_peak_iou'])<1e-12
                            a['iou_sum']+=overlap; a['correct']+=overlap>=.5; a['low']+=overlap<=.1
                        maps_checked+=1
        for name,a in sums.items():
            expected=result['summaries'][ref][name]
            for actual,key in [('events','events'),('valid','valid_GT_events'),('correct','correct_events'),('low','low_events'),('peak_changes','peak_changes'),('write_eligible','write_eligible'),('flips','write_eligibility_flips'),('same_peak_flips','same_peak_write_flips')]:
                assert a[actual]==expected[key],(ref,name,key)
            assert abs(a['iou_sum']/a['valid']-expected['mean_event_iou'])<1e-12
        totals[ref]=sums
    assert maps_checked==6540 and max_parity_bbox<=1e-4 and max_parity_score<=1e-6
    # Existing verifier rereads actual OPE files and real TraX session receipts.
    m62=load_module('m62_entry',BASE/'m62_learned_entry_parity_20260907.py')
    entry_spec, _ = m62.checked()
    r62=json.loads((M62/'result.json').read_text())
    for name,key in [('ope','OPE_receipt_sha256'),('trax','TraX_receipt_sha256')]: assert sha(M62/name/'receipt.json')==r62[key]
    for session in r62['TraX_sessions']:
        assert session['tracking_reports']==201 and session['exit_code']==0
    OUT.mkdir()
    audit=dict(status='completed_event_maps_and_receipt_checks_verified',observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),m61_result_sha256=sha(M61/'result.json'),m62_result_sha256=sha(M62/'result.json'),
        raw_head_outputs_recomputed=maps_checked,reference_events=1308,GT_valid_events_per_reference=572,
        raw_head_peak_and_score_max_error=max_score_error,event_IoU_max_error=max_iou_error,
        all_frame_reference_bbox_max_error=max_parity_bbox,all_frame_reference_score_max_error=max_parity_score,
        paired_event_analysis_sha256=sha(M61/'paired_event_analysis.json'),recomputed=totals,
        M62_source_and_bundle_rechecked=True,M62_completed_receipts_rechecked=True,
        M62_original_verifier_not_rerun_to_preserve_result=True,independent_review_pass=False,
        scope='Executor deterministic evidence checks; not an independent model review. Current-event IoU and uncommitted head maps are not formal tracking scores.')
    write(OUT/'audit_result.json',audit)
    for tag,root in [('m61',M61),('m62',M62)]:
        dest=OUT/tag; dest.mkdir()
        names=['result.json','spec.json']+(['analysis_plan.json','paired_event_analysis.json','analysis.log','analysis.exit','collect_original.log','collect_original.exit','collect_category.log','collect_category.exit','job.exit','run_full.sh'] if tag=='m61' else ['ope.log','ope.exit','trax.log','trax.exit','verification.log','verification.exit','job.exit','run_entries.sh','ope_plan.json','trax_plan.json'])
        for n in names:
            if n=='spec.json':
                data=json.loads((root/n).read_text()); data.pop('cases',None)
                write(dest/'spec_summary.json',dict(source_spec_sha256=sha(root/n),summary=data))
            else: shutil.copyfile(root/n,dest/n)
        receipts=['collect_original','collect_category'] if tag=='m61' else ['ope','trax']
        for n in receipts: shutil.copyfile(root/n/'receipt.json',dest/(n+'_receipt.json'))
        shutil.copyfile(BASE/('m61_fixed_state_language_20260907.py' if tag=='m61' else 'm62_learned_entry_parity_20260907.py'),dest/'run_diagnostic.py')
        if tag=='m61': shutil.copyfile(BASE/'analyze_m61_fixed_state_20260907.py',dest/'analyze_fixed_state.py')
        shutil.copyfile(OUT/'audit_result.json',dest/'audit_result.json')
        shutil.copyfile(Path(__file__),dest/'collect_completed.py')
        write(dest/'evidence_manifest.json',[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(dest.iterdir())])
    archive=OUT/'evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for tag in ['m61','m62']:
            for p in sorted((OUT/tag).iterdir()): tar.add(p,arcname=tag+'/'+p.name)
    print(json.dumps(dict(status=audit['status'],audit_sha256=sha(OUT/'audit_result.json'),archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size),indent=2))


if __name__=='__main__': main()
