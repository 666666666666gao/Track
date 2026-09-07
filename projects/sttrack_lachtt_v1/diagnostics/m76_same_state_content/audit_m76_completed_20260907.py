"""CPU artifact audit; independently recompute geometry, overlap and peak statistics."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math

B = Path('/root/autodl-tmp')
R = B / 'sttrack_m76_same_state_content_20260907'
P = B / 'sttrack_m73_paired_lexical_replication_20260907/seed2027'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())

def overlap(a, b):
    w = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0]))
    h = max(0., min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    intersection = w*h
    return intersection/(a[2]*a[3]+b[2]*b[3]-intersection)

def near(a, b, tolerance=1e-11):
    assert abs(a-b) <= tolerance, (a, b, tolerance)

assert sha(R/'result.json') == 'a8150272c28093b9be9e444c27c0d5e8a50c2e13b81ca41ba0509562ed55d9fd'
assert sha(R/'spec.json') == '638930eb2722b1bee3c7ddd1c1c5579dfbad548341be283aa066c4f9328ca284'
s, receipt, result, evaluated = [read(R/p) for p in ['spec.json','receipt.json','result.json','evaluated_probes.json']]
assert sha(B/'m76_same_state_content_20260907.py') == s['source_sha256'] == receipt['source_sha256']
assert sha(R/'receipt.json') == result['receipt_sha256']
assert sha(R/'spec.json') == receipt['spec_sha256'] == result['spec_sha256']
assert sha(P/'training/category/final.pth') == s['head_sha256'] == receipt['head_sha256']
for n in ['replay.exit','analysis.exit','controller.exit']:
    assert (R/n).read_text().strip() == '0'
for bank in s['banks'].values():
    assert sha(bank['path']) == bank['sha256']
assert receipt['all_public_boxes_scores_exact'] and receipt['probe_state_unchanged']
assert not receipt['subsequent_GT_opened'] and not s['new_training_seeds']
assert s['seed'] == 2027 and not result['public_evaluation_allowed']
training = read(P/'training_spec.json')
ref_receipts = {v['sequence']:v for v in read(P/'category_recursive_receipt.json')['sequences']}
window = [(0.5*(1-math.cos(2*math.pi*(i+1)/17)))*(0.5*(1-math.cos(2*math.pi*(j+1)/17))) for i in range(16) for j in range(16)]
metrics, onset_scores, timeline, files = {}, [], {}, []
total_geometry, total_probes, decoded_boxes = 0, 0, 0
for case, sealed in zip(s['cases'], receipt['sequences']):
    seq = case['sequence']
    assert seq == sealed['sequence']
    path = R/'replay'/(seq+'.json')
    assert sha(path) == sealed['sha256']
    data = read(path)
    original = P/'recursive/category'/(seq+'.json')
    assert sha(original) == ref_receipts[seq]['sha256'] == data['reference_sha256']
    reference = read(original)['rows']
    gt_path = Path(training['dataset_root'])/seq/'groundtruth.txt'
    assert sha(gt_path) == case['gt_sha256']
    gt = [[float(v) for v in line.strip().split(',')] for line in gt_path.read_text().splitlines()]
    assert len(gt) == case['frames']
    valid = [all(math.isfinite(x) for x in g) and g[2]>0 and g[3]>0 for g in gt]
    start, end = s['capture_windows'][seq]
    expected = [i for i in range(1,case['stop_frame_exclusive']) if start<=i<end or i%50==0]
    assert [v['frame'] for v in data['probes']] == expected
    assert len(data['geometry']) == case['stop_frame_exclusive']-1
    assert len(data['probes']) == sealed['probe_frames'] == len(evaluated[seq])
    for i, row in enumerate(data['geometry'],1):
        prior = reference[i-1]['bbox']
        assert row['frame'] == i and row['previous_bbox'] == prior
        assert row['bbox'] == reference[i]['bbox'] and row['score'] == reference[i]['score']
        side = math.ceil(4*math.sqrt(prior[2]*prior[3]))
        assert row['search_rectangle'] == [round(prior[0]+prior[2]/2-side/2),round(prior[1]+prior[3]/2-side/2),side,side]
        assert row['template_write'] == (i%50==0 and row['score']>.75)
    stats = {name:dict(window_valid=0,category_low=0,category_good=0,hann_rescue=0,raw_rescue=0,raw_break_good=0,raw_severe_break_good=0,head_hann_severe_break_good=0,top10_capacity=0,dense_capacity=0,changed_peak=0,scheduled_write_difference=0) for name in s['heads']}
    timeline[seq] = []
    selected_case = next(v for v in s['selected_intervals'] if v['sequence']==seq)
    summary = dict(valid_window_probes=0,category_low_probes=0,category_correct_probes=0,category_low_center_inside=0,heads={name:dict(low_probe_top1_correct=0,low_probe_top10_correct=0,low_probe_dense_correct=0,correct_probe_severely_broken=0) for name in s['heads']})
    for probe, ev in zip(data['probes'], evaluated[seq]):
        i = probe['frame']; geometry = data['geometry'][i-1]
        assert i == ev['frame'] and valid[i] == ev['valid_GT']
        own_index = probe['heads']['category']['selected_index']
        own_iou = overlap(geometry['bbox'],gt[i]) if valid[i] else None
        in_window = start<=i<end and valid[i]
        low = in_window and own_iou<=.1
        good = in_window and own_iou>=.5
        if valid[i]:
            x,y,side,_ = geometry['search_rectangle'];g=gt[i]
            inside = x<=g[0]+g[2]/2<x+side and y<=g[1]+g[3]/2<y+side
            assert ev['center_inside'] == inside and ev['GT'] == g
        if in_window:
            summary['valid_window_probes']+=1;summary['category_low_probes']+=int(low);summary['category_correct_probes']+=int(good)
            summary['category_low_center_inside']+=int(low and inside)
        slim = dict(frame=i,valid_GT=valid[i],heads={})
        for name, head in probe['heads'].items():
            raw,hann,boxes = head['raw_scores'],head['hann_scores'],head['dense_boxes']
            assert len(raw)==len(hann)==len(boxes)==256 and len(head['nms'])==10
            assert all(math.isfinite(x) for box in boxes for x in box)
            assert all(box[2]>0 and box[3]>0 for box in boxes)
            assert all(math.isfinite(x) for x in raw+hann)
            k=max(range(256),key=lambda n:hann[n]);q=max(range(256),key=lambda n:raw[n])
            assert k==head['selected_index'] and q==head['raw_selected_index']
            for n in range(256):near(hann[n],raw[n]*window[n],3e-7)
            values = ev['heads'][name]
            assert values['selected_index']==k and values['peak_changed_vs_category']==(k!=own_index)
            near(values['selected_score'],hann[k])
            assert values['scheduled_write_if_this_head']==(i%50==0 and hann[k]>.75)
            for nms in head['nms']:
                j=16*nms['grid_row']+nms['grid_column'];near(nms['score'],hann[j])
                assert max(abs(a-b) for a,b in zip(nms['bbox'],boxes[j]))<1e-4
            first = head['nms'][0]
            assert 16*first['grid_row']+first['grid_column']==k
            if name=='category':
                near(hann[k],geometry['score']);assert max(abs(a-b) for a,b in zip(boxes[k],geometry['bbox']))<1e-4
            if valid[i]:
                dense_iou=[overlap(b,gt[i]) for b in boxes]
                top_iou=[overlap(v['bbox'],gt[i]) for v in head['nms']]
                first_correct=next((j+1 for j,v in enumerate(top_iou) if v>=.5),None)
                for key,value in dict(selected_iou=dense_iou[k],raw_selected_iou=dense_iou[q],dense_best_iou=max(dense_iou),top10_best_iou=max(top_iou)).items():near(values[key],value)
                assert values['first_correct_rank']==first_correct
                if in_window:
                    z=stats[name];z['window_valid']+=1;z['category_low']+=int(low);z['category_good']+=int(good)
                    z['hann_rescue']+=int(low and dense_iou[k]>=.5);z['raw_rescue']+=int(low and dense_iou[q]>=.5)
                    z['raw_break_good']+=int(good and dense_iou[q]<.5);z['raw_severe_break_good']+=int(good and dense_iou[q]<=.1)
                    z['head_hann_severe_break_good']+=int(good and dense_iou[k]<=.1)
                    z['top10_capacity']+=int(low and max(top_iou)>=.5);z['dense_capacity']+=int(low and max(dense_iou)>=.5)
                    z['changed_peak']+=int(q!=k)
                    u=summary['heads'][name];u['low_probe_top1_correct']+=int(low and dense_iou[k]>=.5);u['low_probe_top10_correct']+=int(low and max(top_iou)>=.5);u['low_probe_dense_correct']+=int(low and max(dense_iou)>=.5);u['correct_probe_severely_broken']+=int(good and dense_iou[k]<=.1)
            if i%50==0:stats[name]['scheduled_write_difference']+=int(values['scheduled_write_if_this_head']!=geometry['template_write'])
            slim['heads'][name]=dict(hann_index=k,raw_index=q,raw_at_hann_peak=raw[k],hann_at_hann_peak=hann[k],raw_at_raw_peak=raw[q],hann_at_raw_peak=hann[q])
            if valid[i]:slim['heads'][name].update(hann_iou=dense_iou[k],raw_iou=dense_iou[q],top10_best=max(top_iou),dense_best=max(dense_iou))
            decoded_boxes+=256
        if start<=i<end:timeline[seq].append(slim)
        if i==selected_case['H10_start']:onset_scores.append(dict(sequence=seq,frame=i,heads=slim['heads']))
    assert summary==result['summary'][seq]
    assert next(v for v in result['onsets'] if v['sequence']==seq)['probe']==next(v for v in evaluated[seq] if v['frame']==selected_case['H10_start'])
    metrics[seq]=stats;total_geometry+=len(data['geometry']);total_probes+=len(data['probes'])
    files.append(dict(sequence=seq,bytes=path.stat().st_size,sha256=sha(path)))
assert total_geometry==s['new_track_calls']==receipt['new_track_calls']==6925
assert total_probes==429
out=dict(status='M76_completed_artifacts_and_independent_scalar_recomputation',observed_utc=datetime.now(timezone.utc).isoformat(),auditor_sha256=sha(__file__),source_sha256=s['source_sha256'],spec_sha256=sha(R/'spec.json'),result_sha256=sha(R/'result.json'),receipt_sha256=sha(R/'receipt.json'),evaluated_probes_sha256=sha(R/'evaluated_probes.json'),replay_files=files,public_geometry_rows=total_geometry,probe_frames=total_probes,decoded_boxes_checked=decoded_boxes,exact_original_prefix_geometry_and_scores=True,independent_summary_recomputation=True,native_Hann_values_verified=True,selected_window_raw_vs_Hann=metrics,onset_peak_scores=onset_scores,new_tracking_calls=0,new_optimizer_steps=0,new_training_seeds=[],public_evaluation_allowed=False,independent_model_review_pass=False,scope='Artifact/scalar verification plus post-result raw-vs-Hann statistics on deliberately selected windows. Not independent recursive rescue, not a new deployment policy, not learned-model review.')
target=R/'completed_evidence_audit.json';assert not target.exists()
target.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
(R/'peak_timeline.json').write_text(json.dumps(timeline,indent=2,allow_nan=False)+'\n')
print(json.dumps(out,indent=2))
