"""CPU-only PR supplement for completed M82/M84 trajectories; no new tracking."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
BASE = Path('/root/autodl-tmp')
METRIC = Path('/home/SRTrack_RGBD_L/lib/test/analysis/depthtrack_pr.py')
METRIC_SHA = '05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc'
PARENTS = {
    'M82': (BASE/'sttrack_m82_native_preservation_20260909', '823d0560db3acfdd594c53ecfae239ebb59ef77b3b2776bc9037dcdaa3a259ea', ['category','empty','category_empty','category_swapped']),
    'M84': (BASE/'sttrack_m84_centered_20260920', 'a9281e0833b8ba1c01220374f3d6a93dbc9d22e6e8648c3495bb009083f2d5f7', ['category','category_empty','category_swapped'])}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare():
    assert not (ROOT/'spec.json').exists()
    assert sha(METRIC) == METRIC_SHA
    families = {}
    for label, (folder, digest, arms) in PARENTS.items():
        assert sha(folder/'recursive_result.json') == digest
        result = read(folder/'recursive_result.json')
        assert result['status'] == 'complete_recursive_development'
        training = read(folder/'training_spec.json'); recursive = read(folder/'recursive_spec.json')
        assert sha(folder/'training_spec.json') == result['training_spec_sha256']
        assert sha(folder/'recursive_spec.json') == result['recursive_spec_sha256']
        for arm in arms:
            rp = folder/(arm+'_recursive_receipt.json')
            assert sha(rp) == result['receipts'][arm]
            receipt = read(rp)
            assert receipt['status'] == 'complete' and receipt['total_frames'] == 33130
            families[label+'_'+arm] = dict(folder=str(folder), arm=arm, result_sha256=digest,
                receipt_path=str(rp), receipt_sha256=sha(rp), head_sha256=receipt['head_sha256'],
                training_spec_sha256=sha(folder/'training_spec.json'), recursive_spec_sha256=sha(folder/'recursive_spec.json'))
    cases = read(PARENTS['M84'][0]/'recursive_spec.json')['cases']
    assert cases == read(PARENTS['M82'][0]/'recursive_spec.json')['cases']
    dataset = read(PARENTS['M84'][0]/'training_spec.json')['dataset_root']
    write(ROOT/'spec.json', dict(status='fixed_before_supplementary_PR', observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__), metric_sha256=METRIC_SHA, metric_path=str(METRIC),
        families=families, cases=cases, dataset_root=dataset, resolution=100, output_decimals=6,
        initial_confidence=1., score='Saved actual Hann peak, no normalization or content intervention',
        scope='All22 repeated DepthTrack Train development sequences. Supplementary reporting only; not Test/CDTB/VOT, not a new gate or checkpoint selection.',
        new_tracking_calls=0,new_optimizer_steps=0, M88_modified=False))
    print(json.dumps({'status':'prepared','spec_sha256':sha(ROOT/'spec.json')}), flush=True)


def run():
    import numpy as np
    spec = read(ROOT/'spec.json')
    assert spec['source_sha256'] == sha(__file__) and spec['metric_sha256'] == sha(METRIC)
    assert not (ROOT/'result.json').exists()
    module_spec = importlib.util.spec_from_file_location('fixed_depthtrack_pr', str(METRIC))
    metric = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(metric)
    metric.cv2.setNumThreads(1)
    sealed = {}
    evidence = ROOT/'evidence'; evidence.mkdir()
    (evidence/'depthtrack_pr.py').write_bytes(METRIC.read_bytes())
    # Complete verification of all seven prediction families precedes GT access.
    for label, family in spec['families'].items():
        folder = Path(family['folder']); rp = Path(family['receipt_path'])
        assert sha(rp) == family['receipt_sha256'] and sha(folder/'recursive_result.json') == family['result_sha256']
        receipt = read(rp); assert len(receipt['sequences']) == len(spec['cases']) == 22
        sealed[label] = {}
        (evidence/(label+'_receipt.json')).write_bytes(rp.read_bytes())
        for case, item in zip(spec['cases'], receipt['sequences']):
            assert case['sequence'] == item['sequence'] and case['frames'] == item['frames']
            path = folder/'recursive'/family['arm']/(case['sequence']+'.json')
            assert sha(path) == item['sha256']
            data = read(path); rows = data['rows']
            assert data['arm'] == family['arm'] and len(rows) == case['frames']
            assert [r['frame'] for r in rows] == list(range(case['frames']))
            assert all(np.isfinite(r['score']) and 0 <= r['score'] <= 1 for r in rows[1:])
            sealed[label][case['sequence']] = rows
    dataset = Path(spec['dataset_root']); geometry = {}
    (evidence/'gt').mkdir()
    for case in spec['cases']:
        name = case['sequence']; path = dataset/name/'groundtruth.txt'
        assert sha(path) == case['gt_sha256']
        (evidence/'gt'/(name+'.txt')).write_bytes(path.read_bytes())
        image = dataset/name/'color/00000001.jpg'
        h,w = metric.cv2.imread(str(image)).shape[:2]
        geometry[name] = dict(width=w,height=h,first_image_sha256=sha(image))
    write(evidence/'geometry.json',geometry)
    results={}; files={}; curves={}
    for label, sequences in sealed.items():
        output=evidence/label; output.mkdir(); files[label]={}; curves[label]={}
        for case in spec['cases']:
            name=case['sequence']; rows=sequences[name]
            boxes=np.asarray([r['bbox'] for r in rows],dtype=np.float64)
            scores=np.asarray([1.]+[r['score'] for r in rows[1:]],dtype=np.float64)
            np.savetxt(output/(name+'.txt'),boxes,fmt='%.6f',delimiter=',')
            np.savetxt(output/(name+'_all_scores.txt'),scores,fmt='%.6f')
            rounded=metric._load_rows(output/(name+'.txt'),4)
            confidence=metric._load_rows(output/(name+'_all_scores.txt'),1).reshape(-1)
            assert np.max(np.abs(rounded-boxes)) <= 5.01e-7
            gt=metric._load_rows(evidence/'gt'/(name+'.txt'),4)
            g=geometry[name]; overlap,visible=metric._vot_overlaps(rounded,gt,g['width'],g['height'])
            curves[label][name]=dict(overlap=overlap.tolist(),visible=visible.tolist(),confidence=confidence.tolist())
            files[label][name]=dict(bbox_sha256=sha(output/(name+'.txt')),score_sha256=sha(output/(name+'_all_scores.txt')))
        results[label]=metric.evaluate_depthtrack_results(dataset,output,resolution=100,sequence_names=[c['sequence'] for c in spec['cases']])
        assert results[label]['sequences']==22 and results[label]['frames']==33130
        print(json.dumps({'family':label,'metrics':results[label]}),flush=True)
    write(evidence/'overlap_score_arrays.json',curves)
    native=results['M84_category_empty']
    historical=read(BASE/'sttrack_m68_reported_confidence_20260907/historical_score_reference/result.json')
    expected=historical['metrics']['native']['raw']
    for k in ['precision_percent','recall_percent','f_score_percent']:
        assert abs(native[k]-expected[k])<1e-10,(k,native[k],expected[k])
    (evidence/'historical_native_PR_reference.json').write_text(json.dumps(historical,indent=2)+'\n')
    result=dict(status='completed_sealed_Train_PR_supplement',observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),spec_sha256=sha(ROOT/'spec.json'),metric_sha256=sha(METRIC),metrics=results,
        input_families=spec['families'],export_sha256=files,arrays_sha256=sha(evidence/'overlap_score_arrays.json'),
        native_historical_PR_exact_to_1e_10=True,scope=spec['scope'],independent_audit_completed=False,
        new_tracking_calls=0,new_optimizer_steps=0,M88_modified=False,
        limitations='Includes initialization and invalid-GT convention of the historical PR evaluator. Best-F threshold is a reporting statistic, not a new deployment threshold. Not comparable to continuous valid-frame IoU.')
    write(ROOT/'result.json',result)
    with (ROOT/'metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['family','precision_percent','recall_percent','f_score_percent','threshold','sequences','frames'])
        writer.writeheader()
        for label,x in results.items():writer.writerow(dict(family=label,**{k:x[k] for k in writer.fieldnames if k!='family'}))
    for n in ['spec.json','result.json','metrics.csv','evaluate_sealed_pr.py']:(evidence/n).write_bytes((ROOT/n).read_bytes())
    manifest=[dict(path=p.relative_to(evidence).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(evidence.rglob('*')) if p.is_file()]
    write(evidence/'manifest.json',manifest)
    with tarfile.open(ROOT/'completed_evidence.tar.gz','w:gz') as archive:
        for p in sorted(evidence.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(evidence).as_posix())
    write(ROOT/'collection_result.json',dict(status='collected',archive_sha256=sha(ROOT/'completed_evidence.tar.gz'),files=len(manifest),bytes=(ROOT/'completed_evidence.tar.gz').stat().st_size))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','run']); args=parser.parse_args()
    prepare() if args.action=='prepare' else run()
