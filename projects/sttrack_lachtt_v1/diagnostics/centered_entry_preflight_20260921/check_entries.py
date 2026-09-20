"""Train-only centered adapter deployment parity; never computes GT metrics."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
M84 = Path('/root/autodl-tmp/sttrack_m84_centered_20260920')
INTERFACE = ROOT / 'interface'
PYTHON = '/root/autodl-tmp/envs/sttrack/bin/python'
sys.path.insert(0, str(INTERFACE))
from initialization_text import initialization_key, sha, vot_wire_bbox


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare():
    import torch
    spec = read(M84 / 'training_spec.json')
    final = M84 / 'training/category/final.pth'
    assert sha(final) == 'c63605ebcb1f702de66f96967255e5301bfdca1e3c40450b6d4b8a952791b1e0'
    assert sha(M84 / 'training_spec.json') == '0f9bb841edd3e2c5171cd78ce9d1030d29a243561006d111d2c98eebfc74abd5'
    assert sha(M84 / 'integration.json') == spec['integration_sha256']
    sources = read(M84 / 'integration.json')['source_sha256']
    for name, digest in sources.items():
        assert sha(M84 / 'code' / name) == digest
    cases = [dict(sequence=r['sequence'], init_bbox=r['first_box'], frames=102)
             for r in spec['sequence_order'][:2]]
    assert [r['sequence'] for r in cases] == ['cube04_indoor', 'bag04_indoor']
    assert all(r['sequence'] not in spec['development_sequences'] for r in cases)
    for case in cases:
        assert vot_wire_bbox(case['init_bbox']) == case['init_bbox']
    write(ROOT / 'cases.json', cases)
    # Actual float32/text wire behavior matters for observation-keyed captions.
    fractional = [269.1234567, 248.7654321, 33.1234567, 32.7654321]
    write(ROOT / 'fractional.json', dict(sent=fractional, received=vot_wire_bbox(fractional)))
    write(ROOT / 'text_protocol.json', dict(scope='M84 historical fit bank reindex only; no regenerated text',
        source_banks=spec['banks']['fit'], fractional_probe='Protocol-only key uses first case text; not an annotated target or metric.'))
    bundle = dict(repository=str(M84 / 'code'), configuration='experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml',
        architecture='semantic_spatial_centered_v1', null_support=True, support_loss_weight=0., arm='category', use_text=True,
        seed=2027, base_checkpoint=spec['native_checkpoint'], base_checkpoint_sha256=spec['native_checkpoint_sha256'],
        adapter_checkpoint=str(final), adapter_checkpoint_sha256=sha(final), training_spec_sha256=sha(M84/'training_spec.json'),
        source_sha256=sources, interface_sha256={p.name:sha(p) for p in sorted(INTERFACE.glob('*.py'))},
        text_protocol_path=str(ROOT/'text_protocol.json'), text_protocol_sha256=sha(ROOT/'text_protocol.json'))
    write(ROOT/'bundle.json', bundle)
    for condition in ['category', 'empty']:
        source = spec['banks']['fit'][condition]
        assert sha(source['path']) == source['sha256']
        old = torch.load(source['path'], map_location='cpu')
        indices = [old['sequences'].index(c['sequence']) for c in cases]
        keys = [initialization_key(sha(Path(spec['dataset_root'])/c['sequence']/'color/00000001.jpg'),c['init_bbox']) for c in cases]
        if condition == 'category':
            indices.append(indices[0])
            keys.append(initialization_key(sha(Path(spec['dataset_root'])/cases[0]['sequence']/'color/00000001.jpg'),vot_wire_bbox(fractional)))
        bank = dict(format='initialization_observation_v1',protocol_sha256=bundle['text_protocol_sha256'],
            keys=keys,tokens=old['tokens'][indices],mask=old['mask'][indices],empty=old['empty'])
        path = ROOT/(condition+'_bank.pt');torch.save(bank,path)
        plan = dict(bundle_path=str(ROOT/'bundle.json'),bundle_sha256=sha(ROOT/'bundle.json'),text_bank_path=str(path),
            text_bank_sha256=sha(path),dataset_root=spec['dataset_root'],cases_path=str(ROOT/'cases.json'),
            cases_sha256=sha(ROOT/'cases.json'),output=str(ROOT/(condition+'_ope')))
        write(ROOT/(condition+'_plan.json'),plan)
    from semantic_runtime import checked_plan, text_bank
    for condition in ['category','empty']:
        p,b = checked_plan(ROOT/(condition+'_plan.json'))
        bank=text_bank(p,b)
        for case in cases:bank.info(Path(p['dataset_root'])/case['sequence']/'color/00000001.jpg',case['init_bbox'])
    write(ROOT/'preparation.json',dict(status='prepared',cases=2,conditions=2,subsequent_gt_opened=False,
        source_sha256=sha(__file__),plan_sha256=sha(ROOT/'EXPERIMENT_PLAN.md'),final_sha256=sha(final)))


def direct(condition):
    import torch
    from types import SimpleNamespace
    spec=read(M84/'training_spec.json')
    sys.path.insert(0,str(M84/'code'))
    from lib.config.sttrack.config import cfg,update_config_from_file
    from lib.test.tracker.sttrack_semantic import STTrackSemantic
    from lib.test.tracker.sttrack import STTrack
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1);torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)
    update_config_from_file(str(M84/'code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params=SimpleNamespace(cfg=cfg,checkpoint=spec['native_checkpoint'],base_checkpoint_sha256=spec['native_checkpoint_sha256'],
        template_factor=2.,template_size=128,search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
    tracker=STTrack(params) if condition=='native' else STTrackSemantic(params,str(M84/'training/category/final.pth'))
    if condition!='native':
        entry=spec['banks']['fit'][condition];assert sha(entry['path'])==entry['sha256']
        bank=torch.load(entry['path'],map_location='cpu')
    outputs=[]
    for case in read(ROOT/'cases.json'):
        rows=[];folder=Path(spec['dataset_root'])/case['sequence']
        for frame in range(case['frames']):
            stem='%08d'%(frame+1)
            image=get_rgbd_frame(str(folder/'color'/(stem+'.jpg')),str(folder/'depth'/(stem+'.png')),dtype='rgbcolormap',depth_clip=True)
            if frame==0:
                info=dict(init_bbox=case['init_bbox'])
                if condition!='native':
                    idx=bank['sequences'].index(case['sequence'])
                    info.update(text_tokens=bank['tokens'][idx],text_mask=bank['mask'][idx],empty_text=bank['empty'])
                tracker.initialize(image,info);rows.append(dict(bbox=list(case['init_bbox']),score=1.))
            else:
                out=tracker.track(image);rows.append(dict(bbox=list(out['target_bbox']),score=float(out['best_score'])))
        outputs.append(dict(sequence=case['sequence'],rows=rows))
    write(ROOT/(condition+'_direct.json'),outputs)


class Frame:
    def __init__(self,folder,index):
        stem='%08d'%(index+1)
        self.paths=dict(color=str(folder/'color'/(stem+'.jpg')),depth=str(folder/'depth'/(stem+'.png')))
    def filename(self,channel):return self.paths[channel]


def client(condition):
    import shlex
    import time
    import importlib.metadata as metadata
    from vot.region import Rectangle
    from vot.tracker import ObjectStatus
    from vot.tracker.trax import TrackerProcess
    assert metadata.version('vot-toolkit')=='0.7.1'
    assert metadata.version('vot-trax')=='4.0.2'
    plan=read(ROOT/(condition+'_plan.json'));cases=read(ROOT/'cases.json')
    expected=read(ROOT/(condition+'_direct.json'));reports=[]
    for index,case in enumerate(cases+([dict(cases[0],fractional=True)] if condition=='category' else [])):
        probe=case.get('fractional',False)
        bbox=read(ROOT/'fractional.json')['sent'] if probe else case['init_bbox']
        log=ROOT/('%s_trax_%d.log'%(condition,index))
        command=shlex.join([PYTHON,str(INTERFACE/'run_semantic_vot.py'),'--plan',str(ROOT/(condition+'_plan.json'))])
        process=TrackerProcess(command,envvars={'CUDA_VISIBLE_DEVICES':'1','OMP_NUM_THREADS':'4'},timeout=60,log=str(log))
        child=process._process;rows=[];max_error=0.
        # Cleanup is required by the existing real TraX subprocess interface.
        try:
            assert process.has_vot_wrapper
            folder=Path(plan['dataset_root'])/case['sequence']
            statuses,_=process.initialize(Frame(folder,0),ObjectStatus(Rectangle(*bbox),{}))
            assert len(statuses)==1
            region=statuses[0].region
            initial=[float(region.x),float(region.y),float(region.width),float(region.height)]
            assert initial==vot_wire_bbox(bbox)
            assert 'confidence' not in statuses[0].properties
            for frame in range(1,1 if probe else case['frames']):
                statuses,_=process.update(Frame(folder,frame));assert len(statuses)==1
                r=statuses[0].region;box=[float(r.x),float(r.y),float(r.width),float(r.height)]
                score=float(statuses[0].properties['confidence'])
                ref=expected[index]['rows'][frame]
                assert box==vot_wire_bbox(ref['bbox']),(condition,index,frame,box,ref['bbox'])
                max_error=max(max_error,abs(score-ref['score']));assert max_error<=1e-6
                rows.append(dict(bbox=box,score=score))
            shutdown_started=time.monotonic()
            process._client.quit()
            code=child.wait(timeout=30)
            shutdown_seconds=time.monotonic()-shutdown_started
            assert code==0,code
        finally:
            process.terminate()
        reports.append(dict(sequence=case['sequence'],fractional_probe=probe,initial_bbox=initial,
            exit_code=code,shutdown_seconds=shutdown_seconds,reports=len(rows),max_score_error=max_error,rows=rows))
    write(ROOT/(condition+'_trax.json'),reports)


def verify():
    import numpy as np
    cases=read(ROOT/'cases.json')
    expected_names=[c['sequence'] for c in cases]
    assert expected_names==['cube04_indoor','bag04_indoor']
    native=read(ROOT/'native_direct.json');empty=read(ROOT/'empty_direct.json')
    for output in [native,empty]:
        assert [r['sequence'] for r in output]==expected_names
        assert all(len(r['rows'])==102 for r in output)
    assert native==empty
    rows=[]
    for condition in ['category','empty']:
        plan=read(ROOT/(condition+'_plan.json'));receipt=read(ROOT/(condition+'_ope/receipt.json'))
        assert receipt['status']=='complete' and receipt['frames']==204
        assert receipt['plan_sha256']==sha(ROOT/(condition+'_plan.json'))
        assert receipt['bundle_sha256']==plan['bundle_sha256'] and receipt['text_bank_sha256']==plan['text_bank_sha256']
        direct_rows=read(ROOT/(condition+'_direct.json'))
        trax=read(ROOT/(condition+'_trax.json'))
        assert [r['sequence'] for r in direct_rows]==expected_names
        assert all(len(r['rows'])==102 for r in direct_rows)
        assert [r['sequence'] for r in receipt['sequences']]==expected_names
        assert [r['sequence'] for r in trax]==expected_names+([expected_names[0]] if condition=='category' else [])
        assert [r['reports'] for r in trax]==[101,101]+([0] if condition=='category' else [])
        assert [r['fractional_probe'] for r in trax]==[False,False]+([True] if condition=='category' else [])
        for index,session in enumerate(trax):
            assert len(session['rows'])==session['reports']
            init=read(ROOT/'fractional.json')['sent'] if session['fractional_probe'] else cases[index]['init_bbox']
            assert session['initial_bbox']==vot_wire_bbox(init)
            for got,ref in zip(session['rows'],direct_rows[index]['rows'][1:] if not session['fractional_probe'] else []):
                assert got['bbox']==vot_wire_bbox(ref['bbox']) and abs(got['score']-ref['score'])<=1e-6
        assert sum(r['reports'] for r in trax)==202 and all(r['exit_code']==0 for r in trax)
        for ref,item in zip(direct_rows,receipt['sequences']):
            name=ref['sequence'];assert item['sequence']==name and item['frames']==102
            path=ROOT/(condition+'_ope')
            assert sha(path/(name+'.txt'))==item['bbox_sha256']
            assert sha(path/(name+'_all_scores.txt'))==item['confidence_sha256']
            boxes=np.loadtxt(path/(name+'.txt'),delimiter=',');scores=np.loadtxt(path/(name+'_all_scores.txt'))
            assert boxes.shape==(102,4) and scores.shape==(102,)
            error_b=float(np.abs(boxes-np.asarray([r['bbox'] for r in ref['rows']])).max())
            error_s=float(np.abs(scores-np.asarray([r['score'] for r in ref['rows']])).max())
            assert max(error_b,error_s)<=5.01e-7
            rows.append(dict(condition=condition,sequence=name,frames=102,box_error=error_b,score_error=error_s))
    assert read(ROOT/'category_trax.json')[-1]['fractional_probe']
    write(ROOT/'result.json',dict(status='real_centered_OPE_TraX_parity_pass',comparisons=rows,
        empty_native_exact_frames=204,real_tracking_calls=1414,initializations=15,trax_sessions=5,trax_reports=404,
        fractional_initialization_round_trip=True,new_captions=0,optimizer_steps=0,subsequent_gt_opened=False,
        formal_evaluation=False,performance_claim=False,checkpoint_sha256=read(ROOT/'preparation.json')['final_sha256']))
    print(json.dumps(read(ROOT/'result.json'),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','direct','client','verify'])
    parser.add_argument('--condition',choices=['category','empty','native']);args=parser.parse_args()
    if args.mode in ['direct','client']:globals()[args.mode](args.condition)
    else:globals()[args.mode]()
