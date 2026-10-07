"""Bind confirmed external words to actual legal RGB bytes and initialization boxes."""
import argparse,hashlib,json,struct
from pathlib import Path


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024**2),b''):digest.update(data)
    return digest.hexdigest()


def initialization_key(rgb_sha,bbox):
    return hashlib.sha256(bytes.fromhex(rgb_sha)+struct.pack('>4d',*bbox)).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    for name in ['labels','ope-inputs','vot-plan','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    labels=json.loads(args.labels.read_text());ope=json.loads(args.ope_inputs.read_text());vot=json.loads(args.vot_plan.read_text())
    assert labels['status']=='prepared_M122_external_human_labels_CPU_only' and labels['human_confirmed']
    assert sha(args.ope_inputs)=='61541e35f7b9e3c40427df79067fc0be20b8622cf275e93025e4a1547bf68601'
    assert sha(args.vot_plan)=='b4c23250a9363475746cb92023c1b67a39219f2babe95c6a98ea1c40109ea568'
    assert vot['coordinate_convention']=='vot_toolkit_xywh' and len(vot['rows'])==1765
    assert all(r['phrases'][0].strip() and r['human_confirmed'] for rows in labels['datasets'].values() for r in rows)
    records=[]
    for name,native_name,count,frames in [('depthtrack_test','depthtrack',50,76373),('cdtb','cdtb',80,101956)]:
        confirmed={r['sequence']:r for r in labels['datasets'][name]};cases=ope[native_name]
        assert len(confirmed)==len(cases)==count and sum(c['frames'] for c in cases)==frames
        assert set(confirmed)=={c['sequence'] for c in cases}
        for case in cases:
            row=confirmed[case['sequence']];assert row['init_frame']=='00000001' and row['frame_index']==0
            image=Path(case['root'])/case['sequence']/'color/00000001.jpg';digest=sha(image)
            key=initialization_key(digest,case['init_bbox'])
            if name=='cdtb':assert key==row['review_id']
            records.append(dict(row,key=name+':'+key,observation_key=key,dataset=name,image=str(image),image_sha256=digest,
                init_bbox=case['init_bbox'],frames=case['frames'],coordinate_convention='ope_raw_xywh'))
    confirmed={r['review_id']:r for r in labels['datasets']['vot']}
    assert len(confirmed)==len(vot['rows'])==1765 and set(confirmed)=={r['key'] for r in vot['rows']}
    for case in vot['rows']:
        row=confirmed[case['key']];image=Path(case['image'])
        assert image.parent.parent.name==row['sequence'] and image.stem==row['init_frame']
        assert int(image.stem)-1==row['frame_index']
        digest=sha(image);assert digest==case['image_sha256']
        assert initialization_key(digest,case['init_bbox'])==row['review_id']
        records.append(dict(row,key='vot:'+case['key'],observation_key=case['key'],dataset='vot',image=str(image),image_sha256=digest,
            init_bbox=case['init_bbox'],coordinate_convention='vot_observed_wire_xywh'))
    assert len(records)==len({r['key'] for r in records})==1895
    result=dict(status='complete_M122_human_official_initialization_binding',rows=records,
        labels_sha256=sha(args.labels),source_sha256=sha(__file__),ope_inputs_sha256=sha(args.ope_inputs),vot_plan_sha256=sha(args.vot_plan),
        counts=dict(depthtrack_test=50,cdtb=80,vot=1765),human_confirmed=True,
        RGB_bytes_and_exact_legal_bbox_bound=True,subsequent_GT_opened=False,optimizer_steps=0,neural_executions=0,
        human_review_used_multiframe_aids=True,uncertain_or_later_frame_fields_encoded=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],counts=result['counts'],sha256=sha(args.output))),flush=True)


if __name__=='__main__':main()
