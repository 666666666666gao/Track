"""Prepare final human external inputs on CPU; do not encode or evaluate them."""
import argparse,csv,hashlib,io,json,re
from pathlib import Path

SOURCES={
    'depthtrack_test':(50,'6ccfab037bc24ca44245bb88739a77bf668f931283da12cb18e37402d0a0443a'),
    'cdtb':(80,'dea42aa507134894bd3e267672b2a69f662a1d26ecd959406257e4b2e5895381'),
    'vot':(1765,'b83f0b4240865d609921c1bf434621577bbe35ace4aceaa0f1242c981445cb9c')}


def main():
    parser=argparse.ArgumentParser()
    for name in ['reviews','site-data','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists();prepared={};source=[]
    for dataset,(count,digest) in SOURCES.items():
        path=args.reviews/('rgbd_'+dataset+'_gb_'+str(count)+'_review.csv');data=path.read_bytes()
        assert hashlib.sha256(data).hexdigest()==digest,path
        rows=list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
        site=json.loads((args.site_data/(dataset+'.json')).read_text(encoding='utf-8'))
        assert len(rows)==site['count']==len(site['rows'])==count
        lookup={row['id']:row for row in site['rows']}
        assert len({row['key'] for row in rows})==count and {row['key'] for row in rows}==set(lookup)
        output=[]
        for row in sorted(rows,key=lambda r:int(r['anchor_number'])):
            case=lookup[row['key']];confirmed=case['human_reviews']['gb']
            assert row['human_confirmed']=='true' and confirmed['human_confirmed'] is True
            assert row['dataset']==dataset and row['sequence']==case['sequence']
            assert row['init_frame']==case['frame'] and int(row['anchor_number'])==case['number']
            for field in ['confirmed_category','stable_attributes','uncertain_attributes','status','updated_at']:
                assert row[field]==confirmed[field],(dataset,row['key'],field)
            category=row['confirmed_category'].strip()
            attributes=[phrase.strip() for phrase in re.split(r'[|;；]',row['stable_attributes']) if phrase.strip()]
            output.append(dict(review_id=row['key'],sequence=row['sequence'],init_frame=row['init_frame'],
                frame_index=case['frame_index'],phrases=[category]+attributes[:4],
                omitted_confirmed_attributes=attributes[4:],human_reviewer=row['reviewer_id'],
                human_confirmed=True,unknown_category_kept_empty=not bool(category),
                human_status_is_proposal_review_not_pending_label=row['status']))
        assert all(len(r['phrases'])<=5 for r in output)
        assert len({r['sequence'] for r in output})=={'depthtrack_test':50,'cdtb':80,'vot':127}[dataset]
        assert sum(r['unknown_category_kept_empty'] for r in output)==0
        prepared[dataset]=output;source.append(dict(dataset=dataset,file=path.name,sha256=digest,
            site_manifest_sha256=hashlib.sha256((args.site_data/(dataset+'.json')).read_bytes()).hexdigest(),count=count))
    result=dict(status='prepared_M122_external_human_labels_CPU_only',datasets=prepared,sources=source,
        human_confirmed=True,first_slot_is_category_including_explicit_empty=True,attribute_limit=4,
        stable_attribute_separators=['|',';','；'],uncertain_or_later_frame_fields_encoded=False,
        provenance='Human review used multiframe/video aids; disclose this input protocol separately from automatic captions.',
        legal_RGB_and_bbox_binding_verified=False,text_encoder_executed=False,optimizer_steps=0,
        pending='Bind actual legal initialization RGB bytes and OPE/raw or VOT/wire bbox before encoding/evaluation.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],counts={k:len(v) for k,v in prepared.items()},
        unknown_category=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
