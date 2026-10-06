"""Bind confirmed Train CSV rows to initialization IDs and the existing split."""
import argparse,csv,io,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    for name in ['reviews','initializations','split-manifest','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    rows=list(csv.DictReader(io.StringIO(a.reviews.read_text(encoding='utf-8-sig'))))
    init=json.loads(a.initializations.read_text(encoding='utf-8'))
    assert init['dataset']=='depthtrack' and init['count']==len(rows)==152
    lookup={r['id']:r for r in init['rows']}
    assert len({r['key'] for r in rows})==152 and {r['key'] for r in rows}==set(lookup)
    partition=json.loads(a.split_manifest.read_text(encoding='utf-8'))['initial']
    splits={r['sequence']:r['split'] for r in partition}
    assert len(splits)==152 and set(splits)=={r['sequence'] for r in rows}
    result=[]
    for r in sorted(rows,key=lambda row:int(row['anchor_number'])):
        case=lookup[r['key']]
        assert r['dataset']=='depthtrack' and r['sequence']==case['sequence']
        assert r['init_frame']==case['frame'] and int(r['anchor_number'])==case['number']
        assert r['human_confirmed']=='true' and r['review_round']=='20261006'
        assert r['review_subject']=='gpt_proposal' and r['status'] in ['supported','conflicting','uncertain']
        category=r['confirmed_category'].strip();assert category
        attributes=[s.strip() for s in r['stable_attributes'].split('|') if s.strip()]
        result.append(dict(audit_id=r['key'],sequence=r['sequence'],split=splits[r['sequence']],
            init_frame=r['init_frame'],phrases=[category]+attributes[:4],
            omitted_confirmed_attributes=attributes[4:],human_reviewer=r['reviewer_id'],
            human_status=r['status'],human_confirmed=True,review_round=r['review_round']))
    assert sum(r['split']=='fit' for r in result)==130
    assert sum(r['split']=='development' for r in result)==22
    labels=dict(status='prepared_human_confirmed_train_initialization',human_confirmed=True,
        dataset='depthtrack',review_round='20261006',initial=result,
        source_review_csv=a.reviews.name,legacy_manifest_used_for_partition_only=True,
        uncertain_or_later_frame_fields_used=False,test_cdtb_vot_rows_used=False,
        current_candidate_semantic_labels=0)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(labels,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=labels['status'],rows=152,fit=130,development=22,
        phrases=sum(len(r['phrases']) for r in result),omitted_attributes=sum(len(r['omitted_confirmed_attributes']) for r in result)),ensure_ascii=False))


if __name__=='__main__':main()
