"""Import submitted human initialization records without changing GPT proposals."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import csv,hashlib,io,json

SITE=Path(__file__).parent
PROJECT=Path(r'C:\Users\gb\Desktop\document\RGBD_LANGUAGE_TRACKING')


def main():
    counts={'depthtrack':152,'depthtrack_test':50,'cdtb':80,'vot':1765}
    sources=[];summary={};flat=[]
    folder=SITE/'data/source_human_reviews_20261007';folder.mkdir(exist_ok=True)
    for dataset,count in counts.items():
        path=PROJECT/'人工审核结果'/f'rgbd_{dataset}_gb_{count}_review.csv'
        data=path.read_bytes();rows=list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
        target=SITE/'data'/f'{dataset}.json';manifest=json.loads(target.read_text(encoding='utf-8'))
        lookup={r['id']:r for r in manifest['rows']}
        assert len(rows)==len({r['key'] for r in rows})==count and {r['key'] for r in rows}==set(lookup)
        for r in rows:
            item=lookup[r['key']]
            assert r['dataset']==dataset and r['sequence']==item['sequence'] and r['init_frame']==item['frame']
            assert int(r['anchor_number'])==item['number'] and r['human_confirmed']=='true'
            assert r['status'] in ['supported','conflicting','uncertain'] and r['reviewer_id']=='gb'
            round=r.get('review_round','initial');assert round==manifest['human_review_round']
            record={field:r[field] for field in ['status','confirmed_category','stable_attributes','uncertain_attributes','note','updated_at']}
            record.update(human_confirmed=True,review_round=round,review_subject=manifest['review_subject'])
            item['human_reviews']['gb']=record
        manifest['published_reviewers']=['gb']
        target.write_text(json.dumps(manifest,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
        (folder/path.name).write_bytes(data)
        sources.append(dict(path='data/source_human_reviews_20261007/'+path.name,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),dataset=dataset,count=count))
        summary[dataset]=dict(count=count,confirmed=count,review_round=manifest['human_review_round'],status=dict(Counter(r['status'] for r in rows)))
        for item in manifest['rows']:
            m=item['model_review'];h=item['human_reviews']['gb']
            flat.append(dict(dataset=dataset,anchor_number=item['number'],key=item['id'],sequence=item['sequence'],init_frame=item['frame'],
                model_status=m['status'],model_category=m['category'],model_stable_attributes=m['stable_attributes'],model_uncertain_attributes=m['uncertain_attributes'],
                model_conflicting_attributes=m['conflicting_attributes'],model_context_only_notes=m['context_only_notes'],model_evidence=m['evidence'],
                model_evidence_frames=m['evidence_frames'],model_initialization_observability=m['initialization_observability'],model_reviewer=m['reviewer'],model_review_date=m['review_date'],
                human_status=h['status'],human_confirmed=True,human_reviewer='gb',review_round=manifest['human_review_round'],
                confirmed_category=h['confirmed_category'],confirmed_stable_attributes=h['stable_attributes'],confirmed_uncertain_attributes=h['uncertain_attributes'],human_note=h['note']))
    with (SITE/'data/gpt_initial_reviews.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    assert len(flat)==2047
    receipt=dict(status='complete_submitted_human_initialization_import',observed_at=datetime.now().astimezone().isoformat(),total=2047,confirmed=2047,sources=sources,datasets=summary,old_depthtrack_round_used=False,training_only_depthtrack_train=True,human_status_not_assumed_to_measure_generator_accuracy=True)
    (SITE/'HUMAN_REVIEW_IMPORT_20261007.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (PROJECT/'人工核验文本导入回执_20261007.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (PROJECT/'三数据集_GPT及人工审核_2047条_20261007.csv').write_bytes((SITE/'data/gpt_initial_reviews.csv').read_bytes())
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
