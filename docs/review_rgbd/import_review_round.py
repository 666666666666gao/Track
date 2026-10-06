"""Import the supplied DepthTrack re-review and submitted external human CSVs."""
from pathlib import Path
from collections import Counter
from PIL import Image
import csv, hashlib, io, json, shutil, zipfile

SITE = Path(__file__).parent
PROJECT = Path(r'C:\Users\gb\Desktop\document\RGBD_LANGUAGE_TRACKING')
ROUND = '20261006'
VERSION = '20261006-depthtrack-rereview'
TRAIN_PACKET = PROJECT / 'DepthTrack_GPT初审包_20261005'
TEST_PACKET = PROJECT / 'DepthTrack_Test50_GPT初审包_20261005'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(data):
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')


def model_review(row):
    assert row['human_status']=='pending' and row['human_confirmed']=='false'
    assert row['reviewed_category_status'] in {'supported','conflicting','uncertain','not_visualized'}
    assert row['reviewed_category_or_coarse_label'] and row['reviewer'] and row['review_note']
    return dict(status=row['reviewed_category_status'],category=row['reviewed_category_or_coarse_label'],
                stable_attributes=row['supported_stable_attributes'],uncertain_attributes=row['unobservable_attributes'],
                conflicting_attributes=row['conflicting_attributes'],context_only_notes=row['video_only_attributes']+'\n'+row['multiframe_review_note'],
                evidence=row['review_note'],evidence_frames=row['evidence_frames'],initialization_observability=row['first_frame_category_observability'],
                reviewer=row['reviewer'],review_date=row['review_date'],image_reviewed=row['model_image_reviewed'],images_viewed=row['model_images_viewed'],
                video_reviewed=row['model_video_reviewed'],video_frames=row['model_video_reviewed_frames'])


def bind_human(data, manifest):
    rows=read_csv(data)
    by_id={r['key']:r for r in rows}
    assert len(rows)==len(by_id)==manifest['count']
    assert set(by_id)=={r['id'] for r in manifest['rows']}
    result={}
    for item in manifest['rows']:
        r=by_id[item['id']]
        assert r['dataset']==manifest['dataset'] and r['sequence']==item['sequence'] and r['init_frame']==item['frame']
        assert int(r['anchor_number'])==item['number']
        assert r['human_confirmed']=='true' and r['status'] in {'supported','conflicting','uncertain'}
        assert r['reviewer_id'] and r['updated_at']
        result[item['id']]={r['reviewer_id']:dict(status=r['status'],confirmed_category=r['confirmed_category'],stable_attributes=r['stable_attributes'],
                                               uncertain_attributes=r['uncertain_attributes'],note=r['note'],updated_at=r['updated_at'],human_confirmed=True,
                                               review_round='initial',review_subject='original_caption')}
    return result


def verify_model_inputs(rows, template):
    assert len(rows)==len(template)
    original={r['audit_id']:r for r in template}
    assert len(original)==len({r['audit_id'] for r in rows})==len(rows)
    assert set(original)=={r['audit_id'] for r in rows}
    fields=['record_type','audit_id','generated_category','generated_attributes','evidence_frames','dataset','sequence','init_frame','init_frame_index','sequence_frames',
            'initialization_board_path','video_path','initialization_context_frames']
    for r in rows:
        expected=original[r['audit_id']]
        for field in fields:
            assert r[field]==expected[field], (r['audit_id'],field)
        model_review(r)
    return {r['audit_id']:r for r in rows}


def main():
    paths={dataset:SITE/'data'/f'{dataset}.json' for dataset in ['depthtrack','cdtb','vot']}
    manifests={key:json.loads(path.read_text(encoding='utf-8')) for key,path in paths.items()}
    assert manifests['depthtrack'].get('model_review_version')!=VERSION, 'Do not re-import the same round over its history'
    before={key:sha(path.read_bytes()) for key,path in paths.items()}
    source_train=PROJECT/'DepthTrack_152_GPT初审结果.csv'
    source_test=PROJECT/'DepthTrack_Test50_完整初审结果_20261006.zip'
    train_bytes=source_train.read_bytes()
    with zipfile.ZipFile(source_test) as archive:
        member='DepthTrack_Test50_完整初审结果_20261006/DepthTrack_Test50_GPT初审结果.csv'
        test_bytes=archive.read(member)
    train_rows=read_csv(train_bytes)
    test_rows=read_csv(test_bytes)
    assert len(train_rows)==152 and len(test_rows)==50
    train=verify_model_inputs(train_rows,read_csv((TRAIN_PACKET/'DepthTrack_152_初审表_待填写.csv').read_bytes()))
    test=verify_model_inputs(test_rows,read_csv((TEST_PACKET/'DepthTrack_Test50_初审表_待填写.csv').read_bytes()))
    for r in test_rows:
        assert r['dataset']=='depthtrack_test' and r['split']=='test' and r['reference_caption_available']=='no'
        assert r['generated_category']==r['generated_attributes']==''
    human_sources={key:PROJECT/'人工审核结果'/name for key,name in [('cdtb','rgbd_cdtb_gb_80_review.csv'),('vot','rgbd_vot_gb_1765_review.csv')]}
    human_bytes={key:path.read_bytes() for key,path in human_sources.items()}
    humans={key:bind_human(data,manifests[key]) for key,data in human_bytes.items()}
    old_train_models=[dict(id=r['id'],sequence=r['sequence'],frame=r['frame'],model_review=r['model_review']) for r in manifests['depthtrack']['rows']]
    hashes=json.loads((SITE/'file_hashes.json').read_text(encoding='utf-8'))
    new_media=[]

    def copy_media(packet, source_rel, target_rel, expected):
        data=(packet/source_rel).read_bytes()
        assert sha(data)==expected['sha256'] and len(data)==expected['bytes']
        destination=SITE/target_rel
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(data)
        assert destination.read_bytes()==data
        hashes[target_rel]={'bytes':len(data),'sha256':sha(data)}
        new_media.append(target_rel)

    train_packet=json.loads((TRAIN_PACKET/'package_manifest.json').read_text(encoding='utf-8'))
    test_packet=json.loads((TEST_PACKET/'package_manifest.json').read_text(encoding='utf-8'))
    test_cases={r['audit_id']:r for r in test_packet['cases']}
    for item in manifests['depthtrack']['rows']:
        row=train[item['id']]
        assert item['sequence']==row['sequence'] and item['frame']==row['init_frame'] and item['frame_index']==int(row['init_frame_index'])
        assert item['sequence_frames']==int(row['sequence_frames']) and item['category']==row['generated_category']
        assert item['attributes']==[s.strip() for s in row['generated_attributes'].split('|') if s.strip()]
        item['model_review']=model_review(row)
        item['human_reviews']={}
        item['temporal_board']=f'media/depthtrack/target_crops_20261005/{item["id"]}.jpg'
        item['temporal_board_label']='Train九帧目标裁剪 · 后帧由Train GT定位，仅作离线审核'
        item['temporal_board_frames']=row['evidence_frames']
        item['temporal_uses_train_gt']=True
        item['video_sample_fps']=8
        copy_media(TRAIN_PACKET,row['train_target_crops_path'],item['temporal_board'],train_packet['files'][row['train_target_crops_path']])
    manifests['depthtrack'].update(model_review_version=VERSION,human_review_round=ROUND,review_subject='gpt_proposal',reference_caption_available=True,display_name='DepthTrack Train',published_reviewers=[])

    test_manifest=dict(dataset='depthtrack_test',display_name='DepthTrack Test',count=50,sequence_count=50,initialization_only=True,review_fps=25,
                       review_fps_is_a_playback_setting_not_measured_capture_rate=True,video_sample_fps=2,reference_caption_available=False,
                       model_review_version=VERSION,human_review_round=ROUND,review_subject='gpt_proposal',published_reviewers=[],rows=[])
    for i in range(1,51):
        row=test[f'TEST{i:03}']
        case=test_cases[row['audit_id']]
        item=dict(number=i,id=row['audit_id'],sequence=row['sequence'],frame=row['init_frame'],frame_index=0,sequence_frames=int(row['sequence_frames']),
                  category='',attributes=[],model_review=model_review(row),human_reviews={},video_start_seconds=0,video_sample_fps=2,
                  board=f'media/depthtrack_test/boards/{row["audit_id"]}.jpg',board_detail=f'media/depthtrack_test/details_20261005/{row["audit_id"]}.jpg',
                  temporal_board=f'media/depthtrack_test/overviews_20261005/{row["audit_id"]}.jpg',
                  temporal_board_label='整条序列九帧全图 · 后帧无GT框、无目标裁剪',temporal_board_frames=row['evidence_frames'],temporal_uses_train_gt=False,
                  video=f'media/depthtrack_test/videos_20261005/{row["audit_id"]}.mp4',original_frames=[])
        assert case['sequence']==item['sequence'] and case['frame_count']==item['sequence_frames']
        assert row['initial_bbox_xywh']==' | '.join(map(str,case['init_bbox']))
        copy_media(TEST_PACKET,row['initialization_board_path'],item['board_detail'],test_packet['files'][row['initialization_board_path']])
        preview=SITE/item['board']
        preview.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(SITE/item['board_detail']) as image:
            image.resize((960,565),Image.Resampling.LANCZOS).save(preview,'JPEG',quality=72,optimize=True)
        hashes[item['board']]={'bytes':preview.stat().st_size,'sha256':sha(preview.read_bytes())}
        new_media.append(item['board'])
        copy_media(TEST_PACKET,row['sequence_overview_path'],item['temporal_board'],test_packet['files'][row['sequence_overview_path']])
        copy_media(TEST_PACKET,row['video_path'],item['video'],test_packet['files'][row['video_path']])
        for frame in case['selected_rgb_frames']:
            rel=f'original_frames/{item["id"]}/{frame:08}.jpg'
            target=f'media/depthtrack_test/original_frames/{item["id"]}/{frame:08}.jpg'
            copy_media(TEST_PACKET,rel,target,test_packet['files'][rel])
            item['original_frames'].append(dict(frame_index=frame,path=target))
        test_manifest['rows'].append(item)
    assert not {r['sequence'] for r in manifests['depthtrack']['rows']} & {r['sequence'] for r in test_manifest['rows']}
    manifests['depthtrack_test']=test_manifest
    for dataset in ['cdtb','vot']:
        for item in manifests[dataset]['rows']:
            item['human_reviews']=humans[dataset][item['id']]
        manifests[dataset].update(human_review_round='initial',review_subject='original_caption',reference_caption_available=True,display_name='CDTB' if dataset=='cdtb' else 'VOT-RGBD2022',published_reviewers=['gb'])
    write_json(SITE/'data/previous_model_depthtrack_20261003.json',old_train_models)
    for key,manifest in manifests.items():
        write_json(SITE/'data'/f'{key}.json',manifest)
    write_json(SITE/'file_hashes.json',hashes)
    sources=[]
    source_dir=SITE/'data/source_reviews_20261006'
    source_dir.mkdir(exist_ok=True)
    for name,data in [('depthtrack_train_gpt.csv',train_bytes),('depthtrack_test_gpt.csv',test_bytes),('cdtb_gb_human.csv',human_bytes['cdtb']),('vot_gb_human.csv',human_bytes['vot'])]:
        (source_dir/name).write_bytes(data)
        sources.append(dict(path='data/source_reviews_20261006/'+name,bytes=len(data),sha256=sha(data)))
    local_sources=PROJECT/'人工审核结果/DepthTrack_GPT重审_20261006'
    local_sources.mkdir(exist_ok=True)
    (local_sources/'DepthTrack_152_GPT初审结果.csv').write_bytes(train_bytes)
    (local_sources/'DepthTrack_Test50_GPT初审结果.csv').write_bytes(test_bytes)
    receipt=dict(version=VERSION,sources=sources,test_source_zip=dict(name=source_test.name,sha256=sha(source_test.read_bytes())),
                 original_manifest_sha256=before,previous_depthtrack_human_used=False,human_confirmation_fabricated=False,
                 new_media_files=len(new_media),new_media_bytes=sum(hashes[p]['bytes'] for p in new_media),datasets={})
    flat=[]
    for dataset in ['depthtrack','depthtrack_test','cdtb','vot']:
        manifest=manifests[dataset]
        submitted=sum(bool(item['human_reviews']) for item in manifest['rows'])
        receipt['datasets'][dataset]=dict(count=manifest['count'],model_statuses=dict(Counter(item['model_review']['status'] for item in manifest['rows'])),
                                         human_review_round=manifest['human_review_round'],current_submitted_human=submitted,current_human_pending=manifest['count']-submitted)
        for item in manifest['rows']:
            m=item['model_review']
            human=item['human_reviews'].get('gb',{})
            flat.append(dict(dataset=dataset,anchor_number=item['number'],key=item['id'],sequence=item['sequence'],init_frame=item['frame'],
                             model_status=m['status'],model_category=m['category'],model_stable_attributes=m['stable_attributes'],model_uncertain_attributes=m['uncertain_attributes'],
                             model_conflicting_attributes=m['conflicting_attributes'],model_context_only_notes=m['context_only_notes'],model_evidence=m['evidence'],
                             model_evidence_frames=m['evidence_frames'],model_initialization_observability=m['initialization_observability'],model_reviewer=m['reviewer'],model_review_date=m['review_date'],
                             human_status=human.get('status','pending'),human_confirmed=bool(human),human_reviewer='gb' if human else '',review_round=manifest['human_review_round']))
    with (SITE/'data/gpt_initial_reviews.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]))
        writer.writeheader();writer.writerows(flat)
    receipt.update(total=len(flat),current_submitted_human=1845,current_depthtrack_human_pending=202)
    write_json(SITE/'data/model_review_receipt.json',receipt)
    write_json(SITE/'REVIEW_ROUND_IMPORT_20261006.json',receipt)
    (PROJECT/'DepthTrack重审与外部人审导入回执_20261006.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
