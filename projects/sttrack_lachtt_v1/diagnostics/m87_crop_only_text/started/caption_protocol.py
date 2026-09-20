"""Frozen crop-only initialization categories; no tracking or later image access."""
import hashlib
import json
from pathlib import Path
import sys
import time

R=Path(__file__).parent
B=Path('/root/autodl-tmp')
OLD=B/'sttrack_m58_semantic_spatial_v1_20260906'
MODEL=B/'qwen/Qwen2.5-VL-3B-Instruct'
PROMPT='''This image is a tight crop of the target object selected for tracking. Identify the target object in this crop. Return one JSON object with exactly one key, "category", whose value is a short lowercase English object category. Do not describe attributes, background, the person holding the object, or objects outside this crop. Do not infer a brand or a function that is not visible. If the object category cannot be determined from the visible evidence, use "object". Do not use Markdown.'''

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def prepare():
    from PIL import Image
    assert not (R/'caption_spec.json').exists()
    old=read(OLD/'initial_captions_v2/plan.json')
    assert sha(OLD/'initial_captions_v2/plan.json')=='50d83fe2811acbec045fd35dee255f634ada284aa39fc43e62142d2b36a2addb'
    for name,digest in old['model_sha256'].items():assert sha(MODEL/name)==digest,name
    for row in old['rows']:
        assert sha(row['image'])==row['image_sha256']
        with Image.open(row['image']) as im:
            assert list(im.size)==row['image_size']
            x1,y1,x2,y2=row['target_xyxy'];assert 0<=x1<x2<=im.width and 0<=y1<y2<=im.height
    spec=dict(status='frozen_before_generation',source_sha256=sha(__file__),plan_sha256=sha(R/'EXPERIMENT_PLAN.md'),
        original_caption_plan_sha256=sha(OLD/'initial_captions_v2/plan.json'),model_path=str(MODEL),
        model_sha256=old['model_sha256'],prompt=PROMPT,rows=old['rows'],seed=2027,
        generation=dict(max_new_tokens=48,do_sample=False,use_cache=True),
        image_pixels=dict(min_pixels=128*28*28,max_pixels=256*28*28),
        full_frame_provided=False,old_text_provided=False,review_labels_provided=False,
        sequence_name_provided=False,later_images_provided=False,optimizer_steps=0)
    save(R/'caption_spec.json',spec)
    print(json.dumps(dict(status=spec['status'],rows=len(spec['rows']),spec_sha256=sha(R/'caption_spec.json'))),flush=True)

def parse_category(raw):
    payload=raw.strip()
    if payload.startswith('```json\n') and payload.endswith('\n```'):
        payload=payload[8:-4]
    value=json.loads(payload)
    assert set(value)=={'category'} and isinstance(value['category'],str)
    category=value['category'];assert category==category.strip().lower() and category
    return category


def generate():
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    spec=read(R/'caption_spec.json')
    assert sha(__file__)==spec['source_sha256'] and spec['prompt']==PROMPT
    assert sha(R/'EXPERIMENT_PLAN.md')==spec['plan_sha256']
    for name,digest in spec['model_sha256'].items():assert sha(MODEL/name)==digest
    output=R/'captions'
    reuse=read(R/'format_amendment.json')
    first_raw=output/(spec['rows'][0]['sequence']+'.raw.txt')
    assert sha(first_raw)==reuse['first_raw_sha256']
    assert reuse['new_source_sha256']==sha(__file__)
    assert reuse['new_plan_sha256']==sha(R/'EXPERIMENT_PLAN.md')
    torch.set_num_threads(4);torch.manual_seed(spec['seed']);torch.cuda.manual_seed_all(spec['seed'])
    started=time.time()
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(MODEL),local_files_only=True,
        torch_dtype=torch.float16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
    processor=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True,use_fast=False)
    count=0
    with (output/'records.jsonl').open('x') as stream:
        for row in spec['rows']:
            assert sha(row['image'])==row['image_sha256']
            image=Image.open(row['image']).convert('RGB').crop(row['target_xyxy'])
            content=[dict(type='image',image=image,**spec['image_pixels']),dict(type='text',text=PROMPT)]
            messages=[dict(role='user',content=content)]
            chat=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            images,videos=process_vision_info(messages)
            assert len(images)==1 and videos is None
            inputs=processor(text=[chat],images=images,videos=videos,padding=True,return_tensors='pt').to(model.device)
            if count==0:
                raw=first_raw.read_text()
            else:
                with torch.inference_mode():generated=model.generate(**inputs,**spec['generation'])
                raw=processor.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)[0]
                (output/(row['sequence']+'.raw.txt')).write_text(raw)
            category=parse_category(raw)
            record=dict(sequence=row['sequence'],split=row['split'],image_sha256=row['image_sha256'],
                target_xyxy=row['target_xyxy'],raw=raw,category=category,input_image_count=1,
                original_crop_size=list(image.size),processor_grid_thw=inputs.image_grid_thw.cpu().tolist(),
                elapsed_seconds=time.time()-started)
            stream.write(json.dumps(record)+'\n');stream.flush();count+=1
            print(json.dumps(dict(completed=count,**record)),flush=True)
    assert count==152
    result=dict(status='all_crop_only_categories_generated',rows=count,seed=spec['seed'],
        spec_sha256=sha(R/'caption_spec.json'),source_sha256=sha(__file__),
        records_sha256=sha(output/'records.jsonl'),elapsed_seconds=time.time()-started,
        actual_qwen_calls=count,new_qwen_calls_this_launch=count-1,reused_previous_raw_responses=1,format_amendment_sha256=sha(R/'format_amendment.json'),optimizer_steps=0,semantic_correctness_verified=False)
    save(R/'caption_result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    assert sys.argv[1] in ['prepare','generate']
    prepare() if sys.argv[1]=='prepare' else generate()
