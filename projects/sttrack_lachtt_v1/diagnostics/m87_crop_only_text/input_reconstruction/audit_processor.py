from pathlib import Path
import hashlib, inspect, json, os, shutil, sys, time

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
from PIL import Image
import transformers, qwen_vl_utils.vision_process as vision
from transformers import AutoProcessor

R=Path(__file__).resolve().parent
W=R.parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
digest=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
torch.set_num_threads(2)
started=time.time()
spec=read(W/'caption_spec.json')
records=[json.loads(x) for x in (W/'captions/records.jsonl').read_text().splitlines()]
assert len(records)==len(spec['rows'])==152
assert sha(W/'caption_protocol.py')==spec['source_sha256']
assert sha(W/'captions/records.jsonl')=='8141f3746b6ddec25beab547fe2bbb7d242b5bb617da830f971af530678208f2'
model=Path(spec['model_path'])
for name in ['config.json','preprocessor_config.json','tokenizer_config.json','tokenizer.json']:
    assert sha(model/name)==spec['model_sha256'][name],name
processor=AutoProcessor.from_pretrained(str(model),local_files_only=True,use_fast=False)
ip=processor.image_processor
config=read(model/'config.json')
assert config['vision_config']['patch_size']==ip.patch_size==14
assert config['vision_config']['temporal_patch_size']==ip.temporal_patch_size==2
assert config['vision_config']['spatial_merge_size']==ip.merge_size==2
assert ip.do_convert_rgb and ip.do_rescale and ip.do_normalize
selected={'010','028','042','062','068','071','105','128'}
out=R/'output';out.mkdir();(out/'images').mkdir();(out/'sources').mkdir()
sources={}
for label,obj in [('image_processor',type(ip)),('processor',type(processor)),('vision_process',vision)]:
    source=Path(inspect.getfile(obj));target=out/'sources'/(label+'.py')
    shutil.copyfile(source,target);sources[label]=dict(installed_path=str(source),sha256=sha(source))
rows=[]
for i,(row,rec) in enumerate(zip(spec['rows'],records)):
    audit_id='%03d'%(i+1)
    assert row['sequence']==rec['sequence'] and row['target_xyxy']==rec['target_xyxy']
    assert sha(Path(row['image']))==row['image_sha256']==rec['image_sha256']
    with Image.open(row['image']) as im:
        assert list(im.size)==row['image_size']
        crop=im.convert('RGB').crop(row['target_xyxy'])
    assert list(crop.size)==rec['original_crop_size']
    content=[dict(type='image',image=crop,**spec['image_pixels']),dict(type='text',text=spec['prompt'])]
    messages=[dict(role='user',content=content)]
    chat=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    assert chat.count('<|image_pad|>')==1 and chat.count(spec['prompt'])==1
    images,videos=vision.process_vision_info(messages)
    assert len(images)==1 and videos is None
    inputs=processor(text=[chat],images=images,videos=videos,padding=True,return_tensors='pt')
    assert inputs.image_grid_thw.tolist()==rec['processor_grid_thw']
    gt,gh,gw=inputs.image_grid_thw[0].tolist()
    p=ip.patch_size;t=ip.temporal_patch_size;m=ip.merge_size
    flat=inputs.pixel_values.numpy()
    assert gt==1 and flat.shape==(gt*gh*gw,3*t*p*p) and np.isfinite(flat).all()
    tokens=int((inputs.input_ids==config['image_token_id']).sum())
    assert tokens==gt*gh*gw//(m*m)
    assert int((inputs.input_ids==config['vision_start_token_id']).sum())==1
    assert int((inputs.input_ids==config['vision_end_token_id']).sum())==1
    packed=flat.reshape(gt,gh//m,gw//m,m,m,3,t,p,p)
    frames=packed.transpose(0,6,5,1,3,7,2,4,8).reshape(gt*t,3,gh*p,gw*p)
    assert np.array_equal(frames[0],frames[1])
    rgb=(frames[0].transpose(1,2,0)*np.asarray(ip.image_std,dtype=np.float32)+np.asarray(ip.image_mean,dtype=np.float32))/ip.rescale_factor
    actual=np.asarray(images[0])
    assert actual.shape==rgb.shape
    restored=np.clip(np.rint(rgb),0,255).astype(np.uint8)
    maxerror=float(np.abs(rgb-actual.astype(np.float32)).max())
    assert np.array_equal(restored,actual),row['sequence']
    item=dict(audit_id=audit_id,sequence=row['sequence'],split=row['split'],category=rec['category'],
        image_sha256=row['image_sha256'],crop_size=list(crop.size),crop_rgb_sha256=digest(np.asarray(crop)),
        resized_size=list(images[0].size),grid_thw=[gt,gh,gw],historical_grid_matches=True,
        input_shape=list(flat.shape),input_dtype=str(flat.dtype),image_tokens=tokens,
        input_ids_sha256=digest(inputs.input_ids.numpy()),pixel_values_sha256=digest(flat),
        pixel_min=float(flat.min()),pixel_max=float(flat.max()),pixel_std=float(flat.std()),
        rgb_roundtrip_exact=True,max_denormalization_float_error=maxerror,temporal_duplicate_exact=True,
        reconstructed_rgb_sha256=digest(restored))
    rows.append(item)
    if audit_id in selected:
        crop.save(out/'images'/(audit_id+'_crop.png'))
        Image.fromarray(restored).save(out/'images'/(audit_id+'_reconstructed.png'))
(out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
result=dict(status='completed_cpu_processor_reconstruction',rows=len(rows),
    checks=dict(source_image_hash=152,crop_size=152,historical_grid=152,single_image_markers=152,
                image_token_count=152,finite_pixel_values=152,rgb_roundtrip_exact=152,temporal_duplicate_exact=152),
    generator_calls=0,model_weight_loads=0,optimizer_steps=0,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
    image_tokens_range=[min(r['image_tokens'] for r in rows),max(r['image_tokens'] for r in rows)],
    crop_short_side_quantiles=np.quantile([min(r['crop_size']) for r in rows],[0,.25,.5,.75,1]).tolist(),
    max_denormalization_float_error=max(r['max_denormalization_float_error'] for r in rows),
    output_rows_sha256=sha(out/'rows.json'),script_sha256=sha(Path(__file__)),plan_sha256=sha(R/'INPUT_AUDIT_PLAN.md'),
    caption_spec_sha256=sha(W/'caption_spec.json'),caption_records_sha256=sha(W/'captions/records.jsonl'),
    library_sources=sources,python=sys.version,torch=torch.__version__,transformers=transformers.__version__,
    elapsed_seconds=time.time()-started,selected_visual_ids=sorted(selected),
    historical_pixel_values_saved=False,historical_generation_replayed=False,caption_semantic_accuracy_estimated=False,
    limitations=['CPU reconstruction verifies current preprocessing, not unrecorded historical tensors',
        'Historical grid and source image hashes match; model visual attention and generation semantics not checked',
        'No conclusion that caption errors are caused by crop size or loss of context'])
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
manifest=[dict(path=str(p.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()]
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(result,indent=2))
