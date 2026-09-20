"""Change only category content in the original five-slot text interface."""
from pathlib import Path
import hashlib,json,sys,time
import torch
from caption_protocol import parse_category

R=Path(__file__).parent;B=Path('/root/autodl-tmp')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
    started=time.time()
    result=read(R/'caption_result.json');spec=read(R/'caption_spec.json')
    assert result['status']=='all_crop_only_categories_generated' and result['rows']==152
    assert result['spec_sha256']==sha(R/'caption_spec.json') and result['source_sha256']==sha(R/'caption_protocol.py')
    assert spec['plan_sha256']==sha(R/'EXPERIMENT_PLAN.md')
    assert sha(R/'captions/records.jsonl')==result['records_sha256']
    records=[json.loads(x) for x in (R/'captions/records.jsonl').read_text().splitlines()]
    assert len(records)==152 and [x['sequence'] for x in records]==[x['sequence'] for x in spec['rows']]
    for rec,row in zip(records,spec['rows']):
        assert rec['image_sha256']==row['image_sha256'] and rec['target_xyxy']==row['target_xyxy']
        assert rec['split']==row['split'] and rec['input_image_count']==1
        assert parse_category(rec['raw'])==rec['category']
        assert (R/'captions'/(rec['sequence']+'.raw.txt')).read_text()==rec['raw']
    by_sequence={r['sequence']:r for r in records}
    oldspec=read(B/'sttrack_m84_centered_20260920/training_spec.json')
    cache={};oldbanks={};empty=None;original_hashes={}
    for split in ['fit','development']:
        p=B/'sttrack_m58_semantic_spatial_v2_20260906'/('text_'+split+'.pt')
        original=torch.load(p,map_location='cpu');original_hashes[split]=sha(p)
        item=oldspec['banks'][split]['category'];assert sha(item['path'])==item['sha256']
        old=torch.load(item['path'],map_location='cpu');oldbanks[split]=old
        assert original['sequences']==old['sequences'] and torch.equal(original['mask'],old['mask'])
        assert original_hashes[split]==old['source_bank_sha256']
        if empty is None:empty=old['empty'].clone()
        assert torch.equal(old['empty'],empty) and torch.equal(original['empty'],empty)
        for i,phrases in enumerate(original['initialization_phrases']):
            for k,phrase in enumerate(phrases):
                vector=original['tokens'][i,k].clone()
                if phrase in cache:assert torch.equal(cache[phrase],vector)
                cache[phrase]=vector
    existing=set(cache)
    categories=sorted({r['category'] for r in records})
    assert all(s and s==s.strip().lower() for s in categories)
    new=[s for s in categories if s not in cache]
    weight=B/'sutrack_assets/weights/ViT-L-14.pt'
    assert sha(weight)=='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
    if new:
        sys.path.insert(0,str(B/'sutrack_assets/openai-clip'))
        import clip
        torch.set_num_threads(4)
        encoder,_=clip.load(str(weight),device='cpu',jit=False)
        encoder=encoder.float().eval().requires_grad_(False)
        ids=clip.tokenize(new,truncate=False)
        with torch.no_grad():vectors=torch.cat([encoder.encode_text(batch).float() for batch in ids.split(32)])
        assert vectors.shape==(len(new),768) and bool(torch.isfinite(vectors).all())
        cache.update(dict(zip(new,vectors)))
    out=R/'banks';out.mkdir()
    bindings={};mapping=[];changed={}
    for split,old in oldbanks.items():
        cats=[by_sequence[s]['category'] for s in old['sequences']]
        assert all(by_sequence[s]['split']==split for s in old['sequences'])
        tokens=old['tokens'].clone()
        for i,c in enumerate(cats):tokens[i,0]=cache[c]
        assert torch.equal(tokens[:,1:],old['tokens'][:,1:])
        changed[split]=sum(not torch.equal(tokens[i,0],old['tokens'][i,0]) for i in range(len(cats)))
        category=dict(sequences=old['sequences'],tokens=tokens,mask=old['mask'].clone(),empty=empty.clone(),
            categories=cats,caption_records_sha256=result['records_sha256'],caption_spec_sha256=sha(R/'caption_spec.json'),
            encoder_sha256=sha(weight),source_category_bank_sha256=oldspec['banks'][split]['category']['sha256'],
            lexical_policy='Only category slot0 replaced; original mask, valid empty attributes, padding and empty vector unchanged.')
        p=out/(split+'_category.pt');torch.save(category,p)
        bindings[split]={'category':dict(path=str(p),sha256=sha(p))}
        # Reuse the exact historical Empty bank, without reserializing vectors.
        item=oldspec['banks'][split]['empty'];assert sha(item['path'])==item['sha256']
        eb=torch.load(item['path'],map_location='cpu')
        assert eb['sequences']==old['sequences'] and torch.equal(eb['mask'],old['mask']) and torch.equal(eb['empty'],empty)
        assert torch.equal(eb['tokens'][eb['mask']],empty.expand(int(eb['mask'].sum()),-1))
        assert torch.equal(eb['tokens'][~eb['mask']],old['tokens'][~old['mask']])
        bindings[split]['empty']=item
        if split=='development':
            bindings[split]['old']=oldspec['banks'][split]['category']
            swap=dict(category);swap['tokens']=tokens.clone();donated=[]
            for i,c in enumerate(cats):
                donor=next((i+k)%len(cats) for k in range(1,len(cats)) if cats[(i+k)%len(cats)]!=c)
                swap['tokens'][i,0]=tokens[donor,0];donated.append(cats[donor])
                mapping.append(dict(sequence=old['sequences'][i],original_category=c,donor_sequence=old['sequences'][donor],donated_category=cats[donor]))
            swap['categories']=donated;swap['lexical_policy']='Different-string donor in frozen cyclic sequence order; slot0 only; not a guaranteed semantic contradiction.'
            p=out/'development_swapped.pt';torch.save(swap,p);bindings[split]['swapped']=dict(path=str(p),sha256=sha(p))
    save(R/'swapped_mapping.json',mapping)
    report=dict(status='crop_only_text_banks_complete',source_sha256=sha(__file__),caption_spec_sha256=sha(R/'caption_spec.json'),
        caption_result_sha256=sha(R/'caption_result.json'),banks=bindings,changed_category_vectors=changed,
        original_full_text_bank_sha256=original_hashes,unique_categories=len(categories),cached_categories=len(categories)-len(new),newly_encoded_categories=len(new),
        vector_sources={s:'historical_exact_string' if s in existing else 'same_frozen_CLIP_CPU' for s in categories},
        original_mask_padding_empty_and_other_slots_exact=True,swapped_mapping_sha256=sha(R/'swapped_mapping.json'),
        subsequent_GT_or_metrics_read=False,training_started=False,elapsed_seconds=time.time()-started)
    save(R/'bank_result.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
