"""Encode the bound human external phrases; launch only on an available GPU."""
import argparse,json,time
from pathlib import Path
import clip,torch
from bind_m122_official_initializations import sha


def main():
    parser=argparse.ArgumentParser()
    for name in ['binding','clip-weight','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    binding=json.loads(args.binding.read_text());rows=binding['rows']
    assert binding['status']=='complete_M122_human_official_initialization_binding' and binding['RGB_bytes_and_exact_legal_bbox_bound']
    assert len(rows)==len({r['key'] for r in rows})==1895 and all(1<=len(r['phrases'])<=5 and r['phrases'][0].strip() for r in rows)
    assert sha(args.clip_weight)=='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
    torch.set_num_threads(1);started=time.time()
    phrases=sorted({''}|{p for row in rows for p in row['phrases']})
    # Match Train encoding: no translation, invented attributes, or silent truncation.
    tokens=clip.tokenize(phrases,truncate=False)
    model,_=clip.load(str(args.clip_weight),device='cuda',jit=False);model=model.float().eval().requires_grad_(False)
    with torch.no_grad():vectors=torch.cat([model.encode_text(part.cuda()).float().cpu() for part in tokens.split(32)])
    assert vectors.shape==(len(phrases),768) and bool(torch.isfinite(vectors).all())
    lookup=dict(zip(phrases,vectors));text=torch.zeros(1895,5,768);mask=torch.zeros(1895,5,dtype=torch.bool)
    for index,row in enumerate(rows):
        text[index,:len(row['phrases'])]=torch.stack([lookup[p] for p in row['phrases']]);mask[index,:len(row['phrases'])]=True
    assert bool(mask[:,0].all())
    bank=dict(format='M122_human_official_initialization_v1',keys=[r['key'] for r in rows],
        datasets=[r['dataset'] for r in rows],tokens=text,mask=mask,empty=lookup[''],human_confirmed=True,
        binding_sha256=sha(args.binding),encoder_sha256=sha(args.clip_weight),source_sha256=sha(__file__))
    torch.save(bank,args.output)
    result=dict(status='complete_M122_official_human_text_encoding',bank_sha256=sha(args.output),binding_sha256=sha(args.binding),
        encoder_sha256=sha(args.clip_weight),source_sha256=sha(__file__),initializations=1895,unique_phrases=len(phrases),
        optimizer_steps=0,subsequent_GT_opened=False,human_review_used_multiframe_aids=True,elapsed_seconds=time.time()-started)
    args.output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
