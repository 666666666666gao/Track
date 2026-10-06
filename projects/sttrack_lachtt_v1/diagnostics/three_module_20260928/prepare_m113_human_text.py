"""Encode confirmed Train initialization phrases with the existing frozen CLIP."""
import argparse,json,time
from pathlib import Path
import clip,torch
from analyze_train_states import sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    labels=json.loads(a.labels.read_text())
    assert labels['status']=='prepared_human_confirmed_train_initialization'
    assert labels['human_confirmed'] and labels['dataset']=='depthtrack'
    rows=labels['initial'];assert len(rows)==152
    assert len({r['sequence'] for r in rows})==152
    assert all(r['split'] in ('fit','development') and 1<=len(r['phrases'])<=5 for r in rows)
    weight=Path('/root/autodl-tmp/sutrack_assets/weights/ViT-L-14.pt')
    assert sha(weight)=='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
    torch.set_num_threads(4);started=time.time()
    model,_=clip.load(str(weight),device='cuda',jit=False)
    model=model.float().eval().requires_grad_(False)
    phrases=sorted({'','object'}|{s for r in rows for s in r['phrases']})
    tokens=clip.tokenize(phrases,truncate=False)
    with torch.no_grad():
        encoded=torch.cat([model.encode_text(b.cuda()).float().cpu() for b in tokens.split(32)])
    assert encoded.shape==(len(phrases),768) and bool(torch.isfinite(encoded).all())
    vectors=dict(zip(phrases,encoded))
    text=torch.zeros(152,5,768);mask=torch.zeros(152,5,dtype=torch.bool)
    for i,r in enumerate(rows):
        text[i,:len(r['phrases'])]=torch.stack([vectors[s] for s in r['phrases']])
        mask[i,:len(r['phrases'])]=True
    bank=dict(sequences=[r['sequence'] for r in rows],splits=[r['split'] for r in rows],
              tokens=text,mask=mask,empty=vectors[''],generic=vectors['object'],
              labels_sha256=sha(a.labels),encoder_sha256=sha(weight),source_sha256=sha(__file__),
              human_confirmed=True,dataset='depthtrack',review_round='20261006')
    torch.save(bank,a.output)
    receipt=dict(status='complete_M113_human_text_encoding',human_confirmed=True,
                 dataset='depthtrack',sequences=152,unique_phrases=len(phrases),
                 bank_sha256=sha(a.output),labels_sha256=sha(a.labels),encoder_sha256=sha(weight),
                 source_sha256=sha(__file__),elapsed_seconds=time.time()-started)
    a.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
