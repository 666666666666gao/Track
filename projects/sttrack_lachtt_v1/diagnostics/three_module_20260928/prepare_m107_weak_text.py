"""Encode user-authorized model weak labels with the existing frozen CLIP."""
import argparse,json,time
from pathlib import Path
import clip,torch
from analyze_train_states import sha

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();assert not a.output.exists()
 torch.set_num_threads(4)
 labels=json.loads(a.labels.read_text())
 assert labels['status']=='prepared_user_authorized_model_weak_labels' and not labels['human_confirmed']
 rows=labels['initial'];assert len(rows)==152
 weight=Path('/root/autodl-tmp/sutrack_assets/weights/ViT-L-14.pt')
 assert sha(weight)=='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
 started=time.time()
 model,_=clip.load(str(weight),device='cuda',jit=False)
 model=model.float().eval().requires_grad_(False)
 phrases=sorted({''}|{s for r in rows for s in r['phrases']})
 tokens=clip.tokenize(phrases,truncate=False)
 with torch.no_grad():
  encoded=torch.cat([model.encode_text(b.cuda()).float().cpu() for b in tokens.split(32)])
 assert encoded.shape==(len(phrases),768) and bool(torch.isfinite(encoded).all())
 vectors=dict(zip(phrases,encoded));text=torch.zeros(152,5,768);mask=torch.zeros(152,5,dtype=torch.bool)
 for i,r in enumerate(rows):
  assert 1<=len(r['phrases'])<=5
  text[i,:len(r['phrases'])]=torch.stack([vectors[s] for s in r['phrases']])
  mask[i,:len(r['phrases'])]=True
 bank=dict(sequences=[r['sequence'] for r in rows],splits=[r['split'] for r in rows],tokens=text,mask=mask,empty=vectors[''],labels_sha256=sha(a.labels),encoder_sha256=sha(weight),source_sha256=sha(__file__),human_confirmed=False)
 torch.save(bank,a.output)
 receipt=dict(status='complete_model_weak_text_encoding',human_confirmed=False,sequences=152,unique_phrases=len(phrases),bank_sha256=sha(a.output),labels_sha256=sha(a.labels),encoder_sha256=sha(weight),source_sha256=sha(__file__),elapsed_seconds=time.time()-started)
 a.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
