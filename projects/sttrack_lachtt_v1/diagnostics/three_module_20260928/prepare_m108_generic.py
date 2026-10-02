"""Add one generic nonempty CLIP vector to the unchanged private M107 bank."""
import argparse,json,time
from pathlib import Path
import clip,torch
from analyze_train_states import sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    assert sha(a.bank)=='7acb2f5509b23bc0a900f6c252017672f04961394054a85deb9b01e697ef388c'
    bank=torch.load(a.bank,map_location='cpu')
    assert len(bank['sequences'])==152 and not bank['human_confirmed']
    weight=Path('/root/autodl-tmp/sutrack_assets/weights/ViT-L-14.pt')
    assert sha(weight)==bank['encoder_sha256']=='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
    torch.set_num_threads(4);started=time.time()
    model,_=clip.load(str(weight),device='cuda',jit=False)
    model=model.float().eval().requires_grad_(False)
    with torch.no_grad():
        vector=model.encode_text(clip.tokenize(['object']).cuda()).float().cpu()[0]
    assert vector.shape==(768,) and bool(torch.isfinite(vector).all())
    bank['generic']=vector
    bank['generic_phrase']='object'
    bank['m107_bank_sha256']=sha(a.bank)
    torch.save(bank,a.output)
    receipt=dict(status='complete_M108_generic_encoding',source_sha256=sha(__file__),
                 input_bank_sha256=sha(a.bank),bank_sha256=sha(a.output),
                 generic_phrase='object',human_confirmed=False,elapsed_seconds=time.time()-started)
    a.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
