"""Encode the explicit M111 category queries using the existing frozen CLIP."""
import argparse
import json
import time
from pathlib import Path
import clip
import torch
from analyze_train_states import sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    manifest = json.loads(a.manifest.read_text())
    assert manifest['status'] == 'prepared_M111_first_frame_model_weak_categories'
    assert not manifest['human_confirmed'] and len(manifest['rows']) == 117
    weight = Path('/root/autodl-tmp/sutrack_assets/weights/ViT-L-14.pt')
    assert sha(weight) == 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
    torch.set_num_threads(4)
    started = time.time()
    model, _ = clip.load(str(weight), device='cuda', jit=False)
    model = model.float().eval().requires_grad_(False)
    queries = [r['query'] for r in manifest['rows']]
    with torch.no_grad():
        vectors = torch.cat([model.encode_text(b.cuda()).float().cpu()
                             for b in clip.tokenize(queries, truncate=False).split(32)])
    assert vectors.shape == (117, 768) and bool(torch.isfinite(vectors).all())
    bank = dict(tokens=vectors, sequences=[r['sequence'] for r in manifest['rows']],
                manifest_sha256=sha(a.manifest), encoder_sha256=sha(weight),
                source_sha256=sha(__file__), human_confirmed=False)
    torch.save(bank, a.output)
    receipt = dict(status='complete_M111_original_category_encoding', sequences=117,
                   bank_sha256=sha(a.output), manifest_sha256=sha(a.manifest),
                   encoder_sha256=sha(weight), source_sha256=sha(__file__),
                   human_confirmed=False, elapsed_seconds=time.time() - started)
    a.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
