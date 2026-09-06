from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import clip
import torch

root = Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v1_20260906')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
source = root / 'initial_captions_v2'
result = json.loads((source / 'result.json').read_text())
assert result['status'] == 'all_initial_captions_generated' and result['sequences'] == 152
assert sha(source / 'records.jsonl') == result['records_sha256']
records = [json.loads(line) for line in (source / 'records.jsonl').read_text().splitlines()]
rows = [dict(sequence=r['sequence'], split=r['split'], phrases=[r['category']] + r['attributes'],
             first_image_sha256=r['image_sha256']) for r in records]
assert len({r['sequence'] for r in rows}) == 152
assert all(1 <= len(r['phrases']) <= 5 for r in rows)
torch.set_num_threads(4)
started = time.time()
weight = Path('/root/autodl-tmp/sutrack_assets/weights/ViT-L-14.pt')
assert sha(weight) == 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
encoder, _ = clip.load(str(weight), device='cpu', jit=False)
encoder = encoder.float().eval().requires_grad_(False)
phrases = sorted({''} | {p for r in rows for p in r['phrases']})
ids = clip.tokenize(phrases, truncate=False)
with torch.no_grad():
    embeddings = torch.cat([encoder.encode_text(batch).float() for batch in ids.split(32)])
assert embeddings.shape == (len(phrases), 768) and bool(torch.isfinite(embeddings).all())
vectors = dict(zip(phrases, embeddings))
for split, expected in [('fit', 130), ('development', 22)]:
    selected = [r for r in rows if r['split'] == split]
    assert len(selected) == expected
    tokens = torch.zeros(expected, 5, 768)
    mask = torch.zeros(expected, 5, dtype=torch.bool)
    for i, row in enumerate(selected):
        tokens[i, :len(row['phrases'])] = torch.stack([vectors[p] for p in row['phrases']])
        mask[i, :len(row['phrases'])] = True
    bank = dict(sequences=[r['sequence'] for r in selected], tokens=tokens, mask=mask, empty=vectors[''],
                initialization_phrases=[r['phrases'] for r in selected],
                caption_records_sha256=result['records_sha256'], encoder_sha256=sha(weight))
    torch.save(bank, root / ('text_' + split + '.pt'))
(root / 'text_manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
record = dict(status='first_frame_only_text_banks_complete', preparer_sha256=sha(__file__),
    caption_plan_sha256=result['plan_sha256'], caption_records_sha256=result['records_sha256'],
    encoder_sha256=sha(weight), clip_version=getattr(clip, '__version__', None),
    clip_source_sha256={str(p): sha(p) for p in Path(clip.__file__).parent.glob('*.py')},
    file_sha256={n: sha(root / n) for n in ['text_manifest.json', 'text_fit.pt', 'text_development.pt']},
    sequences=dict(Counter(r['split'] for r in rows)), valid_slot_counts=dict(Counter(len(r['phrases']) for r in rows)),
    unique_phrases_including_empty=len(phrases), encoding_device='cpu', elapsed_seconds=time.time()-started,
    manual_text_changes=False, semantic_accuracy_verified=False, text_encoder_trained=False,
    scope='Initialization RGB only; no filename/category hint or later images in caption prompt')
(root / 'text_preparation.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
