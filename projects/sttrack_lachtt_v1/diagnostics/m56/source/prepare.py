"""Bind existing Train attributes and encode a frozen CPU CLIP phrase bank."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from PIL import Image


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    assert not (root / 'text_bank.pt').exists()
    parent = Path('/root/autodl-tmp/sttrack_m44_candidate_set_v1_20260905')
    rich = Path('/home/OSTrack_RGBD_L_dataset_modified/annotations/depthtrack_train_first_rich_reviewed_qwen3_v5.jsonl')
    short = Path('/root/autodl-tmp/sutrack_rgbd_language_short_prompt_fixed6_v1/source/depthtrack_train_language_short.jsonl')
    short_receipt = short.with_suffix('.receipt.json')
    assert sha(rich) == '56c03871b8e2a005bbd9ca12c32ed8821c5445d16af9e2631963267b89785b58'
    assert sha(short) == 'c85a38b9d570096d61c2f60bbde1b9b8b8ab8ff9d2a18f4e735c857610c923d1'
    materialization = json.loads(short_receipt.read_text())
    assert materialization['source_sha256'] == sha(rich)
    assert materialization['output_sha256'] == sha(short)
    parent_spec = json.loads((parent / 'spec.json').read_text())
    assert sha(parent / 'spec.json') == '8572eca25d04291186980c947400106ad6db91705222f2d7f1153ca6c8fdbd18'
    inputs = json.loads((parent / 'inference_inputs.json').read_text())
    assert sha(parent / 'inference_inputs.json') == parent_spec['inference_inputs_sha256']
    rich_rows = {row['sequence']: row for row in map(json.loads, rich.read_text().splitlines())}
    assert len(inputs) == 85 and len(rich_rows) == 152
    manifest = []
    for row in sorted(inputs, key=lambda x: x['sequence']):
        source = rich_rows[row['sequence']]
        annotation = source['annotation']
        assert source['frame_index'] == 0 and source['split'] == 'train'
        assert source['provenance']['generated_from_target_crop']
        phrases = [annotation['category']] + annotation['stable_attributes']
        assert all(isinstance(x, str) and x and x.strip() == x for x in phrases)
        image_path = sorted((Path(parent_spec['dataset_root']) / row['sequence'] / 'color').glob('*.jpg'))[0]
        with Image.open(image_path) as image:
            width, height = image.size
        x, y, w, h = row['init_bbox']
        clipped = [max(0., x), max(0., y), min(width, x + w) - max(0., x), min(height, y + h) - max(0., y)]
        assert list(map(float, source['bbox'])) in [row['init_bbox'], clipped], row['sequence']
        manifest.append(dict(sequence=row['sequence'], fold=row['fold'], split=row['split'],
            category=annotation['category'], stable_attributes=annotation['stable_attributes'], phrases=phrases,
            category_source=annotation['category_source'], review_status=source['provenance']['review_status'],
            source_frame_index=0, annotation_bbox_equals_init=source['bbox'] == row['init_bbox'],
            annotation_bbox_equals_clipped_init=source['bbox'] == clipped,
            first_image_sha256=sha(image_path), first_image_size=[width, height]))
    write(root / 'text_manifest.json', manifest)
    # Parsing the existing mixed label file is disclosed; only fit rows are materialized.
    assert sha(parent / 'training_labels.json') == parent_spec['labels_sha256']
    labels = json.loads((parent / 'training_labels.json').read_text())
    fit_names = {x['sequence'] for x in manifest if x['split'] == 'fit'}
    fit_labels = {key: value for key, value in labels.items() if value['sequence'] in fit_names}
    assert len(fit_labels) == 1511 and all(x['fold'] in [2, 3, 4] for x in fit_labels.values())
    write(root / 'fit_labels.json', fit_labels)
    del labels, fit_labels, rich_rows
    import clip
    weight = Path('/root/autodl-tmp/sutrack_assets/weights/ViT-L-14.pt')
    assert sha(weight) == 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
    clip_root = Path(clip.__file__).parent
    clip_sources = {str(path): sha(path) for path in sorted(clip_root.glob('*.py'))}
    clip_sources.update({str(path): sha(path) for path in clip_root.glob('*.gz')})
    torch.set_num_threads(4)
    started = time.time()
    encoder, _ = clip.load(str(weight), device='cpu', jit=False)
    encoder = encoder.float().eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    phrases = sorted({''} | {phrase for row in manifest for phrase in row['phrases']})
    ids = clip.tokenize(phrases, truncate=False)
    with torch.no_grad():
        embeddings = torch.cat([encoder.encode_text(batch).float() for batch in ids.split(32)])
    assert embeddings.shape == (len(phrases), 768) and torch.isfinite(embeddings).all()
    del encoder
    vectors = {phrase: embeddings[i] for i, phrase in enumerate(phrases)}
    slots = max(len(row['phrases']) for row in manifest)
    assert slots == 5
    tokens = torch.zeros(len(manifest), slots, 768)
    mask = torch.zeros(len(manifest), slots, dtype=torch.bool)
    for i, row in enumerate(manifest):
        tokens[i, :len(row['phrases'])] = torch.stack([vectors[p] for p in row['phrases']])
        mask[i, :len(row['phrases'])] = True
    bank = dict(sequences=[x['sequence'] for x in manifest], tokens=tokens, mask=mask,
        empty=vectors[''], phrases=phrases, token_ids=ids, phrase_embeddings=embeddings,
        text_manifest_sha256=sha(root / 'text_manifest.json'), encoder_sha256=sha(weight))
    torch.save(bank, root / 'text_bank.pt')
    receipt = dict(status='complete', scope='Existing DepthTrack Train annotations, frozen phrase embeddings; no optimizer or tracking evaluation',
        preparer_sha256=sha(__file__), parent_spec_sha256=sha(parent / 'spec.json'),
        input_sha256={str(path): sha(path) for path in [rich, short, short_receipt, parent / 'inference_inputs.json', parent / 'training_labels.json']},
        files={name: sha(root / name) for name in ['text_manifest.json', 'text_bank.pt', 'fit_labels.json']},
        encoder_path=str(weight), encoder_sha256=sha(weight), clip_source_sha256=clip_sources,
        torch_version=torch.__version__, cpu_threads=4, clip_device='cpu', phrase_embedding_dimension=768,
        unique_phrases_including_empty=len(phrases), maximum_phrase_slots=slots,
        sequence_counts=dict(Counter(x['split'] for x in manifest)), phrase_counts=dict(Counter(len(x['phrases']) for x in manifest)),
        review_status_counts=dict(Counter(x['review_status'] for x in manifest)),
        clipped_annotation_sequences=[x['sequence'] for x in manifest if not x['annotation_bbox_equals_init']],
        fit_events=1511, development_labels_in_training_file=0,
        mixed_label_source_parsed_and_filtered=True, future_frame_annotations_claimed_causal=False,
        language_fields=['annotation.category', 'annotation.stable_attributes'],
        excluded_from_model=['sequence', 'fold', 'paths', 'bbox', 'depth_relation', 'occlusion', 'motion', 'ordinal'],
        elapsed_seconds=time.time() - started)
    write(root / 'text_preparation.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
