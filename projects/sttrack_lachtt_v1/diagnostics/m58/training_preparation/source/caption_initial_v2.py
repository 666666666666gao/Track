"""Freeze and caption only DepthTrack Train initialization images."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v1_20260906')
MODEL = Path('/root/autodl-tmp/qwen/Qwen2.5-VL-3B-Instruct')
PROMPT = '''The first image is the initialization frame with a red rectangle marking the target. The second image is a tight crop of that target from exactly the same frame. Describe only that specific target object. Use no sequence name or outside information. Return one JSON object, without Markdown, with exactly these keys: "category" (a short English object category), "attributes" (zero to four short English phrases about clearly visible color, material, shape, texture, or distinctive markings). Prefer specific visible evidence over generic properties. Do not invent unreadable text, brand names, hidden parts, or properties. Do not describe the red annotation, background, people holding the target, screen position, ordinal position, movement, depth, or future visibility. If a property is uncertain, omit it. If the category is unclear, use "object". Use lowercase strings. Example format: {"category":"bottle","attributes":["red cap","clear body"]}.'''


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare():
    from PIL import Image
    inventory = json.loads((ROOT / 'data_inventory.json').read_text())
    folder = ROOT / 'initial_captions_v2'
    folder.mkdir()
    rows = []
    for row in inventory['sequences_detail']:
        image = Path(row['initial_rgb'])
        assert image.name == '00000001.jpg' and row['first_box_valid']
        with Image.open(image) as im:
            w, h = im.size
        x, y, bw, bh = row['first_box']
        box = [max(0, int(x)), max(0, int(y)), min(w, int(x + bw + .999999)), min(h, int(y + bh + .999999))]
        assert box[2] > box[0] and box[3] > box[1]
        rows.append(dict(sequence=row['sequence'], split=row['split'], image=str(image),
                         image_sha256=sha(image), image_size=[w, h], target_xyxy=box))
    assert len(rows) == 152 and sum(r['split'] == 'fit' for r in rows) == 130
    model_files = ['config.json', 'generation_config.json', 'preprocessor_config.json',
                   'tokenizer_config.json', 'tokenizer.json', 'model.safetensors.index.json',
                   'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors',
                   'PINNED_DOWNLOAD_MANIFEST.json']
    plan = dict(status='frozen_before_generation', observed_utc=datetime.now(timezone.utc).isoformat(),
                inventory_sha256=sha(ROOT / 'data_inventory.json'), script_sha256=sha(__file__),
                model_path=str(MODEL), model_sha256={n: sha(MODEL / n) for n in model_files},
                prompt=PROMPT, generation=dict(max_new_tokens=160, do_sample=False, use_cache=True),
                image_pixels=dict(full_max=512 * 28 * 28, crop_min=128 * 28 * 28, crop_max=256 * 28 * 28),
                torch_dtype='float16', attention='sdpa', seed=2026, rows=rows,
                first_frame_only=True, sequence_name_in_model_prompt=False,
                category_hint_in_model_prompt=False, subsequent_images_used=False,
                manual_adjudication_used=False, independent_annotation_review=False,
                intended_scope='DepthTrack Train initialization text; no public-test annotation')
    save(folder / 'plan.json', plan)
    print(json.dumps(dict(status=plan['status'], sequences=len(rows), plan_sha256=sha(folder / 'plan.json'))), flush=True)


def run():
    import importlib.metadata as metadata
    import torch
    from PIL import Image, ImageDraw
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    folder = ROOT / 'initial_captions_v2'
    plan = json.loads((folder / 'plan.json').read_text())
    assert sha(__file__) == plan['script_sha256'] and PROMPT == plan['prompt']
    for name, digest in plan['model_sha256'].items():
        assert sha(MODEL / name) == digest, name
    for row in plan['rows']:
        assert sha(row['image']) == row['image_sha256'], row['sequence']
    output = folder / 'records.jsonl'
    assert not output.exists()
    torch.set_num_threads(4)
    torch.manual_seed(plan['seed'])
    started = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(MODEL), local_files_only=True,
             torch_dtype=torch.float16, device_map={'': 'cuda:0'}, attn_implementation='sdpa').eval()
    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True, use_fast=False)
    records = []
    with output.open('x') as stream:
        for row in plan['rows']:
            image = Image.open(row['image']).convert('RGB')
            crop = image.crop(row['target_xyxy'])
            full = image.copy()
            ImageDraw.Draw(full).rectangle(row['target_xyxy'], outline='red', width=2)
            pixels = plan['image_pixels']
            messages = [{'role': 'user', 'content': [
                {'type': 'image', 'image': full, 'max_pixels': pixels['full_max']},
                {'type': 'image', 'image': crop, 'min_pixels': pixels['crop_min'], 'max_pixels': pixels['crop_max']},
                {'type': 'text', 'text': PROMPT}]}]
            chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            inputs = processor(text=[chat], images=images, videos=videos, padding=True, return_tensors='pt').to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, **plan['generation'])
            raw = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0]
            # Preserve the raw reply before enforcing the frozen format contract.
            (folder / (row['sequence'] + '.raw.txt')).write_text(raw)
            serialized = raw.strip()
            if serialized.startswith('```json\n') and serialized.endswith('\n```'):
                serialized = serialized[8:-4]
            value = json.loads(serialized)
            assert set(value) == {'category', 'attributes'}, row['sequence']
            assert isinstance(value['category'], str) and value['category'].strip()
            assert isinstance(value['attributes'], list) and len(value['attributes']) <= 4
            assert all(isinstance(a, str) and a.strip() for a in value['attributes'])
            record = dict(sequence=row['sequence'], split=row['split'], image_sha256=row['image_sha256'],
                          category=value['category'], attributes=value['attributes'], raw=raw,
                          elapsed_seconds=time.time() - started)
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            records.append(record)
            print(json.dumps(dict(completed=len(records), total=len(plan['rows']), **record)), flush=True)
    result = dict(status='all_initial_captions_generated', plan_sha256=sha(folder / 'plan.json'),
                  records_sha256=sha(output), sequences=len(records), elapsed_seconds=time.time() - started,
                  package_versions={n: metadata.version(n) for n in ['torch', 'transformers', 'qwen-vl-utils', 'Pillow', 'accelerate']},
                  actual_dataset_optimizer_steps=0, semantic_correctness_independently_verified=False)
    save(folder / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    assert sys.argv[1] in ['prepare', 'run']
    prepare() if sys.argv[1] == 'prepare' else run()
