"""Prepare initialization-only captions and banks under M58's frozen protocol."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
INTERFACE = Path('/root/autodl-tmp/sttrack_m58_evaluation_preparation_20260906')
PROTOCOL_SHA = 'd08acfb068ac5f7c428d5decb5f17af4655383543eeb4b4277e65cfda14bcac1'
INDEX_SHA = '5ffd459fe13242537571809f8a77b1a57c2b5dea97291221f4c5b4738eedf941'
WIRE_SHA = '20878b6ada788d609605c6328570f19e66d6b3a93e735ca86b1c6ec0646402cc'
MODEL = Path('/root/autodl-tmp/qwen/Qwen2.5-VL-3B-Instruct')
CLIP_WEIGHT = Path('/root/autodl-tmp/sutrack_assets/weights/ViT-L-14.pt')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def protocol():
    path = INTERFACE / 'text_protocol.json'
    assert sha(path) == PROTOCOL_SHA
    assert sha(INTERFACE / 'initialization_text.py') == INDEX_SHA
    assert sha(HERE / 'vot_rectangle_transport.py') == WIRE_SHA
    return json.loads(path.read_text())


def prepare(inputs_path, output):
    from PIL import Image
    protocol()
    sys.path.insert(0, str(INTERFACE))
    from initialization_text import initialization_key
    from vot_rectangle_transport import rectangle_wire_bbox
    inputs_path, output = Path(inputs_path).resolve(), Path(output).resolve()
    inputs = json.loads(inputs_path.read_text())
    convention = inputs['coordinate_convention']
    assert convention in ['ope_raw_xywh', 'vot_toolkit_xywh']
    assert inputs['cases'] and len({c['id'] for c in inputs['cases']}) == len(inputs['cases'])
    rows, cases, indices = [], [], {}
    for case in inputs['cases']:
        bbox = case['bbox']
        assert len(bbox) == 4 and all(math.isfinite(v) for v in bbox)
        assert bbox[2] > 0 and bbox[3] > 0
        received = rectangle_wire_bbox(bbox) if convention == 'vot_toolkit_xywh' else list(bbox)
        image = Path(case['image']).resolve()
        digest = sha(image)
        key = initialization_key(digest, received)
        cases.append(dict(id=case['id'], key=key))
        if key in indices:
            continue
        with Image.open(image) as im:
            w, h = im.size
        x, y, bw, bh = received
        xyxy = [max(0, int(x)), max(0, int(y)), min(w, int(x + bw + .999999)), min(h, int(y + bh + .999999))]
        assert xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]
        indices[key] = len(rows)
        rows.append(dict(key=key, image=str(image), image_sha256=digest, image_size=[w, h],
                         init_bbox=received, target_xyxy=xyxy))
    output.mkdir()
    plan = dict(status='initialization_inputs_frozen_before_generation',
        observed_utc=datetime.now(timezone.utc).isoformat(), inputs_path=str(inputs_path),
        inputs_sha256=sha(inputs_path), coordinate_convention=convention,
        protocol_path=str(INTERFACE / 'text_protocol.json'), protocol_sha256=PROTOCOL_SHA,
        source_sha256={n:sha(HERE / n) for n in ['initialization_captions.py', 'vot_rectangle_transport.py']},
        initialization_index_sha256=INDEX_SHA, model_path=str(MODEL), output=str(output),
        cases=cases, rows=rows, initialization_only=True, public_caption_generation_started=False)
    save(output / 'plan.json', plan)
    return plan


def checked_plan(path):
    plan = json.loads(Path(path).read_text())
    p = protocol()
    assert plan['protocol_sha256'] == PROTOCOL_SHA and plan['initialization_index_sha256'] == INDEX_SHA
    assert sha(plan['inputs_path']) == plan['inputs_sha256']
    for name, digest in plan['source_sha256'].items():
        assert sha(HERE / name) == digest
    return plan, p


def parse_reply(raw):
    serialized = raw.strip()
    if serialized.startswith('```json\n') and serialized.endswith('\n```'):
        serialized = serialized[8:-4]
    value = json.loads(serialized)
    assert set(value) == {'category', 'attributes'}
    assert isinstance(value['category'], str) and value['category'].strip()
    assert isinstance(value['attributes'], list) and len(value['attributes']) <= 4
    assert all(isinstance(a, str) and a.strip() for a in value['attributes'])
    return value


def model_inputs(p, row, processor, device):
    from PIL import Image, ImageDraw
    from qwen_vl_utils import process_vision_info
    image = Image.open(row['image']).convert('RGB')
    crop = image.crop(row['target_xyxy'])
    full = image.copy()
    ImageDraw.Draw(full).rectangle(row['target_xyxy'], outline='red', width=2)
    pixels = p['image_pixels']
    messages = [{'role':'user', 'content':[
        {'type':'image', 'image':full, 'max_pixels':pixels['full_max']},
        {'type':'image', 'image':crop, 'min_pixels':pixels['crop_min'], 'max_pixels':pixels['crop_max']},
        {'type':'text', 'text':p['prompt']}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=p['processor']['add_generation_prompt'])
    images, videos = process_vision_info(messages)
    inputs = processor(text=[chat], images=images, videos=videos,
                       padding=p['processor']['padding'], return_tensors='pt').to(device)
    return chat, inputs


def generate(path):
    import importlib.metadata as metadata
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    plan, p = checked_plan(path)
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    assert gpu in ['0', '1']
    used = int(subprocess.check_output(['nvidia-smi', '--id=' + gpu, '--query-gpu=memory.used', '--format=csv,noheader,nounits']))
    assert used < 500, used
    assert {n:metadata.version(n) for n in p['caption_package_versions']} == p['caption_package_versions']
    for name, digest in p['model_sha256'].items():
        assert sha(MODEL / name) == digest, name
    for row in plan['rows']:
        assert sha(row['image']) == row['image_sha256']
    folder = Path(plan['output'])
    output = folder / 'records.jsonl'
    assert not output.exists()
    torch.set_num_threads(4)
    torch.manual_seed(p['seed'])
    started = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(MODEL), local_files_only=True,
        torch_dtype=torch.float16, device_map={'':'cuda:0'}, attn_implementation=p['attention']).eval()
    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True, use_fast=p['processor']['use_fast'])
    with output.open('x') as stream:
        for index, row in enumerate(plan['rows']):
            _, inputs = model_inputs(p, row, processor, model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, **p['generation'])
            raw = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], **p['decoding'])[0]
            (folder / (row['key'] + '.raw.txt')).write_text(raw)
            value = parse_reply(raw)
            record = dict(key=row['key'], image_sha256=row['image_sha256'], raw=raw,
                category=value['category'], attributes=value['attributes'], elapsed_seconds=time.time() - started)
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            print(json.dumps(dict(completed=index + 1, total=len(plan['rows']), key=row['key'])), flush=True)
    result = dict(status='all_initialization_captions_generated', plan_sha256=sha(path),
        records_sha256=sha(output), unique_observations=len(plan['rows']), cases=len(plan['cases']),
        elapsed_seconds=time.time() - started, package_versions=p['caption_package_versions'],
        actual_tracking_calls=0, optimizer_steps=0, semantic_correctness_independently_verified=False)
    save(folder / 'generation_result.json', result)
    return result


def encode_records(plan, records, output):
    import clip
    import torch
    p = protocol()
    assert [r['key'] for r in records] == [r['key'] for r in plan['rows']]
    for row, record in zip(plan['rows'], records):
        assert record['image_sha256'] == row['image_sha256']
        assert parse_reply(record['raw']) == {k:record[k] for k in ['category', 'attributes']}
    assert sha(CLIP_WEIGHT) == p['text_encoder']['weight_sha256']
    for name, digest in p['text_encoder']['python_source_sha256'].items():
        assert sha(Path(clip.__file__).parent / name) == digest
    assert not Path(output).exists()
    torch.set_num_threads(4)
    started = time.time()
    encoder, _ = clip.load(str(CLIP_WEIGHT), device='cpu', jit=False)
    encoder = encoder.float().eval().requires_grad_(False)
    phrases_by_row = [[r['category']] + r['attributes'] for r in records]
    phrases = sorted({''} | {phrase for row in phrases_by_row for phrase in row})
    ids = clip.tokenize(phrases, truncate=False)
    with torch.no_grad():
        embeddings = torch.cat([encoder.encode_text(batch).float() for batch in ids.split(32)])
    assert embeddings.shape == (len(phrases), 768) and bool(torch.isfinite(embeddings).all())
    vectors = dict(zip(phrases, embeddings))
    tokens = torch.zeros(len(records), 5, 768)
    mask = torch.zeros(len(records), 5, dtype=torch.bool)
    for index, phrases in enumerate(phrases_by_row):
        tokens[index, :len(phrases)] = torch.stack([vectors[phrase] for phrase in phrases])
        mask[index, :len(phrases)] = True
    bank = dict(format='initialization_observation_v1', protocol_sha256=PROTOCOL_SHA,
        keys=[r['key'] for r in records], tokens=tokens, mask=mask, empty=vectors[''].clone())
    torch.save(bank, output)
    assert not torch.cuda.is_initialized()
    return dict(bank_sha256=sha(output), bank_bytes=Path(output).stat().st_size,
        observations=len(records), unique_phrases_including_empty=len(vectors),
        elapsed_seconds=time.time() - started, encoding_device='cpu', encoder_weight_sha256=sha(CLIP_WEIGHT))


def encode(path):
    plan, _ = checked_plan(path)
    folder = Path(plan['output'])
    result = json.loads((folder / 'generation_result.json').read_text())
    assert result['status'] == 'all_initialization_captions_generated' and result['plan_sha256'] == sha(path)
    assert sha(folder / 'records.jsonl') == result['records_sha256']
    records = [json.loads(line) for line in (folder / 'records.jsonl').read_text().splitlines()]
    summary = encode_records(plan, records, folder / 'text_bank.pt')
    summary.update(status='initialization_text_bank_complete', plan_sha256=sha(path),
        generation_result_sha256=sha(folder / 'generation_result.json'), protocol_sha256=PROTOCOL_SHA)
    save(folder / 'encoding_result.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--inputs', required=True)
    prep.add_argument('--output', required=True)
    for name in ['generate', 'encode']:
        commands.add_parser(name).add_argument('--plan', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        plan = prepare(args.inputs, args.output)
        print(json.dumps(dict(cases=len(plan['cases']), unique_observations=len(plan['rows']),
            plan_sha256=sha(Path(args.output) / 'plan.json'), actual_caption_generations=0)))
    else:
        print(json.dumps(generate(args.plan) if args.command == 'generate' else encode(args.plan), indent=2))
