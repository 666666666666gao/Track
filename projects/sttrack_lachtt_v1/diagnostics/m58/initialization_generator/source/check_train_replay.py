"""Compare new input preparation and CPU encoding with the frozen Train run."""
import argparse
import ast
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import initialization_captions as app


ROOT = Path(__file__).resolve().parent
TRAIN = Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v1_20260906')
LEGACY_SHA = '42789e10651e70b0f4bf6685a14d0833d797174f121d6c01642202bf19f73335'
CAPTIONS_SHA = 'ae92f492bcc1e9ca08bed3a734f934210f0c1a9bdd212f0c701c3686d1e6286e'
INVENTORY_SHA = '529f10ebd1122596a6e8832578f1cd26c3315a78fac59664fdb0d0fb367b85c1'


def prepare_check():
    import torch
    from PIL import Image, ImageDraw
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    started = time.time()
    p = app.protocol()
    assert {n:metadata.version(n) for n in p['caption_package_versions']} == p['caption_package_versions']
    assert app.sha(TRAIN / 'data_inventory.json') == INVENTORY_SHA
    assert app.sha(TRAIN / 'caption_initial_v2.py') == LEGACY_SHA
    assert app.sha(TRAIN / 'initial_captions_v2/records.jsonl') == CAPTIONS_SHA
    for name, digest in p['model_sha256'].items():
        if not name.endswith('.safetensors'):
            assert app.sha(app.MODEL / name) == digest
    inventory = json.loads((TRAIN / 'data_inventory.json').read_text())
    old_plan = json.loads((TRAIN / 'initial_captions_v2/plan.json').read_text())
    assert old_plan['script_sha256'] == LEGACY_SHA
    assert all(old_plan[k] == p[k] for k in ['prompt', 'model_sha256', 'generation', 'image_pixels', 'torch_dtype', 'attention', 'seed'])
    inputs = dict(coordinate_convention='ope_raw_xywh', cases=[dict(id=r['sequence'],
        image=r['initial_rgb'], bbox=r['first_box']) for r in inventory['sequences_detail']])
    app.save(ROOT / 'train_inputs.json', inputs)
    plan = app.prepare(ROOT / 'train_inputs.json', ROOT / 'train_replay')
    app.checked_plan(ROOT / 'train_replay/plan.json')
    assert len(plan['rows']) == len(plan['cases']) == 152
    old_by_image = {r['image']:r for r in old_plan['rows']}
    for row in plan['rows']:
        old = old_by_image[row['image']]
        for field in ['image_sha256', 'image_size', 'target_xyxy']:
            assert row[field] == old[field]
    old_records = [json.loads(line) for line in (TRAIN / 'initial_captions_v2/records.jsonl').read_text().splitlines()]
    by_sequence = {r['sequence']:r for r in old_records}
    by_key = {c['key']:by_sequence[c['id']] for c in plan['cases']}
    replay = []
    for row in plan['rows']:
        old = by_key[row['key']]
        assert app.parse_reply(old['raw']) == {k:old[k] for k in ['category', 'attributes']}
        replay.append(dict(key=row['key'], image_sha256=old['image_sha256'], raw=old['raw'],
                           category=old['category'], attributes=old['attributes']))
    (ROOT / 'replay_records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in replay))

    # Execute the exact trusted legacy preprocessing statements, stopping before
    # model.generate. This checks processor tensors, not only a rewritten formula.
    tree = ast.parse((TRAIN / 'caption_initial_v2.py').read_text())
    legacy_prompt = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'PROMPT' for t in n.targets))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    loop = next(n for n in ast.walk(run) if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'row'
                and isinstance(n.body[0], ast.Assign) and n.body[0].targets[0].id == 'image')
    body = loop.body[:9]
    assert isinstance(body[-1], ast.Assign) and body[-1].targets[0].id == 'inputs'
    source = compile(ast.Module(body=body, type_ignores=[]), '<frozen_legacy_preprocessing>', 'exec')
    processor = AutoProcessor.from_pretrained(str(app.MODEL), local_files_only=True, use_fast=False)
    ordered = sorted(plan['rows'], key=lambda r:(r['target_xyxy'][2]-r['target_xyxy'][0]) * (r['target_xyxy'][3]-r['target_xyxy'][1]))
    selected = [ordered[0], ordered[len(ordered)//2], ordered[-1]]
    checks = []
    torch.set_num_threads(4)
    for row in selected:
        ns = dict(Image=Image, ImageDraw=ImageDraw, row=old_by_image[row['image']],
            plan=old_plan, PROMPT=legacy_prompt, processor=processor,
            process_vision_info=process_vision_info, model=SimpleNamespace(device='cpu'))
        exec(source, ns)
        chat, tensors = app.model_inputs(p, row, processor, 'cpu')
        assert chat == ns['chat'] and set(tensors) == set(ns['inputs'])
        assert all(torch.equal(tensors[k], ns['inputs'][k]) for k in tensors)
        checks.append(dict(key=row['key'], tensor_shapes={k:list(v.shape) for k,v in tensors.items()},
                           all_tensor_values_and_chat_equal=True))

    vot = dict(coordinate_convention='vot_toolkit_xywh', cases=[
        dict(id='same_a_1', image=plan['rows'][0]['image'], bbox=[1.123456789, 2.234567891, 30.345678912, 40.456789123]),
        dict(id='same_a_2', image=plan['rows'][0]['image'], bbox=[1.123456789, 2.234567891, 30.345678912, 40.456789123]),
        dict(id='different_b', image=plan['rows'][0]['image'], bbox=[19.234567891, 7.123456789, 20.7654321, 15.4321])])
    app.save(ROOT / 'synthetic_vot_inputs.json', vot)
    vot_plan = app.prepare(ROOT / 'synthetic_vot_inputs.json', ROOT / 'synthetic_vot_preparation')
    assert len(vot_plan['cases']) == 3 and len(vot_plan['rows']) == 2
    assert vot_plan['cases'][0]['key'] == vot_plan['cases'][1]['key'] != vot_plan['cases'][2]['key']
    assert vot_plan['rows'][0]['init_bbox'] == [1.1234999895095825, 2.234600067138672, 30.345699310302734, 40.4567985534668]
    assert not torch.cuda.is_initialized()
    result = dict(status='train_initialization_preprocessing_replay_passed',
        observed_utc=datetime.now(timezone.utc).isoformat(), source_sha256=plan['source_sha256'],
        checker_sha256=app.sha(__file__), legacy_caption_source_sha256=LEGACY_SHA,
        caption_records_sha256=CAPTIONS_SHA, inventory_sha256=INVENTORY_SHA,
        protocol_sha256=app.PROTOCOL_SHA, train_plan_sha256=app.sha(ROOT / 'train_replay/plan.json'),
        replay_records_sha256=app.sha(ROOT / 'replay_records.jsonl'), unchanged_crop_rows=152,
        unchanged_parsed_captions=152, exact_processor_replays=checks,
        synthetic_vot_cases=3, synthetic_vot_unique_observations=2,
        synthetic_vot_has_correct_four_decimal_transport=True,
        actual_qwen_generate_calls=0, qwen_weights_loaded=False, cuda_initialized=False,
        subsequent_dataset_images_used=0, public_dataset_images_used=0, actual_tracking_calls=0,
        elapsed_seconds=time.time()-started, independent_review_pass=False)
    app.save(ROOT / 'preprocessing_result.json', result)
    print(json.dumps(result, indent=2), flush=True)


def encoding_check():
    import torch
    before = json.loads((ROOT / 'preprocessing_result.json').read_text())
    assert before['checker_sha256'] == app.sha(__file__)
    assert before['train_plan_sha256'] == app.sha(ROOT / 'train_replay/plan.json')
    assert before['replay_records_sha256'] == app.sha(ROOT / 'replay_records.jsonl')
    plan, _ = app.checked_plan(ROOT / 'train_replay/plan.json')
    records = [json.loads(line) for line in (ROOT / 'replay_records.jsonl').read_text().splitlines()]
    summary = app.encode_records(plan, records, ROOT / 'train_replay/text_bank.pt')
    sys.path.insert(0, str(app.INTERFACE))
    from initialization_text import InitializationTextBank
    bank = InitializationTextBank(ROOT / 'train_replay/text_bank.pt', summary['bank_sha256'], app.PROTOCOL_SHA)
    expected = {}
    frozen_banks = dict(fit='d7ce0833b8e0e96c881a31ced85be43b6eeccb777adc577144fad0fb2420e3d0',
        development='81879fa1ace5da373c77b898a76dd5fea23c5911c94184322fee4f2755d56e89')
    for split, digest in frozen_banks.items():
        path = TRAIN / ('text_' + split + '.pt')
        assert app.sha(path) == digest
        original = torch.load(path, map_location='cpu')
        for index, name in enumerate(original['sequences']):
            expected[name] = dict(tokens=original['tokens'][index], mask=original['mask'][index], empty=original['empty'])
    row_by_key = {r['key']:r for r in plan['rows']}
    for case in plan['cases']:
        row = row_by_key[case['key']]
        info = bank.info(row['image'], row['init_bbox'])
        old = expected[case['id']]
        assert info['init_bbox'] == row['init_bbox']
        assert torch.equal(info['text_tokens'], old['tokens'])
        assert torch.equal(info['text_mask'], old['mask'])
        assert torch.equal(info['empty_text'], old['empty'])
    assert not torch.cuda.is_initialized()
    summary.update(status='exact_frozen_train_bank_encoding_and_routing_replay_passed',
        observed_utc=datetime.now(timezone.utc).isoformat(), preprocessing_result_sha256=app.sha(ROOT / 'preprocessing_result.json'),
        checker_sha256=app.sha(__file__), equal_token_mask_empty_and_bbox_routes=152,
        frozen_training_banks_sha256=frozen_banks, qwen_weights_loaded=False,
        actual_qwen_generate_calls=0, clip_weights_loaded_on_cpu=True, cuda_initialized=False,
        public_dataset_images_used=0, actual_tracking_calls=0, optimizer_steps=0, independent_review_pass=False)
    app.save(ROOT / 'encoding_replay_result.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'encode'])
    args = parser.parse_args()
    prepare_check() if args.phase == 'prepare' else encoding_check()
