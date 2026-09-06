"""Freeze a post-development category-retention candidate and its input checks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/root/autodl-tmp/sttrack_m64_category_candidate_20260907')
GENERATOR = Path('/root/autodl-tmp/sttrack_m58_initialization_generator_20260906')
INTERFACE = Path('/root/autodl-tmp/sttrack_m58_evaluation_preparation_20260906')
M59 = Path('/root/autodl-tmp/sttrack_m59_fixed_head_content_diagnostic_20260906')
M60 = Path('/root/autodl-tmp/sttrack_m60_category_isolation_20260906')
M62 = Path('/root/autodl-tmp/sttrack_m62_learned_entry_parity_20260907')
LOW_INPUTS = Path('/root/autodl-tmp/sttrack_m58_vot_initialization_export_20260906/low22_inputs/caption_inputs.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def retain_category(bank, protocol_sha):
    out = dict(bank)
    out['tokens'] = bank['tokens'].clone()
    for i in range(len(out['keys'])):
        for j in range(1, 5):
            if bool(out['mask'][i, j]): out['tokens'][i, j] = out['empty']
    out['protocol_sha256'] = protocol_sha
    return out


def prepare():
    import torch
    sys.path.insert(0, str(INTERFACE))
    from semantic_runtime import checked_plan, text_bank
    sys.path.insert(0, str(GENERATOR))
    import initialization_captions as app
    assert sha(M60 / 'result.json') == 'febddd93eda9f25f3530377948751c1999030db9a08656f9bb24ae151ad64287'
    assert sha(M62 / 'result.json') == '731bf649920f91087f242442f384b6611c4b7a7d42186aaedb1dc4bb67325d22'
    assert (M62 / 'job.exit').read_text().strip() == '0'
    plan, bundle = checked_plan(M59 / 'category_plan.json')
    expected = text_bank(plan, bundle).bank
    original_plan, original_bundle = checked_plan(M59 / 'original_plan.json')
    original = text_bank(original_plan, original_bundle).bank
    ROOT.mkdir()
    protocol = dict(observation_protocol_path=str(INTERFACE / 'text_protocol.json'), observation_protocol_sha256=app.PROTOCOL_SHA,
        lexical_policy='Retain the generated category in slot0. Replace each valid attribute slot1..4 by the frozen CLIP empty-string vector. Preserve slot count, mask, padding, initialization key and all other tensors.',
        initialization_only=True, online_caption_updates=False, category_hints=False, manual_caption_corrections=False,
        image_specific_exceptions=False, subsequent_GT_used=False,
        selection_provenance='Selected after M59 on the repeatedly reused DepthTrack Train development22; M60 supplied same-weight category-content evidence. This is a new candidate protocol, not retrospective promotion of M58 full attributes.')
    write(ROOT / 'text_protocol.json', protocol)
    protocol_sha = sha(ROOT / 'text_protocol.json')
    converted = retain_category(original, protocol_sha)
    for k in ['tokens', 'mask', 'empty']: assert torch.equal(converted[k], expected[k]), k
    assert converted['keys'] == expected['keys'] and len(converted['keys']) == 22
    bundle['text_protocol_path'] = str(ROOT / 'text_protocol.json')
    bundle['text_protocol_sha256'] = protocol_sha
    write(ROOT / 'bundle.json', bundle)
    torch.save(converted, ROOT / 'development_category.pt')
    dev_plan = dict(bundle_path=str(ROOT / 'bundle.json'), bundle_sha256=sha(ROOT / 'bundle.json'),
        text_bank_path=str(ROOT / 'development_category.pt'), text_bank_sha256=sha(ROOT / 'development_category.pt'))
    write(ROOT / 'development_plan.json', dev_plan)
    checked_plan(ROOT / 'development_plan.json')
    preprocessing = json.loads((GENERATOR / 'preprocessing_result.json').read_text())
    old = json.loads((GENERATOR / 'train_replay/plan.json').read_text())
    wanted = [r['key'] for r in preprocessing['exact_processor_replays']]
    rows = {r['key']: r for r in old['rows']}
    ids = {r['key']: r['id'] for r in old['cases']}
    write(ROOT / 'train_replay_inputs.json', dict(coordinate_convention='ope_raw_xywh', cases=[
        dict(id=ids[k], image=rows[k]['image'], bbox=rows[k]['init_bbox']) for k in wanted]))
    replay_plan = app.prepare(ROOT / 'train_replay_inputs.json', ROOT / 'train_generator_replay')
    assert [r['key'] for r in replay_plan['rows']] == wanted and len(wanted) == 3
    assert sha(LOW_INPUTS) == '5149bee97236f96587d90191087eef7434373fc714d6f5dbfe4d00583a6859e8'
    spec = dict(status='candidate_frozen_before_public_caption_generation_and_tracking', observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__), bundle_sha256=sha(ROOT / 'bundle.json'), text_protocol_sha256=protocol_sha,
        parent_m60_result_sha256=sha(M60 / 'result.json'), learned_entry_m62_result_sha256=sha(M62 / 'result.json'),
        unchanged_development_text_tensors=22, new_learned_parameters=0, new_optimizer_steps=0,
        training='Reuse the M58 text adapter already trained on130 DepthTrack Train fit sequences, 186694 track calls and5798 optimizer steps; no architecture change.',
        low22_inputs_path=str(LOW_INPUTS), low22_inputs_sha256=sha(LOW_INPUTS),
        low22_baseline=dict(EAO=57.135993, ACC=75.719622, ROB=73.022401, confirmed_failures=124, sequences=22, anchors=303),
        full_evaluation_gate=dict(EAO_min=57.635993, ACC_min=75.719622, ROB_min=73.022401, confirmed_failures_max=118,
            preserve_all_seven_native_zero_failure_sequences=True, all303_anchors_complete=True, identical_bundle_and_text_protocol=True),
        gate_interpretation='Before any M64 public outcome: require at least0.5pp EAO, no ACC/ROB regression, at least6 fewer failures and preserve native-success sequences. No threshold sweep or alternate captions after evaluation.',
        sequence='Actual GPU caption replay on three preselected Train observations must reproduce frozen raw replies; then low22 initialization captions and fixed-runtime tracking. Run full DepthTrack Test, CDTB and VOT127 only if the new low22 gate passes.',
        independence='M64 does not select from M63 crossed arms. Swapped donor categories never enter this candidate.',
        same_bundle_across_three_datasets=True, reused_development_sets_not_unseen=True, independent_review_pass=False,
        public_tracking_started=False, full_three_dataset_evaluation_allowed=False)
    write(ROOT / 'spec.json', spec)
    print(json.dumps(dict(status=spec['status'], source_sha256=spec['source_sha256'], spec_sha256=sha(ROOT / 'spec.json'),
                         bundle_sha256=spec['bundle_sha256'], exact_development_tensor_matches=22, prepared_GPU_replay_observations=3), indent=2))


def checked():
    spec = json.loads((ROOT / 'spec.json').read_text())
    assert sha(__file__) == spec['source_sha256']
    assert sha(ROOT / 'bundle.json') == spec['bundle_sha256']
    assert sha(ROOT / 'text_protocol.json') == spec['text_protocol_sha256']
    return spec


def verify_replay():
    checked()
    root = ROOT / 'train_generator_replay'
    result = json.loads((root / 'generation_result.json').read_text())
    assert result['plan_sha256'] == sha(root / 'plan.json') and result['records_sha256'] == sha(root / 'records.jsonl')
    old = {r['key']: r for r in map(json.loads, (GENERATOR / 'replay_records.jsonl').read_text().splitlines())}
    current = list(map(json.loads, (root / 'records.jsonl').read_text().splitlines()))
    assert len(current) == 3
    rows = []
    for r in current:
        equal = {k: r[k] == old[r['key']][k] for k in ['raw', 'category', 'attributes', 'image_sha256']}
        rows.append(dict(key=r['key'], equal=equal))
    passed = all(all(r['equal'].values()) for r in rows)
    write(ROOT / 'generator_replay_result.json', dict(status='verified' if passed else 'replay_mismatch', rows=rows,
        generation_result_sha256=sha(root / 'generation_result.json'), actual_Qwen_generate_calls=3,
        public_images_used=0, public_preparation_allowed=passed, independent_review_pass=False))
    assert passed, rows
    print((ROOT / 'generator_replay_result.json').read_text())


def low22_prepare():
    spec = checked()
    assert json.loads((ROOT / 'generator_replay_result.json').read_text())['public_preparation_allowed']
    sys.path.insert(0, str(GENERATOR))
    import initialization_captions as app
    assert sha(LOW_INPUTS) == spec['low22_inputs_sha256']
    plan = app.prepare(LOW_INPUTS, ROOT / 'low22_captions')
    assert len(plan['cases']) == 303
    print(json.dumps(dict(cases=len(plan['cases']), unique_observations=len(plan['rows']), plan_sha256=sha(ROOT / 'low22_captions/plan.json'))))


def low22_bank():
    import torch
    spec = checked()
    root = ROOT / 'low22_captions'
    encoded = json.loads((root / 'encoding_result.json').read_text())
    assert encoded['bank_sha256'] == sha(root / 'text_bank.pt')
    bank = torch.load(root / 'text_bank.pt', map_location='cpu')
    converted = retain_category(bank, spec['text_protocol_sha256'])
    torch.save(converted, ROOT / 'low22_category.pt')
    plan = dict(bundle_path=str(ROOT / 'bundle.json'), bundle_sha256=spec['bundle_sha256'],
        text_bank_path=str(ROOT / 'low22_category.pt'), text_bank_sha256=sha(ROOT / 'low22_category.pt'))
    write(ROOT / 'low22_plan.json', plan)
    sys.path.insert(0, str(INTERFACE))
    from semantic_runtime import checked_plan, text_bank
    checked, bundle = checked_plan(ROOT / 'low22_plan.json')
    lookup = text_bank(checked, bundle)
    observations = json.loads((root / 'plan.json').read_text())
    for row in observations['rows']: lookup.info(row['image'], row['init_bbox'])
    write(ROOT / 'low22_bank_result.json', dict(status='category_bank_bound_to_actual_initializations', cases=303,
        unique_observations=len(converted['keys']), bundle_sha256=spec['bundle_sha256'], text_protocol_sha256=spec['text_protocol_sha256'],
        bank_sha256=plan['text_bank_sha256'], encoding_result_sha256=sha(root / 'encoding_result.json'), tracking_calls=0))
    print((ROOT / 'low22_bank_result.json').read_text())


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['prepare', 'verify_replay', 'low22_prepare', 'low22_bank'])
    args = p.parse_args()
    {'prepare': prepare, 'verify_replay': verify_replay, 'low22_prepare': low22_prepare, 'low22_bank': low22_bank}[args.action]()
