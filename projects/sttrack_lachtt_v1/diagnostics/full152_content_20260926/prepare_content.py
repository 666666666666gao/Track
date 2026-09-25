"""Bind Full152 final weights to same-weight OPE text interventions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


EVAL = Path('/root/autodl-tmp/sttrack_full152_evaluation_20260925')
ROOT = Path('/root/autodl-tmp/sttrack_full152_content_20260926')
sys.path.insert(0, str(EVAL / 'interface'))
from semantic_runtime import checked_plan


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def bind(model, dataset, variant):
    source = EVAL / model / dataset
    plan, bundle = checked_plan(source / 'plan.json')
    assert bundle['seed'] == 2027 and bundle['arm'] == {'M67': 'control', 'M82': 'category'}[model]
    assert plan['text_bank_path'] == str(source / 'category.pt')
    bank = torch.load(plan['text_bank_path'], map_location='cpu')
    records_path = EVAL / 'inputs_bfloat16' / dataset / 'captions' / 'records.jsonl'
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    assert [row['key'] for row in records] == bank['keys']
    assert bank['tokens'].shape == (len(records), 5, 768)
    assert all(bool(bank['mask'][i, 0]) for i in range(len(records)))
    assert torch.equal(bank['tokens'][:, 1:][bank['mask'][:, 1:]],
                       bank['empty'].expand_as(bank['tokens'][:, 1:])[bank['mask'][:, 1:]])

    target = ROOT / model / dataset / variant
    target.mkdir(parents=True)
    changed = dict(bank)
    changed['tokens'] = bank['tokens'].clone()
    donors = []
    if variant == 'empty':
        changed['tokens'][:, 0] = bank['empty']
    else:
        names = [row['category'].strip().casefold() for row in records]
        assert len(set(names)) > 1
        for index, name in enumerate(names):
            donor = (index + 1) % len(names)
            while names[donor] == name:
                donor = (donor + 1) % len(names)
            changed['tokens'][index, 0] = bank['tokens'][donor, 0]
            donors.append(dict(key=bank['keys'][index], donor_key=bank['keys'][donor],
                               category=records[index]['category'], donor_category=records[donor]['category']))
    assert torch.equal(changed['mask'], bank['mask'])
    assert torch.equal(changed['tokens'][:, 1:], bank['tokens'][:, 1:])
    assert bool((changed['tokens'][:, 0] != bank['tokens'][:, 0]).any(dim=1).all())
    bank_path = target / 'text_bank.pt'
    torch.save(changed, bank_path)
    output = target / 'predictions'
    intervention_plan = dict(plan, text_bank_path=str(bank_path), text_bank_sha256=sha(bank_path),
                             output=str(output))
    write(target / 'plan.json', intervention_plan)
    checked_plan(target / 'plan.json')
    report = dict(status='prepared', model=model, dataset=dataset, variant=variant,
                  source_plan_sha256=sha(source / 'plan.json'),
                  checkpoint_sha256=bundle['adapter_checkpoint_sha256'],
                  source_bank_sha256=plan['text_bank_sha256'], bank_sha256=sha(bank_path),
                  records_sha256=sha(records_path), observations=len(records),
                  category_vectors_changed=len(records), donors=donors,
                  note='Swapped categories differ as strings; they are not verified semantic contradictions.')
    write(target / 'preparation.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'donors'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['M67', 'M82'], required=True)
    parser.add_argument('--dataset', choices=['depthtrack', 'cdtb'], required=True)
    parser.add_argument('--variant', choices=['empty', 'swapped'], required=True)
    args = parser.parse_args()
    bind(args.model, args.dataset, args.variant)
