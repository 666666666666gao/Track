"""Bind M89 Candidate to the sealed Empty and Swapped OPE inputs."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


MAIN = Path('/root/autodl-tmp/sttrack_m89_evaluation_20260926')
OLD_CONTENT = Path('/root/autodl-tmp/sttrack_full152_content_20260926')
ROOT = Path('/root/autodl-tmp/sttrack_m89_content_20260927')
sys.path.insert(0, '/root/autodl-tmp/sttrack_full152_evaluation_20260925/interface')
from semantic_runtime import checked_plan


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(dataset, variant):
    source = MAIN / 'candidate' / dataset / 'plan.json'
    plan, bundle = checked_plan(source)
    old = OLD_CONTENT / 'M82' / dataset / variant
    old_preparation = json.loads((old / 'preparation.json').read_text())
    old_plan = json.loads((old / 'plan.json').read_text())
    assert bundle['candidate_weight'] == 1
    assert old_preparation['status'] == 'prepared'
    assert old_preparation['source_bank_sha256'] == plan['text_bank_sha256']
    assert sha(old_plan['text_bank_path']) == old_plan['text_bank_sha256']
    target = ROOT / dataset / variant
    target.mkdir(parents=True)
    changed = dict(plan, text_bank_path=old_plan['text_bank_path'],
                   text_bank_sha256=old_plan['text_bank_sha256'],
                   output=str(target / 'predictions'))
    assert {key for key in plan if plan[key] != changed[key]} == {
        'text_bank_path', 'text_bank_sha256', 'output'}
    (target / 'plan.json').write_text(json.dumps(changed, indent=2) + '\n')
    checked_plan(target / 'plan.json')
    receipt = dict(model='m89_candidate', dataset=dataset, variant=variant,
                   category_plan_sha256=sha(source),
                   category_bank_sha256=plan['text_bank_sha256'],
                   intervention_bank_sha256=changed['text_bank_sha256'],
                   checkpoint_sha256=bundle['adapter_checkpoint_sha256'],
                   note='Swapped strings differ; semantic contradiction was not verified.')
    (target / 'preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['depthtrack', 'cdtb'], required=True)
    parser.add_argument('--variant', choices=['empty', 'swapped'], required=True)
    args = parser.parse_args()
    bind(args.dataset, args.variant)
