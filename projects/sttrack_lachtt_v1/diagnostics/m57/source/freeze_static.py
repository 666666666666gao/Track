"""Predeclare fixed-mask M57 snapshot counterfactuals after four-head sealing."""
import ast
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from run_recursive import bound
from train import sha, write


root = Path('/root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906')
spec, recursive, training, cases = bound(root)
assert not (root / 'static_spec.json').exists()
assert (root / 'training_seal.json').exists()
ast.parse((root / 'analyze_static.py').read_text(), feature_version=(3, 8))
bank = torch.load(root / 'text_development.pt', map_location='cpu')
groups = defaultdict(list)
for index, name in enumerate(bank['sequences']):
    groups[tuple(bank['mask'][index].tolist())].append(name)
donors, unpaired = {}, []
for mask, names in sorted(groups.items()):
    names = sorted(names)
    if len(names) == 1:
        unpaired.extend(names)
    else:
        donors.update({name: names[(i + 1) % len(names)] for i, name in enumerate(names)})
assert len(donors) >= 2 and all(name != donor for name, donor in donors.items())
parent = Path(spec['source_root'])
parent_spec = json.loads((parent / 'spec.json').read_text())
assert sha(parent / 'training_labels.json') == parent_spec['labels_sha256']
receipts = []
for shard in [0, 1]:
    receipts += json.loads((parent / ('shard%d_receipt.json' % shard)).read_text())['sequences']
names = {case['sequence'] for case in cases}
features = [dict(sequence=row['sequence'], sha256=row['feature_sha256'])
            for row in sorted(receipts, key=lambda x: x['sequence']) if row['sequence'] in names]
assert len(features) == 22 and {row['sequence'] for row in features} == names
mode_list = [dict(name=arm, variant=arm, content='native') for arm in spec['variants']]
mode_list += [dict(name=arm + '_' + content, variant=arm, content=content)
              for arm in ['candidate_text', 'initial_text'] for content in ['empty', 'same_mask_shuffled']]
inputs = [parent / name for name in ['spec.json', 'training_labels.json', 'shard0_receipt.json', 'shard1_receipt.json']]
inputs += [root / name for name in ['text_development.pt', 'initial_references/initial_development.pt', 'training_seal.json']]
frozen = dict(schema='m57_same_mask_static_content_diagnostic_v1',
    frozen_utc=datetime.now(timezone.utc).isoformat(), recursive_spec_sha256=sha(root / 'recursive_spec.json'),
    analyzer_sha256=sha(root / 'analyze_static.py'), preparer_sha256=sha(__file__),
    input_sha256={str(path): sha(path) for path in inputs}, development_features=features,
    modes=mode_list, same_mask_donors=donors, shuffle_unpaired_sequences=sorted(unpaired),
    donor_rule='Lexicographically sorted cyclic donors within each identical slot-mask group; singleton groups excluded only from shuffling',
    original_mask_used_for_all_modes=True,
    interpretation='Shuffling changes lexical content and can change category; not an annotated false-attribute intervention',
    numerical_development_analysis_started=False, promotion_decision=False,
    same_runtime_t0_protocol_as_training=True)
write(root / 'static_spec.json', frozen)
print(json.dumps(dict(status='frozen_static_spec', spec_sha256=sha(root / 'static_spec.json'),
    modes=len(mode_list), shuffle_sequences=len(donors), shuffle_unpaired_sequences=sorted(unpaired))))
