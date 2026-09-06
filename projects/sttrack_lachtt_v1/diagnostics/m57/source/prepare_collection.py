"""Freeze same-native-base sources and initialization-only M57 collection inputs."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path('/root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906')
decision_path = root / 'base_decision.json'
assert sha(decision_path) == 'bdeaa840a47dca0588bca214829150f1dee79b870c2258bcb942cb0133f79c1b'
decision = json.loads(decision_path.read_text())
assert decision['status'] == 'native_visual_base_fixed_for_m57'
previous = Path('/root/autodl-tmp/sttrack_m56_attribute_alignment_v1_20260906')
assert sha(previous / 'spec.json') == 'cafc4471d87c8c8de5c9cb051ae659f1d14404691d2d178cff2131fa7f30a45e'
m56 = json.loads((previous / 'spec.json').read_text())
assert decision['checkpoint_sha256'] == m56['base_checkpoint_sha256'] == sha(decision['checkpoint'])
code = root / 'code'
code.mkdir()
sources = {}
for source_root, items in [(Path(decision['visual_code_root']), decision['visual_source_manifest']),
                           (Path(m56['repository']), m56['source_sha256'])]:
    for name, digest in items.items():
        source = source_root / name
        assert sha(source) == digest, str(source)
        if name in sources:
            assert sources[name] == digest
        destination = code / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        sources[name] = digest
prototype = Path('/root/autodl-tmp/sttrack_m57_initial_binding_preparation_v2_20260906')
additions = [
    (prototype / 'model.py', 'lib/models/sttrack/lachtt_initial_instance_alignment.py', '59af4516e854e686dc0425c7e57dc70df7238f7afbd12e68fb170df6eaf06746'),
    (prototype / 'initial_observation.py', 'lib/test/tracker/sttrack_initial_instance_observation.py', '2f63f49f90b08f3e205c9bb48cb22aad1d800b6bc7d1d823746c638bf9ffddde'),
    (root / 'runtime.py', 'lib/test/tracker/sttrack_initial_instance_candidate_set.py', sha(root / 'runtime.py')),
]
for source, name, digest in additions:
    assert sha(source) == digest and name not in sources
    target = code / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    sources[name] = digest
parent = Path(m56['source_root'])
original_inputs = parent / 'inference_inputs.json'
assert sha(original_inputs) == '0e9a9bd34d4fb5682ba1f90966c2a8d495d0a10cef5ba2e02bbc8c29039c2ae0'
inputs = [{key: row[key] for key in ['sequence', 'split', 'frames', 'init_bbox']}
          for row in json.loads(original_inputs.read_text())]
assert len(inputs) == 85 and len({row['sequence'] for row in inputs}) == 85
input_path = root / 'initialization_inputs.json'
input_path.write_text(json.dumps(inputs, indent=2) + '\n')
images = {}
for row in inputs:
    folder = Path(m56['dataset_root']) / row['sequence']
    images[row['sequence']] = dict(rgb=sha(folder / 'color/00000001.jpg'), depth=sha(folder / 'depth/00000001.png'))
assert sha(previous / 'text_bank.pt') == '93cd72856177f338d14ae05959459e36bf3082ddda340e9b332e5a33089ae967'
bank = torch.load(previous / 'text_bank.pt', map_location='cpu')
assert set(bank['sequences']) == {row['sequence'] for row in inputs}
text_files = {}
for split, count in [('fit', 63), ('development', 22)]:
    names = [row['sequence'] for row in inputs if row['split'] == split]
    assert len(names) == count
    index = torch.tensor([bank['sequences'].index(name) for name in names])
    target = root / ('text_' + split + '.pt')
    torch.save(dict(sequences=names, split=split, tokens=bank['tokens'][index], mask=bank['mask'][index],
        empty=bank['empty'], original_bank_sha256=sha(previous / 'text_bank.pt')), target)
    text_files[split] = dict(path=str(target), sha256=sha(target), sequences=count)
assert sha(previous / 'fit_labels.json') == '0913d2121efc2e5a5261eebc58d932fa763d039623c11cbe72e14c9796742080'
shutil.copyfile(previous / 'fit_labels.json', root / 'fit_labels.json')
spec = dict(schema='m57_initial_reference_collection_v1', frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    code_root=str(code), source_sha256=sources, checkpoint=decision['checkpoint'], checkpoint_sha256=decision['checkpoint_sha256'],
    base_decision=str(decision_path), base_decision_sha256=sha(decision_path), dataset_root=m56['dataset_root'],
    initialization_inputs=str(input_path), initialization_inputs_sha256=sha(input_path),
    original_initialization_input_sha256=sha(original_inputs), initial_image_sha256=images,
    collector_sha256=sha(root / 'collect_initial.py'), cuda_visible_devices='1', output=str(root / 'initial_references'),
    experiment_plan_sha256=sha(root / 'EXPERIMENT_PLAN.md'), text_files=text_files,
    target_file_sha256=sha(root / 'fit_labels.json'), feature_source_root=str(parent),
    source_state='Native cached trajectory; only the initial reference will be replaced in memory. Full cache/runtime agreement must be checked before fitting.')
(root / 'collection_spec.json').write_text(json.dumps(spec, indent=2) + '\n')
print(json.dumps(dict(status='collection_inputs_frozen', sequences=85, source_files=len(sources),
    collection_spec_sha256=sha(root / 'collection_spec.json'), initialization_inputs_sha256=sha(input_path),
    text_files=text_files, formal_training_started=False), indent=2))
