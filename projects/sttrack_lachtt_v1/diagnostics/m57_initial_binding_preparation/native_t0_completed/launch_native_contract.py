"""Check the fixed native reference on idle GPU1 while M55 Control finishes."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess


base = Path('/root/autodl-tmp/sttrack_m55_tsg_direction_v2_20260906')
assert (base / 'recursive_queue_gpu1_controller.exit').read_text().strip() == '0'
preparation = json.loads((base / 'preparation.json').read_text())
source = preparation['code_sha256']['control']
code = base / 'code/control'
for name, digest in source.items():
    assert hashlib.sha256((code / name).read_bytes()).hexdigest() == digest, name
m56 = json.loads(Path('/root/autodl-tmp/sttrack_m56_attribute_alignment_v1_20260906/spec.json').read_text())
common = set(source) & set(m56['source_sha256'])
for name in common:
    assert source[name] == m56['source_sha256'][name], name
checkpoint = Path(m56['base_checkpoint'])
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == m56['base_checkpoint_sha256']
prep = Path('/root/autodl-tmp/sttrack_m57_t0_contract_preparation_20260906')
checker = prep / 'verify_t0_extraction.py'
manifest = prep / 'initialization_only.json'
extractor = Path('/root/autodl-tmp/sttrack_m57_initial_binding_preparation_v2_20260906/initial_observation.py')
expected = {
    checker: '272fb28aa96e86f47d4354360e58e8f3dbca3de79441a55e5808f95e680e3598',
    manifest: '2bff3239f6a3a43a718ec53d7fc0c50a06c727d77168fd99c29591c917e4f981',
    extractor: '2f63f49f90b08f3e205c9bb48cb22aad1d800b6bc7d1d823746c638bf9ffddde',
}
for path, digest in expected.items():
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, str(path)
used = int(subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=memory.used',
                                  '--format=csv,noheader,nounits'], universal_newlines=True).strip())
assert used < 500, used
root = Path('/root/autodl-tmp/sttrack_m57_native_t0_contract_20260906')
root.mkdir()
spec = dict(checker_sha256=expected[checker], cuda_visible_devices='1', code_root=str(code), source_sha256=source,
    checkpoint=str(checkpoint), checkpoint_sha256=m56['base_checkpoint_sha256'], extractor=str(extractor),
    extractor_sha256=expected[extractor], initialization_manifest=str(manifest),
    initialization_manifest_sha256=expected[manifest], dataset_root=m56['dataset_root'], output=str(root / 'output'),
    sequence='chair01_indoor', frames=102,
    scope='Native reference engineering check only; final language base selection still pending M55 paired recursion',
    protocol_adjustment='Run this required native reference check on idle GPU1 while Control completes; repeat on Clone if that base is later adopted',
    formal_feature_collection=False, formal_training=False, shared_m56_source_files_verified=len(common))
(root / 'spec.json').write_text(json.dumps(spec, indent=2) + '\n')
wrapper = ('#!/bin/bash\ncd ' + str(code) + '\nCUDA_VISIBLE_DEVICES=1 /root/autodl-tmp/envs/sttrack/bin/python -u '
    + str(checker) + ' --spec ' + str(root / 'spec.json') + ' > ' + str(root / 'run.log') + ' 2>&1 &\n'
    + 'child=$!\nprintf "%s\\n" "$child" > ' + str(root / 'run.pid') + '\nwait "$child"\n'
    + 'status=$?\nprintf "%s\\n" "$status" > ' + str(root / 'run.exit') + '\nexit "$status"\n')
(root / 'run.sh').write_text(wrapper)
session = 'm57_native_t0_20260906'
subprocess.run(['screen', '-dmS', session, 'bash', str(root / 'run.sh')], check=True)
record = dict(status='dispatched', screen=session, gpu=1, used_mib_before=used,
    observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    spec_sha256=hashlib.sha256((root / 'spec.json').read_bytes()).hexdigest(),
    wrapper_sha256=hashlib.sha256((root / 'run.sh').read_bytes()).hexdigest(),
    launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    source_files=len(source), common_m56_source_files=len(common), estimated_seconds=120,
    preflight_correction='The first preflight looked for a not-installed overlay extractor and stopped before launch; this run binds the existing frozen preparation extractor with the same published SHA')
(root / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
