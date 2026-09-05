"""Download sealed M54 recursion and raw independent-review inputs."""
import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko


remote = '/root/autodl-tmp/sttrack_m54_template_reader_v1_20260906'
local = Path(r'C:\Users\gb\.codex_remote_staging\m54_recursive_completed_20260906')
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.RejectPolicy())
client.connect('connect.nmb2.seetacloud.com', port=35786, username='root',
               password=os.environ['TRACK_SSH_PASSWORD'], timeout=25)
sftp = client.open_sftp()
for name in ['recursive.exit', 'analysis.exit', 'controller.exit']:
    assert sftp.open(remote + '/' + name, 'rb').read().strip() == b'0'
receipt = json.loads(sftp.open(remote + '/recursive_receipt.json', 'rb').read())
binding = json.loads(sftp.open(remote + '/review_inputs/binding.json', 'rb').read())
assert len(receipt['sequences']) == binding['sequences'] == 22
assert binding['frames'] == 33130 and binding['all_recursive_outputs_sealed_before_gt_copy']
expected = {'recursive/' + item['sequence'] + '.json': item['sha256'] for item in receipt['sequences']}
expected.update({'review_inputs/' + name: item['sha256'] for name, item in binding['files'].items()})
expected['recursive_result.json'] = binding['recursive_result_sha256']
expected['spec.json'] = '1ca1387e6eb33c897e12d5e0c10b746d48257e7745906c431e8ac857f63d7267'
fixed = ['recursive.exit', 'analysis.exit', 'controller.exit', 'recursive.log', 'analysis.log',
         'recursive_receipt.json', 'recursive_result.json', 'spec.json', 'review_inputs/binding.json']
local.mkdir()
files = {}
for name in sorted(set(fixed) | set(expected)):
    data = sftp.open(remote + '/' + name, 'rb').read()
    digest = hashlib.sha256(data).hexdigest()
    if name in expected:
        assert digest == expected[name], name
    path = local / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    files[name] = {'bytes': len(data), 'sha256': digest}
sftp.close()
client.close()
result = json.loads((local / 'recursive_result.json').read_text())
assert result['recursive_receipt_sha256'] == files['recursive_receipt.json']['sha256']
assert result['checkpoint_sha256'] == '53374903f5c7314dbc08c857d9ac6d56f2adc914908e62831176fb7c2b7d0c05'
download = dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_root=remote,
                files=files, groundtruth_source='Dataset files copied after all 22 recursive trajectories were sealed',
                claim='Raw review bundle; downloading does not independently validate metrics.')
(local / 'download_binding.json').write_text(json.dumps(download, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'files': len(files), 'bytes': sum(v['bytes'] for v in files.values()),
                  'aggregates': result['aggregates'], 'primary_pass': result['primary_pass']}, indent=2))
