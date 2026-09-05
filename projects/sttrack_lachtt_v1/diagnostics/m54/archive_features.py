"""Copy sealed M54 RAM feature packets to the user's persistent local workspace."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import paramiko


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-dir', type=Path, required=True)
    parser.add_argument('--spec-sha256', required=True)
    args = parser.parse_args()
    root = '/root/autodl-tmp/sttrack_m54_template_reader_v1_20260906'
    first = datetime.datetime(2026, 9, 5, 21, 0, tzinfo=datetime.timezone.utc).timestamp()
    print('First collection-exit check at 2026-09-05 21:00:00 UTC; then every 240 seconds.', flush=True)
    time.sleep(max(0., first - time.time()))
    while True:
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        client.connect('connect.nmb2.seetacloud.com', port=35786, username='root',
                       password=os.environ['TRACK_SSH_PASSWORD'], timeout=25)
        sftp = client.open_sftp()
        spec_bytes = sftp.open(root + '/spec.json', 'rb').read()
        assert hashlib.sha256(spec_bytes).hexdigest() == args.spec_sha256
        if 'collection.exit' in sftp.listdir(root):
            break
        sftp.close()
        client.close()
        print(datetime.datetime.now(datetime.timezone.utc).isoformat(), 'Collection not complete; next check in 240 seconds.', flush=True)
        time.sleep(240)
    assert sftp.open(root + '/collection.exit').read().decode().strip() == '0'
    receipt_bytes = sftp.open(root + '/collection_receipt.json', 'rb').read()
    receipt = json.loads(receipt_bytes)
    spec = json.loads(spec_bytes)
    assert receipt['status'] == 'complete' and receipt['spec_sha256'] == args.spec_sha256
    assert receipt['events'] == 10615 and receipt['frames'] == 93362 and not receipt['labels_opened']
    assert len(receipt['sequences']) == len({x['sequence'] for x in receipt['sequences']}) == 63
    assert shutil.disk_usage(args.local_dir).free > sum(x['bytes'] for x in receipt['sequences'])
    destination = args.local_dir / 'features'
    destination.mkdir()
    (args.local_dir / 'spec.json').write_bytes(spec_bytes)
    (args.local_dir / 'collection_receipt.json').write_bytes(receipt_bytes)
    copied = []
    for item in receipt['sequences']:
        name = item['sequence'] + '.pt'
        path = destination / name
        assert path.resolve().parent == destination.resolve()
        temporary = path.with_suffix('.partial')
        sftp.get(spec['feature_directory'] + '/' + name, str(temporary))
        assert temporary.stat().st_size == item['bytes'] and digest(temporary) == item['feature_sha256']
        temporary.rename(path)
        copied.append(dict(sequence=item['sequence'], bytes=item['bytes'], sha256=item['feature_sha256']))
        print(json.dumps(dict(copied=len(copied), total=63, sequence=item['sequence'])), flush=True)
    sftp.close()
    client.close()
    result = dict(status='complete', completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        spec_sha256=args.spec_sha256, collection_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        source_sha256=digest(Path(__file__)), remote_feature_directory=spec['feature_directory'],
        sequences=copied, bytes=sum(x['bytes'] for x in copied), content_hashes_verified=True,
        source_feature_files_deleted=False, ground_truth_copied=False)
    (args.local_dir / 'archive_receipt.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'sequences'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
