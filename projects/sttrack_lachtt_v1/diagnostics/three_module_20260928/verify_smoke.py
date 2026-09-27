"""Check M90 GPU smoke parity and capacity before two-shard collection."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    preparation = json.loads((root / 'preparation.json').read_text())
    receipt = json.loads((root / 'smoke_shard0.json').read_text())
    assert receipt['status'] == 'complete' and receipt['smoke'] and receipt['events'] == 3
    assert receipt['model_updates'] == 0 and not receipt['labels_loaded']
    assert receipt['preparation_sha256'] == sha(root / 'preparation.json')
    assert len(receipt['sequences']) == 1
    row = receipt['sequences'][0]
    assert row['maximum_box_error_px'] <= 1e-4 and row['maximum_score_error'] <= 1e-6
    path = root / 'smoke_features' / f"{row['sequence']}.pt"
    assert sha(path) == row['feature_sha256']
    data = torch.load(path, map_location='cpu')
    assert data['event_frames'] == [10, 50, 60]
    assert data['candidate_rois'].shape == (3, 10, 2, 16, 768)
    assert data['candidate_rois'].dtype == torch.float16
    assert data['initial_rois'].shape == (2, 16, 768)
    assert data['dense_boxes'].shape == (3, 256, 4)
    assert data['candidate_cells'].shape == (3, 10)
    assert data['score_map'].shape == (3, 1, 16, 16)
    assert data['response_map'].shape == (3, 1, 16, 16)
    assert data['size_map'].shape == (3, 2, 16, 16)
    assert data['offset_map'].shape == (3, 2, 16, 16)
    assert data['public_bbox'].shape == (3, 4)
    assert data['feature_candidate_count'] == 10 and data['dense_candidate_count'] == 256
    assert not data['labels_loaded']
    estimated_bytes = path.stat().st_size / 3 * preparation['event_count']
    free_bytes = shutil.disk_usage(root).free
    required_bytes = estimated_bytes * 1.25 + 250 * 1024 * 1024
    assert free_bytes >= required_bytes, (free_bytes, required_bytes)
    result = dict(status='verified', smoke_sha256=sha(root / 'smoke_shard0.json'),
                  smoke_feature_sha256=sha(path), smoke_feature_bytes=path.stat().st_size,
                  estimated_full_feature_bytes=int(estimated_bytes),
                  disk_free_bytes=free_bytes, required_free_bytes=int(required_bytes),
                  candidate_rois_shape=list(data['candidate_rois'].shape),
                  dense_boxes_shape=list(data['dense_boxes'].shape),
                  max_bbox_error_px=row['maximum_box_error_px'],
                  max_score_error=row['maximum_score_error'],
                  no_gt_used=True, no_training=True)
    (root / 'smoke_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
