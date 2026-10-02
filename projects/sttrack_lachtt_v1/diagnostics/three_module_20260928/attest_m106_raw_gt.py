"""Read dataset GT directly and verify the original M90 current-box cache."""
from pathlib import Path
import hashlib
import json
import math

SPEC=Path('/root/autodl-tmp/sttrack_full152_paired_20260925/M82/training_spec.json')
CACHE=Path('/root/autodl-tmp/sttrack_m90_train_states_20260928')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    preparation=json.loads((CACHE/'preparation.json').read_text())
    assert sha(SPEC)==preparation['source_spec_sha256']
    labels=json.loads((CACHE/'training_labels.json').read_text())
    assert sha(CACHE/'training_labels.json')==preparation['training_labels_sha256']
    spec=json.loads(SPEC.read_text())
    by_sequence={}
    for key,label in labels.items():
        by_sequence.setdefault(label['sequence'],[]).append((key,label))
    files=[];valid=0;checked=0
    for entry in spec['sequence_order']:
        p=Path(spec['dataset_root'])/entry['sequence']/'groundtruth.txt'
        assert sha(p)==entry['groundtruth_sha256']
        gt=[[float(v) for v in line.split(',')] for line in p.read_text().splitlines() if line.strip()]
        assert len(gt)>=entry['rgb_frames'] and all(len(box)==4 for box in gt)
        for key,label in by_sequence[entry['sequence']]:
            frame=label['frame'];box=gt[frame]
            current=box if all(math.isfinite(v) for v in box) and box[2]>0 and box[3]>0 else None
            assert current==label['current'],key
            checked+=1;valid+=int(current is not None)
        files.append(dict(sequence=entry['sequence'],path=str(p),bytes=p.stat().st_size,sha256=sha(p),
                          expected_spec_sha256=entry['groundtruth_sha256'],raw_rows=len(gt),used_rgb_frames=entry['rgb_frames']))
    assert len(files)==152 and checked==3502 and valid==3039
    print(json.dumps(dict(status='complete_direct_dataset_GT_attestation',source_sha256=sha(Path(__file__)),
                          source_spec_sha256=sha(SPEC),training_labels_sha256=sha(CACHE/'training_labels.json'),
                          current_events_checked=checked,valid_events_exact=valid,files=files,
                          no_prediction_derived_GT=True,scope='Dataset localization labels only; no semantic or physical identity truth.')))


if __name__=='__main__':
    main()
