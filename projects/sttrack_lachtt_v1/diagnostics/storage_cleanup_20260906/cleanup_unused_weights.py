from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil


BASE = Path('/root/autodl-tmp').resolve()
OUT = BASE / 'sttrack_disk_cleanup_20260906_1550'
M55 = BASE / 'sttrack_m55_tsg_direction_v2_20260906'
M57 = BASE / 'sttrack_m57_initial_binding_v1_20260906'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def disk():
    return dict(zip(('total', 'used', 'free'), shutil.disk_usage(str(BASE))))


assert json.loads((M55 / 'recursive_result.json').read_text())['primary_pass'] is False
for name in ('train_control.exit', 'train_clone.exit', 'control_recursive.exit', 'clone_recursive.exit', 'recursive_analysis.exit'):
    assert (M55 / name).read_text().strip() == '0'
assert json.loads((M57 / 'base_decision.json').read_text())['checkpoint'] == str(BASE / 'sttrack_checkpoints/STTrack_Vot22.pth.tar')
assert (M57 / 'runtime_contract/contract.json').is_file()
assert (BASE / 'pretrained/DropTrack_k700_800E_alldata.pth.tar').is_file()
assert Path('/home/SRTrack_RGBD_L/tools/build_droptrack_initializer.py').is_file()

targets = [
    (M55 / 'training/control/model_final.pth', 'fe1d4b2faddc7f0dc523b6ec9cacbbf878f523cfc1985cd2a0d8a3fb4c4ba284', 'Completed M55 negative promotion experiment; results and source retained.'),
    (M55 / 'training/clone/model_final.pth', 'd1ce44c82593fe5f508954a536762ba9ef772e9264da12500e62f80064e46c81', 'Completed M55 negative promotion experiment; results and source retained.'),
    (BASE / 'srtrack_clean_droptrack_initializer_seed2026/DropTrack_mapped_visual.pth.tar', '7def9f2df523356b86ab5b120fb13fbcd96f107802067eb277105963b5bc672a', 'Old SRTrack derived initializer; original source, seed, mapping manifest and builder retained.'),
    (BASE / 'dinov2_vits14_pretrain.pth', 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9', 'Retired M32 global descriptor diagnostic; not used by current language experiments.'),
]
for arm in ('candidate_text', 'candidate_visual', 'initial_text', 'initial_visual'):
    targets.append((M57 / ('runtime_contract/' + arm + '_temporary.pth'), None, 'Untrained temporary runtime wiring check; formal trained head retained.'))

protected = [BASE / 'sttrack_checkpoints/STTrack_Vot22.pth.tar', BASE / 'sutrack_assets/weights/ViT-L-14.pt', BASE / 'sutrack_assets/weights/SUTRACK_ep0180_l384.pth.tar', BASE / 'pretrained/DropTrack_k700_800E_alldata.pth.tar']
for folder in ('qwen/Qwen3_8B', 'qwen/Qwen2.5-VL-3B-Instruct'):
    files = sorted((BASE / folder).glob('*.safetensors'))
    assert files
    protected.extend(files)
protected.extend(M57 / ('training/' + arm + '_final.pth') for arm in ('candidate_text', 'candidate_visual', 'initial_text', 'initial_visual'))
protected_before = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in protected}

entries = []
for path, expected, reason in targets:
    assert BASE in path.resolve().parents
    assert path.resolve() == path and path.is_file() and path.stat().st_nlink == 1
    digest = sha(path)
    assert expected is None or digest == expected, str(path)
    entries.append({'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': digest, 'reason': reason})

OUT.mkdir()
plan = {'schema': 'track_unused_weight_cleanup_v1', 'created_utc': datetime.now(timezone.utc).isoformat(), 'authorization': 'User requested deleting unused weights and explicitly preserving both Qwen models.', 'disk_before': disk(), 'deleted_weight_files': entries, 'protected_weight_metadata_before': protected_before, 'm55_result_sha256': sha(M55 / 'recursive_result.json'), 'm57_base_decision_sha256': sha(M57 / 'base_decision.json'), 'script_sha256': sha(Path(__file__)), 'retirement_limit': 'Deleted M55 weights require retraining for exact reruns; predictions, metrics, configs and logs are retained.'}
(OUT / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
for entry in entries:
    Path(entry['path']).unlink()
assert all(not Path(e['path']).exists() for e in entries)
assert protected_before == {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in protected}
result = {'status': 'complete', 'completed_utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha(OUT / 'plan.json'), 'deleted_files': len(entries), 'deleted_bytes': sum(e['size_bytes'] for e in entries), 'disk_after': disk(), 'all_protected_weights_present_and_metadata_unchanged': True}
(OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
