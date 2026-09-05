"""Screen sealed M45 fitting proposals using native two-stream continuity."""
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


source = Path('/root/autodl-tmp/sttrack_m44_candidate_set_v1_20260905')
control = Path('/root/autodl-tmp/sttrack_m45_default_priority_v1_20260905')
output = Path('/root/autodl-tmp/sttrack_m48_native_continuity_audit_v1_20260905')
spec = json.loads((source / 'spec.json').read_text())
sys.path.insert(0, spec['repository'])
from tools.train_sttrack_m44 import check_binding
from tools.train_sttrack_m42 import overlaps

torch.set_num_threads(1)
check_binding(source, spec)
assert sha(control / 'geometry_result.json') == 'e2647ea8d9738632d93f99d0108e638025b1c38894f8560cc9a532eb4f13f39d'
sealed = json.loads((control / 'geometry_result.json').read_text())
assert sha(control / 'geometry_final.pth') == sealed['checkpoint_sha256']
proposals = {r['key']: r for r in sealed['fit']['rows']}
assert len(proposals) == 1511
assert sha(source / 'training_labels.json') == spec['labels_sha256']
labels = json.loads((source / 'training_labels.json').read_text())
cases = {r['sequence'] for r in json.loads((source / 'inference_inputs.json').read_text()) if r['split'] == 'fit'}
assert len(cases) == 63
receipts = [r for shard in [0, 1] for r in json.loads((source / f'shard{shard}_receipt.json').read_text())['sequences'] if r['sequence'] in cases]
assert len(receipts) == 63
rows = []
for receipt in sorted(receipts, key=lambda r: r['sequence']):
    path = source / 'features' / (receipt['sequence'] + '.pt')
    assert sha(path) == receipt['feature_sha256']
    data = torch.load(path, map_location='cpu')
    assert data['fold'] in spec['fit_folds']
    assert data['current'].dtype == data['previous'].dtype == torch.float16
    current = data['current'].float().flatten(-2)
    previous = data['previous'].float()[:, 0].flatten(-2)
    assert current.shape[1:] == (10, 2, 12288)
    similarity = F.cosine_similarity(current, previous[:, None], dim=-1)
    assert torch.isfinite(similarity).all()
    for i, record in enumerate(data['records']):
        assert record['previous_choice'] == 0 and record['previous_frame'] == record['frame'] - 1
        row = proposals[record['key']]
        label = labels[record['key']]
        assert label['sequence'] in cases and label['fold'] in spec['fit_folds']
        if label['current'] is None:
            ious = torch.zeros(10)
        else:
            gt = torch.tensor(label['current'])
            ious = overlaps(data['current_boxes'][i], gt)
            ious[0] = overlaps(data['public_bbox'][i], gt)
        chosen = row['chosen']
        assert float(ious[0]) == row['default_iou']
        assert float(ious[chosen]) == row['selected_iou']
        delta = similarity[i, chosen] - similarity[i, 0]
        accepted = bool((delta >= 0).all())
        selected = chosen if accepted else 0
        rows.append(dict(key=record['key'], sequence=data['sequence'], frame=record['frame'],
            proposal=chosen, selected=selected, accepted=accepted, vetoed=chosen != 0 and not accepted,
            action_none=row['action_none'], default_similarity=similarity[i, 0].tolist(),
            proposal_similarity=similarity[i, chosen].tolist(), similarity_delta=delta.tolist(),
            default_iou=row['default_iou'], proposal_iou=row['selected_iou'], selected_iou=float(ious[selected])))
assert len(rows) == 1511 and {r['key'] for r in rows} == set(proposals)


def metrics(values, selection):
    return dict(events=len(values), changes=sum(r[selection] != 0 for r in values),
        mean_iou=sum(r['proposal_iou' if selection == 'proposal' else 'selected_iou'] for r in values) / len(values),
        rescues=sum(r['default_iou'] <= .1 and r['proposal_iou' if selection == 'proposal' else 'selected_iou'] >= .5 for r in values),
        breaks=sum(r['default_iou'] >= .5 and r['proposal_iou' if selection == 'proposal' else 'selected_iou'] <= .1 for r in values),
        beneficial=sum(r['proposal_iou' if selection == 'proposal' else 'selected_iou'] > r['default_iou'] for r in values),
        harmful=sum(r['proposal_iou' if selection == 'proposal' else 'selected_iou'] < r['default_iou'] for r in values),
        correct=sum(r['proposal_iou' if selection == 'proposal' else 'selected_iou'] >= .5 for r in values))


summary = dict(events=1511, sequences=63, default_mean_iou=sum(r['default_iou'] for r in rows) / len(rows),
    proposal=metrics(rows, 'proposal'), screened=metrics(rows, 'selected'),
    vetoed=sum(r['vetoed'] for r in rows),
    vetoed_rescues=sum(r['vetoed'] and r['default_iou'] <= .1 and r['proposal_iou'] >= .5 for r in rows),
    vetoed_breaks=sum(r['vetoed'] and r['default_iou'] >= .5 and r['proposal_iou'] <= .1 for r in rows),
    vetoed_beneficial=sum(r['vetoed'] and r['proposal_iou'] > r['default_iou'] for r in rows),
    vetoed_harmful=sum(r['vetoed'] and r['proposal_iou'] < r['default_iou'] for r in rows),
    sequences_with_accepted_changes=len({r['sequence'] for r in rows if r['selected'] != 0}),
    sequences_with_retained_rescues=len({r['sequence'] for r in rows if r['default_iou'] <= .1 and r['selected_iou'] >= .5}))
result = dict(status='complete', integrity_pass=True, summary=summary,
    scope='Original M45 1511 fitting pairs in 63 DepthTrack Train sequences only. No development/public metric, no optimization or inference-policy deployment.',
    rule='Accept a nondefault M45 proposal only when BOTH unprojected native RGB and depth 4x4 RoI cosines to previous selected RoI are at least those of candidate0; zero margin, no sweep. All cached previous choices are0.',
    limitation='These streams come from the fused tracker and are not independent measurements. Fitting proposals may already overfit and contain no severe regressions. Static screening cannot establish recursive safety or public gain.',
    source_spec_sha256=sha(source / 'spec.json'), labels_sha256=sha(source / 'training_labels.json'),
    control_result_sha256=sha(control / 'geometry_result.json'), checkpoint_sha256=sha(control / 'geometry_final.pth'),
    audit_source_sha256=sha(__file__), rows=rows)
output.mkdir(exist_ok=True)
(output / 'native_continuity_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(status='complete', summary=summary, result_sha256=sha(output / 'native_continuity_audit.json')), indent=2))
