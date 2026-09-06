"""Verify sealed report arithmetic, not fresh region overlaps or tracking."""

import hashlib
import json
from pathlib import Path


def verify(root):
    result = json.loads((root / 'result.json').read_text())
    spec = json.loads((root / 'spec.json').read_text())
    analysis_path = root / 'run/master/analysis/sttrack_default_full127_analysis.json'
    analysis = json.loads(analysis_path.read_text())
    assert hashlib.sha256(analysis_path.read_bytes()).hexdigest() == result['analysis_sha256']
    assert hashlib.sha256((root / 'spec.json').read_bytes()).hexdigest() == result['spec_sha256']
    assert result['status'] == 'complete' and result['integrity_pass']
    assert all((root / name).read_text().strip() == '0'
               for name in ('evaluate.exit', 'finalize.exit', 'controller.exit'))
    assert not result['new_checkpoint'] and not result['language_enabled']
    assert result['optimizer_steps'] == 0

    values = analysis['results']['baseline']['results']
    official = {'eao': values[0][0][0], 'acc': values[2][0][0], 'rob': values[2][0][1]}
    assert official == result['metrics_fraction']
    assert {k: v * 100 for k, v in official.items()} == result['metrics_percent']
    curve = values[1][0][0]
    assert len(curve) == 755
    selected = curve[115:755 + 1]  # Exact slice used by the bound official source.
    eao = sum(selected) / len(selected)
    assert abs(eao - official['eao']) < 1e-12

    outcomes = result['failure_outcomes']
    sequences = result['per_sequence_failures']
    assert len(outcomes) == result['anchors'] == spec['anchors'] == 1765
    assert len(sequences) == len(analysis['sequences']) == 127
    assert sum(x['failed'] for x in outcomes.values()) == result['confirmed_failures'] == 183
    weighted_robustness = 0.0
    total_length = 0
    for sequence, counts in sequences.items():
        rows = [x for x in outcomes.values() if x['sequence'] == sequence]
        assert len(rows) == counts['anchors']
        assert sum(x['failed'] for x in rows) == counts['confirmed_failures']
        for row in rows:
            assert row['failed'] == (row['progress'] < row['run_length'])
        length = analysis['sequences'][sequence]['length']
        robustness = sum(x['progress'] for x in rows) / sum(x['run_length'] for x in rows)
        weighted_robustness += robustness * length
        total_length += length
    robustness = weighted_robustness / total_length
    assert abs(robustness - official['rob']) < 1e-12
    passed = {k: v * 100 > spec['target_thresholds_percent'][k] for k, v in official.items()}
    assert passed == result['exceeds_requested_vot_thresholds']
    return {
        'status': 'pass',
        'scope': 'Sealed official report, EAO returned-curve reaggregation and ROB outcome reaggregation',
        'fresh_raster_overlaps_recomputed': False,
        'tracking_rerun': False,
        'metrics_percent': result['metrics_percent'],
        'reaggregated_eao_fraction': eao,
        'reaggregated_rob_fraction': robustness,
        'eao_curve_length': len(curve),
        'eao_selected_values': len(selected),
        'sequences': len(sequences),
        'anchors': len(outcomes),
        'confirmed_failures': result['confirmed_failures'],
        'strict_target_pass': passed,
    }


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
