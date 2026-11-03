"""Check official analysis identity before accepting positional metric extraction."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def check(analysis, reference, manifest, sealed):
    """Check analysis metadata and sealed numbers, not trajectory completeness."""
    assert len(manifest["sequences"]) == len(set(manifest["sequences"])) == 127
    assert manifest["total_anchor_count"] == sealed["anchors"] == 1765
    assert sealed["sequences"] == 127
    assert sealed["status"] == "complete_full127"
    assert set(analysis["trackers"]) == {manifest["tracker"]}
    assert set(analysis["sequences"]) == set(manifest["sequences"])
    assert analysis["sequences"] == reference["sequences"]
    for name in manifest["sequences"]:
        assert analysis["sequences"][name]["length"] == manifest["source"][name]["frame_count"], name
    assert analysis["toolkit"] == reference["toolkit"]
    baseline = analysis["results"]["baseline"]
    expected = reference["results"]["baseline"]
    assert baseline["type"] == expected["type"] == "vot.experiment.multistart.MultiStartExperiment"
    assert baseline["parameters"] == expected["parameters"], "Official protocol differs from reference"
    descriptors = baseline["parameters"]["analyses"]
    assert [row["type"] for row in descriptors] == [
        "vot.analysis.multistart.EAOScore",
        "vot.analysis.multistart.EAOCurve",
        "vot.analysis.multistart.AverageAccuracyRobustness",
    ], "Positional extraction requires the declared analysis order"
    values = baseline["results"]
    assert len(values) == 3 and all(len(row) == 1 for row in values)
    assert len(values[0][0]) == 1 and len(values[2][0]) == 5
    metrics = {
        "EAO": float(values[0][0][0]) * 100,
        "ACC": float(values[2][0][0]) * 100,
        "ROB": float(values[2][0][1]) * 100,
    }
    assert all(math.isfinite(value) and 0 <= value <= 100 for value in metrics.values())
    assert metrics == sealed["metrics_percent"], "Sealed metrics differ from official analysis"
    return {
        "status": "analysis_identity_and_metric_binding_checked",
        "tracker": manifest["tracker"],
        "sequences": 127,
        "metrics_percent": metrics,
        "toolkit": analysis["toolkit"],
        "protocol": baseline["parameters"],
        "trajectory_coverage_checked": False,
        "checkpoint_source_binding_checked": False,
        "independent_completed_audit": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sealed-result", type=Path, required=True)
    arguments = parser.parse_args()
    payloads = {}
    hashes = {}
    for name in ("analysis", "reference", "manifest", "sealed_result"):
        path = getattr(arguments, name)
        data = path.read_bytes()
        payloads[name] = json.loads(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    assert hashes["analysis"] == payloads["sealed_result"]["analysis_sha256"]
    result = check(payloads["analysis"], payloads["reference"], payloads["manifest"], payloads["sealed_result"])
    result["input_sha256"] = hashes
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
