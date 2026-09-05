#!/usr/bin/env python3
"""Analyze the paired M39 VOT low22 default/no-update experiment."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(
    "/root/autodl-tmp/sttrack_lachtt_m39_vot_low22_template_ablation_"
    "v1_20260902")
PYTHON = "/root/miniconda3/envs/mplt/bin/python"
SUTRACK_REPO = Path("/home/SUTrack_RGBD_L")
ARMS = {
    "default": "sttrack_m39_default_low22",
    "no_update": "sttrack_m39_no_update_low22",
}

sys.path.insert(0, str(SUTRACK_REPO))
from tools.finalize_vot_transaction_low22 import collect_confirmed_failure_outcomes


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(arm, tracker):
    workspace = ROOT / arm / "master"
    name = "m39_{}_low22_analysis".format(arm)
    analysis_path = workspace / "analysis" / (name + ".json")
    command = [
        PYTHON, "-m", "vot", "analysis", "--workspace", str(workspace),
        "--format", "json", "--name", name, tracker]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SUTRACK_REPO)
    with (ROOT / (arm + "_analysis.log")).open("ab", buffering=0) as log:
        subprocess.run(command, cwd=str(SUTRACK_REPO), env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    results = payload["results"]["baseline"]["results"]
    metrics = {
        "eao": float(results[0][0][0]),
        "acc": float(results[2][0][0]),
        "rob": float(results[2][0][1]),
    }
    outcomes, failures, per_sequence, settings = (
        collect_confirmed_failure_outcomes(workspace, tracker, expected_anchors=303))
    return {
        "tracker": tracker,
        "metrics_fraction": metrics,
        "metrics_percent": {key: value * 100.0 for key, value in metrics.items()},
        "confirmed_failures": failures,
        "per_sequence_failures": per_sequence,
        "failure_outcomes": outcomes,
        "failure_settings": settings,
        "analysis": str(analysis_path),
        "analysis_sha256": sha256_file(analysis_path),
        "merge_sha256": sha256_file(ROOT / arm / "merge_result.json"),
    }


def main():
    arms = {name: analyze(name, tracker) for name, tracker in ARMS.items()}
    before = arms["default"]
    after = arms["no_update"]
    new_failures = []
    rescues = []
    for key in sorted(before["failure_outcomes"]):
        default_failed = before["failure_outcomes"][key]["failed"]
        no_update_failed = after["failure_outcomes"][key]["failed"]
        if not default_failed and no_update_failed:
            new_failures.append(key)
        if default_failed and not no_update_failed:
            rescues.append(key)
    delta = {
        key: (after["metrics_fraction"][key] - before["metrics_fraction"][key]) * 100.0
        for key in before["metrics_fraction"]}
    checks = {
        "eao_strictly_improved": delta["eao"] > 0,
        "rob_strictly_improved": delta["rob"] > 0,
        "acc_within_minus_0_10_pp": delta["acc"] >= -0.10,
        "confirmed_failures_not_increased": (
            after["confirmed_failures"] <= before["confirmed_failures"]),
        "new_confirmed_failures_zero": len(new_failures) == 0,
    }
    result = {
        "schema": "sttrack_m39_vot_low22_template_ablation_result_v1",
        "status": "complete",
        "scope": {"sequence_count": 22, "anchor_count": 303, "low22_only": True},
        "arms": arms,
        "delta_no_update_vs_default_percent_points": delta,
        "failure_delta": after["confirmed_failures"] - before["confirmed_failures"],
        "rescued_anchors": rescues,
        "newly_failed_anchors": new_failures,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
        "full127_authorized": all(checks.values()),
        "automatic_full127_launch": False,
    }
    output = ROOT / "m39_result.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "default_percent": before["metrics_percent"],
        "no_update_percent": after["metrics_percent"],
        "delta_pp": delta,
        "default_failures": before["confirmed_failures"],
        "no_update_failures": after["confirmed_failures"],
        "rescues": len(rescues),
        "new_failures": len(new_failures),
        "gate_passed": result["gate_passed"],
        "result_sha256": sha256_file(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
