#!/usr/bin/env python3
"""Derive immutable per-sequence diagnostics from the completed M39 result."""

import json
from pathlib import Path


ROOT = Path(
    "/root/autodl-tmp/"
    "sttrack_lachtt_m39_vot_low22_template_ablation_v1_20260902")
HISTORICAL_SUTRACK = {
    "metrics_percent": {
        "eao": 43.274104354018916,
        "acc": 72.06551125207067,
        "rob": 54.38802182117735,
    },
    "confirmed_failures": 195,
}


def main():
    result = json.loads((ROOT / "m39_result.json").read_text(encoding="utf-8"))
    default = result["arms"]["default"]
    no_update = result["arms"]["no_update"]
    rows = []
    for sequence in sorted(default["per_sequence_failures"]):
        keys = [
            key for key, value in default["failure_outcomes"].items()
            if value["sequence"] == sequence]
        default_progress = sum(
            default["failure_outcomes"][key]["progress"] /
            default["failure_outcomes"][key]["run_length"]
            for key in keys) / len(keys)
        no_update_progress = sum(
            no_update["failure_outcomes"][key]["progress"] /
            no_update["failure_outcomes"][key]["run_length"]
            for key in keys) / len(keys)
        default_failures = default["per_sequence_failures"][sequence][
            "confirmed_failures"]
        no_update_failures = no_update["per_sequence_failures"][sequence][
            "confirmed_failures"]
        new_failures = [
            key for key in result["newly_failed_anchors"]
            if no_update["failure_outcomes"][key]["sequence"] == sequence]
        rescues = [
            key for key in result["rescued_anchors"]
            if no_update["failure_outcomes"][key]["sequence"] == sequence]
        rows.append({
            "sequence": sequence,
            "anchors": len(keys),
            "default_confirmed_failures": default_failures,
            "no_update_confirmed_failures": no_update_failures,
            "failure_delta": no_update_failures - default_failures,
            "default_mean_progress_ratio": default_progress,
            "no_update_mean_progress_ratio": no_update_progress,
            "progress_ratio_delta": no_update_progress - default_progress,
            "newly_failed_anchors": new_failures,
            "rescued_anchors": rescues,
        })

    default_metrics = default["metrics_percent"]
    diagnosis = {
        "schema": "sttrack_m39_vot_low22_template_ablation_diagnosis_v1",
        "source_result": str(ROOT / "m39_result.json"),
        "historical_sutrack_identity_only_reference": HISTORICAL_SUTRACK,
        "default_sttrack_delta_vs_historical_sutrack_percent_points": {
            key: default_metrics[key] - HISTORICAL_SUTRACK["metrics_percent"][key]
            for key in default_metrics},
        "default_sttrack_failure_delta_vs_historical_sutrack": (
            default["confirmed_failures"] -
            HISTORICAL_SUTRACK["confirmed_failures"]),
        "summary": {
            "worsened_sequence_count": sum(row["failure_delta"] > 0 for row in rows),
            "improved_sequence_count": sum(row["failure_delta"] < 0 for row in rows),
            "unchanged_sequence_count": sum(row["failure_delta"] == 0 for row in rows),
            "new_failure_count": len(result["newly_failed_anchors"]),
            "rescue_count": len(result["rescued_anchors"]),
            "net_failure_delta": result["failure_delta"],
        },
        "per_sequence": rows,
    }
    output = ROOT / "m39_sequence_diagnosis.json"
    output.write_text(
        json.dumps(diagnosis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(diagnosis["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
