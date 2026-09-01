#!/usr/bin/env python3
"""Compare the sealed M4 and M5 setwise pilot outputs."""

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile


POLICY = {
    "candidate_selection_probability_min": 0.50,
    "selection_probability_margin_min": 0.10,
    "beneficial_probability_min": 0.80,
    "catastrophic_probability_max": 0.05,
    "predicted_gain_min": 0.05,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m4-output", required=True, type=Path)
    parser.add_argument("--m5-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def distribution(values):
    if not values or any(not math.isfinite(float(value)) for value in values):
        raise ValueError("invalid distribution")
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
    }


def summarize(output):
    result_path = output / "result.json"
    events_path = output / "evaluation_events.jsonl.gz"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows = read_jsonl_gz(events_path)
    threshold_pass = {
        "candidate_beats_abstain": 0,
        "candidate_selection_probability": 0,
        "selection_probability_margin": 0,
        "beneficial_probability": 0,
        "catastrophic_probability": 0,
        "predicted_gain": 0,
        "all_conditions": 0,
    }
    values = {key: [] for key in (
        "top_candidate_selection_probability",
        "abstain_probability",
        "top_candidate_margin",
        "top_candidate_beneficial_probability",
        "top_candidate_catastrophic_probability",
        "top_candidate_predicted_gain",
    )}
    outcome = {
        "beneficial_candidates": 0,
        "catastrophic_candidates": 0,
        "beneficial_events": 0,
        "catastrophic_events": 0,
    }
    beneficial_event_details = []
    for row in rows:
        actions = row["actions"]
        beneficial_actions = [action for action in actions
                              if action["actual_beneficial"]]
        catastrophic_actions = [action for action in actions
                                 if action["actual_catastrophic"]]
        outcome["beneficial_candidates"] += len(beneficial_actions)
        outcome["catastrophic_candidates"] += len(catastrophic_actions)
        outcome["beneficial_events"] += int(bool(beneficial_actions))
        outcome["catastrophic_events"] += int(bool(catastrophic_actions))
        top = max(actions, key=lambda action:
                  (action["selection_probability"], action["name"]))
        candidate_probabilities = sorted(
            (action["selection_probability"] for action in actions),
            reverse=True)
        competitor = max(row["abstain_probability"],
                         candidate_probabilities[1])
        margin = top["selection_probability"] - competitor
        checks = {
            "candidate_beats_abstain":
                top["selection_probability"] > row["abstain_probability"],
            "candidate_selection_probability":
                top["selection_probability"] >=
                POLICY["candidate_selection_probability_min"],
            "selection_probability_margin":
                margin >= POLICY["selection_probability_margin_min"],
            "beneficial_probability":
                top["beneficial_probability"] >=
                POLICY["beneficial_probability_min"],
            "catastrophic_probability":
                top["catastrophic_probability"] <=
                POLICY["catastrophic_probability_max"],
            "predicted_gain":
                top["predicted_gain"] >= POLICY["predicted_gain_min"],
        }
        for key, accepted in checks.items():
            threshold_pass[key] += int(accepted)
        threshold_pass["all_conditions"] += int(all(checks.values()))
        values["top_candidate_selection_probability"].append(
            top["selection_probability"])
        values["abstain_probability"].append(row["abstain_probability"])
        values["top_candidate_margin"].append(margin)
        values["top_candidate_beneficial_probability"].append(
            top["beneficial_probability"])
        values["top_candidate_catastrophic_probability"].append(
            top["catastrophic_probability"])
        values["top_candidate_predicted_gain"].append(top["predicted_gain"])
        if beneficial_actions:
            best = max(beneficial_actions,
                       key=lambda action: action["actual_gain"])
            beneficial_event_details.append({
                "sequence": row["sequence"],
                "trigger_frame": row["trigger_frame"],
                "candidate": best["name"],
                "actual_gain": best["actual_gain"],
                "selection_probability": best["selection_probability"],
                "abstain_probability": row["abstain_probability"],
                "beneficial_probability": best["beneficial_probability"],
                "catastrophic_probability":
                    best["catastrophic_probability"],
                "predicted_gain": best["predicted_gain"],
            })
    if len(rows) != result["evaluated_events"]:
        raise ValueError("evaluation event count drifted")
    return {
        "result_sha256": sha256_file(result_path),
        "events_sha256": sha256_file(events_path),
        "accepted": result["accepted"],
        "decision": result["decision"],
        "training_updates": result["training_updates"],
        "evaluated_events": result["evaluated_events"],
        "evaluation_sequences": result["evaluation_sequences"],
        "loss_first": result["loss_first"],
        "loss_last": result["loss_last"],
        "gate": result["gate"],
        "outcome_capacity": outcome,
        "threshold_pass_counts": threshold_pass,
        "top_candidate_distributions": {
            key: distribution(value) for key, value in values.items()
        },
        "beneficial_event_details": beneficial_event_details,
    }


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=False)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True,
                      allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    m4 = summarize(args.m4_output.resolve())
    m5 = summarize(args.m5_output.resolve())
    result = {
        "schema": "sttrack-lachtt-m5-pilot-comparison/v1",
        "complete": True,
        "policy": POLICY,
        "m4": m4,
        "m5": m5,
        "deltas_m5_minus_m4": {
            "loss_last": m5["loss_last"] - m4["loss_last"],
            "selected_actions": (m5["gate"]["selected_actions"] -
                                 m4["gate"]["selected_actions"]),
            "beneficial_actions": (m5["gate"]["beneficial_actions"] -
                                   m4["gate"]["beneficial_actions"]),
            "catastrophic_actions":
                (m5["gate"]["catastrophic_actions"] -
                 m4["gate"]["catastrophic_actions"]),
        },
        "confirmed_failure_mechanisms": [
            {
                "name": "single_event_selection_weight_cancellation",
                "observation": "runner performs one optimizer step per event with batch size one",
                "implemented_reduction": "weighted_loss / event_weight",
                "algebra": "w * CE / w = CE",
                "implication": "the 3.1875 beneficial-event selection weight did not change the direct selection gradient",
            },
            {
                "name": "absolute_probability_gate_misalignment",
                "observation": "positive-weighted BCE raises both beneficial and catastrophic raw sigmoid probabilities",
                "implication": "all M5 top candidates exceeded the frozen catastrophic probability maximum while none reached the beneficial minimum",
            },
        ],
        "decision": "M5 failed; no threshold scan, OOF, replay or public benchmark",
        "next_experiment_ceiling": "a separately preregistered gradient-correct selection objective may be tested on DepthTrack Train only; M5 cannot authorize VOT",
        "qwen_used": False,
        "vot_run": False,
        "automatic_next_stage": False,
    }
    atomic_json(args.output.resolve() / "result.json", result)
    (args.output.resolve() / "result.json").chmod(0o444)
    args.output.resolve().chmod(0o555)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
