"""Validate completed full VOT coverage and compare confirmed failures."""

import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent / "native_vot_full127"
EXPECTED_BUNDLES = {
    "M67": "b613edb60116bbfe0de6c5becb2852027925957230d11161e23778641009c874",
    "M82": "8d5c11a95b80900172d96a22f8d99696ae84a473ca29a44fe866170a7f6cb556",
}
CHECKPOINTS = {
    "M67": "7a8907a45cec672f11563ad7cfaee21cb64acfee3fa13e854621c146cb8d058e",
    "M82": "581a044bbba8514fb5f26d283a9dd038f46e27f8608c08724228eebfff47f859",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((HERE / name).read_text())


def main():
    all_results = read("all_results.json")
    assert all_results["status"] == "six_full_evaluations_complete"
    assert all_results["new_training_steps"] == 0
    result_rows = {(row["model"], row["dataset"]): row for row in all_results["results"]}
    assert set(result_rows) == {(model, dataset) for model in CHECKPOINTS
                                for dataset in ("depthtrack", "cdtb", "vot")}
    outcomes = {}
    per_sequence = {}
    metrics = {}
    for model in CHECKPOINTS:
        data = read(f"{model}_vot_result.json")
        row = result_rows[(model, "vot")]
        assert sha(HERE / f"{model}_vot_result.json") == row["result_sha256"]
        assert data["checkpoint_sha256"] == row["checkpoint_sha256"] == CHECKPOINTS[model]
        assert data["bundle_sha256"] == EXPECTED_BUNDLES[model]
        assert data["status"] == "complete_full127"
        assert data["metrics_percent"] == row["metrics"]
        assert len(data["per_sequence_failures"]) == 127
        assert len(data["failure_outcomes"]) == 1765
        assert sum(value["anchors"] for value in data["per_sequence_failures"].values()) == 1765
        assert sum(value["confirmed_failures"] for value in data["per_sequence_failures"].values()) == data["confirmed_failures"]
        assert sum(bool(value["failed"]) for value in data["failure_outcomes"].values()) == data["confirmed_failures"]
        outcomes[model] = data["failure_outcomes"]
        per_sequence[model] = data["per_sequence_failures"]
        metrics[model] = data["metrics_percent"]

    native_result = json.loads((NATIVE / "result.json").read_text())
    assert native_result["status"] == "complete"
    assert native_result["anchors"] == 1765
    assert native_result["confirmed_failures"] == 183
    with (NATIVE / "per_sequence_failures.csv").open(newline="") as stream:
        native = {row["sequence"]: {key: int(row[key]) for key in ("anchors", "confirmed_failures")}
                  for row in csv.DictReader(stream)}
    assert len(native) == 127 and sum(x["anchors"] for x in native.values()) == 1765
    assert sum(x["confirmed_failures"] for x in native.values()) == 183
    assert set(native) == set(per_sequence["M67"]) == set(per_sequence["M82"])
    assert set(outcomes["M67"]) == set(outcomes["M82"])

    rows = []
    for sequence in sorted(native):
        n = native[sequence]["confirmed_failures"]
        a = per_sequence["M67"][sequence]["confirmed_failures"]
        b = per_sequence["M82"][sequence]["confirmed_failures"]
        assert native[sequence]["anchors"] == per_sequence["M67"][sequence]["anchors"] == per_sequence["M82"][sequence]["anchors"]
        rows.append(dict(sequence=sequence, anchors=native[sequence]["anchors"],
                         native=n, M67=a, M82=b, M82_minus_native=b - n,
                         M82_minus_M67=b - a))
    with (HERE / "M82_vot_per_sequence_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    relation = {
        "M82_vs_native": {
            "worsened_sequences": sum(row["M82_minus_native"] > 0 for row in rows),
            "improved_sequences": sum(row["M82_minus_native"] < 0 for row in rows),
            "same_sequences": sum(row["M82_minus_native"] == 0 for row in rows),
            "added_failure_counts": sum(max(row["M82_minus_native"], 0) for row in rows),
            "reduced_failure_counts": -sum(min(row["M82_minus_native"], 0) for row in rows),
        },
        "M82_vs_M67": {
            "worsened_sequences": sum(row["M82_minus_M67"] > 0 for row in rows),
            "improved_sequences": sum(row["M82_minus_M67"] < 0 for row in rows),
            "same_sequences": sum(row["M82_minus_M67"] == 0 for row in rows),
        },
    }
    anchor_relation = {
        "M82_new_failures_vs_M67": sum(
            not outcomes["M67"][key]["failed"] and outcomes["M82"][key]["failed"]
            for key in outcomes["M67"]
        ),
        "M82_rescued_vs_M67": sum(
            outcomes["M67"][key]["failed"] and not outcomes["M82"][key]["failed"]
            for key in outcomes["M67"]
        ),
    }
    report = dict(
        status="complete_full127_audited",
        source_sha256={name: sha(HERE / name) for name in
                       ("all_results.json", "M67_vot_result.json", "M82_vot_result.json")},
        native_per_sequence_sha256=sha(NATIVE / "per_sequence_failures.csv"),
        coverage=dict(sequences=127, anchors=1765),
        confirmed_failures=dict(native=183, M67=215, M82=212),
        metrics_percent=metrics,
        relation=relation,
        anchor_relation=anchor_relation,
        top_M82_worsened_vs_native=sorted(rows, key=lambda x: (-x["M82_minus_native"], x["sequence"]))[:12],
        top_M82_improved_vs_native=sorted(rows, key=lambda x: (x["M82_minus_native"], x["sequence"]))[:12],
        comparison_csv_sha256=sha(HERE / "M82_vot_per_sequence_comparison.csv"),
    )
    (HERE / "M82_vot_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
