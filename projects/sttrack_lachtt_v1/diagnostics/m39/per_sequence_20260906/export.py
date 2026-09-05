"""Export M39 singleton-sequence metrics with the unchanged VOT 0.7.1 analyses."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path

from cachetools import LRUCache
import vot
from vot.analysis import AnalysisProcessor
import vot.analysis.multistart as multistart
from vot.workspace import Workspace


RESULT_SHA256 = "cf953c0d3c69609bcd83c11cb24ba57f37e30b38d3b3bcad32860b3a9ba9c1b5"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.root / "m39_result.json"
    assert sha256(result_path) == RESULT_SHA256
    original = json.loads(result_path.read_text())
    assert original["status"] == "complete"
    assert vot.__version__ == "0.7.1"
    assert not (args.output / "per_sequence.json").exists()
    args.output.mkdir(parents=True, exist_ok=True)

    bindings = {}
    # Verify both sealed result sets before loading any dataset for analysis.
    for arm in ("default", "no_update"):
        saved = original["arms"][arm]
        analysis_path = Path(saved["analysis"])
        merge_path = args.root / arm / "merge_result.json"
        assert sha256(analysis_path) == saved["analysis_sha256"]
        assert sha256(merge_path) == saved["merge_sha256"]
        merge = json.loads(merge_path.read_text())
        assert merge["status"] == "complete"
        assert merge["anchor_count"] == 303
        assert merge["result_file_count"] == len(merge["result_sha256"]) == 909
        workspace_path = args.root / arm / "master"
        for name, digest in merge["result_sha256"].items():
            assert sha256(workspace_path / name) == digest, name
        bindings[arm] = {
            "analysis_sha256": saved["analysis_sha256"],
            "merge_sha256": saved["merge_sha256"],
            "verified_result_files": 909,
        }
        (args.output / (arm + "_analysis.json")).write_bytes(analysis_path.read_bytes())
        (args.output / (arm + "_merge.json")).write_bytes(merge_path.read_bytes())

    arms = {}
    csv_rows = []
    for arm in ("default", "no_update"):
        saved = original["arms"][arm]
        prior = json.loads((args.output / (arm + "_analysis.json")).read_text())
        assert prior["toolkit"] == "0.7.1"
        workspace = Workspace.load(str(args.root / arm / "master"))
        experiment = workspace.stack["baseline"]
        assert [type(a).__name__ for a in experiment.analyses] == [
            "EAOScore", "EAOCurve", "AverageAccuracyRobustness"]
        assert [dict(a.dump()) for a in experiment.analyses] == [
            {k: v for k, v in a.items() if k != "type"}
            for a in prior["results"]["baseline"]["parameters"]["analyses"]]
        eao, curve, ar = experiment.analyses
        tracker, = workspace.registry.resolve(saved["tracker"])
        names = list(prior["sequences"])
        assert len(names) == 22 and set(names) == set(workspace.dataset.keys())
        sequences = [workspace.dataset[name] for name in names]
        with ThreadPoolExecutor(max_workers=1) as executor:
            processor = AnalysisProcessor(executor, LRUCache(maxsize=1024))
            eao_all = processor.run(eao, experiment, [tracker], sequences)[0, 0][0]
            ar_all = processor.run(ar, experiment, [tracker], sequences)[0, 0]
            aggregate = {"eao": float(eao_all), "acc": float(ar_all[0]),
                         "rob": float(ar_all[1])}
            for key, value in aggregate.items():
                assert abs(value - saved["metrics_fraction"][key]) <= 1e-12, (arm, key)
            sequence_ar = processor.run(ar.analysis, experiment, [tracker], sequences)
            sequence_curves = processor.run(curve.curves, experiment, [tracker], sequences)
            rows = []
            dependencies = {}
            for index, sequence in enumerate(sequences):
                one_eao = processor.run(eao, experiment, [tracker], [sequence])[0, 0][0]
                accuracy, robustness, _, progress, length = sequence_ar[0, index]
                (partial, active), weight = sequence_curves[0, index]
                outcomes = [x for x in saved["failure_outcomes"].values()
                            if x["sequence"] == sequence.name]
                assert len(outcomes) > 0
                assert int(progress) == sum(x["progress"] for x in outcomes)
                assert int(length) == prior["sequences"][sequence.name]["length"]
                row = {
                    "arm": arm, "sequence": sequence.name, "frames": int(length),
                    "anchors": len(outcomes),
                    "confirmed_failures": sum(x["failed"] for x in outcomes),
                    "eao_percent": 100.0 * float(one_eao),
                    "acc_percent": 100.0 * float(accuracy),
                    "rob_percent": 100.0 * float(robustness),
                    "accuracy_weight_progress_frames": int(progress),
                    "robustness_weight_sequence_frames": int(length),
                    "eao_active_bins_in_score_slice": sum(
                        float(x) > 0 for x in active[eao.low:eao.high + 1]),
                }
                rows.append(row)
                csv_rows.append(row)
                dependencies[sequence.name] = {
                    "partial_curve": [float(x) for x in partial],
                    "active_weight": [float(x) for x in active],
                    "sequence_weight": float(weight),
                }
        assert sum(row["anchors"] for row in rows) == 303
        assert sum(row["confirmed_failures"] for row in rows) == saved["confirmed_failures"]
        arms[arm] = {
            "tracker": saved["tracker"], "bindings": bindings[arm],
            "aggregate_fraction_recomputed": aggregate,
            "aggregate_matches_original": True,
            "aggregate_accuracy_weight": int(ar_all[3]),
            "aggregate_robustness_weight": int(ar_all[4]),
            "rows": rows,
        }
        (args.output / (arm + "_eao_dependencies.json")).write_text(
            json.dumps(dependencies, indent=2) + "\n")
        print(json.dumps({"arm": arm, "sequences": len(rows), "aggregate": aggregate}),
              flush=True)

    payload = {
        "schema": "m39_per_sequence_vot071_export_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_result_sha256": RESULT_SHA256,
        "exporter_sha256": sha256(__file__),
        "toolkit": vot.__version__,
        "multistart_source_sha256": sha256(inspect.getfile(multistart)),
        "eao_configured_interval": [115, 755],
        "scope": (
            "Same sealed M39 outputs. EAO is the unchanged toolkit EAOScore "
            "applied to each singleton sequence with the original fixed interval; "
            "unsupported curve bins are zero under the toolkit aggregator. "
            "ACC/ROB are the sequence dependencies of the original analysis. "
            "Aggregate metrics are recomputed from all sequences, not row means. "
            "No tracking, training, checkpoint change, or benchmark promotion."),
        "arms": arms,
    }
    (args.output / "per_sequence.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (args.output / "per_sequence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


if __name__ == "__main__":
    main()
