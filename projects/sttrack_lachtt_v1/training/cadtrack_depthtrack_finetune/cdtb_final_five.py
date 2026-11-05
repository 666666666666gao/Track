"""Evaluate CDTB for all retained epochs, reusing completed DepthTrack results."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess


METRIC_SHA256 = "05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_result(folder, dataset, checkpoint_sha, cases_path):
    folder = Path(folder)
    cases = read(cases_path)
    expected = {"depthtrack_test": (50, 76373), "cdtb": (80, 101956)}[dataset]
    receipt = read(folder / "receipt.json")
    result = read(folder / "metrics.json")
    assert receipt["status"] == "complete_CADTrack_RGBD_OPE_predictions"
    assert result["status"] == "complete_CADTrack_RGBD_OPE_metrics"
    for item in (receipt, result):
        assert item["dataset"] == dataset
        assert item["checkpoint_sha256"] == checkpoint_sha
        assert item["cases_sha256"] == sha(cases_path)
    assert result["receipt_sha256"] == sha(folder / "receipt.json")
    assert result["metric_source_sha256"] == METRIC_SHA256
    assert result["runner_sha256"] == receipt["runner_sha256"]
    assert receipt["groundtruth_opened_during_tracking"] is False
    assert receipt["optimizer_steps"] == 0
    assert (len(cases), sum(row["frames"] for row in cases)) == expected
    assert len(receipt["sequences"]) == expected[0]
    assert receipt["frames"] == expected[1]
    for case, row in zip(cases, receipt["sequences"]):
        assert (row["sequence"], row["frames"]) == (case["sequence"], case["frames"])
        assert row["groundtruth_sha256"] == case["gt_sha256"]
        for suffix, key, columns in (
            (".txt", "boxes_sha256", 4),
            ("_all_scores.txt", "scores_sha256", 1),
        ):
            path = folder / (case["sequence"] + suffix)
            assert sha(path) == row[key]
            lines = path.read_text().splitlines()
            assert len(lines) == case["frames"], str(path)
            assert all(len(line.split(",")) == columns for line in lines), str(path)
    metrics = result["metrics"]
    assert (metrics["sequences"], metrics["frames"]) == expected
    assert all(
        math.isfinite(metrics[key]) and 0 <= metrics[key] <= 100
        for key in ("precision_percent", "recall_percent", "f_score_percent")
    )
    return dict(
        path=str(folder),
        receipt_sha256=sha(folder / "receipt.json"),
        metrics_sha256=sha(folder / "metrics.json"),
        metrics=metrics,
    )


def command(python, evaluator, mode, source, checkpoint, dataset_root, cases, output, metric):
    return [
        str(python), "-B", "-u", str(evaluator), mode,
        "--dataset", "cdtb", "--source", str(source),
        "--checkpoint", str(checkpoint), "--dataset-root", str(dataset_root),
        "--cases", str(cases), "--output", str(output),
        "--metric-source", str(metric),
    ]


def stage(folder, name, cmd, env, gpu):
    launch_path = folder / (name + ".launch.json")
    assert not launch_path.exists()
    launch = dict(
        command=cmd, gpu=gpu, supervisor_pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    with (folder / (name + ".log")).open("xb") as log:
        child = subprocess.Popen(
            cmd, cwd=str(folder), env=env, stdout=log, stderr=subprocess.STDOUT
        )
        launch["child_pid"] = child.pid
        write(launch_path, launch)
        code = child.wait()
    launch.update(exit_code=code, finished_at=datetime.now(timezone.utc).isoformat())
    write(folder / (name + ".receipt.json"), launch)
    (folder / (name + ".exit")).write_text(str(code) + "\n")
    assert code == 0, "Original stage failed; no automatic retry."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "training-root", "training-control", "probe-root", "base-cdtb-plan",
        "output", "tracker-python", "metric-python",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    training = args.training_root.resolve()
    probe = args.probe_root.resolve()
    root = args.output.resolve()
    terminal_path = args.training_control / "terminal.json"
    assert read(terminal_path)["exit_code"] == 0
    assert read(training / "launch.json")["status"] == "complete"
    rows_path = training / "epoch_metrics.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [31, 32, 33, 34, 35]
    base_path = args.base_cdtb_plan.resolve()
    base = read(base_path)
    assert base["dataset"] == "cdtb"
    cases_path = probe / "cdtb_cases.json"
    cases = read(cases_path)
    assert cases == read(base["cases_path"])
    assert sha(base["cases_path"]) == base["cases_sha256"]
    metric = Path(base["metric_source"])
    assert sha(metric) == base["metric_source_sha256"] == METRIC_SHA256
    dataset_root = Path(base["dataset_root"])
    for case in cases:
        assert sha(dataset_root / case["sequence"] / "groundtruth.txt") == case["gt_sha256"]
    source = probe / "source"
    config = source / "experiments/cadtrack/cadtrack.yaml"
    import yaml
    profile = yaml.safe_load(config.read_text())["TEST"]
    assert [profile[key] for key in (
        "TEMPLATE_FACTOR", "TEMPLATE_SIZE", "SEARCH_FACTOR", "SEARCH_SIZE",
        "UPDATE_INTERVALS", "UPDATE_THRESHOLD",
    )] == [2.0, 128, 4.0, 256, 5, 0.65]
    source_hashes = {
        str(path): sha(path)
        for path in sorted(source.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    retained = []
    for row in rows:
        epoch = row["epoch"]
        checkpoint = training / "final_five" / ("epoch_%02d.pth.tar" % epoch)
        digest = sha(checkpoint)
        assert digest == row["checkpoint_sha256"]
        dt = verify_result(
            training / "evaluations" / ("epoch_%02d" % epoch),
            "depthtrack_test", digest, probe / "depthtrack_test_cases.json",
        )
        assert dt["metrics"] == row["metrics"]
        retained.append(dict(
            epoch=epoch, checkpoint=str(checkpoint), checkpoint_sha256=digest,
            depthtrack_test=dt,
        ))
    root.mkdir(parents=True, exist_ok=False)
    evaluator = root / "evaluate_ope.py"
    shutil.copy2(probe / "evaluate_ope.py", evaluator)
    manifest = dict(
        schema="CADTrack_final_five_CDTB_comparison_v1",
        user_request="Compare all five retained epochs on DepthTrack and CDTB",
        final_checkpoint_selected=False,
        selection_bias_disclosed=True,
        comparison_uses_test_sets=True,
        depthtrack_reused=True,
        vot_requested=False,
        source_sha256=source_hashes,
        runner_sha256=sha(__file__),
        evaluator_sha256=sha(evaluator),
        metric_sha256=sha(metric), cases_sha256=sha(cases_path),
        config_sha256=sha(config), inference_profile=profile,
        training_terminal_sha256=sha(terminal_path),
        epoch_metrics_sha256=sha(rows_path),
        retained=retained,
    )
    write(root / "manifest.json", manifest)
    environment = dict(
        os.environ, PYTHONPATH=os.pathsep.join([str(probe), str(source)]),
        PYTHONDONTWRITEBYTECODE="1", PYTHONOPTIMIZE="0",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
    )

    def worker(gpu):
        completed = []
        for item in retained[gpu::2]:
            folder = root / ("epoch_%02d" % item["epoch"])
            folder.mkdir(exist_ok=False)
            output = folder / "predictions"
            track_cmd = command(
                args.tracker_python, evaluator, "track", source, item["checkpoint"],
                dataset_root, cases_path, output, metric,
            )
            stage(folder, "track", track_cmd, dict(environment, CUDA_VISIBLE_DEVICES=str(gpu)), gpu)
            score_cmd = command(
                args.metric_python, evaluator, "score", source, item["checkpoint"],
                dataset_root, cases_path, output, metric,
            )
            stage(folder, "score", score_cmd, dict(environment, CUDA_VISIBLE_DEVICES=""), None)
            cdtb = verify_result(output, "cdtb", item["checkpoint_sha256"], cases_path)
            assert read(output / "receipt.json")["runner_sha256"] == manifest["evaluator_sha256"]
            result = dict(item, cdtb=cdtb)
            write(folder / "result.json", result)
            print(json.dumps(result), flush=True)
            completed.append(result)
        return completed

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, gpu) for gpu in (0, 1)]
        results = [row for future in futures for row in future.result()]
    for path, digest in source_hashes.items():
        assert sha(path) == digest, path
    assert sha(evaluator) == manifest["evaluator_sha256"]
    for item in retained:
        assert sha(item["checkpoint"]) == item["checkpoint_sha256"]
    write(root / "all_results.json", dict(
        status="complete_CADTrack_final_five_DT50_CDTB80",
        manifest_sha256=sha(root / "manifest.json"),
        results=sorted(results, key=lambda row: row["epoch"]),
        final_checkpoint_selected=False, comparison_uses_test_sets=True,
        independent_completed_audit=False,
        finished_at=datetime.now(timezone.utc).isoformat(),
    ))


if __name__ == "__main__":
    main()
