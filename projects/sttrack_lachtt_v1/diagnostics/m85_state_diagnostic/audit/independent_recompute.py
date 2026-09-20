"""Independent CPU-only audit of frozen M85 arrays and exported dataset labels.

This script does not import the author's analyzer, tracker, Torch, or checkpoints.
Its only writes are audit JSON reports in this script's directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


A = Path(__file__).resolve().parent
R = A.parent / "completed"
P = A.parent.parent / "sttrack_m84_centered_20260920" / "completed"
INPUTS = {}
BINDINGS = []
ARMS = ("category", "swapped", "native")


def sha(path):
    path = Path(path)
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    INPUTS[str(path)] = value
    return value


def read(path):
    sha(path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bind(path, expected, label):
    actual = sha(path)
    assert actual == expected, (label, str(path), actual, expected)
    BINDINGS.append({"label": label, "path": str(path), "sha256": actual})


def overlap(boxes, target):
    # Continuous xywh areas in image coordinates, without prediction normalization.
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    x_right = np.minimum(b[:, 0] + b[:, 2], target[0] + target[2])
    y_bottom = np.minimum(b[:, 1] + b[:, 3], target[1] + target[3])
    x_left = np.maximum(b[:, 0], target[0])
    y_top = np.maximum(b[:, 1], target[1])
    area = np.maximum(x_right - x_left, 0) * np.maximum(y_bottom - y_top, 0)
    return area / (b[:, 2] * b[:, 3] + target[2] * target[3] - area)


def boxes_from_maps(maps, row):
    # Independently transcribed from CenterPredictor.cal_bbox and tracker mapping.
    yy, xx = np.indices((16, 16), dtype=np.float32)
    centers_x = ((xx + maps[3]) / np.float32(16)).reshape(-1)
    centers_y = ((yy + maps[4]) / np.float32(16)).reshape(-1)
    cxcywh = np.column_stack((centers_x, centers_y, maps[1].ravel(), maps[2].ravel()))
    # Match the online float32 scale operations before Python image coordinates.
    cxcywh *= np.float32(256)
    cxcywh /= np.float32(256.0 / row["search_side"])
    cxcywh = cxcywh.astype(np.float64)
    x, y, w, h = row["previous_bbox"]
    cxcywh[:, 0] += x + w / 2 - row["search_side"] / 2
    cxcywh[:, 1] += y + h / 2 - row["search_side"] / 2
    left = cxcywh[:, 0] - cxcywh[:, 2] / 2
    top = cxcywh[:, 1] - cxcywh[:, 3] / 2
    right = left + cxcywh[:, 2]
    bottom = top + cxcywh[:, 3]
    ih, iw = row["image_hw"]
    left = np.clip(left, 0, iw - 10)
    top = np.clip(top, 0, ih - 10)
    right = np.clip(right, 10, iw)
    bottom = np.clip(bottom, 10, ih)
    return np.column_stack((left, top, np.maximum(10, right - left), np.maximum(10, bottom - top)))


def low_intervals(records, arm):
    # Boolean transitions preserve invalid-frame breaks, including a final run.
    low = np.array([r["valid"] and r["variants"][arm]["hann_iou"] <= .1 for r in records], dtype=np.int8)
    transitions = np.diff(np.r_[np.int8(0), low, np.int8(0)])
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return [[int(start + 1), int(end + 1)] for start, end in zip(starts, ends) if end - start >= 10]


def summary(records):
    good = [r for r in records if r["valid"]]
    bad = [r for r in good if r["variants"]["category"]["hann_iou"] <= .1]
    result = {
        "valid_frames": len(good),
        "category_low_frames": len(bad),
        "low_center_outside": sum(not r["center_in_crop"] for r in bad),
        "low_center_inside_no_correct_dense": sum(r["center_in_crop"] and r["variants"]["category"]["dense_correct_count"] == 0 for r in bad),
        "low_center_inside_correct_dense": sum(r["center_in_crop"] and r["variants"]["category"]["dense_correct_count"] > 0 for r in bad),
        "raw_rescue": sum(r["variants"]["category"]["raw_iou"] >= .5 for r in bad),
        "raw_harm": sum(r["variants"]["category"]["hann_iou"] >= .5 and r["variants"]["category"]["raw_iou"] <= .1 for r in good),
    }
    for arm in ("swapped", "native"):
        result[arm + "_rescue"] = sum(r["variants"][arm]["hann_iou"] >= .5 for r in bad)
        result[arm + "_harm"] = sum(r["variants"]["category"]["hann_iou"] >= .5 and r["variants"][arm]["hann_iou"] <= .1 for r in good)
        result[arm + "_correct_dense_on_category_low"] = sum(r["variants"][arm]["dense_correct_count"] > 0 for r in bad)
    result["low_all_heads_no_correct_dense"] = sum(all(r["variants"][arm]["dense_correct_count"] == 0 for arm in ARMS) for r in bad)
    assert sum(result[k] for k in ("low_center_outside", "low_center_inside_no_correct_dense", "low_center_inside_correct_dense")) == len(bad)
    return result


def run():
    plan = read(R / "spec.json")
    train = read(P / "training_spec.json")
    result = read(R / "diagnostic_result.json")
    parent_result = read(P / "recursive_result.json")
    parent_spec = read(P / "recursive_spec.json")
    parent_receipt = read(P / "category_recursive_receipt.json")
    parent_frozen = read(P / "frozen.json")
    review = read(R / "review_receipt.json")
    launch = read(R / "launch.json")
    integration = read(P / "integration.json")
    assert plan["seed"] == train["seed"] == 2027
    assert plan["tracking_calls"] == 2616
    for key, name in (("source_sha256", "same_state.py"), ("analysis_sha256", "analyze_saved.py"), ("queue_sha256", "run_replay.sh"), ("plan_sha256", "EXPERIMENT_PLAN.md")):
        bind(R / name, plan[key], "M85 spec " + key)
    for key, name in (("training_spec_sha256", "training_spec.json"), ("parent_result_sha256", "recursive_result.json"), ("head_sha256", "training/category/final.pth")):
        bind(P / name, plan[key], "M85 parent " + key)
    bind(P / "integration.json", train["integration_sha256"], "training source integration")
    bind(P / "training_spec.json", parent_frozen["training_spec_sha256"], "parent frozen training")
    bind(P / "recursive_spec.json", parent_frozen["recursive_spec_sha256"], "parent frozen recursion")
    for name, digest in integration["source_sha256"].items():
        bind(P / "code" / name, digest, "frozen integrated source")
    bind(R / "spec.json", review["spec_sha256"], "code review spec")
    bind(R / "CODE_REVIEW.md", review["report_sha256"], "code review report")
    bind(R / "code_review.json", review["review_json_sha256"], "code review JSON")
    bind(R / "review_receipt.json", launch["review_receipt_sha256"], "launch reviewed receipt")
    bind(R / "spec.json", launch["spec_sha256"], "launch spec")
    assert review["blocking_findings"] == 0
    assert parent_result["head_sha256"] == parent_receipt["head_sha256"] == plan["head_sha256"]
    assert parent_result["all_gates_pass"] is False
    manifest = read(P / "evidence_manifest.json")
    for name, item in manifest.items():
        bind(P / name, item["sha256"], "M84 exported manifest")
        assert (P / name).stat().st_size == item["bytes"]
    m83 = A.parent.parent / "m82_complete_20260920" / "m83_completed" / "m83_same_state.py"
    m83_reference = {"path": str(m83), "available": m83.is_file(), "used_as_executed_code": False}
    if m83.is_file():
        bind(m83, plan["parent_replay_source_sha256"], "nonexecuted M83 reference source")
    for stage in ("preflight", "replay", "controller", "analysis"):
        path = R / (stage + ".exit")
        sha(path)
        assert path.read_text().strip() == "0"
    for name in ("same_state.py", "analyze_saved.py", "launch.py"):
        path = R / name
        sha(path)
        compile(path.read_text(), str(path), "exec")
    receipts = {}
    for mode in ("preflight", "predictions"):
        rec = read(R / mode / "receipt.json")
        receipts[mode] = rec
        bind(R / "spec.json", rec["spec_sha256"], mode + " spec")
        bind(R / "same_state.py", rec["source_sha256"], mode + " runner")
        assert rec["head_sha256"] == plan["head_sha256"]
        assert rec["status"] == "complete"
        for item in rec["sequences"]:
            seq = item["sequence"]
            bind(R / mode / (seq + ".json"), item["sha256"], mode + " selected outputs")
            bind(R / mode / (seq + ".npz"), item["dense_sha256"], mode + " dense outputs")
        logpath = R / ("preflight.log" if mode == "preflight" else "replay.log")
        sha(logpath)
        loglines = [json.loads(line) for line in logpath.read_text().splitlines() if line.startswith("{")]
        assert loglines[-1] == rec
        assert loglines[:-1] == rec["sequences"]
    assert receipts["preflight"]["positions"] == 101 and receipts["preflight"]["mode"] == "preflight"
    assert receipts["predictions"]["positions"] == 2616 and receipts["predictions"]["mode"] == "selected3"
    bind(R / "spec.json", result["spec_sha256"], "analysis spec")
    bind(R / "analyze_saved.py", result["source_sha256"], "analysis source")
    bind(R / "predictions/receipt.json", result["receipt_sha256"], "analysis sealed predictions")
    assert result["status"] == "complete_selected_diagnostic"
    assert [x["sequence"] for x in result["sequences"]] == [x["sequence"] for x in plan["cases"]]
    assert [x["sequence"] for x in receipts["predictions"]["sequences"]] == [x["sequence"] for x in plan["cases"]]
    p_cases = {c["sequence"]: c for c in parent_spec["cases"]}
    p_receipts = {c["sequence"]: c for c in parent_receipt["sequences"]}
    sequence_results = []
    all_metrics = {arm: {kind: [] for kind in ("raw", "hann")} for arm in ARMS}
    for case, stored in zip(plan["cases"], result["sequences"]):
        seq = case["sequence"]
        bind(P / "recursive/category" / (seq + ".json"), case["sealed_category_sha256"], "M85 exact parent category")
        assert p_receipts[seq]["sha256"] == case["sealed_category_sha256"]
        assert p_cases[seq]["frames"] == case["frames"]
        assert p_cases[seq]["gt_sha256"] == case["gt_sha256"]
        assert p_cases[seq]["init_bbox"] == case["init_bbox"]
        # All prediction files have already been hashed above, before label parsing.
        gtpath = P / "dataset_gt" / (seq + ".txt")
        bind(gtpath, case["gt_sha256"], "dataset GT export")
        gt = np.loadtxt(gtpath, delimiter=",", dtype=np.float64)
        assert gt.shape == (case["frames"], 4)
        assert gt[0].tolist() == case["init_bbox"]
        rows = read(R / "predictions" / (seq + ".json"))["rows"]
        parent_rows = read(P / "recursive/category" / (seq + ".json"))["rows"]
        assert [row["frame"] for row in rows] == list(range(1, len(gt)))
        assert len(stored["rows"]) == len(rows)
        with np.load(R / "predictions" / (seq + ".npz"), allow_pickle=False) as z:
            assert set(z.files) == {"window", *ARMS}
            maps = {arm: z[arm] for arm in ARMS}
            window = z["window"]
        assert window.shape == (1, 1, 16, 16) and window.dtype == np.float32
        assert all(m.shape == (len(rows), 5, 16, 16) and m.dtype == np.float32 and np.isfinite(m).all() for m in maps.values())
        hann1d = .5 * (1 - np.cos(2 * np.pi / 17 * np.arange(1, 17)))
        window_formula_error = float(np.abs(window[0, 0] - np.outer(hann1d, hann1d)).max())
        assert window_formula_error < 5e-7
        records = []
        decode_max = 0.
        metric_max = 0.
        mass_max = 0.
        kl_max = 0.
        template_frames = []
        low_capacity_rows = []
        for j, (row, claimed) in enumerate(zip(rows, stored["rows"])):
            i = row["frame"]
            assert row["previous_bbox"] == parent_rows[i - 1]["bbox"]
            category = row["variants"]["category"]
            assert category["hann_bbox"] == parent_rows[i]["bbox"]
            assert category["hann_max"] == parent_rows[i]["score"]
            assert row["variants"]["empty"] == row["variants"]["native"]
            previous = row["previous_bbox"]
            side = math.ceil(math.sqrt(previous[2] * previous[3]) * 4)
            origin = [round(previous[k] + previous[k + 2] / 2 - side / 2) for k in (0, 1)]
            assert row["search_side"] == side and row["crop_origin"] == origin
            eligible = i % 50 == 0 and parent_rows[i]["score"] > .75
            assert row["actual_template_write"] == eligible
            if eligible:
                template_frames.append(i)
            g = gt[i]
            valid = bool(np.isfinite(g).all() and g[2] > 0 and g[3] > 0)
            end = np.asarray(origin) + side
            center = g[:2] + g[2:] / 2
            record = {"frame": i, "valid": valid,
                      "center_in_crop": bool(valid and np.all(center >= origin) and np.all(center < end)),
                      "full_box_in_crop": bool(valid and np.all(g[:2] >= origin) and np.all(g[:2] + g[2:] <= end)),
                      "variants": {}}
            p_native = maps["native"][j, 0].astype(np.float64).ravel()
            p_native /= p_native.sum()
            for arm in ARMS:
                m = maps[arm][j]
                boxes = boxes_from_maps(m, row)
                v = row["variants"][arm]
                raw = m[0].ravel()
                hann = raw * window.ravel()
                for kind, scores in (("raw", raw), ("hann", hann)):
                    idx = int(np.argmax(scores))
                    assert idx == v[kind + "_peak"]
                    assert float(scores[idx]) == v[kind + "_max"]
                    diff = float(np.max(np.abs(boxes[idx] - v[kind + "_bbox"])))
                    decode_max = max(decode_max, diff)
                    assert diff <= 1e-4
                mass_max = max(mass_max, abs(float(raw.astype(np.float64).sum()) - v["raw_mass"]))
                prob = raw.astype(np.float64)
                prob /= prob.sum()
                divergence = float(np.sum(p_native * np.log(p_native / prob)))
                kl_max = max(kl_max, abs(divergence - v["native_spatial_kl"]))
                if valid:
                    overlaps = overlap(boxes, g)
                    vals = {"raw_iou": float(overlap(v["raw_bbox"], g)[0]),
                            "hann_iou": float(overlap(v["hann_bbox"], g)[0]),
                            "dense_best_iou": float(np.max(overlaps)),
                            "dense_correct_count": int(np.count_nonzero(overlaps >= .5)),
                            "dense_best_index": int(np.argmax(overlaps))}
                    record["variants"][arm] = vals
                    for kind in ("raw", "hann"):
                        all_metrics[arm][kind].append(vals[kind + "_iou"])
            assert {k:v for k,v in record.items() if k != "variants"} == {k:v for k,v in claimed.items() if k != "variants"}
            assert set(record["variants"]) == set(claimed["variants"])
            for arm, values in record["variants"].items():
                for key, value in values.items():
                    if key in ("dense_correct_count", "dense_best_index"):
                        assert value == claimed["variants"][arm][key]
                    else:
                        difference = abs(value - claimed["variants"][arm][key])
                        metric_max = max(metric_max, difference)
                        assert difference < 1e-12
            if valid and record["variants"]["category"]["hann_iou"] <= .1 and any(record["variants"][arm]["dense_correct_count"] for arm in ARMS):
                low_capacity_rows.append(record)
            records.append(record)
        calculated_summary = summary(records)
        assert calculated_summary == stored["summary"]
        h10 = {arm: low_intervals(records, arm) for arm in ARMS}
        assert h10 == stored["H10"]
        assert decode_max == stored["max_selected_decoder_error"]
        selected_metrics = {}
        for arm in ARMS:
            selected_metrics[arm] = {}
            for kind in ("raw", "hann"):
                values = np.array([r["variants"][arm][kind + "_iou"] for r in records if r["valid"]])
                selected_metrics[arm][kind] = {"valid_frames": len(values), "iou_sum": float(values.sum()),
                    "mean_iou": float(values.mean()), "low_iou_frames": int(np.sum(values <= .1)), "correct_frames": int(np.sum(values >= .5))}
        # Category single-step values must equal the same frames in the M84 trajectory.
        for key in ("valid_frames", "low_iou_frames"):
            assert selected_metrics["category"]["hann"][key] == parent_result["per_sequence"]["category"][seq][key]
        assert abs(selected_metrics["category"]["hann"]["iou_sum"] - parent_result["per_sequence"]["category"][seq]["iou_sum"]) < 1e-10
        assert len(h10["category"]) == parent_result["per_sequence"]["category"][seq]["failure_episodes"]
        preflight_equal = None
        if seq == plan["cases"][0]["sequence"]:
            pf = read(R / "preflight" / (seq + ".json"))
            assert pf["rows"] == rows[:101]
            with np.load(R / "preflight" / (seq + ".npz"), allow_pickle=False) as z:
                assert np.array_equal(z["window"], window)
                for arm in ARMS:
                    assert np.array_equal(z[arm], maps[arm][:101])
            preflight_equal = True
        sequence_results.append({"sequence": seq, "positions": len(rows), "summary": calculated_summary,
            "H10": h10, "single_step_metrics": selected_metrics,
            "max_selected_decoder_error": decode_max, "max_reported_metric_abs_error": metric_max,
            "max_float64_mass_vs_gpu_float32_abs_error": mass_max,
            "max_float64_KL_vs_gpu_float32_abs_error": kl_max,
            "window_analytic_max_abs_error": window_formula_error,
            "exact_parent_category_bbox_score_frames": len(rows),
            "exact_previous_state_frames": len(rows), "exact_empty_native_saved_scalar_box_records": len(rows),
            "template_write_eligibility_frames": template_frames,
            "low_frames_with_any_correct_dense_candidate": low_capacity_rows,
            "preflight_prefix_array_and_rows_exact": preflight_equal,
            "invalid_gt_frames": len(rows) - calculated_summary["valid_frames"]})
    analysis_log = read(R / "analysis.log")
    assert analysis_log == [{"sequence": s["sequence"], **s["summary"], "max_selected_decoder_error": s["max_selected_decoder_error"]} for s in sequence_results]
    aggregate = {key: sum(s["summary"][key] for s in sequence_results) for key in sequence_results[0]["summary"]}
    total_metrics = {arm: {kind: {"valid_frames": len(v), "iou_sum": float(np.sum(v)), "mean_iou": float(np.mean(v)),
                                   "low_iou_frames": int(np.sum(np.asarray(v) <= .1)), "correct_frames": int(np.sum(np.asarray(v) >= .5))}
                          for kind, v in kinds.items()} for arm, kinds in all_metrics.items()}
    # A final byte pass detects changes to every input observed during this audit.
    for name, before in list(INPUTS.items()):
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == before, name
    output = {"status": "PASS_DETERMINISTIC_SAVED_ARTIFACT_CHECKS", "generated_at": datetime.now(timezone.utc).isoformat(),
        "review_independence": "deterministic", "acceptance_status": "accepted",
        "runtime": {"python": sys.executable, "python_version": platform.python_version(), "numpy_version": np.__version__},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "CPU recomputation of saved artifacts only; no network, GPU, checkpoint execution, or training reproduction.",
        "M84_export_manifest_files_verified": len(manifest), "M84_integration_source_files_verified": len(integration["source_sha256"]),
        "binding_checks": BINDINGS, "M83_reference": m83_reference,
        "positions": sum(s["positions"] for s in sequence_results), "aggregate_summary": aggregate,
        "aggregate_single_step_metrics": total_metrics, "sequences": sequence_results,
        "audited_input_hashes": INPUTS,
        "unverified_runtime_artifacts": {"native_checkpoint": train["native_checkpoint"],
            "development_banks": {k:v for k,v in train["banks"]["development"].items() if isinstance(v,dict)},
            "RGB_depth_images": train["dataset_root"], "full_empty_dense_maps": "Not saved; replay asserts exact equality with native.",
            "query_template_snapshots": "Not saved; source checks query values, bbox, and template object identity at runtime."},
        "all_observed_input_bytes_unchanged": True}
    (A / "independent_recompute.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k:output[k] for k in ("status", "runtime", "positions", "M84_export_manifest_files_verified", "M84_integration_source_files_verified", "aggregate_summary", "aggregate_single_step_metrics")}, indent=2))
    print(json.dumps({"sequences": [{k:v for k,v in s.items() if k not in ("single_step_metrics", "low_frames_with_any_correct_dense_candidate")} for s in sequence_results]}, indent=2))


if __name__ == "__main__":
    run()
