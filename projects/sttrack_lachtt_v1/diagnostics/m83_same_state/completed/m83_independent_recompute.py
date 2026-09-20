"""Independent, standard-library-only audit of sealed M83 local evidence.

Does not import the experiment's analysis/verifier, execute a model, contact a
remote host, or write to any experiment input. Sums use math.fsum. The saved
per-frame response KL/mass scalars can be aggregated, but not reconstructed
without dense response tensors; that boundary is recorded in the output.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import math
import tarfile

R = Path(__file__).resolve().parent
W = R.parent
ARMS = ("category", "empty", "swapped", "native")
ABS_TOL = 1e-8
hashes = {}
hash_checks = []
failures = []
check_counts = Counter()
float_comparisons = {"count": 0, "maximum_absolute_difference": 0.0, "location": None}


def content(path):
    data = path.read_bytes()
    hashes[path.relative_to(W).as_posix()] = hashlib.sha256(data).hexdigest()
    return data


def read(path):
    return json.loads(content(path))


def check(condition, family, detail):
    check_counts[family] += 1
    if not condition:
        failures.append({"family": family, "detail": detail})


def seal(path, expected, provenance):
    data = content(path)
    actual = hashlib.sha256(data).hexdigest()
    hash_checks.append({"path": path.relative_to(W).as_posix(), "provenance": provenance,
                        "expected": expected, "actual": actual, "match": actual == expected})
    check(actual == expected, "sha256", str(path) + " <- " + provenance)


def equal(actual, expected, location, family="recomputed_statistics"):
    if isinstance(expected, dict):
        check(set(actual) == set(expected), family, location + ": dictionary keys")
        for key in expected:
            if key in actual:
                equal(actual[key], expected[key], location + "/" + key, family)
    elif isinstance(expected, list):
        check(len(actual) == len(expected), family, location + ": list length")
        for index, (left, right) in enumerate(zip(actual, expected)):
            equal(left, right, location + "/" + str(index), family)
    elif isinstance(expected, float):
        difference = abs(actual - expected)
        float_comparisons["count"] += 1
        if difference > float_comparisons["maximum_absolute_difference"]:
            float_comparisons.update(maximum_absolute_difference=difference, location=location)
        check(math.isfinite(actual) and difference <= ABS_TOL, family,
              {"location": location, "actual": actual, "expected": expected})
    else:
        check(actual == expected, family, {"location": location, "actual": actual, "expected": expected})


def overlap(box, gt):
    # Independent rectangle intersection/union from dataset xywh, in pixels.
    left, top = max(box[0], gt[0]), max(box[1], gt[1])
    right, bottom = min(box[0] + box[2], gt[0] + gt[2]), min(box[1] + box[3], gt[1] + gt[3])
    intersection = max(right - left, 0.0) * max(bottom - top, 0.0)
    return intersection / (box[2] * box[3] + gt[2] * gt[3] - intersection)


def summarize(entries):
    out = {}
    for arm in ARMS:
        out[arm] = {
            "n": len(entries),
            "iou_sum": math.fsum(e["hann"][arm] for e in entries),
            "low": sum(e["hann"][arm] <= 0.1 for e in entries),
            "correct": sum(e["hann"][arm] >= 0.5 for e in entries),
            "rescue_vs_category": sum(e["hann"]["category"] <= 0.1 and e["hann"][arm] >= 0.5 for e in entries),
            "harm_vs_category": sum(e["hann"]["category"] >= 0.5 and e["hann"][arm] <= 0.1 for e in entries),
            "raw_rescue_hann": sum(e["hann"][arm] <= 0.1 and e["raw"][arm] >= 0.5 for e in entries),
            "raw_harm_hann": sum(e["hann"][arm] >= 0.5 and e["raw"][arm] <= 0.1 for e in entries),
            "raw_hann_peak_changes": sum(e["variants"][arm]["raw_peak"] != e["variants"][arm]["hann_peak"] for e in entries),
            "kl_native_sum": math.fsum(e["variants"][arm]["native_spatial_kl"] for e in entries),
            "response_mass_sum": math.fsum(e["variants"][arm]["raw_mass"] for e in entries),
        }
    return out


spec = read(R / "spec.json")
launch = read(R / "launch.json")
receipt = read(R / "predictions/receipt.json")
result = read(R / "diagnostic_result.json")
saved = read(R / "saved_diagnostic_verification.json")
descriptive = read(R / "descriptive_tables.json")
historical_description = read(W / "posthoc_descriptive.json")
initial_text_probe = read(W / "initial_text_probe_20260920.json")
centered_interface_receipt = read(W / "centered_real_interface_result.json")
manifest = read(R / "export_manifest.json")
export = read(R / "export_receipt.json")
training = read(W / "training_spec.json")
parent = read(W / "recursive_result.json")
inventory = read(W / "data_inventory.json")
integration = read(W / "integration.json")
inventory_by_sequence = {row["sequence"]: row for row in inventory["sequences_detail"]}

for name, digest in launch["sources"].items():
    seal(R / name, digest, "launch.json/sources/" + name)
for path, digest, provenance in [
    (W / "m83_preflight_receipt.json", launch["preflight_sha256"], "launch/preflight"),
    (W / "training_spec.json", spec["training_spec_sha256"], "spec/training_spec"),
    (W / "recursive_result.json", spec["parent_result_sha256"], "spec/parent_result"),
    (R / "m83_same_state.py", spec["source_sha256"], "spec/source"),
    (R / "spec.json", receipt["spec_sha256"], "receipt/spec"),
    (R / "m83_same_state.py", receipt["source_sha256"], "receipt/source"),
    (R / "analyze_m83.py", result["source_sha256"], "result/source"),
    (R / "spec.json", result["spec_sha256"], "result/spec"),
    (R / "predictions/receipt.json", result["receipt_sha256"], "result/receipt"),
    (R / "verify_m83_saved.py", saved["source_sha256"], "verification/source"),
    (R / "diagnostic_result.json", saved["result_sha256"], "verification/result"),
    (R / "diagnostic_result.json", descriptive["result_sha256"], "descriptive/result"),
    (R / "saved_diagnostic_verification.json", descriptive["verification_sha256"], "descriptive/verification"),
    (W / "summarize_m83.py", descriptive["source_sha256"], "descriptive/source"),
    (R / "export_manifest.json", export["manifest_sha256"], "export/manifest"),
    (W / "m83_completed_review_evidence.tar.gz", export["archive_sha256"], "export/archive"),
    (W / "data_inventory.json", training["inventory_sha256"], "training/inventory"),
    (W / "integration.json", training["integration_sha256"], "training/integration"),
    (W / "recursive_result.json", historical_description["result_sha256"], "historical_description/result"),
    (W / "summarize_completed.py", historical_description["source_sha256"], "historical_description/source"),
]:
    seal(path, digest, provenance)
check((W / "m83_completed_review_evidence.tar.gz").stat().st_size == export["archive_bytes"], "archive", "archive size")
check(receipt["head_sha256"] == spec["head_sha256"], "metadata", "head digest identifiers agree; binary not opened")
check(launch["seed"] == spec["seed"] == training["seed"] == 2027, "metadata", "seed identity")
equal(read(R / "predictions_receipt.json"), receipt, "duplicate predictions receipt", "metadata")
preflight = read(W / "m83_preflight_receipt.json")
for field in ("source_sha256", "spec_sha256", "head_sha256"):
    equal(preflight[field], receipt[field], "preflight/full receipt/" + field, "metadata")
for item in preflight["sequences"]:
    path = W / "m83_preflight" / (item["sequence"] + ".json")
    seal(path, item["sha256"], "preflight receipt/" + item["sequence"])
    earlier = read(path)["rows"]
    current = read(R / "predictions" / (item["sequence"] + ".json"))["rows"][:len(earlier)]
    check(earlier == current and len(earlier) == item["positions"] == 101, "preflight_full_parity", item["sequence"] + " all saved fields")
for item in manifest["files"]:
    seal(R / item["path"], item["sha256"], "export_manifest/" + item["path"])
    check((R / item["path"]).stat().st_size == item["bytes"], "manifest_size", item["path"])
with tarfile.open(W / "m83_completed_review_evidence.tar.gz", "r:gz") as archive:
    expected_members = {item["path"] for item in manifest["files"]} | {"export_manifest.json"}
    check(set(archive.getnames()) == expected_members, "archive", "member set")
    for member in archive.getmembers():
        raw = archive.extractfile(member).read()
        check(raw == content(R / member.name), "archive", member.name + " bytes match local file")
integration_missing = []
for name, digest in integration["source_sha256"].items():
    source = W / "code" / name
    if source.is_file():
        seal(source, digest, "integration/source_sha256/" + name)
    else:
        integration_missing.append(name)
for name in ("replay", "analysis", "controller", "verification"):
    check(content(R / (name + ".exit")).strip() == b"0", "terminal_exit", name)
equal(json.loads(content(R / "replay.log").decode().splitlines()[-1]), receipt, "replay log final receipt", "terminal_log")
analysis_log = json.loads(content(R / "analysis.log"))
equal(analysis_log["aggregates"], result["aggregates"], "analysis log aggregates", "terminal_log")
check(analysis_log["result_sha256"] == hashes["m83_completed/diagnostic_result.json"], "terminal_log", "analysis result digest")
equal(json.loads(content(R / "verification.log")), saved, "verification log", "terminal_log")

sequences = [case["sequence"] for case in spec["cases"]]
equal(sequences, training["development_sequences"], "fixed development sequence list", "coverage")
equal(sequences, [item["sequence"] for item in receipt["sequences"]], "receipt sequence list", "coverage")
check(len(sequences) == len(set(sequences)) == 22, "coverage", "22 unique sequences")
check({p.stem for p in (R / "predictions").glob("*.json")} == set(sequences) | {"receipt"}, "coverage", "raw JSON file set")
check(receipt["status"] == "complete" and receipt["mode"] == "full22", "terminal_status", "complete full22 receipt")
check(result["status"] == "complete_fixed_Category_state_diagnostic", "terminal_status", "complete diagnostic result")
check(saved["status"] == "complete_scalar_fixed_state_verification", "terminal_status", "complete saved verifier")

per_sequence = {}
whole_groups = defaultdict(list)
coverage = []
parity = Counter()
raw_events = []
alternative_events = []
qualification_events = []
centre_outside_intersects = 0
centre_outside_disjoint = 0
for case, item in zip(spec["cases"], receipt["sequences"]):
    sequence = case["sequence"]
    predpath = R / "predictions" / (sequence + ".json")
    oldpath = W / "recursive/category" / (sequence + ".json")
    gtpath = W / "dataset_gt" / sequence / "groundtruth.txt"
    seal(predpath, item["sha256"], "prediction receipt/" + sequence)
    seal(oldpath, case["sealed_category_sha256"], "spec/sealed_category/" + sequence)
    seal(gtpath, case["gt_sha256"], "spec/gt/" + sequence)
    inv = inventory_by_sequence[sequence]
    seal(gtpath, inv["groundtruth_sha256"], "training-bound inventory/" + sequence)
    gt = [[float(value) for value in line.split(",")] for line in content(gtpath).decode().splitlines() if line.strip()]
    preds, old = read(predpath), read(oldpath)
    check(preds["sequence"] == old["sequence"] == sequence and old["arm"] == "category", "coverage", sequence + " headers")
    check(len(gt) == len(old["rows"]) == case["frames"] == inv["groundtruth_rows"], "coverage", sequence + " GT/reference length")
    check(gt[0] == case["init_bbox"] == old["rows"][0]["bbox"] == inv["first_box"], "initialization", sequence)
    check(len(preds["rows"]) == item["positions"] == case["frames"] - 1, "coverage", sequence + " positions")
    check([row["frame"] for row in old["rows"]] == list(range(case["frames"])), "coverage", sequence + " original frame indices")
    invalid = Counter()
    groups = defaultdict(list)
    qualifications = {arm: 0 for arm in ARMS}
    disagreements = {arm: 0 for arm in ARMS if arm != "category"}
    opportunities = 0
    for frame, row in enumerate(preds["rows"], 1):
        location = sequence + ":frame=" + str(frame)
        check(row["frame"] == frame, "frame_alignment", location)
        check(row["previous_bbox"] == old["rows"][frame - 1]["bbox"], "previous_bbox_exact", location)
        variants = row["variants"]
        check(set(variants) == set(ARMS), "readout_set", location)
        check(variants["category"]["hann_bbox"] == old["rows"][frame]["bbox"], "category_bbox_exact", location)
        check(variants["category"]["hann_max"] == old["rows"][frame]["score"], "category_score_exact", location)
        previous = row["previous_bbox"]
        side = math.ceil(4 * math.sqrt(previous[2] * previous[3]))
        check(side == row["search_side"], "search_side_exact", location)
        parity.update(positions=1, previous_bbox_coordinates=4, category_bbox_coordinates=4, category_scores=1, search_sides=1)
        for arm, output in variants.items():
            for key in ("raw_bbox", "hann_bbox"):
                box = output[key]
                check(len(box) == 4 and all(math.isfinite(x) for x in box) and box[2] > 0 and box[3] > 0, "finite_positive_boxes", location + "/" + arm + "/" + key)
            check(all(isinstance(output[k], int) and 0 <= output[k] < 256 for k in ("raw_peak", "hann_peak")), "peak_indices", location + "/" + arm)
            check(0 < output["hann_max"] <= output["raw_max"] <= 1, "score_range", location + "/" + arm)
            check(math.isfinite(output["native_spatial_kl"]) and math.isfinite(output["raw_mass"]) and output["raw_mass"] >= output["raw_max"], "saved_response_scalars", location + "/" + arm)
            if arm == "native":
                check(output["native_spatial_kl"] == 0, "native_self_kl", location)
            if output["raw_peak"] == output["hann_peak"]:
                check(output["raw_bbox"] == output["hann_bbox"], "same_peak_same_box", location + "/" + arm)
        if frame % 50 == 0:
            opportunities += 1
            decisions = {arm: variants[arm]["hann_max"] > 0.75 for arm in ARMS}
            for arm in ARMS:
                qualifications[arm] += decisions[arm]
                if arm != "category":
                    disagree = decisions[arm] != decisions["category"]
                    disagreements[arm] += disagree
                    if disagree:
                        qualification_events.append({"sequence": sequence, "frame": frame, "readout": arm,
                            "category_qualifies": decisions["category"], "alternative_qualifies": decisions[arm],
                            "category_hann_max": variants["category"]["hann_max"], "alternative_hann_max": variants[arm]["hann_max"]})
        truth = gt[frame]
        check(len(truth) == 4, "gt_shape", location)
        if not all(math.isfinite(x) for x in truth):
            invalid["nonfinite"] += 1
            continue
        if truth[2] <= 0 or truth[3] <= 0:
            invalid["nonpositive_extent"] += 1
            continue
        crop_x = round(previous[0] + previous[2] / 2 - side / 2)
        crop_y = round(previous[1] + previous[3] / 2 - side / 2)
        gx, gy = truth[0] + truth[2] / 2, truth[1] + truth[3] / 2
        inside = crop_x <= gx < crop_x + side and crop_y <= gy < crop_y + side
        hann = {arm: overlap(variants[arm]["hann_bbox"], truth) for arm in ARMS}
        raw = {arm: overlap(variants[arm]["raw_bbox"], truth) for arm in ARMS}
        if not inside:
            intersects = overlap([crop_x, crop_y, side, side], truth) > 0
            centre_outside_intersects += intersects
            centre_outside_disjoint += not intersects
        entry = {"hann": hann, "raw": raw, "variants": variants}
        labels = ("all_valid", "centre_inside" if inside else "centre_outside", "native_correct" if hann["native"] >= 0.5 else "native_not_correct")
        for label in labels:
            groups[label].append(entry)
            whole_groups[label].append(entry)
        for arm in ARMS:
            rescue = hann["category"] <= 0.1 and hann[arm] >= 0.5
            harm = hann["category"] >= 0.5 and hann[arm] <= 0.1
            if rescue or harm:
                alternative_events.append({"sequence": sequence, "frame": frame, "readout": arm,
                    "type": "rescue" if rescue else "harm", "category_iou": hann["category"], "alternative_iou": hann[arm]})
            rescue_raw = hann[arm] <= 0.1 and raw[arm] >= 0.5
            harm_raw = hann[arm] >= 0.5 and raw[arm] <= 0.1
            if rescue_raw or harm_raw:
                raw_events.append({"sequence": sequence, "frame": frame, "readout": arm,
                    "type": "raw_rescue_hann" if rescue_raw else "raw_harm_hann", "hann_iou": hann[arm], "raw_iou": raw[arm]})
    per_sequence[sequence] = {"groups": {label: summarize(entries) for label, entries in groups.items()},
                             "update_qualification_count": qualifications, "update_qualification_disagreements": disagreements}
    valid = len(groups["all_valid"])
    coverage.append({"sequence": sequence, "gt_rows": len(gt), "positions": len(preds["rows"]),
                     "valid_positions": valid, "invalid_positions": sum(invalid.values()),
                     "invalid_reasons": dict(invalid), "update_opportunities": opportunities,
                     "gt_sha256": hashes[gtpath.relative_to(W).as_posix()]})
    prior = parent["per_sequence"]["category"][sequence]
    current = per_sequence[sequence]["groups"]["all_valid"]["category"]
    equal(current["n"], prior["valid_frames"], sequence + "/parent/valid", "parent_category_metrics")
    equal(current["iou_sum"], prior["iou_sum"], sequence + "/parent/iou_sum", "parent_category_metrics")
    equal(current["low"], prior["low_iou_frames"], sequence + "/parent/low", "parent_category_metrics")
    equal(sum(invalid.values()), prior["invalid_gt_frames"], sequence + "/parent/invalid", "parent_category_metrics")

aggregates = {label: summarize(entries) for label, entries in whole_groups.items()}
for group in aggregates.values():
    for values in group.values():
        values["mean_one_step_iou"] = values["iou_sum"] / values["n"]
equal(per_sequence, result["per_sequence"], "diagnostic_result/per_sequence")
equal(aggregates, result["aggregates"], "diagnostic_result/aggregates")
positions = sum(item["positions"] for item in coverage)
valid_positions = sum(item["valid_positions"] for item in coverage)
check(positions == receipt["positions"] == saved["positions"] == 33108, "coverage", "total positions")
check(valid_positions == saved["valid_positions"] == 28897, "coverage", "valid positions")
updates = {arm: sum(row["update_qualification_count"][arm] for row in per_sequence.values()) for arm in ARMS}
disagreements = {arm: sum(row["update_qualification_disagreements"][arm] for row in per_sequence.values()) for arm in ARMS if arm != "category"}
equal(updates, descriptive["update_qualifications"], "descriptive/update_qualifications")
equal(disagreements, descriptive["update_qualification_disagreements"], "descriptive/disagreements")
for arm in ARMS:
    values = aggregates["all_valid"][arm]
    expected = {"low": values["low"], "correct": values["correct"], "rescue": values["rescue_vs_category"],
                "harm": values["harm_vs_category"], "write": updates[arm], "mean_one_step_iou": values["mean_one_step_iou"]}
    equal(expected, saved["overall"][arm], "saved_verification/overall/" + arm)

group_rows = []
for label, group in aggregates.items():
    for arm in ARMS:
        v = group[arm]
        row = {"group": label, "readout": arm, **{key: v[key] for key in (
            "n", "mean_one_step_iou", "low", "correct", "rescue_vs_category", "harm_vs_category", "raw_rescue_hann", "raw_harm_hann", "raw_hann_peak_changes")},
            "mean_native_spatial_kl": v["kl_native_sum"] / v["n"], "mean_response_mass": v["response_mass_sum"] / v["n"]}
        group_rows.append(row)
equal(group_rows, descriptive["aggregate_rows"], "descriptive/aggregate_rows", "descriptive_tables")
sequence_rows = []
for sequence, data in per_sequence.items():
    for arm in ARMS:
        v = data["groups"]["all_valid"][arm]
        sequence_rows.append({"sequence": sequence, "readout": arm, "n": v["n"], "mean_one_step_iou": v["iou_sum"] / v["n"],
            "low": v["low"], "rescue_vs_category": v["rescue_vs_category"], "harm_vs_category": v["harm_vs_category"],
            "update_qualification_count": data["update_qualification_count"][arm],
            "update_qualification_disagreements": 0 if arm == "category" else data["update_qualification_disagreements"][arm]})
for name, recomputed in (("group_readout_metrics.csv", group_rows), ("sequence_readout_metrics.csv", sequence_rows)):
    rows = list(csv.DictReader(content(R / name).decode().splitlines()))
    check(len(rows) == len(recomputed), "csv_tables", name + " row count")
    converted = []
    for stored, expected in zip(rows, recomputed):
        converted.append({key: type(value)(stored[key]) for key, value in expected.items()})
    equal(converted, recomputed, name, "csv_tables")

all_valid = aggregates["all_valid"]
prior = parent["aggregates"]
narrative_calculations = {
    "one_step_category_minus_empty_percentage_points": 100 * (all_valid["category"]["mean_one_step_iou"] - all_valid["empty"]["mean_one_step_iou"]),
    "one_step_category_minus_swapped_percentage_points": 100 * (all_valid["category"]["mean_one_step_iou"] - all_valid["swapped"]["mean_one_step_iou"]),
    "recursive_category_minus_category_empty_percentage_points": 100 * (prior["category"]["mean_iou"] - prior["category_empty"]["mean_iou"]),
    "recursive_category_minus_category_swapped_percentage_points": 100 * (prior["category"]["mean_iou"] - prior["category_swapped"]["mean_iou"]),
    "outside_centre_share_of_low_positions_percent": 100 * aggregates["centre_outside"]["category"]["low"] / all_valid["category"]["low"],
    "primary_gates_true_total": [sum(parent["gates"].values()), len(parent["gates"])],
    "category_increment_gates_true_total": [sum(parent["preservation_incremental_gates"]["category"].values()), len(parent["preservation_incremental_gates"]["category"])],
    "content_gates_true_total": [sum(value for group in parent["content_gates"].values() for value in group.values()), sum(len(group) for group in parent["content_gates"].values())],
    "native_independent_recursive_mean_iou_from_bound_parent_result": prior["native"]["mean_iou"],
    "last_receipt_elapsed_seconds": receipt["sequences"][-1]["elapsed_seconds"],
    "historical_recursive_update_counts_from_bound_description": historical_description["update_count_reconstructed_from_saved_score_and_verified_runtime_rule"],
    "initial_text_probe_record_count": len(initial_text_probe["rows"]),
    "centered_interface_receipt_status": centered_interface_receipt["status"],
    "centered_interface_receipt_positions": centered_interface_receipt["positions"],
}
content(R / "NARRATIVE_REPORT.md")
content(W / "M83_INTERFACE_AUDIT.md")
content(Path(__file__))
out = {
    "status": "PASS" if not failures else "FAIL", "generated_at": datetime.now(timezone.utc).isoformat(),
    "reviewer": "/root/m83_completed_integrity", "reviewer_model": "gpt-6-astra", "reviewer_reasoning": "max",
    "review_independence": "deterministic", "acceptance_status": "accepted",
    "semantic_review_independence": "same-family", "semantic_acceptance_status": "provisional",
    "method": "Independent stdlib Python; no import or execution of author analysis/verifier; rectangle IoU and math.fsum from raw saved boxes plus frozen dataset GT.",
    "float_absolute_tolerance": ABS_TOL, "float_comparisons": float_comparisons,
    "check_counts": dict(check_counts), "failures": failures,
    "coverage": coverage, "positions": positions, "valid_positions": valid_positions,
    "invalid_positions": positions - valid_positions, "parity_counts": dict(parity),
    "integration_source_files_checked": len(integration["source_sha256"]) - len(integration_missing),
    "integration_source_files_unavailable": integration_missing,
    "sealed_export_file_count": len(manifest["files"]), "archive_member_count": len(manifest["files"]) + 1,
    "hash_checks": hash_checks, "audited_input_hashes": hashes,
    "aggregates": aggregates, "per_sequence": per_sequence,
    "update_qualification_counts": updates, "update_qualification_disagreements": disagreements,
    "update_opportunities": sum(row["update_opportunities"] for row in coverage),
    "alternative_rescue_harm_events": alternative_events, "raw_hann_events": raw_events,
    "update_qualification_disagreement_events": qualification_events,
    "centre_outside_gt_box_intersects_search_square": centre_outside_intersects,
    "centre_outside_gt_box_disjoint_from_search_square": centre_outside_disjoint,
    "narrative_calculations": narrative_calculations,
    "limits": [
        "No model/checkpoint/text-bank binary opened or hashed by this reviewer; their identifiers are traceable to sealed metadata and inference-source assertions only.",
        "No remote observation, model inference, training, seed addition, or new strategy execution.",
        "Full dense response/size/offset/query/template tensors are not saved here: no independent per-frame KL/argmax re-derivation or CUDA state replay. Saved scalar aggregates and all box/GT metrics were independently recomputed.",
        "Local dataset GT matches both frozen spec and training-bound prior inventory; no fresh download or image-level manual annotation validation.",
        "Alternative readouts share Category visited states; qualifications do not mean alternative writes; per-frame low-overlap shares are not failure-cause shares.",
        "Historical non-Category recursive values in narrative_calculations are checked against the hash-bound parent summary; they were not newly rerun as independent trajectories.",
    ],
}
target = R / "m83_independent_recompute.json"
target.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps({key: out[key] for key in ("status", "positions", "valid_positions", "invalid_positions", "sealed_export_file_count", "integration_source_files_checked", "float_comparisons", "update_qualification_counts", "update_qualification_disagreements", "narrative_calculations", "failures")}, indent=2))
raise SystemExit(0 if not failures else 1)
