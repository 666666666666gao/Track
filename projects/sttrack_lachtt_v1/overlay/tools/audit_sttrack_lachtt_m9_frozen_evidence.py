#!/usr/bin/env python3
"""Read-only strict-H10 audit of parameter-free candidate identity evidence."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
import torch.nn.functional as F


SIGNALS = (
    "clip_initial_mean_cosine",
    "clip_text_mean_cosine",
    "native_rgb_top4_mean_cosine",
    "native_depth_top4_validity_adjusted",
    "clip_adjacent_mean_cosine",
    "native_rgb_adjacent_mean_cosine",
    "native_depth_adjacent_validity_adjusted",
    "response_score_mean",
    "response_margin_mean",
    "negative_entropy_mean",
)
IDENTITY_SIGNALS = SIGNALS[:4]
FEATURE_SHAPES = {
    "clip_image": (5, 6, 768),
    "native_depth": (5, 6, 768),
    "native_fused": (5, 6, 768),
    "native_rgb": (5, 6, 768),
    "query_depth": (5, 6, 768),
    "query_rgb": (5, 6, 768),
    "raw_depth": (5, 6, 2, 16, 16),
    "scalars": (5, 6, 15),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def atomic_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_jsonl_gz(path, rows):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True,
                                        allow_nan=False) + "\n")
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_record(record, require_bytes=True):
    path = Path(record["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if require_bytes and "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise ValueError("source size mismatch: %s" % path)
    if sha256_file(path) != record["sha256"]:
        raise ValueError("source hash mismatch: %s" % path)
    return path


def safe_torch_load(path):
    return torch.load(str(path), map_location="cpu", weights_only=False)


def normalize(value, epsilon):
    return F.normalize(value.double(), p=2.0, dim=-1, eps=epsilon)


def anchor_topk(candidate, bank, top_k, epsilon):
    candidate = normalize(candidate, epsilon)
    bank = normalize(bank, epsilon)
    similarity = torch.einsum("hcd,td->hct", candidate, bank)
    return similarity.topk(top_k, dim=-1).values.mean(dim=-1)


def adjacent_cosine(value, epsilon):
    value = normalize(value, epsilon)
    return (value[1:] * value[:-1]).sum(dim=-1)


def evidence_matrix(features, anchor, native, contract):
    epsilon = float(contract["l2_epsilon"])
    top_k = int(contract["native_top_k"])
    missing_floor = float(contract["depth_missing_floor"])
    clip = normalize(features["clip_image"], epsilon)
    initial = normalize(anchor["initial_image"], epsilon).reshape(1, 1, 768)
    text = normalize(anchor["identity_text"], epsilon).reshape(1, 1, 768)
    depth_validity = features["raw_depth"][:, :, 1].double().mean(dim=(-1, -2))

    clip_initial = (clip * initial).sum(dim=-1).mean(dim=0)
    clip_text = (clip * text).sum(dim=-1).mean(dim=0)
    native_rgb = anchor_topk(
        features["native_rgb"], native["native_template_rgb_tokens"],
        top_k, epsilon).mean(dim=0)
    native_depth_age = anchor_topk(
        features["native_depth"], native["native_template_depth_tokens"],
        top_k, epsilon)
    native_depth = (depth_validity * native_depth_age +
                    (1.0 - depth_validity) * missing_floor).mean(dim=0)
    clip_adjacent = adjacent_cosine(features["clip_image"], epsilon).mean(dim=0)
    rgb_adjacent = adjacent_cosine(features["native_rgb"], epsilon).mean(dim=0)
    depth_adjacent_age = adjacent_cosine(features["native_depth"], epsilon)
    adjacent_validity = torch.minimum(depth_validity[1:], depth_validity[:-1])
    depth_adjacent = (adjacent_validity * depth_adjacent_age +
                      (1.0 - adjacent_validity) * missing_floor).mean(dim=0)
    scalars = features["scalars"].double()
    values = torch.stack((
        clip_initial,
        clip_text,
        native_rgb,
        native_depth,
        clip_adjacent,
        rgb_adjacent,
        depth_adjacent,
        scalars[:, :, 0].mean(dim=0),
        scalars[:, :, 1].mean(dim=0),
        -scalars[:, :, 2].mean(dim=0),
    ), dim=1)
    if tuple(values.shape) != (6, len(SIGNALS)):
        raise RuntimeError("evidence matrix shape drifted")
    if not torch.isfinite(values).all().item():
        raise RuntimeError("non-finite evidence")
    return values.numpy()


def dense_rank_descending(values):
    return np.asarray([1 + int(np.sum(values > value)) for value in values],
                      dtype=np.int64)


def consensus(evidence, rule):
    if evidence.shape != (6, len(SIGNALS)):
        raise ValueError("consensus evidence shape drifted")
    votes = np.zeros(6, dtype=np.int64)
    signal_winners = []
    for signal_index in range(evidence.shape[1]):
        column = evidence[:, signal_index]
        maximum = np.max(column)
        winners = np.flatnonzero(column == maximum)
        winner = int(winners[0]) if len(winners) == 1 else None
        signal_winners.append(winner)
        if winner is not None:
            votes[winner] += 1
    order = np.argsort(-votes, kind="stable")
    winner = int(order[0])
    runner_up = int(order[1])
    reason = "selected"
    selected = winner
    if int(votes[winner]) < int(rule["minimum_winner_votes"]):
        selected, reason = None, "insufficient_votes"
    elif int(votes[winner] - votes[runner_up]) < int(rule["minimum_vote_lead"]):
        selected, reason = None, "insufficient_vote_lead"
    else:
        for name in rule["identity_signals"]:
            signal_index = SIGNALS.index(name)
            ranks = dense_rank_descending(evidence[:, signal_index])
            if int(ranks[winner]) > int(rule["maximum_identity_rank"]):
                selected, reason = None, "identity_rank_conflict"
                break
    return {
        "selected_index": selected,
        "reason": reason,
        "votes": votes.tolist(),
        "signal_winners": signal_winners,
    }


def auc_binary(positives, negatives):
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def main():
    args = parse_args()
    spec_path = args.spec.resolve()
    binding_path = args.binding.resolve()
    spec = load_json(spec_path)
    binding = load_json(binding_path)
    runner_path = Path(__file__).resolve()
    if (binding.get("schema") != "sttrack-lachtt-m9-frozen-evidence-binding/v1" or
            binding["spec"]["path"] != str(spec_path) or
            binding["spec"]["sha256"] != sha256_file(spec_path) or
            binding["runner"]["path"] != str(runner_path) or
            binding["runner"]["sha256"] != sha256_file(runner_path) or
            binding["output"] != spec["output"]):
        raise ValueError("binding mismatch")
    if tuple(spec["evidence_contract"]["signals"]) != SIGNALS:
        raise ValueError("signal contract mismatch")
    for record in spec["sources"].values():
        validate_record(record)
    closure_result = load_json(spec["sources"]["closure_result"]["path"])
    native_manifest = load_json(spec["sources"]["native_anchor_manifest"]["path"])
    if (closure_result.get("accepted") is not True or
            native_manifest.get("accepted") is not True or
            native_manifest.get("future_frame_opened") is not False or
            native_manifest.get("future_ground_truth_opened") is not False):
        raise ValueError("source safety receipt rejected")

    native_root = Path(spec["sources"]["native_anchor_index"]["path"]).parent
    native_index = {}
    with Path(spec["sources"]["native_anchor_index"]["path"]).open(
            "r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["sequence"] in native_index:
                raise ValueError("duplicate native sequence")
            native_index[row["sequence"]] = row

    output = Path(spec["output"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    expected = spec["expected"]
    counters = Counter()
    label_counts = Counter()
    sequence_names = set()
    evidence_rows = []
    action_signal_values = {
        name: {"beneficial": [], "catastrophic": []} for name in SIGNALS}
    sequence_signal_values = {
        name: defaultdict(lambda: {"beneficial": [], "catastrophic": []})
        for name in SIGNALS
    }
    signal_top1 = {name: Counter() for name in SIGNALS}
    consensus_labels = Counter()
    consensus_sequences = set()
    beneficial_sequences = set()
    selected_by_sequence = Counter()
    permutation_mismatches = 0
    maximum_permutation_error = 0.0
    rng = random.Random(int(spec["permutation_audit"]["seed"]))
    clip_anchor_cache = {}
    native_anchor_cache = {}

    closure_path = Path(spec["sources"]["closure"]["path"])
    with gzip.open(closure_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            counters["events"] += 1
            sequence = row["sequence"]
            sequence_names.add(sequence)
            branch_order = list(row["branch_order"])
            if branch_order != list(expected["candidate_order"]):
                counters["candidate_axis_mismatches"] += 1
            actions = row["actions"]
            if ([action["branch_id"] for action in actions] != branch_order or
                    len(actions) != 6):
                counters["candidate_axis_mismatches"] += 1
            labels = [action["strict_label"] for action in actions]
            label_counts.update(labels)
            counters["actions"] += len(actions)
            available = [label != "unavailable" for label in labels]
            if all(available):
                counters["fully_available_events"] += 1
            elif not any(available):
                counters["fully_unavailable_events"] += 1
            else:
                counters["partial_availability_events"] += 1

            feature_path = Path(row["feature_path"])
            if (feature_path.stat().st_size != int(row["feature_bytes"]) or
                    sha256_file(feature_path) != row["feature_sha256"]):
                raise ValueError("feature identity mismatch: %s" % feature_path)
            features = safe_torch_load(feature_path)
            if set(features) != set(FEATURE_SHAPES):
                raise ValueError("feature key mismatch")
            for name, shape in FEATURE_SHAPES.items():
                value = features[name]
                if tuple(value.shape) != shape:
                    counters["shape_mismatches"] += 1
                if not torch.isfinite(value.float()).all().item():
                    counters["nonfinite_values"] += 1

            clip_anchor_path = Path(row["anchor_path"])
            if sequence not in clip_anchor_cache:
                clip_anchor_cache[sequence] = safe_torch_load(clip_anchor_path)
                for name in ("initial_image", "identity_text"):
                    value = clip_anchor_cache[sequence].get(name)
                    if value is None or tuple(value.shape) != (1, 768):
                        counters["shape_mismatches"] += 1
                    elif not torch.isfinite(value.float()).all().item():
                        counters["nonfinite_values"] += 1
            clip_anchor = clip_anchor_cache[sequence]

            native_row = native_index.get(sequence)
            if native_row is None:
                raise ValueError("missing native anchor: %s" % sequence)
            native_path = native_root / native_row["path"]
            if sequence not in native_anchor_cache:
                if (native_path.stat().st_size != int(native_row["bytes"]) or
                        sha256_file(native_path) != native_row["sha256"]):
                    raise ValueError("native anchor identity mismatch: %s" % native_path)
                native_anchor_cache[sequence] = safe_torch_load(native_path)
                for name in ("native_template_rgb_tokens",
                             "native_template_depth_tokens"):
                    value = native_anchor_cache[sequence].get(name)
                    if value is None or tuple(value.shape) != (64, 768):
                        counters["shape_mismatches"] += 1
                    elif not torch.isfinite(value.float()).all().item():
                        counters["nonfinite_values"] += 1
            native = native_anchor_cache[sequence]

            evidence = evidence_matrix(
                features, clip_anchor, native, spec["evidence_contract"])
            decision = consensus(evidence, spec["consensus_rule"])
            for _ in range(int(spec["permutation_audit"]["permutations"])):
                permutation = list(range(6))
                rng.shuffle(permutation)
                inverse = np.argsort(np.asarray(permutation))
                permuted = evidence[permutation]
                restored = permuted[inverse]
                maximum_permutation_error = max(
                    maximum_permutation_error,
                    float(np.max(np.abs(restored - evidence))))
                permuted_decision = consensus(permuted, spec["consensus_rule"])
                permuted_selected = permuted_decision["selected_index"]
                mapped_selected = (None if permuted_selected is None else
                                   permutation[int(permuted_selected)])
                restored_votes = np.asarray(permuted_decision["votes"])[inverse]
                if (mapped_selected != decision["selected_index"] or
                        not np.array_equal(restored_votes,
                                           np.asarray(decision["votes"]))):
                    permutation_mismatches += 1

            if all(available):
                for signal_index, name in enumerate(SIGNALS):
                    values = evidence[:, signal_index]
                    winners = np.flatnonzero(values == np.max(values))
                    if len(winners) == 1:
                        signal_top1[name][labels[int(winners[0])]] += 1
                    else:
                        signal_top1[name]["tie"] += 1
                    for candidate_index, label in enumerate(labels):
                        if label in ("beneficial", "catastrophic"):
                            value = float(values[candidate_index])
                            action_signal_values[name][label].append(value)
                            sequence_signal_values[name][sequence][label].append(value)
                selected_index = decision["selected_index"]
                if selected_index is None:
                    consensus_labels["abstain"] += 1
                else:
                    selected_label = labels[int(selected_index)]
                    consensus_labels[selected_label] += 1
                    selected_by_sequence[sequence] += 1
                    consensus_sequences.add(sequence)
                    if selected_label == "beneficial":
                        beneficial_sequences.add(sequence)
            else:
                if decision["selected_index"] is not None:
                    consensus_labels["unavailable_selected"] += 1

            evidence_rows.append({
                "sequence": sequence,
                "event_id": int(row["event_id"]),
                "trigger_frame": int(row["trigger_frame"]),
                "branch_order": branch_order,
                "labels": labels,
                "evidence": {
                    name: [float(value) for value in evidence[:, index]]
                    for index, name in enumerate(SIGNALS)
                },
                "decision": decision,
            })

    signal_diagnostics = {}
    for name in SIGNALS:
        sequence_aucs = []
        for values in sequence_signal_values[name].values():
            value = auc_binary(values["beneficial"], values["catastrophic"])
            if value is not None:
                sequence_aucs.append(value)
        signal_diagnostics[name] = {
            "global_beneficial_vs_catastrophic_auc": auc_binary(
                action_signal_values[name]["beneficial"],
                action_signal_values[name]["catastrophic"]),
            "sequence_macro_auc": (sum(sequence_aucs) / len(sequence_aucs)
                                   if sequence_aucs else None),
            "sequences_with_both_classes": len(sequence_aucs),
            "top1_labels": dict(signal_top1[name]),
        }

    beneficial = int(consensus_labels["beneficial"])
    catastrophic = int(consensus_labels["catastrophic"])
    neutral = int(consensus_labels["neutral"])
    selected = beneficial + catastrophic + neutral
    precision = beneficial / selected if selected else 0.0
    gates = spec["gates"]
    conditions = {
        "source_hashes_exact": True,
        "event_count_exact": counters["events"] == int(expected["events"]),
        "action_count_exact": counters["actions"] == int(expected["actions"]),
        "sequence_count_exact": len(sequence_names) == int(expected["sequences"]),
        "fully_available_events_exact": counters["fully_available_events"] == int(
            expected["fully_available_events"]),
        "fully_unavailable_events_exact": counters["fully_unavailable_events"] == int(
            expected["fully_unavailable_events"]),
        "partial_availability_events_zero": counters["partial_availability_events"] == 0,
        "strict_action_counts_exact": dict(sorted(label_counts.items())) == dict(
            sorted(expected["strict_action_counts"].items())),
        "candidate_axis_mismatches_max": counters["candidate_axis_mismatches"] <= int(
            gates["candidate_axis_mismatches_max"]),
        "shape_mismatches_max": counters["shape_mismatches"] <= int(
            gates["shape_mismatches_max"]),
        "nonfinite_values_max": counters["nonfinite_values"] <= int(
            gates["nonfinite_values_max"]),
        "permutation_selection_mismatches_max": permutation_mismatches <= int(
            gates["permutation_selection_mismatches_max"]),
        "permutation_maximum_absolute_error": maximum_permutation_error <= float(
            spec["permutation_audit"]["maximum_absolute_error"]),
        "selected_actions_min": selected >= int(gates["selected_actions_min"]),
        "beneficial_actions_min": beneficial >= int(gates["beneficial_actions_min"]),
        "beneficial_sequences_min": len(beneficial_sequences) >= int(
            gates["beneficial_sequences_min"]),
        "catastrophic_actions_max": catastrophic <= int(
            gates["catastrophic_actions_max"]),
        "beneficial_precision_min": precision >= float(
            gates["beneficial_precision_min"]),
    }
    accepted = all(conditions.values())
    result = {
        "schema": "sttrack-lachtt-m9-frozen-evidence-audit-result/v1",
        "complete": True,
        "accepted": accepted,
        "decision": ("m9_pass_plan_shallow_calibration_only" if accepted else
                     "m9_fail_stop_cached_selector_move_online_memory"),
        "claim_ceiling": spec["claim_ceiling"],
        "conditions": conditions,
        "counts": {
            **dict(counters),
            "sequences": len(sequence_names),
            "strict_action_counts": dict(label_counts),
        },
        "consensus": {
            "labels": dict(consensus_labels),
            "selected_evaluable_actions": selected,
            "beneficial_precision": precision,
            "selected_sequences": len(consensus_sequences),
            "beneficial_sequences": len(beneficial_sequences),
            "top_selected_sequences": selected_by_sequence.most_common(20),
        },
        "signal_diagnostics": signal_diagnostics,
        "permutation_audit": {
            "permutations_per_event": int(
                spec["permutation_audit"]["permutations"]),
            "selection_mismatches": permutation_mismatches,
            "maximum_absolute_error": maximum_permutation_error,
        },
        "authorization": {
            "shallow_calibration_plan": accepted,
            "training": False,
            "online_replay": False,
            "depthtrack_test": False,
            "cdtb": False,
            "vot_low22": False,
            "vot_full127": False,
            "qwen": False,
        },
        "inputs": {
            "spec": file_record(spec_path),
            "binding": file_record(binding_path),
            "runner": file_record(runner_path),
        },
    }
    evidence_path = output / "evidence_summary.jsonl.gz"
    result_path = output / "result.json"
    manifest_path = output / "manifest.json"
    output.mkdir(parents=True)
    atomic_jsonl_gz(evidence_path, evidence_rows)
    atomic_json(result_path, result)
    manifest = {
        "schema": "sttrack-lachtt-m9-frozen-evidence-audit-manifest/v1",
        "complete": True,
        "accepted": accepted,
        "payload": {
            "result": file_record(result_path),
            "evidence_summary": file_record(evidence_path),
        },
        "unauthorized_actions": {
            "model_loaded": False,
            "training": False,
            "optimizer": False,
            "checkpoint_written": False,
            "original_rgb_or_depth_opened": False,
            "ground_truth_opened": False,
            "depthtrack_test": False,
            "cdtb": False,
            "vot_low22": False,
            "vot_full127": False,
            "qwen": False,
        },
    }
    atomic_json(manifest_path, manifest)
    for path in (result_path, evidence_path, manifest_path):
        path.chmod(0o444)
    output.chmod(0o555)
    print(json.dumps({
        "accepted": accepted,
        "decision": result["decision"],
        "consensus": result["consensus"],
        "result": file_record(result_path),
        "manifest": file_record(manifest_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
