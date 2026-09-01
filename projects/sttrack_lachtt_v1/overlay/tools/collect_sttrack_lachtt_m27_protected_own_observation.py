#!/usr/bin/env python3
"""Collect label-free protected-own native RGB-D observations for M27."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools import run_sttrack_lachtt_train152_collection as base


SCHEMA = "sttrack-lachtt-m27-protected-own-observation-spec/v1"
RESULT_SCHEMA = "sttrack-lachtt-m27-protected-own-observation-result/v1"
EXPECTED_CHECKPOINT_SHA256 = (
    "cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98"
)
EXPECTED_EVENTS = {0: 246, 1: 261}
EXPECTED_SEQUENCES = {0: 37, 1: 39}
FEATURE_AGES = 5
FORBIDDEN_NUMERIC_TARGET_PATHS = {
    Path("/root/autodl-tmp/"
         "sttrack_lachtt_m18_0_causal_survival_target_closure_v1_20260901/"
         "trajectory_targets.jsonl.gz").resolve(),
    Path("/root/autodl-tmp/sttrack_lachtt_train152_gatea_v1_20260831/"
         "labeled_actions.jsonl.gz").resolve(),
    Path("/root/autodl-tmp/"
         "sttrack_lachtt_m22a_sequence_disjoint_causal_survival_v1_20260901/"
         "heldout_predictions.jsonl.gz").resolve(),
    Path("/root/autodl-tmp/"
         "sttrack_lachtt_m23a_r2_unique_hypothesis_direct_selection_v1_20260902/"
         "development_predictions.jsonl.gz").resolve(),
    Path("/root/autodl-tmp/"
         "sttrack_lachtt_m25_sequence_pooled_lofo_direct_router_v1_20260902/"
         "oof_predictions.jsonl.gz").resolve(),
    Path("/root/autodl-tmp/"
         "sttrack_lachtt_m26_nested_sequence_calibrated_harm_v1_20260902/"
         "oof_predictions.jsonl.gz").resolve(),
}
RUNTIME_AUDIT = {
    "forbidden_file_opens": [],
    "groundtruth_file_opens": [],
    "network_connects": [],
}


class ContractError(RuntimeError):
    pass


def runtime_audit_hook(event, args):
    if event == "open":
        target = args[0]
        if isinstance(target, (str, bytes, os.PathLike)):
            resolved = Path(os.fsdecode(target)).resolve()
            if resolved in FORBIDDEN_NUMERIC_TARGET_PATHS:
                RUNTIME_AUDIT["forbidden_file_opens"].append(str(resolved))
            if (resolved.name == "groundtruth.txt" and len(args) > 1 and
                    args[1] is not None):
                RUNTIME_AUDIT["groundtruth_file_opens"].append(str(resolved))
    elif event == "socket.connect":
        RUNTIME_AUDIT["network_connects"].append(str(args))


sys.addaudithook(runtime_audit_hook)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--collection-shard", required=True, type=Path)
    parser.add_argument("--split-ledger", required=True, type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--shard", required=True, type=int, choices=(0, 1))
    parser.add_argument("--device", required=True, type=int, choices=(0, 1))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sequence", action="append")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def atomic_torch_save(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def repository_identity():
    return {
        "path": str(REPOSITORY_ROOT),
        "branch": subprocess.check_output(
            ("git", "-C", str(REPOSITORY_ROOT), "branch", "--show-current"),
            text=True).strip(),
        "commit": subprocess.check_output(
            ("git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"),
            text=True).strip(),
        "clean": not subprocess.check_output(
            ("git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain"),
            text=True).strip(),
    }


def validate_spec(args):
    if args.smoke:
        if args.spec is not None or args.max_events != 1 or not args.sequence:
            raise ContractError(
                "smoke requires one-event limit, an explicit sequence and no spec")
        return None
    if (args.spec is None or not args.spec.is_file() or args.sequence or
            args.max_events is not None):
        raise ContractError("formal M27 collection arguments drifted")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    identity = repository_identity()
    if (spec.get("schema") != SCHEMA or spec.get("complete") is not True or
            spec.get("created_before_execution") is not True or
            spec.get("repository") != identity):
        raise ContractError("formal M27 spec identity drifted")
    expected = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "collection_ledger_shard_%d" % args.shard:
            args.collection_shard / "events.jsonl",
        "split_ledger": args.split_ledger,
        "runner": Path(__file__).resolve(),
        "base_collector": Path(base.__file__).resolve(),
    }
    records = spec.get("bindings", {})
    for name, path in expected.items():
        record = records.get(name)
        if not isinstance(record, dict) or record != file_record(path):
            raise ContractError("formal binding mismatch: %s" % name)
    if spec.get("dataset_root") != {"path": str(args.dataset_root)}:
        raise ContractError("formal dataset-root binding drifted")
    shard = spec.get("shards", {}).get(str(args.shard), {})
    if (Path(shard.get("output", "")).resolve() != args.output or
            int(shard.get("device", -1)) != args.device or
            int(shard.get("events", -1)) != EXPECTED_EVENTS[args.shard] or
            int(shard.get("sequences", -1)) != EXPECTED_SEQUENCES[args.shard]):
        raise ContractError("formal shard contract drifted")
    return spec


def event_key(row):
    return (str(row["sequence"]), int(row["event_id"]),
            int(row["trigger_frame"]))


def load_selected_events(args):
    split = json.loads(args.split_ledger.read_text(encoding="utf-8"))
    if (split.get("schema") !=
            "sttrack-lachtt-m18-0-sequence-split-ledger/v1" or
            split.get("training_folds") != [2, 3, 4, 5] or
            int(split.get("heldout_fold", -1)) != 1 or
            int(split.get("quarantine_fold", -1)) != 0):
        raise ContractError("M18 split identity drifted")
    selected = {
        (str(row["sequence"]), int(row["event_id"]),
         int(row["trigger_frame"])): int(row["fold"])
        for row in split["events"]
        if row.get("partition") == "training" and
        row.get("strict_h10_available") is True
    }
    if len(selected) != 507 or set(selected.values()) != {2, 3, 4, 5}:
        raise ContractError("M27 Train-only event census drifted")
    ledger_path = args.collection_shard / "events.jsonl"
    rows = []
    with ledger_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = event_key(row)
            if key in selected:
                row["fold"] = selected[key]
                rows.append(row)
    if args.sequence:
        allowed = set(args.sequence)
        rows = [row for row in rows if row["sequence"] in allowed]
    rows.sort(key=event_key)
    if args.max_events is not None:
        rows = rows[:args.max_events]
    if not rows or len({event_key(row) for row in rows}) != len(rows):
        raise ContractError("M27 selected event identity drifted")
    return rows


def validate_config_and_network(args):
    if sha256_file(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ContractError("official STTrack checkpoint hash mismatch")
    base.update_config_from_file(str(args.config))
    if (not bool(base.cfg.MODEL.TSG.FIX_QUERY_WINDOW) or
            float(base.cfg.TEST.SEARCH_FACTOR) != 4.0 or
            int(base.cfg.TEST.SEARCH_SIZE) != 256 or
            int(base.cfg.DATA.TEMPLATE.NUMBER) != 2):
        raise ContractError("STTrack protected-observation config drifted")
    network = base.build_sttrack(base.cfg, training=False)
    incompatible = network.load_state_dict(
        torch.load(str(args.checkpoint), map_location="cpu")["net"],
        strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ContractError("official checkpoint strict load failed")
    network = network.cuda(args.device).eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network


def collect_event(network, preprocessor, template, colors, depths, row,
                  search_size, keep_rate):
    searches = []
    priors = []
    resize_factors = []
    raw_rois = []
    scores = []
    if len(row["public"]) < FEATURE_AGES:
        raise ContractError("public trajectory is shorter than five ages")
    for age, public in enumerate(row["public"][:FEATURE_AGES]):
        frame_index = int(public["frame_index"])
        if frame_index != int(row["trigger_frame"]) + age:
            raise ContractError("public frame alignment drifted")
        image, raw_depth = base.read_rgbd(colors[frame_index], depths[frame_index])
        bbox = base.finite_bbox(public["bbox"])
        if bbox is None or public.get("score") is None:
            raise ContractError("protected public observation is incomplete")
        patch, resize_factor, _ = base.sample_target(
            image, bbox, base.cfg.TEST.SEARCH_FACTOR, output_sz=search_size)
        searches.append(preprocessor.process(patch))
        priors.append(bbox)
        resize_factors.append(float(resize_factor))
        raw_rois.append(base.raw_depth_rois(raw_depth, [bbox])[0])
        scores.append(float(public["score"]))
    search = torch.cat(searches, dim=0)
    batch = len(priors)
    templates = template.repeat(batch, 1, 1, 1)
    with torch.no_grad():
        output = network.forward(
            template=[templates, templates], search=[search],
            track_query_before=None, keep_rate=keep_rate,
            return_candidate_features=True)[0]
    native = output["candidate_features"]
    candidates = [{"source_index": index, "bbox": bbox}
                  for index, bbox in enumerate(priors)]
    result = {}
    for source, target in (
            ("search_rgb_tokens", "public_native_rgb"),
            ("search_depth_tokens", "public_native_depth"),
            ("search_fused_tokens", "public_native_fused")):
        result[target] = base.pool_candidate_tokens(
            native[source], candidates, priors, resize_factors,
            search_size).detach().cpu().half()
    result["public_raw_depth"] = torch.stack(raw_rois).half()
    result["public_score"] = torch.tensor(scores, dtype=torch.float32)
    return result


def validate_feature(value):
    expected = {
        "public_native_rgb": ((FEATURE_AGES, 768), torch.float16),
        "public_native_depth": ((FEATURE_AGES, 768), torch.float16),
        "public_native_fused": ((FEATURE_AGES, 768), torch.float16),
        "public_raw_depth": ((FEATURE_AGES, 2, 16, 16), torch.float16),
        "public_score": ((FEATURE_AGES,), torch.float32),
    }
    if set(value) != set(expected):
        raise ContractError("M27 feature key set drifted")
    for name, (shape, dtype) in expected.items():
        tensor = value[name]
        if (tuple(tensor.shape) != shape or tensor.dtype != dtype or
                not torch.isfinite(tensor.float()).all().item()):
            raise ContractError("M27 feature contract drifted: %s" % name)
    valid = value["public_raw_depth"][:, 1].float()
    if valid.min().item() < 0.0 or valid.max().item() > 1.0:
        raise ContractError("M27 raw-depth validity mask drifted")


def seal_output(root):
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def main():
    args = parse_args()
    for name in ("checkpoint", "config", "dataset_root", "collection_shard",
                 "split_ledger", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.spec is not None:
        args.spec = args.spec.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.time()
    formal_spec = validate_spec(args)
    rows = load_selected_events(args)
    selected_sequences = sorted({row["sequence"] for row in rows})
    if not args.smoke:
        if (len(rows) != EXPECTED_EVENTS[args.shard] or
                len(selected_sequences) != EXPECTED_SEQUENCES[args.shard]):
            raise ContractError("formal M27 shard census drifted")
    torch.manual_seed(20260927)
    torch.cuda.manual_seed_all(20260927)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    network = validate_config_and_network(args)
    preprocessor = base.PreprocessorMM(
        mean=base.cfg.DATA.MEAN, std=base.cfg.DATA.STD)
    keep_rate = [value for value in torch.linspace(0.7, 1.0, 3)][::-1]
    search_size = int(base.cfg.TEST.SEARCH_SIZE)

    args.output.mkdir(parents=True)
    feature_root = args.output / "events"
    feature_root.mkdir()
    partial = args.output / "events.jsonl.partial"
    rows_by_sequence = defaultdict(list)
    for row in rows:
        rows_by_sequence[row["sequence"]].append(row)
    output_rows = []
    template_checks = 0
    template_changes = 0
    with partial.open("w", encoding="utf-8") as stream:
        for sequence in selected_sequences:
            sequence_root = args.dataset_root / sequence
            colors, depths = base.resolve_frames(sequence_root)
            init_bbox = base.read_initial_bbox(sequence_root / "groundtruth.txt")
            initial_image, _ = base.read_rgbd(colors[0], depths[0])
            template_patch, _, _ = base.sample_target(
                initial_image, init_bbox, base.cfg.TEST.TEMPLATE_FACTOR,
                output_sz=base.cfg.TEST.TEMPLATE_SIZE)
            template = preprocessor.process(template_patch).detach()
            template_before = template.detach().clone()
            for row in rows_by_sequence[sequence]:
                source_feature = args.collection_shard / row["feature_path"]
                if (sha256_file(source_feature) != row["feature_sha256"] or
                        source_feature.stat().st_size != int(row["feature_bytes"])):
                    raise ContractError("source candidate feature identity drifted")
                feature = collect_event(
                    network, preprocessor, template, colors, depths, row,
                    search_size, keep_rate)
                validate_feature(feature)
                relative = Path("events") / (
                    "%s_event%04d_frame%06d.pt" %
                    (sequence, int(row["event_id"]), int(row["trigger_frame"])))
                feature_path = args.output / relative
                atomic_torch_save(feature_path, feature)
                output_row = {
                    "sequence": sequence,
                    "event_id": int(row["event_id"]),
                    "trigger_frame": int(row["trigger_frame"]),
                    "fold": int(row["fold"]),
                    "feature": file_record(feature_path),
                    "source_candidate_feature": file_record(source_feature),
                    "query_state_contract": (
                        "protected-own native search observation with "
                        "track_query_before=None; not exact protected query state"),
                }
                output_rows.append(output_row)
                stream.write(json.dumps(
                    output_row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                if len(output_rows) % 25 == 0:
                    print(json.dumps({
                        "shard": args.shard,
                        "events_completed": len(output_rows),
                        "events_total": len(rows),
                    }, sort_keys=True), flush=True)
            template_checks += 1
            template_changed = not torch.equal(template, template_before)
            template_changes += int(template_changed)
            if template_changed:
                raise ContractError("immutable initial template changed")
    os.replace(partial, args.output / "events.jsonl")

    groundtruth_opens = Counter(RUNTIME_AUDIT["groundtruth_file_opens"])
    expected_groundtruth = {
        str((args.dataset_root / sequence / "groundtruth.txt").resolve())
        for sequence in selected_sequences}
    source_records_exact = all(
        row["source_candidate_feature"]["sha256"] ==
        next(value for value in rows if event_key(value) == (
            row["sequence"], row["event_id"], row["trigger_frame"]))[
                "feature_sha256"]
        for row in output_rows)
    engineering = {
        "event_count_exact": len(output_rows) == len(rows),
        "unique_event_keys_exact": len({
            (row["sequence"], row["event_id"], row["trigger_frame"])
            for row in output_rows}) == len(rows),
        "sequence_count_exact": len(selected_sequences) == len(rows_by_sequence),
        "source_candidate_feature_records_exact": source_records_exact,
        "groundtruth_open_set_exact": set(groundtruth_opens) == expected_groundtruth,
        "groundtruth_open_once_per_sequence": (
            set(groundtruth_opens.values()) == {1}),
        "forbidden_numeric_target_open_count_zero": not RUNTIME_AUDIT[
            "forbidden_file_opens"],
        "network_connect_count_zero": not RUNTIME_AUDIT["network_connects"],
        "template_checks_exact": template_checks == len(selected_sequences),
        "template_change_count_zero": template_changes == 0,
    }
    if not args.smoke:
        engineering.update({
            "formal_shard_event_count_exact": (
                len(output_rows) == EXPECTED_EVENTS[args.shard]),
            "formal_shard_sequence_count_exact": (
                len(selected_sequences) == EXPECTED_SEQUENCES[args.shard]),
            "repository_identity_exact": repository_identity() ==
            formal_spec["repository"],
        })
    accepted = all(engineering.values())
    result = {
        "schema": RESULT_SCHEMA,
        "complete": True,
        "accepted": accepted,
        "mode": "smoke" if args.smoke else "formal",
        "shard": args.shard,
        "device": args.device,
        "events": len(output_rows),
        "sequences": len(selected_sequences),
        "fold_counts": dict(sorted(Counter(
            row["fold"] for row in output_rows).items())),
        "feature_schema": {
            "public_native_rgb": [5, 768],
            "public_native_depth": [5, 768],
            "public_native_fused": [5, 768],
            "public_raw_depth": [5, 2, 16, 16],
            "public_score": [5],
        },
        "query_state_contract": (
            "track_query_before=None; appearance observation only"),
        "engineering_conditions": engineering,
        "failed_engineering_conditions": sorted(
            name for name, passed in engineering.items() if not passed),
        "runtime_audit": {
            "forbidden_file_opens": sorted(set(
                RUNTIME_AUDIT["forbidden_file_opens"])),
            "groundtruth_file_open_count": sum(groundtruth_opens.values()),
            "groundtruth_file_open_unique": len(groundtruth_opens),
            "network_connect_count": len(RUNTIME_AUDIT["network_connects"]),
        },
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(args.device)),
        "elapsed_seconds": time.time() - started,
        "source": {
            "repository": repository_identity(),
            "runner": file_record(Path(__file__).resolve()),
            "checkpoint": file_record(args.checkpoint),
            "config": file_record(args.config),
            "collection_ledger": file_record(
                args.collection_shard / "events.jsonl"),
            "split_ledger": file_record(args.split_ledger),
        },
        "authorization": {
            "selector_training": False,
            "tracking_checkpoint": False,
            "fold1": False,
            "fold0": False,
            "depthtrack_test": False,
            "cdtb": False,
            "vot_low22": False,
            "vot_full127": False,
            "qwen": False,
            "automatic_next_stage": False,
        },
    }
    atomic_json(args.output / "result.json", result)
    manifest = {
        "schema": "sttrack-lachtt-m27-protected-own-observation-manifest/v1",
        "complete": True,
        "accepted": accepted,
        "files": {
            "events.jsonl": file_record(args.output / "events.jsonl"),
            "result.json": file_record(args.output / "result.json"),
        },
        "event_features": [row["feature"] for row in output_rows],
    }
    atomic_json(args.output / "manifest.json", manifest)
    seal_output(args.output)
    print(json.dumps({
        "accepted": accepted,
        "events": len(output_rows),
        "sequences": len(selected_sequences),
        "output": str(args.output),
    }, sort_keys=True), flush=True)
    raise SystemExit(0 if accepted else 2)


if __name__ == "__main__":
    main()
