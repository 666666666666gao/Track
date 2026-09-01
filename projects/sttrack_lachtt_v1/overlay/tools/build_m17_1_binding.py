#!/usr/bin/env python3
"""Build immutable preflight or post-audit M17-1 execution bindings."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime, timezone


PREFLIGHT_SCHEMA = "sttrack-lachtt-m17-1-preflight-binding/v1"
EXECUTION_SCHEMA = "sttrack-lachtt-m17-1-postaudit-execution-binding/v1"
AUDIT_SCHEMA = "sttrack-lachtt-m17-1-preexecution-audit/v1"
EXPECTED_OUTPUT_FILES = [
    "heldout_predictions.jsonl.gz",
    "manifest.json",
    "result.json",
    "training_trace.jsonl.gz",
]


class BindingError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "execution"), required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--audit", type=Path)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stat_result = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "bytes": stat_result.st_size,
        "sha256": digest.hexdigest(),
    }


def load_json(path):
    value, _ = load_json_snapshot(path)
    return value


def load_json_snapshot(path):
    path = Path(path).resolve()
    chunks = []
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stat_result = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            chunks.append(block)
    payload = b"".join(chunks)
    record = {
        "path": str(path),
        "bytes": stat_result.st_size,
        "sha256": digest.hexdigest(),
    }
    return json.loads(payload.decode("utf-8")), record


def verified_json(record):
    value, actual = load_json_snapshot(record["path"])
    if (actual["path"] != str(Path(record["path"]).resolve()) or
            actual["sha256"] != record["sha256"] or
            ("bytes" in record and
             actual["bytes"] != int(record["bytes"]))):
        raise BindingError("JSON source drifted: %s" % record["path"])
    return value


def verified_jsonl(record):
    path = Path(record["path"]).resolve()
    chunks = []
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stat_result = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            chunks.append(block)
    actual = {
        "path": str(path),
        "bytes": stat_result.st_size,
        "sha256": digest.hexdigest(),
    }
    if (actual["path"] != str(path) or
            actual["sha256"] != record["sha256"] or
            ("bytes" in record and
             actual["bytes"] != int(record["bytes"]))):
        raise BindingError("JSONL source drifted: %s" % path)
    return [
        json.loads(line)
        for line in b"".join(chunks).decode("utf-8").splitlines()
    ]


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def collect_spec_file_records(value, records):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(
                value.get("sha256"), str):
            path = Path(value["path"]).resolve()
            if not path.is_file():
                raise BindingError("spec-bound source is absent: %s" % path)
            actual = file_record(path)
            if actual["sha256"] != value["sha256"]:
                raise BindingError("spec-bound source drifted: %s" % path)
            records[str(path)] = actual
        for child in value.values():
            collect_spec_file_records(child, records)
    elif isinstance(value, list):
        for child in value:
            collect_spec_file_records(child, records)


def load_collection_anchors(spec):
    anchors = {}
    manifest_records = []
    for shard in spec["frozen_inputs"]["collection_shards"]:
        root = Path(shard["root"]).resolve()
        manifest = root / "manifest.json"
        manifest_record = file_record(manifest)
        if manifest_record["sha256"] != shard["manifest_sha256"]:
            raise BindingError("collection manifest drifted")
        manifest_records.append(manifest_record)
        for row in verified_jsonl(shard["event_ledger"]):
            sequence = str(row["sequence"])
            anchor = (root / row["anchor_path"]).resolve()
            try:
                anchor.relative_to(root)
            except ValueError as error:
                raise BindingError("clip anchor escaped shard root") from error
            previous = anchors.setdefault(sequence, anchor)
            if previous != anchor:
                raise BindingError("clip anchor changed within sequence")
    return anchors, manifest_records


def load_native_anchors(spec):
    index_record = spec["frozen_inputs"]["native_anchor_index"]
    index = Path(index_record["path"])
    root = index.parent.resolve()
    records = {}
    for row in verified_jsonl(index_record):
        path = (root / row["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise BindingError("native anchor escaped index root") from error
        record = file_record(path)
        if (record["bytes"] != int(row["bytes"]) or
                record["sha256"] != row["sha256"]):
            raise BindingError("native anchor drifted: %s" % path)
        records[str(row["sequence"])] = record
    return records


def required_sequences(spec):
    ledger = verified_json(
        spec["frozen_inputs"]["m17_0_closure"]["split_ledger"])
    return sorted({str(row["sequence"]) for row in ledger["events"]})


def atomic_readonly_json(path, value):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                value, stream, ensure_ascii=False, indent=2,
                sort_keys=True, allow_nan=False
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        fsync_directory(path.parent)
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fsync_directory(path):
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(str(Path(path)), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main():
    args = parse_args()
    for name in ("spec", "runner", "destination"):
        setattr(args, name, getattr(args, name).resolve())
    spec, spec_record = load_json_snapshot(args.spec)
    if (spec.get("schema") !=
            "sttrack-lachtt-m17-r3-execution-constant-closure-spec/v1" or
            spec.get("complete") is not True):
        raise BindingError("R3 spec identity drifted")
    repo = Path(spec["repository"]["path"]).resolve()
    expected_runner = repo / "tools/run_sttrack_lachtt_m17_1_sequence_disjoint_survival.py"
    if args.runner != expected_runner or not args.runner.is_file():
        raise BindingError("runner path drifted")
    branch = git(repo, "branch", "--show-current")
    commit = git(repo, "rev-parse", "HEAD")
    if branch != spec["repository"]["branch"] or git(
            repo, "status", "--porcelain"):
        raise BindingError("repository branch or clean state drifted")
    output = Path(spec["outputs"]["m17_1_root"]).resolve()
    if output.exists():
        raise FileExistsError(output)

    code_paths = {
        args.runner,
        repo / "lib/models/sttrack/lachtt_cached_strict_router.py",
        repo / "lib/models/sttrack/lachtt_canonical_role_router.py",
        repo / "lib/models/sttrack/lachtt_independent_utility_safety.py",
        repo / "lib/models/sttrack/lachtt_learned_bounded_roi_association.py",
        repo / "lib/models/sttrack/lachtt_rich_roi_relation.py",
        repo / "lib/models/sttrack/lachtt_target_distractor_memory.py",
        repo / "tools/run_sttrack_lachtt_m17_0_target_split_closure.py",
        repo / "tools/smoke_sttrack_lachtt_m8b_cached.py",
    }
    code_records = [file_record(path) for path in sorted(code_paths)]
    sources = {}
    collect_spec_file_records(spec, sources)
    clip_anchors, collection_manifests = load_collection_anchors(spec)
    for record in collection_manifests:
        sources[record["path"]] = record
    for path in code_paths:
        sources.pop(str(path.resolve()), None)
    sources.pop(str(args.spec), None)
    source_records = [sources[name] for name in sorted(sources)]
    sequences = required_sequences(spec)
    native_anchors = load_native_anchors(spec)
    if (set(sequences) != set(clip_anchors) or
            not set(sequences).issubset(native_anchors)):
        raise BindingError("required anchor sequence closure drifted")
    clip_records = [file_record(clip_anchors[name]) for name in sequences]
    native_records = [native_anchors[name] for name in sequences]
    if len(clip_records) != 134 or len(native_records) != 134:
        raise BindingError("anchor payload count drifted")

    binding = {
        "schema": PREFLIGHT_SCHEMA,
        "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "m17_1_execution_authorized": False,
        "claim_ceiling": spec["claim_ceiling"],
        "spec": spec_record,
        "runner": file_record(args.runner),
        "repository": {
            "path": str(repo),
            "branch": branch,
            "commit": commit,
            "clean": True,
        },
        "output": {
            "path": str(output),
            "absent_at_binding": True,
            "expected_files": EXPECTED_OUTPUT_FILES,
        },
        "code_records": code_records,
        "source_records": source_records,
        "clip_anchor_payloads": clip_records,
        "native_anchor_payloads": native_records,
        "counts": {
            "code_records": len(code_records),
            "source_records": len(source_records),
            "clip_anchor_payloads": len(clip_records),
            "native_anchor_payloads": len(native_records),
        },
        "authorization": {
            "independent_preexecution_audit": True,
            "m17_1_execution": False,
            "tracking_checkpoint": False,
            "depthtrack_test": False,
            "cdtb": False,
            "vot_low22": False,
            "vot_full127": False,
            "qwen": False,
            "automatic_next_stage": False,
        },
    }

    if args.mode == "execution":
        if args.preflight is None or args.audit is None:
            raise BindingError("execution mode requires preflight and audit")
        preflight_path = args.preflight.resolve()
        audit_path = args.audit.resolve()
        preflight, preflight_record = load_json_snapshot(preflight_path)
        audit, audit_record = load_json_snapshot(audit_path)
        if (preflight.get("schema") != PREFLIGHT_SCHEMA or
                preflight.get("complete") is not True or
                preflight.get("m17_1_execution_authorized") is not False):
            raise BindingError("preflight identity drifted")
        for key in (
                "spec", "runner", "repository", "output", "code_records",
                "source_records", "clip_anchor_payloads",
                "native_anchor_payloads"):
            if preflight.get(key) != binding.get(key):
                raise BindingError("post-audit dependency drifted: %s" % key)
        if (audit.get("schema") != AUDIT_SCHEMA or
                str(audit.get("overall_verdict", "")).upper() != "PASS" or
                str(audit.get("integrity_verdict", "")).upper() != "PASS" or
                audit.get("authorization", {}).get("m17_1_execution") is not True):
            raise BindingError("preexecution audit did not authorize execution")
        identity = audit.get("audited_identity", {})
        if (identity.get("spec_sha256") != binding["spec"]["sha256"] or
                identity.get("runner_sha256") != binding["runner"]["sha256"] or
                identity.get("repository_commit") != commit or
                identity.get("preflight_binding_sha256") !=
                preflight_record["sha256"]):
            raise BindingError("audit identity does not match current closure")
        binding.update({
            "schema": EXECUTION_SCHEMA,
            "m17_1_execution_authorized": True,
            "preflight_binding": preflight_record,
            "preexecution_audit": audit_record,
        })
        binding["authorization"]["m17_1_execution"] = True
        binding["authorization"]["independent_preexecution_audit"] = False
    elif args.preflight is not None or args.audit is not None:
        raise BindingError("preflight mode cannot accept audit inputs")

    atomic_readonly_json(args.destination, binding)
    print(json.dumps(file_record(args.destination), sort_keys=True))


if __name__ == "__main__":
    main()
