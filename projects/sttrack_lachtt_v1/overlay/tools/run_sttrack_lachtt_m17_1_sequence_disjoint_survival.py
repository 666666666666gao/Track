#!/usr/bin/env python3
"""Run the single preregistered M17-1 sequence-disjoint Train-only pilot.

The executable is deliberately fail closed.  It requires a post-audit
execution binding, never opens held-out numeric targets before optimization is
complete, computes held-out predictions before targets are opened, and writes
no model checkpoint.
"""

import argparse
import atexit
from collections import Counter, defaultdict
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GIT_EXECUTABLE = Path(shutil.which("git") or "/usr/bin/git").resolve()
BRANCH_ORDER = (
    "current_peak0",
    "current_peak1",
    "last_reliable_peak0",
    "last_reliable_peak1",
    "velocity_peak0",
    "velocity_peak1",
)
HORIZONS = (3, 5, 10)
TRAJECTORY_METRICS = (
    "branch_mean_iou",
    "public_mean_iou",
    "gain",
    "low_overlap_fraction",
    "trailing_low_run_fraction",
)
CanonicalRoleIndependentUtilitySafetyRouter = None
build_detached_roi_differences = None
cached_strict_router_loss = None
canonical_rows_sha256 = None
gradient_diagnostics = None
recompute_event_class = None
recompute_label = None
scale_gradients = None
state_digest = None
trajectory_metrics = None
torch = None
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
EXPECTED_OUTPUT_FILES = (
    "heldout_predictions.jsonl.gz",
    "manifest.json",
    "result.json",
    "training_trace.jsonl.gz",
)
EXECUTION_BINDING_SCHEMA = \
    "sttrack-lachtt-m17-1-postaudit-execution-binding/v1"
PREAUDIT_SCHEMA = "sttrack-lachtt-m17-1-preexecution-audit/v1"


class ContractError(RuntimeError):
    pass


class RuntimeSideEffectObserver:
    """Record process writes, subprocesses and network connects after auth."""

    CHECKPOINT_SUFFIXES = (
        ".ckpt", ".onnx", ".pth", ".pt", ".safetensors",
    )

    def __init__(self):
        self.write_paths = []
        self.filesystem_events = []
        self.subprocesses = []
        self.network_connections = []

    def record_filesystem_event(self, event, *paths):
        values = [
            str(path) for path in paths
            if isinstance(path, (str, bytes, os.PathLike))
        ]
        self.filesystem_events.append({"event": event, "paths": values})
        self.write_paths.extend(values)

    def audit_hook(self, event, arguments):
        if event == "open" and arguments:
            path = arguments[0]
            mode = arguments[1] if len(arguments) > 1 else None
            flags = arguments[2] if len(arguments) > 2 else 0
            writable_mode = isinstance(mode, str) and any(
                token in mode for token in ("a", "w", "x", "+"))
            writable_flags = isinstance(flags, int) and bool(
                flags & (os.O_APPEND | os.O_CREAT | os.O_RDWR |
                         os.O_TRUNC | os.O_WRONLY))
            if (writable_mode or writable_flags) and isinstance(
                    path, (str, bytes, os.PathLike)):
                self.record_filesystem_event(event, path)
        elif event == "subprocess.Popen":
            executable = arguments[0] if arguments else None
            command = arguments[1] if len(arguments) > 1 else None
            self.subprocesses.append({
                "executable": str(executable),
                "command": [str(value) for value in command]
                if isinstance(command, (list, tuple)) else str(command),
            })
        elif event == "socket.connect":
            address = arguments[1] if len(arguments) > 1 else None
            self.network_connections.append(repr(address))
        elif event in ("os.rename", "os.replace") and len(arguments) >= 2:
            self.record_filesystem_event(event, arguments[0], arguments[1])
        elif event in ("os.remove", "os.rmdir") and arguments:
            self.record_filesystem_event(event, arguments[0])
        elif event in ("os.mkdir", "os.chmod", "os.chown", "os.truncate",
                       "os.utime") and arguments:
            self.record_filesystem_event(event, arguments[0])
        elif event in ("os.link", "os.symlink") and len(arguments) >= 2:
            self.record_filesystem_event(event, arguments[0], arguments[1])
        elif event in ("shutil.copyfile", "shutil.copymode",
                       "shutil.copystat") and len(arguments) >= 2:
            self.record_filesystem_event(event, arguments[0], arguments[1])

    def install(self):
        sys.addaudithook(self.audit_hook)

    def unexpected_subprocesses(self):
        prefix = (str(GIT_EXECUTABLE), "-C", str(REPOSITORY_ROOT))
        allowed = {
            prefix + ("rev-parse", "HEAD"),
            prefix + ("branch", "--show-current"),
            prefix + ("status", "--porcelain"),
        }
        unexpected = []
        for row in self.subprocesses:
            command = row["command"]
            if (not isinstance(command, list) or tuple(command) not in allowed or
                    Path(row["executable"]).resolve() != GIT_EXECUTABLE):
                unexpected.append(row)
        return unexpected

    def checkpoint_write_paths(self):
        return sorted({
            path for path in self.write_paths
            if path.lower().endswith(self.CHECKPOINT_SUFFIXES)
        })

    def forbidden_write_paths(self, allowed_roots=()):
        resolved_roots = tuple(Path(root).resolve() for root in allowed_roots)
        forbidden = []
        for value in self.write_paths:
            path = Path(value)
            try:
                resolved = path.resolve()
            except (OSError, RuntimeError):
                forbidden.append(value)
                continue
            allowed = False
            for root in resolved_roots:
                try:
                    resolved.relative_to(root)
                    allowed = True
                    break
                except ValueError:
                    continue
            if not allowed:
                forbidden.append(value)
        return sorted(set(forbidden))


def load_project_components():
    """Import project code only after execution authorization is verified."""
    global CanonicalRoleIndependentUtilitySafetyRouter
    global build_detached_roi_differences
    global cached_strict_router_loss
    global canonical_rows_sha256
    global gradient_diagnostics
    global recompute_event_class
    global recompute_label
    global scale_gradients
    global state_digest
    global torch
    global trajectory_metrics
    if CanonicalRoleIndependentUtilitySafetyRouter is not None:
        return
    sys.dont_write_bytecode = True
    import torch as torch_module
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from lib.models.sttrack.lachtt_cached_strict_router import (
        cached_strict_router_loss as strict_loss,
    )
    from lib.models.sttrack.lachtt_canonical_role_router import (
        CanonicalRoleIndependentUtilitySafetyRouter as router_class,
    )
    from lib.models.sttrack.lachtt_independent_utility_safety import (
        HORIZONS as project_horizons,
        TRAJECTORY_METRICS as project_metrics,
    )
    from lib.models.sttrack.lachtt_learned_bounded_roi_association import (
        build_detached_roi_differences as relation_builder,
    )
    from tools.run_sttrack_lachtt_m17_0_target_split_closure import (
        canonical_rows_sha256 as rows_sha256,
        recompute_event_class as event_class_builder,
        recompute_label as label_builder,
        trajectory_metrics as target_builder,
    )
    from tools.smoke_sttrack_lachtt_m8b_cached import (
        gradient_diagnostics as gradient_probe,
        scale_gradients as gradient_scaler,
        state_digest as model_state_digest,
    )
    if (tuple(project_horizons) != HORIZONS or
            tuple(project_metrics) != TRAJECTORY_METRICS):
        raise ContractError("project target-axis constants drifted")
    CanonicalRoleIndependentUtilitySafetyRouter = router_class
    build_detached_roi_differences = relation_builder
    cached_strict_router_loss = strict_loss
    canonical_rows_sha256 = rows_sha256
    gradient_diagnostics = gradient_probe
    recompute_event_class = event_class_builder
    recompute_label = label_builder
    scale_gradients = gradient_scaler
    state_digest = model_state_digest
    trajectory_metrics = target_builder
    torch = torch_module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--preaudit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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


def verified_file_bytes(record):
    path = Path(record["path"]).resolve()
    chunks = []
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stat_result = os.fstat(stream.fileno())
        if ("bytes" in record and
                stat_result.st_size != int(record["bytes"])):
            raise ContractError("source byte count drifted: %s" % path)
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            chunks.append(block)
    if digest.hexdigest() != record["sha256"]:
        raise ContractError("source hash drifted: %s" % path)
    return b"".join(chunks)


def load_verified_json(record):
    return json.loads(verified_file_bytes(record).decode("utf-8"))


def load_verified_jsonl(record, compressed=False):
    payload = verified_file_bytes(record)
    if compressed:
        payload = gzip.decompress(payload)
    return [json.loads(line) for line in payload.decode("utf-8").splitlines()]


def git_output(*arguments):
    return subprocess.check_output(
        [str(GIT_EXECUTABLE), "-C", str(REPOSITORY_ROOT), *arguments],
        text=True).strip()


def validate_file_record(record):
    path = Path(record["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise ContractError("source byte count drifted: %s" % path)
    if sha256_file(path) != record["sha256"]:
        raise ContractError("source hash drifted: %s" % path)
    return path


def finite_tensor(value):
    return isinstance(value, torch.Tensor) and bool(
        torch.isfinite(value.float()).all().item())


def verified_torch_load(record):
    """Hash and deserialize the same open file descriptor."""
    path = Path(record["path"]).resolve()
    with path.open("rb") as stream:
        stat_result = os.fstat(stream.fileno())
        if stat_result.st_size != int(record["bytes"]):
            raise ContractError("torch payload byte count drifted: %s" % path)
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != record["sha256"]:
            raise ContractError("torch payload hash drifted: %s" % path)
        stream.seek(0)
        return torch.load(stream, map_location="cpu", weights_only=True)


def event_key(row):
    return (str(row["sequence"]), int(row["event_id"]),
            int(row["trigger_frame"]))


def action_key(row):
    return event_key(row) + (str(row["branch_id"]),)


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
        with open(temporary, "wb") as raw_stream:
            with gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw_stream,
                    mtime=0) as compressed:
                for row in rows:
                    payload = json.dumps(
                        row, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"), allow_nan=False)
                    compressed.write((payload + "\n").encode("utf-8"))
            raw_stream.flush()
            os.fsync(raw_stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fsync_directory(path):
    """Persist directory entries around the final atomic publication."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(str(Path(path)), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_spec(spec, args):
    if (spec.get("schema") !=
            "sttrack-lachtt-m17-r3-execution-constant-closure-spec/v1" or
            spec.get("complete") is not True or
            spec.get("created_before_implementation") is not True or
            spec.get("created_before_execution") is not True):
        raise ContractError("M17-R3 spec identity or completeness drifted")
    if REPOSITORY_ROOT != Path(spec["repository"]["path"]).resolve():
        raise ContractError("repository absolute path drifted")
    expected_output = Path(spec["outputs"]["m17_1_root"]).resolve()
    if args.output != expected_output or args.output != Path(
            spec["outputs"]["root"]).resolve():
        raise ContractError("M17-1 output root drifted")
    if args.output.exists():
        raise FileExistsError(args.output)
    training = spec["training"]
    if (training.get("device") != "cpu" or
            training.get("dtype") != "float32" or
            float(training["loss"]["pairwise_margin"]) != 0.1):
        raise ContractError("R3 execution constants drifted")
    authorization = spec["authorization"]
    if authorization.get("m17_1_implementation") is not True:
        raise ContractError("M17-1 implementation is not authorized")
    for name in (
            "m17_1_execution", "full_six_fold_oof", "online_replay",
            "tracking_checkpoint", "depthtrack_test", "cdtb",
            "vot_low22", "vot_full127", "qwen", "automatic_next_stage"):
        if authorization.get(name) is not False:
            raise ContractError("unsafe spec authorization: %s" % name)


def collect_required_spec_records(value, records):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(
                value.get("sha256"), str):
            path = Path(value["path"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            records[path] = {
                "path": str(path),
                "sha256": value["sha256"],
            }
        for child in value.values():
            collect_required_spec_records(child, records)
    elif isinstance(value, list):
        for child in value:
            collect_required_spec_records(child, records)


def bound_record_map(binding):
    records = {}
    for record in (
            *binding.get("code_records", []),
            *binding.get("source_records", []),
            *binding.get("clip_anchor_payloads", []),
            *binding.get("native_anchor_payloads", [])):
        path = Path(record["path"]).resolve()
        previous = records.setdefault(path, record)
        if previous != record:
            raise ContractError("binding contains conflicting file records")
    return records


def validate_execution_binding(args, spec, spec_record):
    binding, binding_record = load_json_snapshot(args.binding)
    runner = Path(__file__).resolve()
    runner_record = file_record(runner)
    commit = git_output("rev-parse", "HEAD")
    branch = git_output("branch", "--show-current")
    clean = not git_output("status", "--porcelain")
    if (binding.get("schema") != EXECUTION_BINDING_SCHEMA or
            binding.get("complete") is not True or
            binding.get("m17_1_execution_authorized") is not True):
        raise ContractError("post-audit execution binding is absent")
    expected = {
        "spec": spec_record,
        "runner": runner_record,
        "repository": {
            "path": str(REPOSITORY_ROOT),
            "branch": spec["repository"]["branch"],
            "commit": commit,
            "clean": True,
        },
        "output": {
            "path": str(args.output),
            "absent_at_binding": True,
            "expected_files": list(EXPECTED_OUTPUT_FILES),
        },
    }
    for name, value in expected.items():
        if binding.get(name) != value:
            raise ContractError("execution binding mismatch: %s" % name)
    if branch != spec["repository"]["branch"] or not clean:
        raise ContractError("repository branch/clean state drifted")
    audit, audit_record = load_json_snapshot(args.preaudit)
    if binding.get("preexecution_audit") != audit_record:
        raise ContractError("preexecution audit file identity drifted")
    if (
            audit.get("schema") != PREAUDIT_SCHEMA or
            str(audit.get("overall_verdict", "")).upper() != "PASS" or
            str(audit.get("integrity_verdict", "")).upper() != "PASS" or
            audit.get("authorization", {}).get(
                "m17_1_execution") is not True):
        raise ContractError("independent preexecution audit did not authorize")
    audited = audit.get("audited_identity", {})
    if (audited.get("spec_sha256") != spec_record["sha256"] or
            audited.get("runner_sha256") != runner_record["sha256"] or
            audited.get("repository_commit") != commit or
            audited.get("preflight_binding_sha256") != binding.get(
                "preflight_binding", {}).get("sha256")):
        raise ContractError("preexecution audit identity drifted")
    preflight_path = Path(binding["preflight_binding"]["path"]).resolve()
    preflight, preflight_record = load_json_snapshot(preflight_path)
    if preflight_record != binding["preflight_binding"]:
        raise ContractError("preflight binding file identity drifted")
    if (preflight.get("schema") !=
            "sttrack-lachtt-m17-1-preflight-binding/v1" or
            preflight.get("complete") is not True or
            preflight.get("m17_1_execution_authorized") is not False):
        raise ContractError("audited preflight binding identity drifted")
    for name in (
            "spec", "runner", "repository", "output", "code_records",
            "source_records", "clip_anchor_payloads",
            "native_anchor_payloads"):
        if binding.get(name) != preflight.get(name):
            raise ContractError(
                "execution/preflight dependency mismatch: %s" % name)
    code_records = binding.get("code_records", [])
    source_records = binding.get("source_records", [])
    if not code_records or not source_records:
        raise ContractError("execution binding dependency closure is empty")
    for record in (*code_records, *source_records,
                   *binding.get("clip_anchor_payloads", []),
                   *binding.get("native_anchor_payloads", [])):
        validate_file_record(record)
    bound = bound_record_map(binding)
    required = {}
    collect_required_spec_records(spec, required)
    for shard in spec["frozen_inputs"]["collection_shards"]:
        path = Path(shard["root"]).resolve() / "manifest.json"
        required[path] = {
            "path": str(path), "sha256": shard["manifest_sha256"]}
    for path, required_record in required.items():
        record = bound.get(path)
        if record is None or record["sha256"] != required_record["sha256"]:
            raise ContractError("required spec source is not bound: %s" % path)
    if args.output.exists():
        raise FileExistsError(args.output)
    runtime_identity = {
        "spec": spec_record,
        "binding": binding_record,
        "preexecution_audit": audit_record,
        "preflight_binding": preflight_record,
        "runner": runner_record,
        "repository_commit": commit,
        "repository_branch": branch,
    }
    return binding, audit, commit, runtime_identity


def validate_runtime_identity(args, spec, binding, runtime_identity,
                              observer, additional_records=(),
                              allowed_write_roots=()):
    commit = git_output("rev-parse", "HEAD")
    branch = git_output("branch", "--show-current")
    status = git_output("status", "--porcelain")
    if (commit != runtime_identity["repository_commit"] or
            branch != runtime_identity["repository_branch"] or
            branch != spec["repository"]["branch"] or status):
        raise ContractError("repository identity drifted during execution")
    control_names = (
        "spec", "binding", "preexecution_audit",
        "preflight_binding", "runner",
    )
    for name in control_names:
        record = runtime_identity[name]
        if file_record(record["path"]) != record:
            raise ContractError("control file drifted during execution: %s" % name)
    records = [
        *binding["code_records"], *binding["source_records"],
        *binding["clip_anchor_payloads"],
        *binding["native_anchor_payloads"], *additional_records,
    ]
    mismatches = source_recheck(records)
    if mismatches:
        raise ContractError("bound source drifted during execution")
    if args.output.exists():
        raise FileExistsError(args.output)
    forbidden_writes = observer.forbidden_write_paths(allowed_write_roots)
    unexpected_subprocesses = observer.unexpected_subprocesses()
    qwen_modules = sorted(
        name for name in sys.modules if "qwen" in name.lower())
    tracker_modules = sorted(
        name for name in sys.modules
        if name.startswith("lib.test.tracker"))
    benchmark_modules = sorted(
        name for name in sys.modules
        if (name == "vot" or name.startswith("vot.") or
            "depthtrack" in name.lower() or "cdtb" in name.lower()))
    if (forbidden_writes or unexpected_subprocesses or
            observer.network_connections or qwen_modules or
            tracker_modules or benchmark_modules or
            observer.checkpoint_write_paths()):
        raise ContractError("forbidden runtime side effect observed")
    return {
        "forbidden_write_paths": forbidden_writes,
        "authorized_filesystem_events": list(observer.filesystem_events),
        "checkpoint_write_paths": observer.checkpoint_write_paths(),
        "network_connections": list(observer.network_connections),
        "unexpected_subprocesses": unexpected_subprocesses,
        "qwen_modules": qwen_modules,
        "public_tracker_modules": tracker_modules,
        "public_benchmark_modules": benchmark_modules,
        "git_subprocesses": len(observer.subprocesses),
        "bound_source_mismatches": mismatches,
    }


def validate_frozen_receipts(spec):
    frozen = spec["frozen_inputs"]
    closure = frozen["m17_0_closure"]
    for name in ("result", "manifest", "split_ledger",
                 "trajectory_targets", "result_audit"):
        record = closure[name]
        validate_file_record(record)
    result = load_verified_json(closure["result"])
    result_audit = load_verified_json(closure["result_audit"])
    native_manifest = load_verified_json(frozen["native_anchor_manifest"])
    if (result.get("accepted") is not True or
            str(result_audit.get("overall_verdict", "")).upper() != "PASS" or
            native_manifest.get("accepted") is not True or
            native_manifest.get("future_frame_opened") is not False or
            native_manifest.get("future_ground_truth_opened") is not False or
            int(native_manifest.get("ground_truth_rows_read_per_sequence")) != 1):
        raise ContractError("frozen receipt boundary drifted")
    if not result.get("conditions") or not all(
            value is True for value in result["conditions"].values()):
        raise ContractError("M17-0 did not pass every frozen condition")
    return result


def ensure_within(root, path):
    root = Path(root).resolve()
    path = Path(path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ContractError("payload path escaped shard root") from error
    return path


def load_collection_index(spec):
    records = {}
    sequence_anchors = {}
    total = 0
    for shard in spec["frozen_inputs"]["collection_shards"]:
        root = Path(shard["root"]).resolve()
        ledger_record = shard["event_ledger"]
        validate_file_record(ledger_record)
        manifest_path = root / "manifest.json"
        manifest_record = {
            "path": str(manifest_path),
            "sha256": shard["manifest_sha256"],
        }
        manifest = load_verified_json(manifest_record)
        if (manifest.get("accepted") is not True or
                manifest.get("future_ground_truth_opened") is not False or
                manifest.get("ground_truth_used_after_initialization") is not False or
                manifest.get("metric_computed") is not False):
            raise ContractError("collection safety receipt drifted")
        rows = 0
        for row in load_verified_jsonl(ledger_record):
            key = event_key(row)
            if key in records:
                raise ContractError("duplicate collection event")
            feature_path = ensure_within(root, root / row["feature_path"])
            anchor_path = ensure_within(root, root / row["anchor_path"])
            trajectory = row.get("trajectory", [])
            if (len(trajectory) < 5 or any(
                    tuple(branch["name"] for branch in age["branches"]) !=
                    BRANCH_ORDER for age in trajectory[:5])):
                raise ContractError("collection candidate order drifted")
            records[key] = {
                "sequence": key[0],
                "event_id": key[1],
                "trigger_frame": key[2],
                "feature_path": feature_path,
                "feature_bytes": int(row["feature_bytes"]),
                "feature_sha256": row["feature_sha256"],
                "anchor_path": anchor_path,
            }
            previous = sequence_anchors.setdefault(key[0], anchor_path)
            if previous != anchor_path:
                raise ContractError("sequence anchor path drifted")
            rows += 1
        if rows != int(shard["event_ledger"]["rows"]):
            raise ContractError("collection event row count drifted")
        total += rows
    if total != int(spec["frozen_inputs"]["counts"]["events"]):
        raise ContractError("collection total event count drifted")
    return records, sequence_anchors


def load_split_ledger(spec):
    record = spec["frozen_inputs"]["m17_0_closure"]["split_ledger"]
    validate_file_record(record)
    ledger = load_verified_json(record)
    entries = {}
    for row in ledger["events"]:
        key = event_key(row)
        if key in entries:
            raise ContractError("duplicate split event")
        entries[key] = row
    if len(entries) != int(spec["frozen_inputs"]["counts"]["events"]):
        raise ContractError("split event count drifted")
    storage_contract = ledger.get("target_storage_contract")
    heldout_commitment = ledger.get("target_commitments", {}).get("heldout", {})
    expected_heldout = spec["frozen_inputs"]["m17_0_closure"][
        "trajectory_targets"]
    if (not isinstance(storage_contract, str) or
            "heldout numeric targets are omitted" not in storage_contract or
            heldout_commitment.get("canonical_jsonl_sha256") !=
            expected_heldout["heldout_target_commitment"] or
            int(heldout_commitment.get("action_rows", -1)) != int(
                spec["sequence_split"]["heldout"]["available_actions"]) or
            int(ledger.get("summary", {}).get(
                "serialized_training_target_rows", -1)) != int(
                    spec["sequence_split"]["training"]["available_actions"])):
        raise ContractError("held-out target storage contract drifted")
    return ledger, entries


def validate_target_value(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or \
            not math.isfinite(float(value)):
        raise ContractError("non-finite target: %s" % name)
    value = float(value)
    if name == "gain":
        if value < -1.0 or value > 1.0:
            raise ContractError("gain target outside [-1,1]")
    elif value < 0.0 or value > 1.0:
        raise ContractError("bounded target outside [0,1]")
    return value


def validate_action_target(row, expected_partition):
    if (row.get("record_type") != "action_target" or
            row.get("partition") != expected_partition or
            int(row["candidate_role_id"]) not in range(6) or
            row["branch_id"] != BRANCH_ORDER[int(row["candidate_role_id"])] or
            row["strict_label"] not in
            ("beneficial", "catastrophic", "neutral")):
        raise ContractError("action target contract drifted")
    targets = row.get("targets")
    if set(targets) != {str(value) for value in HORIZONS}:
        raise ContractError("target horizon key set drifted")
    for horizon in (str(value) for value in HORIZONS):
        if set(targets[horizon]) != set(TRAJECTORY_METRICS):
            raise ContractError("target metric key set drifted")
        for name, value in targets[horizon].items():
            validate_target_value(name, value)


def load_training_targets(spec, split_entries):
    record = spec["frozen_inputs"]["m17_0_closure"]["trajectory_targets"]
    validate_file_record(record)
    groups = defaultdict(list)
    record_counts = Counter()
    heldout_numeric_rows = 0
    heldout_commitment = None
    for row in load_verified_jsonl(record, compressed=True):
        record_counts[row["record_type"]] += 1
        if row["record_type"] == "action_target":
            if row.get("partition") != "training":
                heldout_numeric_rows += 1
                continue
            validate_action_target(row, "training")
            key = event_key(row)
            split = split_entries.get(key)
            if (split is None or split["partition"] != "training" or
                    split["strict_h10_available"] is not True):
                raise ContractError("training target split drifted")
            groups[key].append(row)
        elif row["record_type"] == "heldout_target_commitment":
            heldout_commitment = row
    if (heldout_numeric_rows != 0 or len(groups) != int(
            spec["sequence_split"]["training"]["available_events"]) or
            sum(len(value) for value in groups.values()) != int(
                spec["sequence_split"]["training"]["available_actions"])):
        raise ContractError("training target partition count drifted")
    expected_commitment = record["heldout_target_commitment"]
    if (heldout_commitment is None or
            heldout_commitment["canonical_jsonl_sha256"] != expected_commitment or
            int(heldout_commitment["action_rows"]) != int(
                spec["sequence_split"]["heldout"]["available_actions"]) or
            heldout_commitment["numeric_targets_serialized"] is not False):
        raise ContractError("held-out target commitment record drifted")
    for key, rows in groups.items():
        rows.sort(key=lambda row: int(row["candidate_role_id"]))
        if (tuple(row["branch_id"] for row in rows) != BRANCH_ORDER or
                len({row["strict_event_class"] for row in rows}) != 1):
            raise ContractError("training event target axis drifted")
    return dict(groups), heldout_commitment, dict(record_counts)


def load_native_index(spec):
    frozen = spec["frozen_inputs"]
    record = frozen["native_anchor_index"]
    path = validate_file_record(record)
    root = path.parent
    records = {}
    for row in load_verified_jsonl(record):
        sequence = str(row["sequence"])
        if sequence in records:
            raise ContractError("duplicate native anchor sequence")
        records[sequence] = {
            "path": ensure_within(root, root / row["path"]),
            "bytes": int(row["bytes"]),
            "sha256": row["sha256"],
        }
    if len(records) != int(frozen["native_anchor_index"]["rows"]):
        raise ContractError("native anchor index count drifted")
    return records


def validate_anchor_binding(binding, sequence_anchors, native_index,
                            required_sequences):
    clip = {Path(row["path"]).resolve(): row
            for row in binding["clip_anchor_payloads"]}
    native = {Path(row["path"]).resolve(): row
              for row in binding["native_anchor_payloads"]}
    if len(clip) != len(required_sequences) or \
            len(native) != len(required_sequences):
        raise ContractError("bound anchor payload count drifted")
    for sequence in required_sequences:
        clip_path = sequence_anchors.get(sequence)
        native_row = native_index.get(sequence)
        if clip_path is None or native_row is None:
            raise ContractError("required sequence anchor is missing")
        if clip_path not in clip:
            raise ContractError("clip anchor is not bound")
        if native_row["path"] not in native:
            raise ContractError("native anchor is not bound")
        bound_native = native[native_row["path"]]
        if (Path(bound_native["path"]).resolve() != native_row["path"] or
                int(bound_native["bytes"]) != int(native_row["bytes"]) or
                bound_native["sha256"] != native_row["sha256"]):
            raise ContractError("native anchor binding/index identity drifted")
    return clip, native


def load_feature_payload(record, loaded_feature_records):
    path = record["feature_path"]
    payload_record = {
        "path": str(path),
        "bytes": int(record["feature_bytes"]),
        "sha256": record["feature_sha256"],
    }
    payload = verified_torch_load(payload_record)
    if set(payload) != set(FEATURE_SHAPES):
        raise ContractError("feature payload keys drifted")
    for name, shape in FEATURE_SHAPES.items():
        if tuple(payload[name].shape) != shape or not finite_tensor(payload[name]):
            raise ContractError("feature payload tensor drifted: %s" % name)
    loaded_feature_records[str(path)] = {
        "path": str(path), "bytes": path.stat().st_size,
        "sha256": record["feature_sha256"],
    }
    return payload


def load_clip_anchor(record, cache):
    path = Path(record["path"]).resolve()
    if path not in cache:
        value = verified_torch_load(record)
        if set(value) != {"initial_image", "identity_text"}:
            raise ContractError("clip anchor keys drifted")
        for name in ("initial_image", "identity_text"):
            if tuple(value[name].shape) != (1, 768) or not finite_tensor(value[name]):
                raise ContractError("clip anchor tensor drifted")
        cache[path] = value
    return cache[path]


def load_native_anchor(record, cache):
    path = record["path"]
    if path not in cache:
        value = verified_torch_load(record)
        expected_shapes = {
            "native_template_rgb_tokens": (64, 768),
            "native_template_depth_tokens": (64, 768),
            "native_template_rgb_mean": (768,),
            "native_template_depth_mean": (768,),
        }
        if set(value) != set(expected_shapes):
            raise ContractError("native anchor keys drifted")
        for name, shape in expected_shapes.items():
            if tuple(value[name].shape) != shape or not finite_tensor(value[name]):
                raise ContractError("native anchor tensor drifted")
        cache[path] = value
    return cache[path]


def relation_for_event(spec, collection, sequence_anchors, native_index,
                       clip_binding, key, clip_cache, native_cache,
                       loaded_feature_records):
    record = collection[key]
    features = load_feature_payload(record, loaded_feature_records)
    clip = load_clip_anchor(
        clip_binding[sequence_anchors[key[0]]], clip_cache)
    native = load_native_anchor(native_index[key[0]], native_cache)
    builder = spec["architecture"]["relation_builder_parameters"]
    differences, gates, scalar = build_detached_roi_differences(
        {name: value.unsqueeze(0) for name, value in features.items()},
        clip["initial_image"].unsqueeze(0),
        clip["identity_text"].unsqueeze(0),
        native["native_template_rgb_tokens"].unsqueeze(0),
        native["native_template_depth_tokens"].unsqueeze(0),
        ema_alpha=float(builder["ema_alpha"]),
        epsilon=float(builder["l2_epsilon"]),
        soft_distractor_scale=float(builder["soft_distractor_scale"]),
        native_anchor_top_k=int(builder["native_anchor_top_k"]),
        depth_missing_floor=float(builder["depth_missing_floor"]),
    )
    return (differences[0].contiguous(), gates[0].contiguous(),
            scalar[0].contiguous())


def stable_digest_int(*items):
    payload = "\0".join(str(item) for item in items).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def balanced_event_batches(target_groups, spec, epoch):
    composition = spec["training"]["event_batch_composition"]
    pools = defaultdict(list)
    for key, rows in target_groups.items():
        event_class = rows[0]["strict_event_class"]
        pools[event_class].append(key)
    if set(pools) != set(composition):
        raise ContractError("training event class pools drifted")
    seed = int(spec["training"]["seed"])
    for name in pools:
        pools[name].sort(key=lambda key: (
            stable_digest_int(seed, epoch, name, *key), key))
    offsets = {name: 0 for name in pools}
    batches = []
    for step in range(int(spec["training"]["steps_per_epoch"])):
        batch = []
        for name in sorted(composition):
            for _ in range(int(composition[name])):
                pool = pools[name]
                batch.append(pool[offsets[name] % len(pool)])
                offsets[name] += 1
        batch.sort(key=lambda key: (
            stable_digest_int(seed, epoch, step, "batch", *key), key))
        batches.append(batch)
    return batches, {name: len(value) for name, value in pools.items()}


ALL_CANDIDATE_PERMUTATIONS = tuple(itertools.permutations(range(6)))


def training_permutation(seed, epoch, step, batch_index, key):
    index = stable_digest_int(
        seed, epoch, step, batch_index, *key) % len(ALL_CANDIDATE_PERMUTATIONS)
    return torch.tensor(ALL_CANDIDATE_PERMUTATIONS[index], dtype=torch.int64)


def target_tensors(rows):
    trajectory = []
    gain = []
    beneficial = []
    catastrophic = []
    for row in rows:
        trajectory.append([[float(row["targets"][str(horizon)][metric])
                            for metric in TRAJECTORY_METRICS]
                           for horizon in HORIZONS])
        gain.append(float(row["targets"]["10"]["gain"]))
        beneficial.append(row["strict_label"] == "beneficial")
        catastrophic.append(row["strict_label"] == "catastrophic")
    return {
        "trajectory": torch.tensor(trajectory, dtype=torch.float32),
        "gain": torch.tensor(gain, dtype=torch.float32),
        "beneficial": torch.tensor(beneficial, dtype=torch.bool),
        "catastrophic": torch.tensor(catastrophic, dtype=torch.bool),
    }


def make_training_batch(keys, relations, targets, spec, epoch, step):
    differences, gates, scalar = [], [], []
    role_ids, trajectory, gain, benefit, catastrophe = [], [], [], [], []
    event_target = []
    seed = int(spec["training"]["seed"])
    for batch_index, key in enumerate(keys):
        permutation = training_permutation(
            seed, epoch, step, batch_index, key)
        relation = relations[key]
        target = targets[key]
        differences.append(relation[0][:, permutation])
        gates.append(relation[1][:, permutation])
        scalar.append(relation[2][:, permutation])
        role_ids.append(permutation)
        trajectory.append(target["trajectory"][permutation])
        gain.append(target["gain"][permutation])
        benefit.append(target["beneficial"][permutation])
        catastrophe.append(target["catastrophic"][permutation])
        event_target.append(
            targets[key]["event_class"] == "beneficial")
    batch_size = len(keys)
    return {
        "differences": torch.stack(differences),
        "gates": torch.stack(gates),
        "scalar": torch.stack(scalar),
        "role_ids": torch.stack(role_ids),
        "trajectory": torch.stack(trajectory),
        "gain": torch.stack(gain),
        "beneficial": torch.stack(benefit),
        "catastrophic": torch.stack(catastrophe),
        "event_target": torch.tensor(event_target, dtype=torch.bool),
        "candidate_valid": torch.ones(batch_size, 6, dtype=torch.bool),
        "label_available": torch.ones(batch_size, 6, dtype=torch.bool),
    }


def forward_losses(model, batch, spec):
    outputs = model(
        batch["differences"], batch["gates"], batch["scalar"],
        batch["candidate_valid"], batch["role_ids"])
    strict_outputs = {
        "event_commit_logit": outputs["event_commit_logit"],
        "candidate_rank_logits": outputs["candidate_rank_logits"],
        "candidate_benefit_logits": outputs["candidate_benefit_logits"],
        "candidate_catastrophe_logits":
            outputs["candidate_catastrophe_logits"],
        "candidate_h10_gain": outputs["candidate_trajectory"][:, :, 2, 2],
    }
    strict = cached_strict_router_loss(
        strict_outputs, batch["event_target"], batch["gain"],
        batch["beneficial"], batch["catastrophic"],
        batch["label_available"], batch["candidate_valid"],
        pairwise_margin=float(spec["training"]["loss"]["pairwise_margin"]),
    )
    trajectory_l1 = torch.abs(
        outputs["candidate_trajectory"] - batch["trajectory"]).mean()
    total = strict["total"] + float(
        spec["training"]["loss"]["trajectory_l1_weight"]) * trajectory_l1
    if not math.isfinite(float(total.detach())):
        raise ContractError("training loss is non-finite")
    return outputs, {**strict, "trajectory_l1": trajectory_l1,
                     "total_with_trajectory": total}


def count_nonzero_finite_gradients(parameters):
    count = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if torch.isfinite(gradient).all().item() and \
                torch.count_nonzero(gradient).item() > 0:
            count += 1
    return count


def changed_named(before, after, prefix):
    return sum(not torch.equal(value, after[name])
               for name, value in before.items() if name.startswith(prefix))


def identity_batch(keys, relations):
    return {
        "differences": torch.stack([relations[key][0] for key in keys]),
        "gates": torch.stack([relations[key][1] for key in keys]),
        "scalar": torch.stack([relations[key][2] for key in keys]),
        "candidate_valid": torch.ones(len(keys), 6, dtype=torch.bool),
        "role_ids": torch.arange(6, dtype=torch.int64).expand(len(keys), -1),
    }


def model_outputs(model, batch):
    with torch.no_grad():
        return model(
            batch["differences"], batch["gates"], batch["scalar"],
            batch["candidate_valid"], batch["role_ids"])


def policy_decision(outputs, event_index, policy):
    commit_probability = float(torch.sigmoid(
        outputs["event_commit_logit"][event_index]).item())
    rank_probabilities = torch.softmax(
        outputs["candidate_rank_logits"][event_index], dim=0)
    order = torch.argsort(rank_probabilities, descending=True, stable=True)
    top = int(order[0].item())
    second = int(order[1].item())
    top_probability = float(rank_probabilities[top].item())
    margin = top_probability - float(rank_probabilities[second].item())
    benefit = float(torch.sigmoid(
        outputs["candidate_benefit_logits"][event_index, top]).item())
    catastrophe = float(torch.sigmoid(
        outputs["candidate_catastrophe_logits"][event_index, top]).item())
    branch_mean = float(outputs[
        "candidate_trajectory"][event_index, top, 2, 0].item())
    gain = float(outputs[
        "candidate_trajectory"][event_index, top, 2, 2].item())
    gates = {
        "event_commit": commit_probability >= float(
            policy["event_commit_probability_min"]),
        "rank_probability": top_probability >= float(
            policy["candidate_rank_softmax_probability_min"]),
        "rank_margin": margin >= float(
            policy["candidate_rank_probability_margin_min"]),
        "benefit": benefit >= float(
            policy["candidate_benefit_probability_min"]),
        "catastrophe": catastrophe <= float(
            policy["candidate_catastrophe_probability_max"]),
        "h10_branch_mean": branch_mean >= float(
            policy["predicted_h10_branch_mean_iou_min"]),
        "h10_gain": gain >= float(policy["predicted_h10_gain_min"]),
    }
    return {
        "selected_role_id": top if all(gates.values()) else None,
        "top_role_id": top,
        "event_commit_probability": commit_probability,
        "rank_probability": top_probability,
        "rank_margin": margin,
        "benefit_probability": benefit,
        "catastrophe_probability": catastrophe,
        "predicted_h10_branch_mean_iou": branch_mean,
        "predicted_h10_gain": gain,
        "gates": gates,
    }


def max_output_error(first, second):
    return max(float(torch.max(torch.abs(first[name] - second[name])).item())
               for name in first)


def candidate_permutation_audit(model, key, relation, policy, seed, trials):
    original_batch = identity_batch([key], {key: relation})
    original = model_outputs(model, original_batch)
    original_decision = policy_decision(original, 0, policy)
    start = stable_digest_int(seed, "candidate_permutation_audit") % 720
    stride = 23
    maximum = 0.0
    mismatches = 0
    details = []
    for trial in range(int(trials)):
        permutation = torch.tensor(
            ALL_CANDIDATE_PERMUTATIONS[(start + stride * trial) % 720],
            dtype=torch.int64)
        inverse = torch.argsort(permutation)
        batch = {
            "differences": relation[0][:, permutation].unsqueeze(0),
            "gates": relation[1][:, permutation].unsqueeze(0),
            "scalar": relation[2][:, permutation].unsqueeze(0),
            "candidate_valid": torch.ones(1, 6, dtype=torch.bool),
            "role_ids": permutation.unsqueeze(0),
        }
        permuted = model_outputs(model, batch)
        restored = {"event_commit_logit": permuted["event_commit_logit"]}
        for name in (
                "candidate_rank_logits", "candidate_benefit_logits",
                "candidate_catastrophe_logits", "candidate_trajectory"):
            restored[name] = permuted[name][:, inverse]
        error = max_output_error(original, restored)
        maximum = max(maximum, error)
        decision = policy_decision(restored, 0, policy)
        mismatch = decision["selected_role_id"] != original_decision[
            "selected_role_id"]
        mismatches += int(mismatch)
        details.append({
            "trial": trial, "permutation": permutation.tolist(),
            "maximum_absolute_error": error,
            "selection_mismatch": mismatch,
        })
    return {"trials": int(trials), "selection_mismatches": mismatches,
            "maximum_absolute_error": maximum, "details": details}


def event_permutation_audit(model, keys, relations, policy):
    original = model_outputs(model, identity_batch(keys, relations))
    order = torch.arange(len(keys) - 1, -1, -1)
    reversed_keys = [keys[int(index)] for index in order]
    permuted = model_outputs(model, identity_batch(reversed_keys, relations))
    inverse = torch.argsort(order)
    restored = {name: value[inverse] for name, value in permuted.items()}
    mismatches = sum(
        policy_decision(original, index, policy)["selected_role_id"] !=
        policy_decision(restored, index, policy)["selected_role_id"]
        for index in range(len(keys)))
    return {"events": len(keys), "selection_mismatches": mismatches,
            "maximum_absolute_error": max_output_error(original, restored)}


def rederive_heldout_targets(spec, split_entries):
    record = spec["frozen_inputs"]["labeled_actions"]
    validate_file_record(record)
    actions = defaultdict(list)
    for row in load_verified_jsonl(record, compressed=True):
        key = event_key(row)
        split = split_entries.get(key)
        if (split is not None and split["partition"] == "heldout" and
                split["strict_h10_available"] is True):
            actions[key].append(row)
    target_rows = []
    groups = {}
    threshold = float(spec["stage_m17_0_target_and_split_closure"][
        "low_overlap_threshold"])
    for key in sorted(actions):
        rows = {str(row["branch_id"]): row for row in actions[key]}
        if tuple(sorted(rows, key=BRANCH_ORDER.index)) != BRANCH_ORDER:
            raise ContractError("held-out action axis drifted")
        labels = []
        event_targets = []
        for role_id, branch_id in enumerate(BRANCH_ORDER):
            action = rows[branch_id]
            branch_ious = [float(value) for value in action["branch_ious"]]
            public_ious = [float(value) for value in action["public_ious"]]
            if len(branch_ious) != 10 or len(public_ious) != 10:
                raise ContractError("held-out trajectory length drifted")
            label, _, _, _, early_hits = recompute_label(
                branch_ious, public_ious)
            if label != action["label"]:
                raise ContractError("held-out strict label drifted")
            labels.append(label)
            target = {
                "record_type": "action_target",
                "partition": "heldout",
                "fold": int(spec["sequence_split"]["evaluation_fold"]),
                "sequence": key[0], "event_id": key[1],
                "trigger_frame": key[2], "branch_id": branch_id,
                "candidate_role_id": role_id,
                "strict_event_class": None,
                "strict_label": label, "early_hits_h5": early_hits,
                "targets": {
                    str(horizon): trajectory_metrics(
                        branch_ious, public_ious, horizon, threshold)
                    for horizon in HORIZONS
                },
            }
            event_targets.append(target)
        event_class = recompute_event_class(labels)
        for target in event_targets:
            target["strict_event_class"] = event_class
            validate_action_target(target, "heldout")
        groups[key] = event_targets
        target_rows.extend(event_targets)
    expected = spec["sequence_split"]["heldout"]
    commitment = spec["frozen_inputs"]["m17_0_closure"][
        "trajectory_targets"]["heldout_target_commitment"]
    if (len(groups) != int(expected["available_events"]) or
            len(target_rows) != int(expected["available_actions"]) or
            canonical_rows_sha256(target_rows) != commitment):
        raise ContractError("held-out target commitment mismatch")
    return groups, target_rows, commitment


def prediction_rows(keys, outputs, policy):
    rows = []
    for index, key in enumerate(keys):
        decision = policy_decision(outputs, index, policy)
        actions = []
        rank_probabilities = torch.softmax(
            outputs["candidate_rank_logits"][index], dim=0)
        for role_id, branch_id in enumerate(BRANCH_ORDER):
            trajectory = outputs[
                "candidate_trajectory"][index, role_id]
            actions.append({
                "branch_id": branch_id, "candidate_role_id": role_id,
                "rank_probability": float(rank_probabilities[role_id].item()),
                "benefit_probability": float(torch.sigmoid(outputs[
                    "candidate_benefit_logits"][index, role_id]).item()),
                "catastrophe_probability": float(torch.sigmoid(outputs[
                    "candidate_catastrophe_logits"][index, role_id]).item()),
                "predicted_trajectory": {
                    str(horizon): {
                        metric: float(trajectory[horizon_index, metric_index].item())
                        for metric_index, metric in enumerate(TRAJECTORY_METRICS)
                    } for horizon_index, horizon in enumerate(HORIZONS)
                },
            })
        rows.append({
            "record_type": "heldout_event_prediction",
            "sequence": key[0], "event_id": key[1],
            "trigger_frame": key[2], "decision": decision,
            "actions": actions,
        })
    return rows


def attach_actual_targets(predictions, targets):
    for row in predictions:
        key = event_key(row)
        target_rows = targets[key]
        by_role = {int(value["candidate_role_id"]): value
                   for value in target_rows}
        for action in row["actions"]:
            target = by_role[int(action["candidate_role_id"])]
            action["strict_label"] = target["strict_label"]
            action["actual_trajectory"] = target["targets"]
        selected = row["decision"]["selected_role_id"]
        row["strict_event_class"] = target_rows[0]["strict_event_class"]
        if selected is None:
            row.update({"selected_branch_id": None,
                        "selected_strict_label": "abstain",
                        "selected_actual_h10_gain": 0.0,
                        "selected_actual_h10_branch_mean_iou": None,
                        "selected_actual_h10_public_mean_iou": None})
        else:
            target = by_role[int(selected)]
            h10 = target["targets"]["10"]
            row.update({
                "selected_branch_id": BRANCH_ORDER[int(selected)],
                "selected_strict_label": target["strict_label"],
                "selected_actual_h10_gain": float(h10["gain"]),
                "selected_actual_h10_branch_mean_iou": float(
                    h10["branch_mean_iou"]),
                "selected_actual_h10_public_mean_iou": float(
                    h10["public_mean_iou"]),
            })


def scientific_summary(predictions):
    selected = [row for row in predictions
                if row["selected_branch_id"] is not None]
    beneficial = [row for row in selected
                  if row["selected_strict_label"] == "beneficial"]
    catastrophic = [row for row in selected
                    if row["selected_strict_label"] == "catastrophic"]
    mean_gain = (sum(row["selected_actual_h10_gain"] for row in selected) /
                 len(selected) if selected else None)
    branch_mean = (sum(row["selected_actual_h10_branch_mean_iou"]
                       for row in selected) / len(selected)
                   if selected else None)
    public_mean = (sum(row["selected_actual_h10_public_mean_iou"]
                       for row in selected) / len(selected)
                   if selected else None)
    return {
        "selected_actions": len(selected),
        "beneficial_actions": len(beneficial),
        "neutral_actions": sum(row["selected_strict_label"] == "neutral"
                               for row in selected),
        "catastrophic_actions": len(catastrophic),
        "beneficial_sequences": len({row["sequence"] for row in beneficial}),
        "selected_sequences": len({row["sequence"] for row in selected}),
        "beneficial_precision": (len(beneficial) / len(selected)
                                 if selected else 0.0),
        "selected_mean_true_h10_gain": mean_gain,
        "selected_branch_aggregate_h10_mean_iou": branch_mean,
        "selected_public_aggregate_h10_mean_iou": public_mean,
    }


def source_recheck(records):
    mismatches = []
    for record in records:
        path = Path(record["path"])
        if (not path.is_file() or
                path.stat().st_size != int(record["bytes"]) or
                sha256_file(path) != record["sha256"]):
            mismatches.append(str(path))
    return mismatches


def publish_output(output, result, trace, predictions, manifest_builder,
                   prepublish_validator):
    prepublish_validator(())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=output.name + ".tmp.", dir=str(output.parent)))
    try:
        trace_path = temporary / "training_trace.jsonl.gz"
        predictions_path = temporary / "heldout_predictions.jsonl.gz"
        result_path = temporary / "result.json"
        manifest_path = temporary / "manifest.json"
        atomic_jsonl_gz(trace_path, trace)
        atomic_jsonl_gz(predictions_path, predictions)
        atomic_json(result_path, result)
        manifest = manifest_builder(
            result_path, trace_path, predictions_path, output)
        atomic_json(manifest_path, manifest)
        actual = sorted(path.name for path in temporary.iterdir())
        if actual != sorted(EXPECTED_OUTPUT_FILES):
            raise ContractError("M17-1 output file set drifted")
        for path in temporary.iterdir():
            path.chmod(0o444)
        temporary.chmod(0o555)
        fsync_directory(temporary)
        fsync_directory(output.parent)
        prepublish_validator((temporary,))
        if output.exists():
            raise FileExistsError(output)
        os.replace(temporary, output)
        fsync_directory(output.parent)
    finally:
        if temporary.exists():
            temporary.chmod(0o755)
            shutil.rmtree(temporary)


def main():
    args = parse_args()
    started = time.time()
    for name in ("spec", "binding", "preaudit", "output"):
        setattr(args, name, getattr(args, name).resolve())
    spec, spec_record = load_json_snapshot(args.spec)
    validate_spec(spec, args)
    binding, audit, commit, runtime_identity = validate_execution_binding(
        args, spec, spec_record)
    observer = RuntimeSideEffectObserver()
    observer.install()
    if not args.output.parent.is_dir():
        raise ContractError("M17-1 output parent is absent")
    runtime_scratch = Path(tempfile.mkdtemp(
        prefix="m17_1_runtime.", dir=str(args.output.parent))).resolve()
    previous_tmpdir_environment = os.environ.get("TMPDIR")
    previous_tempfile_directory = tempfile.tempdir
    os.environ["TMPDIR"] = str(runtime_scratch)
    tempfile.tempdir = str(runtime_scratch)

    def cleanup_runtime_scratch():
        tempfile.tempdir = previous_tempfile_directory
        if previous_tmpdir_environment is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = previous_tmpdir_environment
        if runtime_scratch.exists():
            shutil.rmtree(runtime_scratch)

    atexit.register(cleanup_runtime_scratch)
    m17_0_result = validate_frozen_receipts(spec)
    collection, sequence_anchors = load_collection_index(spec)
    split_ledger, split_entries = load_split_ledger(spec)
    if set(collection) != set(split_entries):
        raise ContractError("collection/split event keys drifted")
    training_sequences = {
        key[0] for key, row in split_entries.items()
        if row["partition"] == "training"}
    heldout_sequences = {
        key[0] for key, row in split_entries.items()
        if row["partition"] == "heldout"}
    sequence_overlap = sorted(training_sequences & heldout_sequences)
    training_groups, heldout_commitment, target_record_counts = \
        load_training_targets(spec, split_entries)
    native_index = load_native_index(spec)
    required_sequences = sorted({key[0] for key in split_entries})
    clip_binding, _ = validate_anchor_binding(
        binding, sequence_anchors, native_index, required_sequences)
    load_project_components()

    seed = int(spec["training"]["seed"])
    torch.manual_seed(seed)
    architecture = spec["architecture"]
    model = CanonicalRoleIndependentUtilitySafetyRouter(
        hidden_dim=int(architecture["hidden_dim"]),
        residual_scale=float(architecture["residual_scale"]),
        base_projection_seed=int(architecture["base_projection_seed"]),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters()
                          if parameter.requires_grad)
    utility_parameters = list(model.utility_parameters())
    safety_parameters = list(model.safety_parameters())
    parameter_overlap = len(
        {id(value) for value in utility_parameters} &
        {id(value) for value in safety_parameters})
    forbidden_fragments = tuple(architecture["forbidden_modules"])
    forbidden_modules = [
        type(module).__name__ for module in model.modules()
        if any(fragment in type(module).__name__
               for fragment in forbidden_fragments)
    ]
    initial_state = {name: value.detach().clone()
                     for name, value in model.state_dict().items()}
    initial_state_sha256 = state_digest(model)

    clip_cache = {}
    native_cache = {}
    loaded_feature_records = {}
    training_relations = {}
    for key in sorted(training_groups):
        training_relations[key] = relation_for_event(
            spec, collection, sequence_anchors, native_index,
            clip_binding, key,
            clip_cache, native_cache, loaded_feature_records)
    training_targets = {}
    for key, rows in training_groups.items():
        values = target_tensors(rows)
        values["event_class"] = rows[0]["strict_event_class"]
        training_targets[key] = values

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(spec["training"]["learning_rate"]),
        weight_decay=float(spec["training"]["weight_decay"]))
    trace = []
    steps_completed = 0
    gradient_failure = None
    preclip_values = []
    postclip_values = []
    utility_projector_coverage = set()
    safety_projector_coverage = set()
    utility_nonprojector_gradient_seen = False
    safety_nonprojector_gradient_seen = False
    class_pool_counts = None
    model.train()
    for epoch in range(int(spec["training"]["epochs"])):
        batches, observed_pool_counts = balanced_event_batches(
            training_groups, spec, epoch)
        if class_pool_counts is None:
            class_pool_counts = observed_pool_counts
        elif class_pool_counts != observed_pool_counts:
            raise ContractError("training class pools changed by epoch")
        for epoch_step, keys in enumerate(batches):
            global_step = epoch * int(
                spec["training"]["steps_per_epoch"]) + epoch_step + 1
            if len(keys) != int(spec["training"]["event_batch_size"]):
                raise ContractError("event batch size drifted")
            class_counts = Counter(
                training_targets[key]["event_class"] for key in keys)
            if class_counts != Counter(
                    spec["training"]["event_batch_composition"]):
                raise ContractError("event batch composition drifted")
            batch = make_training_batch(
                keys, training_relations, training_targets,
                spec, epoch, epoch_step)
            optimizer.zero_grad(set_to_none=True)
            try:
                outputs, losses = forward_losses(model, batch, spec)
            except Exception as error:
                gradient_failure = {
                    "step": global_step, "phase": "forward_or_loss",
                    "error": type(error).__name__ + ": " + str(error),
                }
                break
            if any(not finite_tensor(value) for value in outputs.values()) or \
                    any(not math.isfinite(float(value.detach()))
                        for value in losses.values()):
                gradient_failure = {
                    "step": global_step, "phase": "nonfinite_output_or_loss"}
                break
            losses["total_with_trajectory"].backward()
            for index, projector in enumerate(model.utility_projectors):
                if count_nonzero_finite_gradients(projector.parameters()) > 0:
                    utility_projector_coverage.add(index)
            for index, projector in enumerate(model.safety_projectors):
                if count_nonzero_finite_gradients(projector.parameters()) > 0:
                    safety_projector_coverage.add(index)
            utility_nonprojector_gradient_seen |= (
                count_nonzero_finite_gradients(
                    model.utility_router.parameters()) > 0)
            safety_nonprojector_gradient_seen |= (
                count_nonzero_finite_gradients(
                    model.safety_critic.parameters()) > 0)
            preclip, nonfinite, _ = gradient_diagnostics(model, 0)
            gradient_gates = spec["training"]["gradient_gates"]
            if (nonfinite != 0 or not math.isfinite(preclip) or
                    preclip <= float(gradient_gates[
                        "preclip_total_l2_min_exclusive"]) or
                    preclip > float(gradient_gates[
                        "preclip_total_l2_max"])):
                gradient_failure = {
                    "step": global_step, "phase": "preclip",
                    "norm": preclip, "nonfinite": nonfinite}
                break
            maximum = float(spec["training"]["gradient_clip_norm"])
            scale_gradients(model, min(1.0, maximum / (preclip + 1e-12)))
            postclip, post_nonfinite, _ = gradient_diagnostics(model, 0)
            if (post_nonfinite != 0 or not math.isfinite(postclip) or
                    postclip > float(gradient_gates[
                        "postclip_total_l2_max"])):
                gradient_failure = {
                    "step": global_step, "phase": "postclip",
                    "norm": postclip, "nonfinite": post_nonfinite}
                break
            optimizer.step()
            steps_completed = global_step
            preclip_values.append(preclip)
            postclip_values.append(postclip)
            trace.append({
                "record_type": "optimizer_step", "global_step": global_step,
                "epoch": epoch, "epoch_step": epoch_step,
                "event_keys": [list(key) for key in keys],
                "event_classes": dict(sorted(class_counts.items())),
                "losses": {name: float(value.detach())
                           for name, value in losses.items()},
                "preclip_total_l2": preclip,
                "postclip_total_l2": postclip,
                "nonfinite_gradients": nonfinite,
                "optimizer_step_executed": True,
            })
        if gradient_failure is not None:
            break

    training_complete = (
        gradient_failure is None and steps_completed == int(
            spec["training"]["optimizer_steps_total"]))
    model.eval()
    frozen_state_sha256 = state_digest(model)
    state_before_heldout = {name: value.detach().clone()
                            for name, value in model.state_dict().items()}
    heldout_targets_opened_before_training_complete = False
    heldout_predictions_computed_before_targets_open = False
    predictions = []
    permutation_audit = None
    event_order_audit = None
    heldout_commitment_observed = None

    if training_complete:
        heldout_keys = sorted(
            key for key, row in split_entries.items()
            if row["partition"] == "heldout" and
            row["strict_h10_available"] is True)
        heldout_relations = {}
        for key in heldout_keys:
            heldout_relations[key] = relation_for_event(
                spec, collection, sequence_anchors, native_index,
                clip_binding, key,
                clip_cache, native_cache, loaded_feature_records)
        permutation_audit = candidate_permutation_audit(
            model, heldout_keys[0], heldout_relations[heldout_keys[0]],
            spec["heldout_policy"], seed,
            int(spec["engineering_gates"]["candidate_permutation_trials"]))
        event_order_audit = event_permutation_audit(
            model, heldout_keys[:8], heldout_relations,
            spec["heldout_policy"])
        output_batches = []
        for start in range(0, len(heldout_keys), 16):
            keys = heldout_keys[start:start + 16]
            output_batches.extend(prediction_rows(
                keys, model_outputs(model, identity_batch(
                    keys, heldout_relations)), spec["heldout_policy"]))
        predictions = output_batches
        heldout_predictions_computed_before_targets_open = True
        heldout_targets, heldout_target_rows, heldout_commitment_observed = \
            rederive_heldout_targets(spec, split_entries)
        attach_actual_targets(predictions, heldout_targets)
    else:
        predictions = [{
            "record_type": "heldout_evaluation_skipped",
            "reason": "training_engineering_gate_failed",
            "heldout_numeric_targets_opened": False,
        }]

    evaluation_state_unchanged = all(
        torch.equal(value, model.state_dict()[name])
        for name, value in state_before_heldout.items())
    current_state = model.state_dict()
    changes = {
        "utility_projectors": changed_named(
            initial_state, current_state, "utility_projectors."),
        "utility_router": changed_named(
            initial_state, current_state, "utility_router."),
        "safety_projectors": changed_named(
            initial_state, current_state, "safety_projectors."),
        "safety_critic": changed_named(
            initial_state, current_state, "safety_critic."),
    }
    scientific = scientific_summary(predictions) if training_complete else {
        "selected_actions": 0, "beneficial_actions": 0,
        "neutral_actions": 0, "catastrophic_actions": 0,
        "beneficial_sequences": 0, "selected_sequences": 0,
        "beneficial_precision": 0.0,
        "selected_mean_true_h10_gain": None,
        "selected_branch_aggregate_h10_mean_iou": None,
        "selected_public_aggregate_h10_mean_iou": None,
    }

    feature_records = list(loaded_feature_records.values())
    cleanup_runtime_scratch()
    atexit.unregister(cleanup_runtime_scratch)
    source_mismatches_after = source_recheck([
        *binding["code_records"], *binding["source_records"],
        *binding["clip_anchor_payloads"],
        *binding["native_anchor_payloads"], *feature_records])
    runtime_observations = validate_runtime_identity(
        args, spec, binding, runtime_identity, observer,
        additional_records=feature_records,
        allowed_write_roots=(runtime_scratch,))
    side_effect_counts = {
        "public_tracker_mutations": (
            len(source_mismatches_after) +
            len(runtime_observations["public_tracker_modules"]) +
            len(runtime_observations["forbidden_write_paths"])),
        "qwen_activity": (
            len(runtime_observations["qwen_modules"]) +
            len(runtime_observations["network_connections"])),
        "tracking_checkpoint_writes": len(
            runtime_observations["checkpoint_write_paths"]),
        "public_benchmark_activity": (
            len(runtime_observations["public_benchmark_modules"]) +
            len(runtime_observations["unexpected_subprocesses"])),
    }
    gates = spec["engineering_gates"]
    engineering_conditions = {
        "m17_0_all_conditions": all(
            value is True
            for value in m17_0_result["conditions"].values()),
        "train_heldout_sequence_overlap": len(sequence_overlap) <= int(
            gates["train_heldout_sequence_overlap_max"]),
        "optimizer_steps_exact": steps_completed == int(
            gates["optimizer_steps_exact"]),
        "gradient_safety": gradient_failure is None,
        "trace_rows_exact": len(trace) == int(
            gates["optimizer_steps_exact"]),
        "model_parameters_exact": parameter_count == int(
            gates["model_parameters_exact"]),
        "utility_safety_parameter_overlap": parameter_overlap == int(
            gates["utility_safety_parameter_overlap_exact"]),
        "forbidden_modules": len(forbidden_modules) <= int(
            gates["forbidden_module_count_max"]),
        "utility_projector_gradient_coverage":
            len(utility_projector_coverage) == int(
                gates["utility_projectors_with_nonzero_finite_gradient_exact"]),
        "safety_projector_gradient_coverage":
            len(safety_projector_coverage) == int(
                gates["safety_projectors_with_nonzero_finite_gradient_exact"]),
        "utility_projectors_changed": changes["utility_projectors"] == int(
            gates["utility_projectors_changed_exact"]),
        "safety_projectors_changed": changes["safety_projectors"] == int(
            gates["safety_projectors_changed_exact"]),
        "utility_nonprojector_gradient": utility_nonprojector_gradient_seen,
        "safety_nonprojector_gradient": safety_nonprojector_gradient_seen,
        "utility_nonprojector_changed": changes["utility_router"] >= 1,
        "safety_nonprojector_changed": changes["safety_critic"] >= 1,
        "heldout_targets_not_opened_before_training_complete":
            not heldout_targets_opened_before_training_complete,
        "heldout_predictions_before_targets":
            heldout_predictions_computed_before_targets_open,
        "heldout_commitment_exact": heldout_commitment_observed ==
            heldout_commitment["canonical_jsonl_sha256"],
        "candidate_permutation_selection": (
            permutation_audit is not None and
            permutation_audit["selection_mismatches"] <= int(
                gates["candidate_permutation_selection_mismatches_max"])),
        "candidate_permutation_error": (
            permutation_audit is not None and
            permutation_audit["maximum_absolute_error"] <= float(
                gates["candidate_permutation_max_absolute_error"])),
        "event_permutation_selection": (
            event_order_audit is not None and
            event_order_audit["selection_mismatches"] == 0),
        "event_permutation_error": (
            event_order_audit is not None and
            event_order_audit["maximum_absolute_error"] <= float(
                gates["event_permutation_max_absolute_error"])),
        "evaluation_state_unchanged": evaluation_state_unchanged,
        "source_hashes_after": not source_mismatches_after,
        "public_tracker_mutations": side_effect_counts[
            "public_tracker_mutations"] <= int(
                gates["public_tracker_mutations_max"]),
        "qwen_calls": side_effect_counts["qwen_activity"] <= int(
            gates["qwen_calls_max"]),
        "tracking_checkpoints": side_effect_counts[
            "tracking_checkpoint_writes"] <= int(
                gates["tracking_checkpoints_max"]),
        "public_benchmark_runs": side_effect_counts[
            "public_benchmark_activity"] <= int(
                gates["public_benchmark_runs_max"]),
    }
    scientific_gates = spec["scientific_gates"]
    scientific_conditions = {
        "selected_actions_min": scientific["selected_actions"] >= int(
            scientific_gates["heldout_selected_actions_min"]),
        "beneficial_actions_min": scientific["beneficial_actions"] >= int(
            scientific_gates["heldout_beneficial_actions_min"]),
        "beneficial_sequences_min": scientific["beneficial_sequences"] >= int(
            scientific_gates["heldout_beneficial_sequences_min"]),
        "beneficial_precision_min": scientific["beneficial_precision"] >= float(
            scientific_gates["heldout_beneficial_precision_min"]),
        "catastrophic_actions_max": scientific["catastrophic_actions"] <= int(
            scientific_gates["heldout_catastrophic_actions_max"]),
        "selected_mean_true_h10_gain_min": (
            scientific["selected_mean_true_h10_gain"] is not None and
            scientific["selected_mean_true_h10_gain"] >= float(
                scientific_gates["heldout_selected_mean_true_h10_gain_min"])),
        "branch_aggregate_gt_public": (
            scientific["selected_branch_aggregate_h10_mean_iou"] is not None and
            scientific["selected_public_aggregate_h10_mean_iou"] is not None and
            scientific["selected_branch_aggregate_h10_mean_iou"] >
            scientific["selected_public_aggregate_h10_mean_iou"]),
        "all_abstain_is_not_pass": scientific["selected_actions"] > 0,
    }
    engineering_pass = all(engineering_conditions.values())
    scientific_pass = training_complete and all(scientific_conditions.values())
    accepted = engineering_pass and scientific_pass
    result = {
        "schema": "sttrack-lachtt-m17-1-sequence-disjoint-survival-result/v1",
        "complete": True, "accepted": accepted,
        "decision": ("m17_1_pass_authorize_six_fold_plan_spec_only"
                     if accepted else
                     "m17_1_fail_stop_fixed_family_without_rescan"),
        "claim_ceiling": spec["claim_ceiling"],
        "repository": {"path": str(REPOSITORY_ROOT), "commit": commit,
                       "branch": spec["repository"]["branch"], "clean": True},
        "identity": {
            "spec": runtime_identity["spec"],
            "binding": runtime_identity["binding"],
            "preexecution_audit": runtime_identity[
                "preexecution_audit"],
            "preflight_binding": runtime_identity[
                "preflight_binding"],
            "runner": runtime_identity["runner"],
        },
        "training": {
            "seed": seed, "device": "cpu", "dtype": "float32",
            "steps_requested": int(spec["training"]["optimizer_steps_total"]),
            "steps_completed": steps_completed,
            "trace_rows": len(trace), "class_pool_counts": class_pool_counts,
            "gradient_failure": gradient_failure,
            "preclip_max": max(preclip_values) if preclip_values else None,
            "postclip_max": max(postclip_values) if postclip_values else None,
        },
        "data_isolation": {
            "training_sequence_count": len(training_sequences),
            "heldout_sequence_count": len(heldout_sequences),
            "train_heldout_sequence_overlap": sequence_overlap,
            "training_target_events": len(training_groups),
            "training_target_actions": sum(len(value)
                                           for value in training_groups.values()),
            "target_file_record_counts": target_record_counts,
            "heldout_numeric_targets_serialized_before_training": 0,
            "heldout_targets_opened_before_training_complete":
                heldout_targets_opened_before_training_complete,
            "heldout_predictions_computed_before_targets_open":
                heldout_predictions_computed_before_targets_open,
            "heldout_target_commitment_expected":
                heldout_commitment["canonical_jsonl_sha256"],
            "heldout_target_commitment_observed": heldout_commitment_observed,
            "training_feature_events_loaded": len(training_relations),
            "heldout_feature_events_loaded": (
                int(spec["sequence_split"]["heldout"]["available_events"])
                if training_complete else 0),
        },
        "model": {
            "class": architecture["class"],
            "parameter_count": parameter_count,
            "utility_safety_parameter_overlap": parameter_overlap,
            "forbidden_modules": forbidden_modules,
            "initial_state_sha256": initial_state_sha256,
            "frozen_state_sha256": frozen_state_sha256,
            "projector_gradient_coverage": {
                "utility": sorted(utility_projector_coverage),
                "safety": sorted(safety_projector_coverage)},
            "changes": changes,
        },
        "permutation_audit": {
            "candidate": permutation_audit,
            "event_order": event_order_audit,
        },
        "scientific_summary": scientific,
        "engineering_conditions": engineering_conditions,
        "scientific_conditions": scientific_conditions,
        "engineering_pass": engineering_pass,
        "scientific_pass": scientific_pass,
        "failed_engineering_conditions": sorted(
            name for name, value in engineering_conditions.items() if not value),
        "failed_scientific_conditions": sorted(
            name for name, value in scientific_conditions.items() if not value),
        "source_mismatches_after": source_mismatches_after,
        "runtime_side_effect_observations": runtime_observations,
        "runtime_side_effect_counts": side_effect_counts,
        "elapsed_seconds": time.time() - started,
        "tracking_checkpoint_written": side_effect_counts[
            "tracking_checkpoint_writes"] > 0,
        "public_tracker_mutations": side_effect_counts[
            "public_tracker_mutations"],
        "qwen_calls": side_effect_counts["qwen_activity"],
        "depthtrack_test_run": any(
            "depthtrack" in name.lower()
            for name in runtime_observations["public_benchmark_modules"]),
        "cdtb_run": any(
            "cdtb" in name.lower()
            for name in runtime_observations["public_benchmark_modules"]),
        "vot_low22_run": any(
            name == "vot" or name.startswith("vot.")
            for name in runtime_observations["public_benchmark_modules"]),
        "vot_full127_run": any(
            name == "vot" or name.startswith("vot.")
            for name in runtime_observations["public_benchmark_modules"]),
        "automatic_next_stage": False,
        "authorization": {
            "independent_result_audit": True,
            "six_fold_plan_spec_only": accepted,
            "six_fold_execution": False,
            "online_replay": False, "tracking_checkpoint": False,
            "depthtrack_test": False, "cdtb": False,
            "vot_low22": False, "vot_full127": False, "qwen": False,
        },
    }

    def published_record(temporary_path, published_path):
        record = file_record(temporary_path)
        record["path"] = str(Path(published_path).resolve())
        return record

    def manifest_builder(result_path, trace_path, predictions_path,
                         published_root):
        return {
            "schema": "sttrack-lachtt-m17-1-sequence-disjoint-survival-manifest/v1",
            "complete": True, "accepted": accepted,
            "identity": result["identity"],
            "preflight_binding": binding["preflight_binding"],
            "code_records": binding["code_records"],
            "source_records": binding["source_records"],
            "payload": {
                "result": published_record(
                    result_path, published_root / "result.json"),
                "training_trace": published_record(
                    trace_path, published_root / "training_trace.jsonl.gz"),
                "heldout_predictions": published_record(
                    predictions_path,
                    published_root / "heldout_predictions.jsonl.gz"),
            },
            "unauthorized_actions": {
                "tracking_checkpoint_written": result[
                    "tracking_checkpoint_written"],
                "public_tracker_mutated": result[
                    "public_tracker_mutations"] > 0,
                "original_rgb_or_depth_opened": False,
                "new_ground_truth_file_opened": False,
                "qwen": result["qwen_calls"] > 0,
                "depthtrack_test": result["depthtrack_test_run"],
                "cdtb": result["cdtb_run"],
                "vot_low22": result["vot_low22_run"],
                "vot_full127": result["vot_full127_run"],
                "automatic_next_stage": False,
            },
        }

    def prepublish_validator(allowed_write_roots):
        validate_runtime_identity(
            args, spec, binding, runtime_identity, observer,
            additional_records=feature_records,
            allowed_write_roots=(runtime_scratch, *allowed_write_roots))

    publish_output(
        args.output, result, trace, predictions, manifest_builder,
        prepublish_validator)
    print(json.dumps({
        "accepted": accepted, "decision": result["decision"],
        "engineering_pass": engineering_pass,
        "scientific_pass": scientific_pass,
        "steps_completed": steps_completed,
        "scientific_summary": scientific,
        "result": file_record(args.output / "result.json"),
        "manifest": file_record(args.output / "manifest.json"),
        "training_trace": file_record(
            args.output / "training_trace.jsonl.gz"),
        "heldout_predictions": file_record(
            args.output / "heldout_predictions.jsonl.gz"),
    }, sort_keys=True))
    raise SystemExit(0 if accepted else 2)


if __name__ == "__main__":
    main()
