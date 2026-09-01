#!/usr/bin/env python3
"""Run one immutable, phase-closed causal-survival runtime smoke."""

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import traceback
import types


REPOSITORY_ROOT = Path(
    "/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1").resolve()
EXPECTED_PLAN_PATH = Path(
    "/home/SUTrack_RGBD_L/refine-logs/"
    "EXPERIMENT_PLAN_M21_IMMUTABLE_PHASE_CLOSED_RUNTIME_SMOKE_"
    "20260901_210242.md").resolve()
EXPECTED_SPEC_PATH = Path(
    "/home/SUTrack_RGBD_L/refine-logs/"
    "STTRACK_LACHTT_M21A_IMMUTABLE_PHASE_CLOSED_RUNTIME_SMOKE_SPEC_"
    "20260901.json").resolve()
EXPECTED_BINDING_PATH = Path(
    "/home/SUTrack_RGBD_L/refine-logs/"
    "STTRACK_LACHTT_M21A_IMMUTABLE_PHASE_CLOSED_RUNTIME_SMOKE_BINDING_"
    "20260901.json").resolve()
EXPECTED_ATTEMPT_ROOT = Path(
    "/root/autodl-tmp/"
    "sttrack_lachtt_m21a_immutable_phase_closed_runtime_smoke_attempt_v1_"
    "20260901").resolve()
EXPECTED_SCIENTIFIC_OUTPUT = Path(
    "/root/autodl-tmp/"
    "sttrack_lachtt_m21a_immutable_phase_closed_runtime_scientific_output_v1_"
    "20260901").resolve()
EXPECTED_PYTHON = Path(
    "/root/autodl-tmp/envs/sttrack/bin/python3.8").resolve()
AUTHORIZED_ACTION = (
    "run exactly one M21a immutable phase-closed eight-event zero-step "
    "model-runtime "
    "smoke if the binding passes its own preflight"
)
BOOTSTRAP_PHASES = {
    "observer_installed", "subprocess_wrapper_installed", "torch_import",
}


class ContractError(RuntimeError):
    """Raised when the frozen M21a contract is not exact."""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--attempt-root", required=True, type=Path)
    parser.add_argument("--scientific-output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file_record(path):
    path = Path(path).resolve()
    metadata = path.lstat()
    if (not stat.S_ISREG(metadata.st_mode) or path.is_symlink() or
            metadata.st_nlink != 1):
        raise ContractError(
            "input is not an independent regular file: {}".format(path))
    return {
        "path": str(path),
        "bytes": metadata.st_size,
        "sha256": sha256_file(path),
        "mode": format(stat.S_IMODE(metadata.st_mode), "04o"),
    }


def read_verified_bytes(record):
    path = Path(record["path"]).resolve()
    actual = regular_file_record(path)
    for key in ("path", "bytes", "sha256", "mode"):
        if actual[key] != record[key]:
            raise ContractError(
                "bound file identity drifted: {}".format(path))
    return path.read_bytes(), actual


def load_verified_json(record):
    payload, actual = read_verified_bytes(record)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("invalid bound JSON: {}".format(error))
    return value, actual


def write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2,
                  sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def load_verified_module(name, record):
    payload, actual = read_verified_bytes(record)
    module = types.ModuleType(name)
    module.__file__ = actual["path"]
    module.__package__ = ""
    sys.modules[name] = module
    try:
        code = compile(payload, actual["path"], "exec")
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, actual


def collect_file_records(value, output=None):
    if output is None:
        output = []
    if isinstance(value, dict):
        if "sha256" in value:
            required = {
                "path": str,
                "bytes": int,
                "sha256": str,
                "mode": str,
            }
            if any(not isinstance(value.get(key), kind)
                   for key, kind in required.items()):
                raise ContractError(
                    "partial frozen file record in M21a spec")
        if (isinstance(value.get("path"), str) and
                isinstance(value.get("sha256"), str) and
                isinstance(value.get("mode"), str) and
                isinstance(value.get("bytes"), int)):
            output.append(value)
        for nested in value.values():
            collect_file_records(nested, output)
    elif isinstance(value, list):
        for nested in value:
            collect_file_records(nested, output)
    return output


def exact_record_set(records):
    observed = []
    seen = set()
    for record in records:
        key = (record["path"], record["sha256"])
        if key in seen:
            continue
        seen.add(key)
        _, actual = read_verified_bytes(record)
        observed.append(actual)
    return observed


def first_attribution_caller(stack, skipped_paths):
    skipped = {str(Path(path).resolve()) for path in skipped_paths}
    for row in reversed(stack):
        frame_path = row["file"]
        normalized = frame_path.replace("\\", "/")
        if frame_path in skipped or frame_path.startswith("<"):
            continue
        if ("/lib/python3." in normalized and
                "/site-packages/" not in normalized):
            continue
        return {
            "file": frame_path,
            "line": int(row["line"]),
            "function": row["function"],
        }
    return None


def bootstrap_event_from_observation(observation, runner_path,
                                     provenance_path):
    popen_calls = observation["popen_calls"]
    subprocess_rows = observation["subprocess_audit_events"]
    devnull_rows = observation["devnull_write_events"]
    if (len(popen_calls) != 1 or len(subprocess_rows) != 1 or
            len(devnull_rows) != 1):
        raise ContractError("bootstrap event cardinality drifted")
    popen = popen_calls[0]
    subprocess_row = subprocess_rows[0]
    devnull = devnull_rows[0]
    correlation = popen["correlation_id"]
    if (subprocess_row.get("correlation_id") != correlation or
            devnull.get("correlation_id") != correlation):
        raise ContractError("bootstrap event correlation drifted")
    caller = first_attribution_caller(
        devnull.get("stack", []), (runner_path, provenance_path))
    return {
        "phase": popen["phase"],
        "name": subprocess_row["event"],
        "executable": subprocess_row["executable"],
        "argv": popen["command"],
        "effective_cwd": popen["effective_cwd"],
        "stdio": {
            "stdin": popen["stdin"],
            "stdout": popen["stdout"],
            "stderr": popen["stderr"],
        },
        "devnull": {
            "path": devnull["path"],
            "flags": devnull["flags"],
            "mode_semantics": devnull["mode_semantics"],
        },
        "correlation_id": correlation,
        "popen_wrapper_event_id": popen["event_id"],
        "devnull_event_id": devnull["event_id"],
        "popen_audit_event_id": subprocess_row["event_id"],
        "first_attribution_caller": caller,
    }


def record_without_resolved_identity(event):
    output = dict(event)
    output.pop("resolved_executable_identity", None)
    return output


def validate_receipt(receipt, actual_event):
    if (receipt.get("schema") !=
            "sttrack-lachtt-exact-bootstrap-receipt/v1" or
            receipt.get("complete") is not True or
            receipt.get("claim_ceiling") !=
            "exact bootstrap receipt only; no model or benchmark claim" or
            receipt.get("wildcards_allowed") is not False or
            receipt.get("prefix_or_directory_allowance") is not False or
            receipt.get("model_runtime_allowance") != []):
        raise ContractError("M19b receipt permission surface drifted")
    expected_event = receipt["event"]
    if record_without_resolved_identity(expected_event) != actual_event:
        raise ContractError("observed bootstrap event differs from receipt")
    read_verified_bytes(expected_event["resolved_executable_identity"])


def frozen_observation(observer):
    return copy.deepcopy(observer.snapshot())


def phase_events(observation, phase):
    def selected(rows):
        return [row for row in rows if row.get("phase") == phase]

    return {
        "popen_calls": selected(observation["popen_calls"]),
        "subprocess_audit_events": selected(
            observation["subprocess_audit_events"]),
        "devnull_write_events": selected(
            observation["devnull_write_events"]),
        "all_write_events": selected(observation["all_write_events"]),
        "unresolved_write_events": selected(
            observation["unresolved_write_events"]),
        "all_mutation_events": selected(
            observation["all_mutation_events"]),
        "network_events": selected(observation["network_events"]),
        "sensitive_read_events": selected(
            observation["sensitive_read_events"]),
    }


def model_runtime_events(observation):
    return phase_events(observation, "model_runtime")


def stack_has_postflight_git_identity(event, provenance_path):
    provenance_path = str(Path(provenance_path).resolve())
    functions = [
        row.get("function") for row in event.get("stack", [])
        if str(Path(row.get("file", "")).resolve()) == provenance_path
    ]
    return ("direct_git_identity" in functions and
            "_git_worktree_clean" in functions)


def validate_postflight_events(events, provenance_path):
    if (events["popen_calls"] or events["subprocess_audit_events"] or
            events["devnull_write_events"] or events["all_write_events"] or
            events["unresolved_write_events"] or
            events["all_mutation_events"] or events["network_events"] or
            not events["sensitive_read_events"]):
        return False
    for event in events["sensitive_read_events"]:
        try:
            Path(event["path"]).resolve().relative_to(REPOSITORY_ROOT)
        except ValueError:
            return False
        if not stack_has_postflight_git_identity(event, provenance_path):
            return False
    return True


def validate_control_contract(args, spec, binding):
    expected_spec_authorization = {
        "implementation_and_independent_review": True,
        "m21a_execution_after_preaudit_pass": True,
        "m21b_training": False,
        "public_evaluation": False,
        "automatic_next_stage": False,
    }
    expected_forbidden_actions = {
        "optimizer_or_training": True,
        "checkpoint_or_prediction_output": True,
        "online_replay": True,
        "public_tracker_mutation": True,
        "qwen_or_online_text": True,
        "depthtrack_test": True,
        "cdtb": True,
        "vot_low22_or_full127": True,
        "automatic_m21b": True,
    }
    expected_binding_authorization = {
        "m21a_zero_step_model_runtime_smoke": True,
        "authorized_run_count": 1,
        "optimizer_steps": 0,
        "training": False,
        "checkpoint": False,
        "public_evaluation": False,
        "automatic_next_stage": False,
    }
    if (args.spec != EXPECTED_SPEC_PATH or
            args.binding != EXPECTED_BINDING_PATH or
            args.attempt_root != EXPECTED_ATTEMPT_ROOT or
            args.scientific_output != EXPECTED_SCIENTIFIC_OUTPUT):
        raise ContractError("M21a invocation path drifted")
    if (spec.get("schema") !=
            "sttrack-lachtt-m21a-immutable-phase-closed-runtime-smoke-spec/v1" or
            spec.get("complete") is not True or
            binding.get("schema") !=
            "sttrack-lachtt-m21a-immutable-phase-closed-runtime-smoke-binding/v1" or
            binding.get("complete") is not True):
        raise ContractError("M21a control schema drifted")
    if (spec.get("authorization") != expected_spec_authorization or
            spec.get("forbidden_actions") != expected_forbidden_actions or
            binding.get("binding_path") != str(args.binding) or
            binding.get("claim_ceiling") != spec.get("claim_ceiling") or
            binding.get("authorization") !=
            expected_binding_authorization):
        raise ContractError("M21a binding authorization drifted")
    required_records = {
        "plan": binding["plan"],
        "spec": binding["spec"],
        "runner": binding["runner"],
        "model": binding["model"],
        "m18_runner": binding["m18_runner"],
        "m19_provenance_runner": binding["m19_provenance_runner"],
        "receipt": binding["receipt"],
        "m19b_result_audit": binding["m19b_result_audit"],
        "preexecution_audit": binding["preexecution_audit"],
        "python": binding["python"],
    }
    expected_paths = {
        "plan": EXPECTED_PLAN_PATH,
        "spec": args.spec,
        "runner": Path(__file__).resolve(),
        "model": REPOSITORY_ROOT /
            "lib/models/sttrack/lachtt_causal_quantile_survival.py",
        "m18_runner": REPOSITORY_ROOT /
            "tools/run_sttrack_lachtt_m18a_architecture_journal_smoke.py",
        "m19_provenance_runner": REPOSITORY_ROOT /
            "tools/run_sttrack_lachtt_m19a_bootstrap_attribution.py",
        "receipt": Path(spec["bootstrap_receipt"]["path"]).resolve(),
        "m19b_result_audit": Path(
            spec["m19b_result_audit"]["path"]).resolve(),
        "preexecution_audit": Path(
            binding["preexecution_audit"]["path"]).resolve(),
        "python": EXPECTED_PYTHON,
    }
    actual_records = {}
    for name, record in required_records.items():
        if Path(record["path"]).resolve() != expected_paths[name]:
            raise ContractError("{} path drifted".format(name))
        _, actual_records[name] = read_verified_bytes(record)
    if (binding["plan"] != spec["plan"] or
            binding["runner"] != spec["runner"] or
            binding["model"] != spec["model"] or
            binding["m18_runner"] != spec["m18_runner"] or
            binding["m19_provenance_runner"] !=
            spec["m19_provenance_runner"] or
            binding["receipt"] != spec["bootstrap_receipt"] or
            binding["m19b_result_audit"] != spec["m19b_result_audit"] or
            binding["python"] != spec["python"]):
        raise ContractError("spec/binding identity mismatch")
    audit, _ = load_verified_json(binding["preexecution_audit"])
    allowed = audit.get("authorization_boundary", {}).get(
        "authorized_next_actions_after_pass", [])
    expected_audit_identity = {
        "plan_sha256": binding["plan"]["sha256"],
        "spec_sha256": binding["spec"]["sha256"],
        "runner_sha256": binding["runner"]["sha256"],
        "model_sha256": binding["model"]["sha256"],
        "receipt_sha256": binding["receipt"]["sha256"],
        "repository_commit": binding["repository"]["commit"],
        "attempt_root": str(args.attempt_root),
        "scientific_output": str(args.scientific_output),
    }
    if (audit.get("overall_verdict") != "PASS" or
            str(audit.get("integrity_status", "")).upper() != "PASS" or
            audit.get("claim_ceiling") != spec["claim_ceiling"] or
            allowed != [AUTHORIZED_ACTION] or
            any(audit.get("audited_identity", {}).get(key) != value
                for key, value in expected_audit_identity.items())):
        raise ContractError("M21a preexecution audit identity drifted")
    m19b_audit, _ = load_verified_json(binding["m19b_result_audit"])
    if (m19b_audit.get("overall_verdict") != "PASS" or
            m19b_audit.get("integrity_status") != "PASS" or
            m19b_audit.get("claim_supported") is not True or
            m19b_audit.get("claim_ceiling") !=
            "exact bootstrap receipt only; no model or benchmark claim"):
        raise ContractError("M19b result audit contract drifted")
    receipt, _ = load_verified_json(binding["receipt"])
    provenance, _ = load_verified_module(
        "m21_bound_m19_provenance", binding["m19_provenance_runner"])
    m18, _ = load_verified_module(
        "m21_bound_m18_smoke", binding["m18_runner"])
    repository = provenance.direct_git_identity(REPOSITORY_ROOT)
    if (repository["commit"] != binding["repository"]["commit"] or
            repository["branch"] != binding["repository"]["branch"] or
            repository["clean"] is not True or
            binding["repository"].get("clean_required") is not True):
        raise ContractError("M21a repository identity drifted")
    if (Path(sys.executable).resolve() != EXPECTED_PYTHON or
            regular_file_record(EXPECTED_PYTHON) != binding["python"]):
        raise ContractError("M21a Python identity drifted")
    source_records = exact_record_set(collect_file_records(spec))
    if (not source_records or args.attempt_root.exists() or
            args.scientific_output.exists()):
        raise ContractError("M21a output or source precondition drifted")
    return {
        "audit": audit,
        "receipt": receipt,
        "provenance": provenance,
        "m18": m18,
        "repository": repository,
        "records": actual_records,
        "source_records_checked": len(source_records),
    }


def exact_journal(root, expected):
    names = []
    for path in Path(root).iterdir():
        metadata = path.lstat()
        if (not stat.S_ISREG(metadata.st_mode) or path.is_symlink() or
                metadata.st_nlink != 1):
            return False, sorted(item.name for item in Path(root).iterdir())
        names.append(path.name)
    return set(names) == set(expected), sorted(names)


def seal_journal(root, start_path, terminal, observer):
    terminal_path = root / "terminal.json"
    manifest_path = root / "manifest.json"
    observer.set_phase("journal_publication")
    write_json_atomic(terminal_path, terminal)
    os.chmod(start_path, 0o444)
    os.chmod(terminal_path, 0o444)
    start_record = regular_file_record(start_path)
    terminal_record = regular_file_record(terminal_path)
    post_terminal_observation = frozen_observation(observer)
    observer.capture = False
    manifest = {
        "schema": "sttrack-lachtt-m21a-immutable-phase-closed-runtime-smoke-"
                  "manifest/v1",
        "complete": True,
        "status": terminal["status"],
        "accepted": terminal["accepted"],
        "exit_code": terminal["exit_code"],
        "expected_file_set": ["manifest.json", "start.json", "terminal.json"],
        "files": {
            "start.json": start_record,
            "terminal.json": terminal_record,
        },
        "post_terminal_runtime_observation": post_terminal_observation,
        "claim_ceiling": terminal["claim_ceiling"],
    }
    write_json_atomic(manifest_path, manifest)
    os.chmod(manifest_path, 0o444)
    exact, names = exact_journal(
        root, {"manifest.json", "start.json", "terminal.json"})
    if not exact:
        raise ContractError(
            "M21a sealed journal file set drifted: {}".format(names))
    for path in (start_path, terminal_path, manifest_path):
        if regular_file_record(path)["mode"] != "0444":
            raise ContractError("M21a journal mode drifted")
    os.chmod(root, 0o555)


def publication_safe_terminal(terminal, claim_ceiling):
    try:
        json.dumps(terminal, ensure_ascii=False, sort_keys=True,
                   allow_nan=False)
        return terminal
    except (TypeError, ValueError) as error:
        return {
            "schema": (
                "sttrack-lachtt-m21a-immutable-phase-closed-runtime-smoke-"
                "terminal/v1"),
            "complete": True,
            "status": "publication_payload_failure",
            "accepted": False,
            "exit_code": 1,
            "claim_ceiling": claim_ceiling,
            "exception": {
                "type": error.__class__.__name__,
                "message": str(error),
            },
            "optimizer_steps": terminal.get("optimizer_steps"),
            "checkpoint_written": terminal.get("checkpoint_written"),
            "authorization": {
                "independent_result_audit": True,
                "training": False,
                "checkpoint": False,
                "public_evaluation": False,
                "automatic_next_stage": False,
            },
        }


def recover_failed_journal(root, start_path, start, error, observer,
                           claim_ceiling, optimizer_steps=None,
                           checkpoint_written=None):
    observer.capture = False
    os.chmod(root, 0o755)
    for name in ("start.json.tmp", "terminal.json.tmp", "manifest.json.tmp"):
        temporary = root / name
        if temporary.exists():
            temporary.unlink()
    terminal = {
        "schema": (
            "sttrack-lachtt-m21a-immutable-phase-closed-runtime-smoke-"
            "terminal/v1"),
        "complete": True,
        "status": "journal_publication_failure",
        "accepted": False,
        "exit_code": 1,
        "claim_ceiling": claim_ceiling,
        "exception": {
            "type": error.__class__.__name__,
            "message": str(error),
        },
        "optimizer_steps": optimizer_steps,
        "checkpoint_written": checkpoint_written,
        "authorization": {
            "independent_result_audit": True,
            "training": False,
            "checkpoint": False,
            "public_evaluation": False,
            "automatic_next_stage": False,
        },
    }
    write_json_atomic(start_path, start)
    terminal_path = root / "terminal.json"
    manifest_path = root / "manifest.json"
    write_json_atomic(terminal_path, terminal)
    os.chmod(start_path, 0o444)
    os.chmod(terminal_path, 0o444)
    manifest = {
        "schema": "sttrack-lachtt-m21a-immutable-phase-closed-runtime-smoke-"
                  "manifest/v1",
        "complete": True,
        "status": terminal["status"],
        "accepted": False,
        "exit_code": 1,
        "expected_file_set": ["manifest.json", "start.json", "terminal.json"],
        "files": {
            "start.json": regular_file_record(start_path),
            "terminal.json": regular_file_record(terminal_path),
        },
        "post_terminal_runtime_observation": None,
        "claim_ceiling": claim_ceiling,
    }
    write_json_atomic(manifest_path, manifest)
    os.chmod(manifest_path, 0o444)
    exact, names = exact_journal(
        root, {"manifest.json", "start.json", "terminal.json"})
    if not exact:
        raise ContractError(
            "M21a recovery journal file set drifted: {}".format(names))
    os.chmod(root, 0o555)
    return terminal


def main():
    args = parse_args()
    args.spec = args.spec.resolve()
    args.binding = args.binding.resolve()
    args.attempt_root = args.attempt_root.resolve()
    args.scientific_output = args.scientific_output.resolve()
    spec_record = regular_file_record(args.spec)
    binding_record = regular_file_record(args.binding)
    spec, _ = load_verified_json(spec_record)
    binding, _ = load_verified_json(binding_record)
    controls = validate_control_contract(args, spec, binding)

    args.attempt_root.mkdir(mode=0o755, parents=False, exist_ok=False)
    root = args.attempt_root
    start_path = root / "start.json"
    provenance = controls["provenance"]
    observer = provenance.ProvenanceObserver(root, Path(__file__).resolve())
    sys.addaudithook(observer.hook)
    start = {
        "schema": "sttrack-lachtt-m21a-immutable-phase-closed-runtime-smoke-"
                  "start/v1",
        "complete": True,
        "phase": "observer_installed",
        "argv": list(sys.argv),
        "pid": os.getpid(),
        "python": str(Path(sys.executable).resolve()),
        "requested_paths": {
            "spec": str(args.spec),
            "binding": str(args.binding),
            "attempt_root": str(args.attempt_root),
            "scientific_output": str(args.scientific_output),
        },
        "repository": controls["repository"],
        "spec": spec_record,
        "binding": binding_record,
        "source_records_checked": controls["source_records_checked"],
        "optimizer_steps": 0,
        "checkpoint_written": False,
    }
    try:
        write_json_atomic(start_path, start)
    except Exception as error:
        terminal = recover_failed_journal(
            root, start_path, start, error, observer, spec["claim_ceiling"])
        return terminal["exit_code"]

    observer.allow_source_paths(
        record["path"] for record in collect_file_records(spec))
    status = "exception"
    exit_code = 1
    accepted = False
    smoke_result = None
    execution_counts = None
    bootstrap_event = None
    bootstrap_observation = None
    runtime_observation = None
    runtime_events = None
    postflight_observation = None
    postflight_events = None
    final_repository = None
    controls_exact = False
    exception_record = None
    original_popen = None
    subprocess_module = None
    try:
        observer.set_phase("subprocess_wrapper_installed")
        subprocess_module, original_popen = provenance.install_traced_popen(
            observer)
        observer.set_phase("torch_import")
        torch_module = importlib.import_module("torch")
        torch_identity = {
            "version": str(torch_module.__version__),
            "module_path": str(Path(torch_module.__file__).resolve()),
            "cuda_version": str(torch_module.version.cuda),
        }
        if torch_identity != spec["torch"]:
            raise ContractError("M21a torch identity drifted")
        bootstrap_observation = frozen_observation(observer)
        bootstrap_event = bootstrap_event_from_observation(
            bootstrap_observation, Path(__file__).resolve(),
            binding["m19_provenance_runner"]["path"])
        validate_receipt(controls["receipt"], bootstrap_event)
        if (bootstrap_observation["forbidden_write_events"] or
                bootstrap_observation["unresolved_write_events"] or
                bootstrap_observation["forbidden_mutation_events"] or
                bootstrap_observation["network_events"] or
                bootstrap_observation["sensitive_read_events"] or
                bootstrap_observation["forbidden_modules"]):
            raise ContractError("M21a bootstrap side-effect surface drifted")

        observer.set_phase("model_runtime")
        instrumentation = provenance.ProjectExecutionInstrumentation(
            torch_module)
        try:
            instrumentation.install()
            controls["m18"].load_project_components(
                REPOSITORY_ROOT, spec, binding)
            smoke_result = controls["m18"].run_smoke(spec)
        finally:
            instrumentation.restore()
        execution_counts = dict(instrumentation.counts)
        runtime_observation = frozen_observation(observer)
        runtime_events = model_runtime_events(runtime_observation)
        runtime_clean = (
            not runtime_events["popen_calls"] and
            not runtime_events["subprocess_audit_events"] and
            not runtime_events["devnull_write_events"] and
            not runtime_events["all_write_events"] and
            not runtime_events["unresolved_write_events"] and
            not runtime_events["all_mutation_events"] and
            not runtime_events["network_events"] and
            not runtime_events["sensitive_read_events"] and
            not runtime_observation["forbidden_modules"])
        observer.set_phase("postflight_identity")
        final_repository = provenance.direct_git_identity(REPOSITORY_ROOT)
        controls_exact = (
            regular_file_record(args.binding) == binding_record and
            all(
                regular_file_record(record["path"]) == record
                for record in controls["records"].values()))
        observer.set_phase("postflight_complete")
        postflight_observation = frozen_observation(observer)
        postflight_events = phase_events(
            postflight_observation, "postflight_identity")
        postflight_clean = (
            validate_postflight_events(
                postflight_events,
                binding["m19_provenance_runner"]["path"]) and
            postflight_observation["imports"] ==
            runtime_observation["imports"] and
            not postflight_observation["forbidden_modules"])
        snapshots_phase_closed = (
            bootstrap_observation["phase_at_snapshot"] == "torch_import" and
            runtime_observation["phase_at_snapshot"] == "model_runtime" and
            not any(
                row.get("phase") in {
                    "model_runtime", "postflight_identity",
                    "postflight_complete",
                }
                for key in (
                    "popen_calls", "subprocess_audit_events",
                    "devnull_write_events", "all_write_events",
                    "unresolved_write_events", "all_mutation_events",
                    "network_events", "sensitive_read_events",
                )
                for row in bootstrap_observation[key]) and
            not any(
                row.get("phase") in {
                    "postflight_identity", "postflight_complete",
                }
                for key in (
                    "popen_calls", "subprocess_audit_events",
                    "devnull_write_events", "all_write_events",
                    "unresolved_write_events", "all_mutation_events",
                    "network_events", "sensitive_read_events",
                )
                for row in runtime_observation[key]))
        required_model_gate_keys = spec["smoke"]["required_model_gate_keys"]
        model_gate_set_exact = (
            len(required_model_gate_keys) ==
            int(spec["smoke"]["required_model_gate_count"]) and
            set(smoke_result["gates"]) == set(required_model_gate_keys))
        engineering_gates = {
            "bootstrap_receipt_exact": True,
            "bootstrap_event_only_in_frozen_phase":
                bootstrap_event["phase"] in BOOTSTRAP_PHASES,
            "model_runtime_new_side_effects_zero": runtime_clean,
            "observation_snapshots_immutable_and_phase_closed":
                snapshots_phase_closed,
            "postflight_identity_reads_exact_and_isolated": postflight_clean,
            "postflight_sensitive_reads_observed":
                len(postflight_events["sensitive_read_events"]) > 0,
            "m18_model_gate_set_exact": model_gate_set_exact,
            "m18_model_gates_all_pass": (
                model_gate_set_exact and
                all(smoke_result["gates"].values())),
            "model_instantiations_positive":
                execution_counts["model_instantiations"] > 0,
            "forward_calls_positive":
                execution_counts["forward_call_entries"] > 0,
            "tensor_dispatch_positive":
                execution_counts["tensor_dispatch_ops"] > 0,
            "optimizer_constructions_zero":
                execution_counts["optimizer_constructions"] == 0,
            "optimizer_steps_zero":
                execution_counts["optimizer_step_entries"] == 0,
            "checkpoint_writes_zero":
                execution_counts["checkpoint_write_entries"] == 0,
            "repository_identity_still_exact":
                final_repository == controls["repository"],
            "control_identities_still_exact": controls_exact,
            "scientific_output_absent": not args.scientific_output.exists(),
        }
        accepted = all(engineering_gates.values())
        smoke_result["m21_engineering_gates"] = engineering_gates
        smoke_result["accepted"] = accepted
        status = "success" if accepted else "gate_failure"
        exit_code = 0 if accepted else 2
    except BaseException as error:
        exception_record = {
            "type": error.__class__.__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        if runtime_observation is None:
            runtime_observation = frozen_observation(observer)
        runtime_events = model_runtime_events(runtime_observation)
        postflight_observation = frozen_observation(observer)
        postflight_events = phase_events(
            postflight_observation, "postflight_identity")
        status = "exception"
        exit_code = 1
    finally:
        if subprocess_module is not None and original_popen is not None:
            subprocess_module.Popen = original_popen

    terminal = {
        "schema": "sttrack-lachtt-m21a-immutable-phase-closed-runtime-smoke-"
                  "terminal/v1",
        "complete": True,
        "status": status,
        "accepted": accepted,
        "exit_code": exit_code,
        "claim_ceiling": spec["claim_ceiling"],
        "bootstrap_event": bootstrap_event,
        "bootstrap_observation": bootstrap_observation,
        "model_runtime_observation": runtime_observation,
        "model_runtime_events": runtime_events,
        "postflight_observation": postflight_observation,
        "postflight_identity_events": postflight_events,
        "final_repository": final_repository,
        "control_identities_still_exact": controls_exact,
        "execution_counts": execution_counts,
        "smoke_result": smoke_result,
        "exception": exception_record,
        "optimizer_steps": (
            None if execution_counts is None else
            execution_counts["optimizer_step_entries"]),
        "checkpoint_written": (
            None if execution_counts is None else
            execution_counts["checkpoint_write_entries"] > 0),
        "scientific_output_exists": args.scientific_output.exists(),
        "authorization": {
            "independent_result_audit": True,
            "training": False,
            "checkpoint": False,
            "public_evaluation": False,
            "automatic_next_stage": False,
        },
    }
    terminal = publication_safe_terminal(terminal, spec["claim_ceiling"])
    try:
        seal_journal(root, start_path, terminal, observer)
    except Exception as error:
        terminal = recover_failed_journal(
            root, start_path, start, error, observer, spec["claim_ceiling"],
            optimizer_steps=terminal.get("optimizer_steps"),
            checkpoint_written=terminal.get("checkpoint_written"))
    return terminal["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
