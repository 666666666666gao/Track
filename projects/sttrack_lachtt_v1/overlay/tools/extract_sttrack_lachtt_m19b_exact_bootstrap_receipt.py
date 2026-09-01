#!/usr/bin/env python3
"""Mechanically derive an exact bootstrap receipt from the sealed M19a journal.

This program deliberately uses only the Python standard library.  It does not
import torch or project model modules and it never executes a subprocess.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SPEC_SCHEMA = "sttrack-lachtt-m19b-exact-bootstrap-receipt-spec/v1"
BINDING_SCHEMA = "sttrack-lachtt-m19b-exact-bootstrap-receipt-binding/v1"
RECEIPT_SCHEMA = "sttrack-lachtt-exact-bootstrap-receipt/v1"
START_SCHEMA = "sttrack-lachtt-m19b-exact-bootstrap-receipt-start/v1"
TERMINAL_SCHEMA = "sttrack-lachtt-m19b-exact-bootstrap-receipt-terminal/v1"
MANIFEST_SCHEMA = "sttrack-lachtt-m19b-exact-bootstrap-receipt-manifest/v1"
EXPECTED_OUTPUTS = ("manifest.json", "receipt.json", "start.json", "terminal.json")
EXPECTED_M19A_INPUT_KEYS = {
    "binding",
    "manifest_json",
    "plan",
    "preexecution_audit",
    "result_audit",
    "spec",
    "start_json",
    "terminal_json",
}


class ContractError(RuntimeError):
    """Raised when a frozen input or derivation invariant does not hold."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mode_string(path: Path) -> str:
    return f"{stat.S_IMODE(path.stat().st_mode):04o}"


def file_identity(path: Path) -> Dict[str, Any]:
    require(path.is_absolute(), f"identity path must be absolute: {path}")
    info = os.lstat(str(path))
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ContractError(f"not a no-follow regular file: {path}")
    require(info.st_nlink == 1, f"hard-linked identity input is forbidden: {path}")
    return {
        "path": str(path),
        "bytes": info.st_size,
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "sha256": sha256_file(path),
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def validate_identity(record: Mapping[str, Any], *, expected_path: Path | None = None) -> Dict[str, Any]:
    path = Path(os.path.abspath(str(record["path"])))
    if expected_path is not None:
        expected_absolute = Path(os.path.abspath(str(expected_path)))
        require(path == expected_absolute, f"path mismatch: {path} != {expected_absolute}")
    actual = file_identity(path)
    for field in ("bytes", "mode", "sha256"):
        require(actual[field] == record[field], f"identity mismatch for {path}: {field}")
    return actual


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    payload = canonical_bytes(value)
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def resolve_git_dir(repository: Path) -> Path:
    marker = repository / ".git"
    if marker.is_dir():
        return marker.resolve()
    require(marker.is_file(), f"missing .git marker: {repository}")
    text = marker.read_text(encoding="utf-8").strip()
    require(text.startswith("gitdir:"), f"unsupported .git file: {marker}")
    target = Path(text.split(":", 1)[1].strip())
    if not target.is_absolute():
        target = marker.parent / target
    return target.resolve()


def resolve_common_git_dir(git_dir: Path) -> Path:
    marker = git_dir / "commondir"
    if not marker.is_file():
        return git_dir
    text = marker.read_text(encoding="utf-8").strip()
    require(bool(text), f"empty commondir file: {marker}")
    common_dir = Path(text)
    if not common_dir.is_absolute():
        common_dir = git_dir / common_dir
    common_dir = common_dir.resolve()
    require(common_dir.is_dir(), f"invalid common git directory: {common_dir}")
    return common_dir


def read_repository_head(repository: Path) -> Tuple[str, str]:
    git_dir = resolve_git_dir(repository)
    common_dir = resolve_common_git_dir(git_dir)
    head_text = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if head_text.startswith("ref:"):
        ref_name = head_text.split(":", 1)[1].strip()
        loose_ref = common_dir / ref_name
        if loose_ref.is_file():
            commit = loose_ref.read_text(encoding="utf-8").strip()
        else:
            commit = ""
            packed_refs = common_dir / "packed-refs"
            if packed_refs.is_file():
                for line in packed_refs.read_text(encoding="utf-8").splitlines():
                    if not line or line.startswith("#") or line.startswith("^"):
                        continue
                    candidate, candidate_ref = line.split(" ", 1)
                    if candidate_ref == ref_name:
                        commit = candidate
                        break
        require(bool(commit), f"cannot resolve git ref: {ref_name}")
        branch = ref_name[len("refs/heads/") :] if ref_name.startswith("refs/heads/") else ref_name
    else:
        commit = head_text
        branch = "HEAD"
    require(len(commit) == 40 and all(ch in "0123456789abcdef" for ch in commit.lower()), "invalid commit")
    return commit.lower(), branch


def find_exact(items: Iterable[Mapping[str, Any]], predicate, label: str) -> Mapping[str, Any]:
    matches = [item for item in items if predicate(item)]
    require(len(matches) == 1, f"expected exactly one {label}, found {len(matches)}")
    return matches[0]


def frozen_caller(spec: Mapping[str, Any], stack: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    expected = spec["expected_event"]["first_attribution_caller"]
    matches = [
        frame
        for frame in stack
        if frame.get("file") == expected["file"]
        and frame.get("function") == expected["function"]
        and frame.get("line") == expected["line"]
    ]
    require(len(matches) == 1, f"expected one frozen attribution caller, found {len(matches)}")
    return {
        "file": matches[0]["file"],
        "function": matches[0]["function"],
        "line": matches[0]["line"],
    }


def build_receipt(
    spec: Mapping[str, Any],
    popen: Mapping[str, Any],
    audit_event: Mapping[str, Any],
    devnull: Mapping[str, Any],
    caller: Mapping[str, Any],
) -> Dict[str, Any]:
    expected = spec["expected_event"]
    require(popen["correlation_id"] == audit_event["correlation_id"] == devnull["correlation_id"], "correlation mismatch")
    require(popen["correlation_id"] == expected["correlation_id"], "unexpected correlation id")
    require(popen["phase"] == audit_event["phase"] == devnull["phase"] == expected["phase"], "phase mismatch")
    require(popen["command"] == audit_event["command"] == expected["argv"], "argv mismatch")
    require(popen["effective_executable"] == expected["executable"], "executable mismatch")
    require(popen["effective_cwd"] == expected["effective_cwd"], "cwd mismatch")
    require(popen["stdin"] == expected["stdin"], "stdin mode mismatch")
    require(popen["stdout"] == expected["stdout"], "stdout mode mismatch")
    require(popen["stderr"] == expected["stderr"], "stderr mode mismatch")
    require(popen["event_id"] == expected["popen_wrapper_event_id"], "Popen wrapper event id mismatch")
    require(audit_event["event_id"] == expected["popen_audit_event_id"], "Popen audit event id mismatch")
    require(devnull["event_id"] == expected["devnull_event_id"], "devnull event id mismatch")
    require(audit_event["event"] == "subprocess.Popen", "audit event mismatch")
    require(devnull["path"] == expected["devnull_path"], "devnull path mismatch")
    require(devnull["flags"] == expected["devnull_flags"], "devnull flags mismatch")
    require(devnull["mode_semantics"] == expected["devnull_mode_semantics"], "devnull mode mismatch")
    require(caller == expected["first_attribution_caller"], "attribution caller mismatch")
    return {
        "schema": RECEIPT_SCHEMA,
        "complete": True,
        "scope": "exact dependency bootstrap prefix for the frozen identity only",
        "claim_ceiling": spec["claim_ceiling"],
        "wildcards_allowed": False,
        "prefix_or_directory_allowance": False,
        "model_runtime_allowance": [],
        "python": spec["python"],
        "torch": spec["torch"],
        "causal_source_identities": spec["causal_source_identities"],
        "event": {
            "phase": popen["phase"],
            "name": audit_event["event"],
            "correlation_id": popen["correlation_id"],
            "popen_wrapper_event_id": popen["event_id"],
            "popen_audit_event_id": audit_event["event_id"],
            "devnull_event_id": devnull["event_id"],
            "executable": popen["effective_executable"],
            "resolved_executable_identity": spec["resolved_executable_identity"],
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
            "first_attribution_caller": dict(caller),
        },
        "source_journal": spec["m19a_inputs"],
    }


def derive_path_a(spec: Mapping[str, Any], terminal: Mapping[str, Any]) -> Dict[str, Any]:
    analysis = terminal["analysis"]
    observation = terminal["observation"]
    pair = find_exact(analysis["linked_pairs"], lambda _: True, "linked pair")
    popen = find_exact(observation["popen_calls"], lambda item: item["event_id"] == pair["popen_call_event_id"], "linked Popen")
    devnull = find_exact(observation["devnull_write_events"], lambda item: item["event_id"] == pair["devnull_event_id"], "linked devnull write")
    audit_event = find_exact(
        observation["subprocess_audit_events"],
        lambda item: item["correlation_id"] == pair["correlation_id"],
        "linked Popen audit event",
    )
    caller = dict(pair["first_attribution_caller"])
    frozen_caller(spec, popen["stack"])
    frozen_caller(spec, audit_event["stack"])
    frozen_caller(spec, devnull["stack"])
    return build_receipt(spec, popen, audit_event, devnull, caller)


def derive_path_b(spec: Mapping[str, Any], terminal: Mapping[str, Any]) -> Dict[str, Any]:
    observation = terminal["observation"]
    expected = spec["expected_event"]
    popen = find_exact(
        observation["popen_calls"],
        lambda item: item["phase"] == expected["phase"] and item["command"] == expected["argv"],
        "raw Popen",
    )
    audit_event = find_exact(
        observation["subprocess_audit_events"],
        lambda item: item["correlation_id"] == popen["correlation_id"]
        and item["phase"] == popen["phase"]
        and item["command"] == popen["command"],
        "raw Popen audit event",
    )
    devnull = find_exact(
        observation["all_write_events"],
        lambda item: item.get("correlation_id") == popen["correlation_id"]
        and item.get("phase") == popen["phase"]
        and item.get("path") == expected["devnull_path"],
        "raw correlated devnull write",
    )
    caller = frozen_caller(spec, popen["stack"])
    require(caller == frozen_caller(spec, audit_event["stack"]), "Popen/audit caller mismatch")
    require(caller == frozen_caller(spec, devnull["stack"]), "Popen/devnull caller mismatch")
    return build_receipt(spec, popen, audit_event, devnull, caller)


def validate_m19a_inputs(spec: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    inputs = spec["m19a_inputs"]
    require(set(inputs) == EXPECTED_M19A_INPUT_KEYS, "M19a input key set is not exact")
    for record in inputs.values():
        validate_identity(record)
    start = read_json(Path(inputs["start_json"]["path"]))
    terminal = read_json(Path(inputs["terminal_json"]["path"]))
    manifest = read_json(Path(inputs["manifest_json"]["path"]))
    require(start["complete"] is True, "M19a start incomplete")
    require(terminal["status"] == "success" and terminal["accepted"] is True, "M19a terminal not accepted")
    require(terminal["exit_code"] == 0 and all(terminal["gates"].values()), "M19a gates not all true")
    require(manifest["complete"] is True and manifest["status"] == "success" and manifest["exit_code"] == 0, "M19a manifest not successful")
    require(tuple(sorted(manifest["expected_file_set"])) == ("manifest.json", "start.json", "terminal.json"), "M19a expected file set mismatch")
    root = Path(spec["m19a_attempt_root"])
    root_info = os.lstat(str(root))
    require(not stat.S_ISLNK(root_info.st_mode) and stat.S_ISDIR(root_info.st_mode), "M19a root is not a no-follow directory")
    require(mode_string(root) == "0555", "M19a root mode mismatch")
    actual_names = tuple(sorted(path.name for path in root.iterdir()))
    require(actual_names == ("manifest.json", "start.json", "terminal.json"), "M19a actual file set mismatch")
    return start, terminal, manifest


def validate_preflight(spec_path: Path, binding_path: Path, attempt_root: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    require(not attempt_root.exists(), f"attempt root already exists: {attempt_root}")
    spec = read_json(spec_path)
    binding = read_json(binding_path)
    require(spec.get("schema") == SPEC_SCHEMA and spec.get("complete") is True, "invalid spec")
    require(binding.get("schema") == BINDING_SCHEMA and binding.get("complete") is True, "invalid binding")
    require(Path(spec["runtime"]["attempt_root"]).resolve() == attempt_root.resolve(), "attempt root differs from spec")
    validate_identity(binding["spec"], expected_path=spec_path)
    require(binding["plan"] == spec["plan"], "binding/spec plan identity mismatch")
    require(binding["extractor"] == spec["extractor"], "binding/spec extractor identity mismatch")
    require(binding["m19a_result_audit"] == spec["m19a_inputs"]["result_audit"], "binding/spec result-audit identity mismatch")
    validate_identity(binding["plan"])
    validate_identity(binding["extractor"], expected_path=Path(__file__))
    validate_identity(binding["preexecution_audit"])
    validate_identity(binding["m19a_result_audit"])
    preexecution_audit = read_json(Path(binding["preexecution_audit"]["path"]))
    require(preexecution_audit.get("complete") is True, "preexecution audit incomplete")
    require(preexecution_audit.get("overall_verdict") == "PASS", "preexecution audit did not PASS")
    require(str(preexecution_audit.get("integrity_status", "")).lower() == "pass", "preexecution audit integrity did not pass")
    require(
        preexecution_audit.get("claim_ceiling") == "exact bootstrap receipt only; no model or benchmark claim",
        "preexecution audit claim ceiling mismatch",
    )
    authorized_actions = preexecution_audit.get("authorization_boundary", {}).get("authorized_next_actions_after_pass", [])
    require(
        len(authorized_actions) == 1 and "M19b" in authorized_actions[0] and "exact bootstrap receipt" in authorized_actions[0],
        "preexecution audit does not authorize exactly one M19b receipt extraction",
    )
    result_audit = read_json(Path(binding["m19a_result_audit"]["path"]))
    require(result_audit.get("verdict") == "PASS", "M19a result audit did not PASS")
    require(result_audit.get("integrity_status") == "PASS", "M19a result audit integrity did not PASS")
    require(result_audit.get("claim_supported") is True, "M19a result audit claim is unsupported")
    require(
        result_audit.get("claim_ceiling") == "import-only runtime provenance; no model or benchmark claim",
        "M19a result audit claim ceiling mismatch",
    )
    for record in spec["causal_source_identities"].values():
        validate_identity(record)
    validate_identity(spec["resolved_executable_identity"])
    validate_identity(spec["python"])
    for field in ("path", "branch", "commit"):
        require(binding["repository"][field] == spec["repository"][field], f"binding/spec repository {field} mismatch")
    require(spec["repository"]["clean_preexecution_required"] is True, "spec does not require a clean repository")
    repository_path = Path(os.path.abspath(binding["repository"]["path"]))
    repository_info = os.lstat(str(repository_path))
    require(not stat.S_ISLNK(repository_info.st_mode) and stat.S_ISDIR(repository_info.st_mode), "repository is not a no-follow directory")
    repository = repository_path
    commit, branch = read_repository_head(repository)
    require(commit == binding["repository"]["commit"], "repository commit mismatch")
    require(branch == binding["repository"]["branch"], "repository branch mismatch")
    require(binding["repository"]["clean_preexecution"] is True, "repository clean preexecution not attested")
    require(binding["authorization"]["m19b_receipt"] is True, "M19b is not authorized")
    for forbidden in ("torch_import", "model_smoke", "training", "checkpoint", "public_evaluation", "automatic_next_stage"):
        require(binding["authorization"][forbidden] is False, f"forbidden authorization enabled: {forbidden}")
    return spec, binding


def seal_output_root(attempt_root: Path) -> None:
    for name in EXPECTED_OUTPUTS:
        path = attempt_root / name
        require(path.is_file() and not path.is_symlink(), f"missing or unsafe output: {path}")
        require(path.stat().st_nlink == 1, f"hard link not allowed: {path}")
        os.chmod(path, 0o444)
    require(tuple(sorted(path.name for path in attempt_root.iterdir())) == EXPECTED_OUTPUTS, "unexpected output file set")
    manifest = read_json(attempt_root / "manifest.json")
    require(manifest.get("schema") == MANIFEST_SCHEMA and manifest.get("complete") is True, "invalid manifest contract")
    require(manifest.get("status") in {"success", "contract_failure"}, "invalid manifest status")
    require(tuple(sorted(manifest.get("expected_file_set", []))) == EXPECTED_OUTPUTS, "manifest output set mismatch")
    for name, recorded in manifest.get("files", {}).items():
        require(name in {"receipt.json", "start.json", "terminal.json"}, f"unexpected manifest file: {name}")
        require(recorded == file_identity(attempt_root / name), f"manifest identity mismatch: {name}")
    require(set(manifest.get("files", {})) == {"receipt.json", "start.json", "terminal.json"}, "manifest files incomplete")
    os.chmod(attempt_root, 0o555)


def success_manifest(attempt_root: Path) -> Dict[str, Any]:
    files = {
        name: file_identity(attempt_root / name)
        for name in ("receipt.json", "start.json", "terminal.json")
    }
    return {
        "schema": MANIFEST_SCHEMA,
        "complete": True,
        "status": "success",
        "exit_code": 0,
        "expected_file_set": list(EXPECTED_OUTPUTS),
        "files": files,
        "manifest_self_write_contract": {
            "path": str(attempt_root / "manifest.json"),
            "external_sha256_required": True,
        },
    }


def failure_manifest(attempt_root: Path, error: str) -> Dict[str, Any]:
    files = {
        name: file_identity(attempt_root / name)
        for name in ("receipt.json", "start.json", "terminal.json")
    }
    return {
        "schema": MANIFEST_SCHEMA,
        "complete": True,
        "status": "contract_failure",
        "exit_code": 2,
        "error": error,
        "expected_file_set": list(EXPECTED_OUTPUTS),
        "files": files,
        "manifest_self_write_contract": {
            "path": str(attempt_root / "manifest.json"),
            "external_sha256_required": True,
        },
    }


def publish_failure(
    attempt_root: Path,
    spec_path: Path,
    binding_path: Path,
    error: BaseException,
) -> None:
    """Best-effort fail-closed publication after the attempt root is consumed."""
    message = f"{type(error).__name__}: {error}"
    attempt_root.mkdir(mode=0o700, parents=False, exist_ok=True)
    for name in EXPECTED_OUTPUTS:
        path = attempt_root / name
        if path.exists() and not path.is_symlink():
            os.chmod(path, 0o600)
    start_path = attempt_root / "start.json"
    if not start_path.exists():
        atomic_write_json(
            start_path,
            {
                "schema": START_SCHEMA,
                "complete": True,
                "spec_path": str(spec_path),
                "binding_path": str(binding_path),
                "requested_attempt_root": str(attempt_root),
                "publication": "failure_recovery",
            },
        )
    atomic_write_json(
        attempt_root / "receipt.json",
        {
            "schema": RECEIPT_SCHEMA,
            "complete": False,
            "accepted": False,
            "error": message,
            "model_runtime_allowance": [],
            "wildcards_allowed": False,
        },
    )
    atomic_write_json(
        attempt_root / "terminal.json",
        {
            "schema": TERMINAL_SCHEMA,
            "complete": True,
            "status": "contract_failure",
            "accepted": False,
            "exit_code": 2,
            "error": message,
            "authorization": {
                "model_smoke": False,
                "training": False,
                "checkpoint": False,
                "public_evaluation": False,
                "automatic_next_stage": False,
            },
        },
    )
    for name in ("receipt.json", "start.json", "terminal.json"):
        os.chmod(attempt_root / name, 0o444)
    atomic_write_json(attempt_root / "manifest.json", failure_manifest(attempt_root, message))
    seal_output_root(attempt_root)


def run(spec_path: Path, binding_path: Path, attempt_root: Path) -> int:
    spec, binding = validate_preflight(spec_path, binding_path, attempt_root)
    _, terminal_m19a, _ = validate_m19a_inputs(spec)

    attempt_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    try:
        start = {
            "schema": START_SCHEMA,
            "complete": True,
            "spec": file_identity(spec_path),
            "binding": file_identity(binding_path),
            "extractor": file_identity(Path(os.path.abspath(__file__))),
            "source_attempt_root": spec["m19a_attempt_root"],
            "requested_attempt_root": str(attempt_root),
            "execution_contract": {
                "torch_imports": 0,
                "project_model_imports": 0,
                "subprocesses": 0,
                "network_events": 0,
                "sensitive_data_reads": 0,
                "model_instantiations": 0,
                "forward_calls": 0,
                "optimizer_steps": 0,
                "checkpoint_operations": 0,
            },
        }
        atomic_write_json(attempt_root / "start.json", start)

        receipt_a = derive_path_a(spec, terminal_m19a)
        receipt_b = derive_path_b(spec, terminal_m19a)
        bytes_a = canonical_bytes(receipt_a)
        bytes_b = canonical_bytes(receipt_b)
        require(bytes_a == bytes_b, "independent receipt derivations are not byte-identical")
        receipt_hash = sha256_bytes(bytes_a)
        atomic_write_json(attempt_root / "receipt.json", receipt_a)

        terminal = {
        "schema": TERMINAL_SCHEMA,
        "complete": True,
        "status": "success",
        "accepted": True,
        "exit_code": 0,
        "claim_ceiling": "exact bootstrap receipt only; no model or benchmark claim",
        "derivation": {
            "path_a": "analysis.linked_pairs joined to raw Popen/devnull/audit events",
            "path_b": "raw event-set reconstruction without linked_pairs",
            "path_a_bytes": len(bytes_a),
            "path_b_bytes": len(bytes_b),
            "path_a_sha256": receipt_hash,
            "path_b_sha256": sha256_bytes(bytes_b),
            "byte_identical": True,
        },
        "gates": {
            "m19a_inputs_exact": True,
            "m19a_terminal_accepted": True,
            "m19a_all_gates_true": True,
            "m19a_seal_exact": True,
            "single_popen_event": True,
            "single_devnull_event": True,
            "correlation_exact": True,
            "phase_exact": True,
            "parameters_exact": True,
            "attribution_caller_exact": True,
            "causal_source_identities_exact": True,
            "independent_derivations_byte_identical": True,
            "no_wildcard_or_prefix_allowance": True,
            "model_runtime_allowance_empty": True,
            "zero_torch_imports": True,
            "zero_project_model_imports": True,
            "zero_subprocesses": True,
            "zero_network_events": True,
            "zero_sensitive_data_reads": True,
            "zero_model_execution": True,
            "zero_checkpoint_operations": True,
            "no_automatic_next_stage": True,
        },
        "authorization": {
            "model_smoke": False,
            "training": False,
            "checkpoint": False,
            "public_evaluation": False,
            "automatic_next_stage": False,
        },
        }
        atomic_write_json(attempt_root / "terminal.json", terminal)
        for name in ("receipt.json", "start.json", "terminal.json"):
            os.chmod(attempt_root / name, 0o444)
        atomic_write_json(attempt_root / "manifest.json", success_manifest(attempt_root))
        seal_output_root(attempt_root)
        return 0
    except Exception as error:
        publish_failure(attempt_root, spec_path, binding_path, error)
        return 2


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return run(args.spec.resolve(), args.binding.resolve(), args.attempt_root.resolve())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as error:
        print(f"M19b contract failure: {error}", file=sys.stderr)
        raise SystemExit(2)
