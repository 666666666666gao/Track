#!/usr/bin/env python3
"""Run the frozen M18a eight-event, zero-optimizer-step engineering smoke."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import traceback
import types
import zlib


REPOSITORY_ROOT = Path(
    "/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1").resolve()
EXPECTED_SPEC_PATH = Path(
    "/home/SUTrack_RGBD_L/refine-logs/"
    "STTRACK_LACHTT_M18A_CAUSAL_SURVIVAL_ARCHITECTURE_JOURNAL_"
    "SMOKE_SPEC_20260901.json").resolve()
EXPECTED_BINDING_PATH = Path(
    "/home/SUTrack_RGBD_L/refine-logs/"
    "STTRACK_LACHTT_M18A_CAUSAL_SURVIVAL_ARCHITECTURE_JOURNAL_"
    "SMOKE_BINDING_20260901.json").resolve()
EXPECTED_ATTEMPT_ROOT = Path(
    "/root/autodl-tmp/"
    "sttrack_lachtt_m18a_architecture_journal_smoke_attempt_v1_20260901"
).resolve()
EXPECTED_SCIENTIFIC_OUTPUT = Path(
    "/root/autodl-tmp/"
    "sttrack_lachtt_m18a_architecture_smoke_scientific_output_v1_20260901"
).resolve()
BRANCH_ORDER = (
    "current_peak0", "current_peak1",
    "last_reliable_peak0", "last_reliable_peak1",
    "velocity_peak0", "velocity_peak1",
)
HORIZONS = (3, 5, 10)
TARGET_METRICS = (
    "branch_mean_iou", "public_mean_iou", "gain",
    "max_consecutive_low_overlap_run_fraction",
)
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

torch = None
build_detached_roi_differences = None
CausalQuantileSurvivalRouter = None
trainable_parameter_count = None
FORBIDDEN_MODULE_NAMES = None


class ContractError(RuntimeError):
    pass


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


def file_record(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def bytes_record(path, payload):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def read_verified_bytes(record):
    path = Path(record["path"]).resolve()
    with path.open("rb") as stream:
        payload = stream.read()
    actual = bytes_record(path, payload)
    if (actual["sha256"] != record["sha256"] or
            (record.get("bytes") is not None and
             actual["bytes"] != int(record["bytes"]))):
        raise ContractError("bound byte payload identity drifted")
    return payload, actual


def load_json_record(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        payload = stream.read()
    return json.loads(payload.decode("utf-8")), bytes_record(path, payload)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def finite_number(value):
    return (not isinstance(value, bool) and
            isinstance(value, (int, float)) and math.isfinite(float(value)))


def collect_hashed_records(value, records=None):
    if records is None:
        records = []
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(
                value.get("sha256"), str):
            records.append({
                "path": value["path"],
                "bytes": value.get("bytes"),
                "sha256": value["sha256"],
            })
        for nested in value.values():
            collect_hashed_records(nested, records)
    elif isinstance(value, list):
        for nested in value:
            collect_hashed_records(nested, records)
    return records


def verify_hashed_records(value):
    checked = []
    seen = set()
    for expected in collect_hashed_records(value):
        key = (expected["path"], expected["sha256"])
        if key in seen:
            continue
        seen.add(key)
        path = Path(expected["path"]).resolve()
        actual_sha = sha256_file(path) if path.is_file() else None
        actual_bytes = path.stat().st_size if path.is_file() else None
        match = (actual_sha == expected["sha256"] and
                 (expected["bytes"] is None or
                  actual_bytes == int(expected["bytes"])))
        checked.append({
            "path": str(path),
            "expected_sha256": expected["sha256"],
            "actual_sha256": actual_sha,
            "expected_bytes": expected["bytes"],
            "actual_bytes": actual_bytes,
            "match": match,
        })
    return checked


def _git_loose_object(common_dir, object_id):
    path = common_dir / "objects" / object_id[:2] / object_id[2:]
    if not path.is_file():
        raise ContractError("bound git object is not loose")
    raw = zlib.decompress(path.read_bytes())
    header, payload = raw.split(b"\0", 1)
    kind, size = header.split(b" ", 1)
    if int(size) != len(payload):
        raise ContractError("git object size drifted")
    return kind.decode("ascii"), payload


def _git_head(git_dir, common_dir):
    head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
    if head.startswith("ref: "):
        reference = head[5:]
        ref_path = git_dir / reference
        if not ref_path.is_file():
            ref_path = common_dir / reference
        if ref_path.is_file():
            commit = ref_path.read_text(encoding="ascii").strip()
        else:
            commit = None
            packed = common_dir / "packed-refs"
            if packed.is_file():
                for line in packed.read_text(encoding="ascii").splitlines():
                    if line and not line.startswith(("#", "^")):
                        object_id, name = line.split(" ", 1)
                        if name == reference:
                            commit = object_id
                            break
            if commit is None:
                raise ContractError("git HEAD reference is unresolved")
        prefix = "refs/heads/"
        branch = reference[len(prefix):] if reference.startswith(
            prefix) else reference
    else:
        commit = head
        branch = "HEAD"
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise ContractError("git HEAD commit drifted")
    return commit, branch


def _git_index_entries(git_dir):
    payload = (git_dir / "index").read_bytes()
    if len(payload) < 32 or hashlib.sha1(payload[:-20]).digest() != payload[-20:]:
        raise ContractError("git index checksum drifted")
    signature, version, count = struct.unpack(">4sLL", payload[:12])
    if signature != b"DIRC" or version not in (2, 3):
        raise ContractError("unsupported git index format")
    offset = 12
    entries = []
    for _ in range(count):
        start = offset
        fixed = payload[offset:offset + 62]
        if len(fixed) != 62:
            raise ContractError("git index entry truncated")
        values = struct.unpack(">LLLLLLLLLL20sH", fixed)
        mode, object_id, flags = values[6], values[10].hex(), values[11]
        offset += 62
        if version >= 3 and flags & 0x4000:
            offset += 2
        terminator = payload.find(b"\0", offset, len(payload) - 20)
        if terminator < 0:
            raise ContractError("git index path is unterminated")
        path = payload[offset:terminator].decode("utf-8", "surrogateescape")
        offset = terminator + 1
        while (offset - start) % 8:
            offset += 1
        if ((flags >> 12) & 0x3) != 0:
            raise ContractError("git index contains an unmerged entry")
        entries.append({"path": path, "mode": mode, "object_id": object_id})
    return entries


def _git_index_tree(entries):
    root = {}
    for entry in entries:
        node = root
        parts = entry["path"].split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ContractError("git index tree collision")
        if parts[-1] in node:
            raise ContractError("duplicate git index entry")
        node[parts[-1]] = (entry["mode"], entry["object_id"])

    def digest_tree(node):
        rows = []
        ordered = sorted(
            node.items(),
            key=lambda row: row[0].encode("utf-8", "surrogateescape") +
            (b"/" if isinstance(row[1], dict) else b""))
        for name, value in ordered:
            encoded = name.encode("utf-8", "surrogateescape")
            if isinstance(value, dict):
                mode, object_id = "40000", digest_tree(value)
            else:
                raw_mode, object_id = value
                mode = format(raw_mode, "o")
            rows.append(mode.encode("ascii") + b" " + encoded + b"\0" +
                        bytes.fromhex(object_id))
        body = b"".join(rows)
        return hashlib.sha1(
            b"tree " + str(len(body)).encode("ascii") + b"\0" + body
        ).hexdigest()

    return digest_tree(root)


def _git_worktree_clean(repo, entries):
    tracked = {entry["path"]: entry for entry in entries}
    submodules = {entry["path"] for entry in entries
                  if entry["mode"] & 0o170000 == 0o160000}
    for relative, entry in tracked.items():
        path = repo / relative
        kind = entry["mode"] & 0o170000
        if kind == 0o160000:
            if not path.is_dir():
                return False
            try:
                submodule_identity = git_identity(path)
            except (OSError, ContractError):
                return False
            if (submodule_identity["commit"] != entry["object_id"] or
                    submodule_identity["clean"] is not True):
                return False
            continue
        if kind == 0o120000:
            if not path.is_symlink():
                return False
            payload = os.readlink(path).encode("utf-8", "surrogateescape")
        else:
            if not path.is_file() or path.is_symlink():
                return False
            payload = path.read_bytes()
            executable = bool(path.stat().st_mode & stat.S_IXUSR)
            if executable != bool(entry["mode"] & 0o111):
                return False
        blob = hashlib.sha1(
            b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
        ).hexdigest()
        if blob != entry["object_id"]:
            return False
    for root, directories, files in os.walk(repo, followlinks=False):
        root_path = Path(root)
        relative_root = root_path.relative_to(repo)
        if relative_root == Path("."):
            directories[:] = [name for name in directories
                              if name not in (".git", ".aris")]
        for name in list(directories):
            relative = str((relative_root / name)).replace("\\", "/")
            if relative.startswith("./"):
                relative = relative[2:]
            path = root_path / name
            if path.is_symlink():
                entry = tracked.get(relative)
                if entry is None or entry["mode"] & 0o170000 != 0o120000:
                    return False
                directories.remove(name)
        directories[:] = [name for name in directories if
                          (str((relative_root / name)).replace("\\", "/")
                           not in submodules)]
        for name in files:
            relative = str((relative_root / name)).replace("\\", "/")
            if relative.startswith("./"):
                relative = relative[2:]
            if relative == ".git":
                continue
            if relative not in tracked:
                return False
    return True


def git_identity(repo):
    dot_git = repo / ".git"
    if dot_git.is_file():
        pointer = dot_git.read_text(encoding="utf-8").strip()
        if not pointer.startswith("gitdir: "):
            raise ContractError("repository gitfile drifted")
        git_dir = Path(pointer[8:])
        if not git_dir.is_absolute():
            git_dir = (repo / git_dir).resolve()
    elif dot_git.is_dir():
        git_dir = dot_git.resolve()
    else:
        raise ContractError("repository git directory drifted")
    common_dir = git_dir
    common_pointer = git_dir / "commondir"
    if common_pointer.is_file():
        common_dir = (git_dir / common_pointer.read_text(
            encoding="utf-8").strip()).resolve()
    commit, branch = _git_head(git_dir, common_dir)
    kind, commit_payload = _git_loose_object(common_dir, commit)
    if kind != "commit":
        raise ContractError("git HEAD is not a commit")
    tree_line = commit_payload.splitlines()[0].decode("ascii")
    if not tree_line.startswith("tree "):
        raise ContractError("git commit tree drifted")
    head_tree = tree_line[5:]
    entries = _git_index_entries(git_dir)
    clean = (_git_index_tree(entries) == head_tree and
             _git_worktree_clean(repo, entries))
    return {"commit": commit, "branch": branch, "clean": clean}


def validate_binding(args, spec, binding):
    claimed_binding_path = binding.get("binding_path")
    if (binding.get("schema") !=
            "sttrack-lachtt-m18a-architecture-journal-smoke-binding/v1" or
            binding.get("complete") is not True or
            not isinstance(claimed_binding_path, str) or
            Path(claimed_binding_path).resolve() != args.binding or
            binding.get("authorization", {}).get(
                "m18a_zero_step_smoke") is not True):
        raise ContractError("M18a binding contract drifted")
    spec_record = binding["spec"]
    runner_record = binding["runner"]
    model_record = binding["model"]
    audit_record = binding["preexecution_audit"]
    if (Path(spec_record["path"]).resolve() != args.spec or
            spec_record["sha256"] != sha256_file(args.spec) or
            Path(runner_record["path"]).resolve() != Path(__file__).resolve() or
            runner_record["sha256"] != sha256_file(Path(__file__).resolve()) or
            model_record["sha256"] != sha256_file(model_record["path"]) or
            audit_record["sha256"] != sha256_file(audit_record["path"])):
        raise ContractError("M18a bound control identity drifted")
    audit_payload, audit_actual_record = read_verified_bytes(audit_record)
    audit = json.loads(audit_payload.decode("utf-8"))
    exact_authorization = (
        "run exactly one M18a eight-event zero-step architecture/journal "
        "smoke if the binding passes its own preflight")
    allowed = audit.get("authorization_boundary", {}).get(
        "authorized_next_actions_after_pass", [])
    expected_audit_identity = {
        "spec_sha256": spec_record["sha256"],
        "runner_sha256": runner_record["sha256"],
        "model_sha256": model_record["sha256"],
        "repository_commit": binding["repository"]["commit"],
        "repository_branch": binding["repository"]["branch"],
        "attempt_root": str(args.attempt_root),
        "scientific_output": str(args.scientific_output),
    }
    if (audit.get("overall_verdict") != "PASS" or
            exact_authorization not in allowed or
            any(audit.get("audited_identity", {}).get(name) != value
                for name, value in expected_audit_identity.items())):
        raise ContractError("M18a preexecution audit identity drifted")
    repo = Path(binding["repository"]["path"]).resolve()
    identity = git_identity(repo)
    if (repo != REPOSITORY_ROOT or
            identity["commit"] != binding["repository"]["commit"] or
            identity["branch"] != binding["repository"]["branch"] or
            identity["clean"] is not True or
            identity["branch"] != spec["repository"]["branch"]):
        raise ContractError("M18a repository identity drifted")
    if (Path(model_record["path"]).resolve() !=
            repo / spec["repository"]["model_relative_path"] or
            Path(__file__).resolve() !=
            repo / spec["repository"]["runner_relative_path"]):
        raise ContractError("M18a code path drifted")
    if (args.attempt_root != Path(
            spec["runtime_journal"]["attempt_root"]).resolve() or
            args.scientific_output != Path(
                spec["scientific_output"]["root"]).resolve() or
            args.attempt_root != Path(binding["attempt_root"]["path"]).resolve() or
             args.scientific_output != Path(
                 binding["scientific_output"]["path"]).resolve() or
             not args.attempt_root.is_dir() or
             args.scientific_output.exists()):
        raise ContractError("M18a output-root precondition drifted")
    return audit, repo, identity, audit_actual_record


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2,
                  sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


class RuntimeAuditObserver:
    MUTATION_PATH_ARGUMENTS = {
        "os.mkdir": ((0, 2),),
        "os.rename": ((0, 2), (1, 3)),
        "os.remove": ((0, 1),),
        "os.rmdir": ((0, 1),),
        "os.symlink": ((1, 2),),
        "os.link": ((0, 2), (1, 3)),
        "os.chdir": ((0, None),),
        "os.chmod": ((0, 2),),
        "os.chown": ((0, 3),),
        "os.utime": ((0, 3),),
        "os.truncate": ((0, None),),
        "os.setxattr": ((0, None),),
        "os.removexattr": ((0, None),),
        "os.fchmod": ((0, None),),
        "os.fchown": ((0, None),),
        "os.ftruncate": ((0, None),),
    }

    def __init__(self, attempt_root):
        self.attempt_root = Path(attempt_root).resolve()
        self.write_paths = set()
        self.unresolved_write_targets = set()
        self.mutation_paths = set()
        self.mutation_events = set()
        self.unresolved_mutation_targets = set()
        self.subprocess_events = set()
        self.network_events = set()
        self.imported_modules = set()
        self._inside_hook = False

    @staticmethod
    def _write_open(mode, flags):
        if isinstance(mode, str) and any(
                token in mode for token in ("w", "a", "x", "+")):
            return True
        if isinstance(flags, int):
            mask = (os.O_WRONLY | os.O_RDWR | os.O_CREAT |
                    os.O_TRUNC | os.O_APPEND)
            return bool(flags & mask)
        return False

    @staticmethod
    def _fd_path(file_descriptor):
        try:
            return Path(os.readlink(
                "/proc/self/fd/{}".format(int(file_descriptor)))).resolve()
        except (OSError, TypeError, ValueError):
            return None

    def _mutation_path(self, value, dir_fd):
        if isinstance(value, int):
            return self._fd_path(value)
        if not isinstance(value, (str, bytes, os.PathLike)):
            return None
        decoded = Path(os.fsdecode(value))
        if decoded.is_absolute():
            return decoded.resolve()
        base = None
        if isinstance(dir_fd, int) and dir_fd >= 0:
            base = self._fd_path(dir_fd)
        if base is None:
            base = Path.cwd()
        return (base / decoded).resolve()

    def hook(self, event, arguments):
        if self._inside_hook:
            return
        self._inside_hook = True
        try:
            if event == "open" and arguments:
                path = arguments[0]
                mode = arguments[1] if len(arguments) > 1 else None
                flags = arguments[2] if len(arguments) > 2 else None
                if self._write_open(mode, flags):
                    resolved = None
                    if isinstance(path, int):
                        resolved = self._fd_path(path)
                    elif isinstance(path, (str, bytes, os.PathLike)):
                        candidate = Path(os.fsdecode(path))
                        if candidate.is_absolute():
                            resolved = candidate.resolve()
                    if resolved is None:
                        self.unresolved_write_targets.add(
                            "open:{}".format(repr(path)))
                    else:
                        self.write_paths.add(str(resolved))
            elif event in self.MUTATION_PATH_ARGUMENTS:
                self.mutation_events.add(event)
                for path_index, dir_fd_index in self.MUTATION_PATH_ARGUMENTS[
                        event]:
                    value = arguments[path_index] if path_index < len(
                        arguments) else None
                    dir_fd = (arguments[dir_fd_index]
                              if dir_fd_index is not None and
                              dir_fd_index < len(arguments) else None)
                    resolved = self._mutation_path(value, dir_fd)
                    if resolved is None:
                        self.unresolved_mutation_targets.add(
                            "{}:{}".format(event, repr(value)))
                    else:
                        self.mutation_paths.add(str(resolved))
            elif event.startswith("subprocess"):
                self.subprocess_events.add(event)
            elif event.startswith("socket.connect") or \
                    event.startswith("socket.getaddrinfo"):
                self.network_events.add(event)
            elif event == "import" and arguments:
                self.imported_modules.add(str(arguments[0]))
        finally:
            self._inside_hook = False

    def snapshot(self):
        forbidden_writes = []
        for value in sorted(self.write_paths):
            resolved = Path(value)
            try:
                resolved.relative_to(self.attempt_root)
            except ValueError:
                forbidden_writes.append(str(resolved))
        forbidden_mutations = []
        for value in sorted(self.mutation_paths):
            resolved = Path(value)
            try:
                resolved.relative_to(self.attempt_root)
            except ValueError:
                forbidden_mutations.append(str(resolved))
        forbidden_modules = sorted(name for name in self.imported_modules
            if ("qwen" in name.lower() or name.startswith("lib.test") or
                name == "vot" or name.startswith("vot.")))
        return {
            "write_paths": sorted(self.write_paths),
            "forbidden_write_paths": forbidden_writes,
            "unresolved_write_targets": sorted(
                self.unresolved_write_targets),
            "mutation_events": sorted(self.mutation_events),
            "mutation_paths": sorted(self.mutation_paths),
            "forbidden_mutation_paths": forbidden_mutations,
            "unresolved_mutation_targets": sorted(
                self.unresolved_mutation_targets),
            "subprocess_events": sorted(self.subprocess_events),
            "network_events": sorted(self.network_events),
            "imported_modules": sorted(self.imported_modules),
            "forbidden_modules": forbidden_modules,
        }


def install_stub_packages(repo):
    packages = {
        "lib": repo / "lib",
        "lib.models": repo / "lib" / "models",
        "lib.models.sttrack": repo / "lib" / "models" / "sttrack",
    }
    for name, path in packages.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module


def _load_verified_module(name, record):
    payload, actual_record = read_verified_bytes(record)
    module = types.ModuleType(name)
    module.__file__ = actual_record["path"]
    module.__package__ = name.rsplit(".", 1)[0]
    sys.modules[name] = module
    try:
        code = compile(payload, actual_record["path"], "exec")
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, actual_record


def load_project_components(repo, spec, binding):
    global torch
    global build_detached_roi_differences
    global CausalQuantileSurvivalRouter
    global trainable_parameter_count
    global FORBIDDEN_MODULE_NAMES
    sys.dont_write_bytecode = True
    torch = importlib.import_module("torch")
    install_stub_packages(repo)
    dependencies = {Path(record["path"]).name: record for record in
                    spec["frozen_inputs"]["relation_dependencies"]}
    expected_dependencies = {
        "lachtt_target_distractor_memory.py",
        "lachtt_rich_roi_relation.py",
        "lachtt_learned_bounded_roi_association.py",
    }
    if set(dependencies) != expected_dependencies:
        raise ContractError("relation dependency set drifted")
    loaded_records = []
    for name, filename in (
            ("lib.models.sttrack.lachtt_target_distractor_memory",
             "lachtt_target_distractor_memory.py"),
            ("lib.models.sttrack.lachtt_rich_roi_relation",
             "lachtt_rich_roi_relation.py"),
            ("lib.models.sttrack.lachtt_learned_bounded_roi_association",
             "lachtt_learned_bounded_roi_association.py")):
        _, actual_record = _load_verified_module(name, dependencies[filename])
        loaded_records.append(actual_record)
    relation = sys.modules[
        "lib.models.sttrack.lachtt_learned_bounded_roi_association"]
    model, model_actual_record = _load_verified_module(
        "lib.models.sttrack.lachtt_causal_quantile_survival",
        binding["model"])
    loaded_records.append(model_actual_record)
    build_detached_roi_differences = relation.build_detached_roi_differences
    CausalQuantileSurvivalRouter = model.CausalQuantileSurvivalRouter
    trainable_parameter_count = model.trainable_parameter_count
    FORBIDDEN_MODULE_NAMES = tuple(model.FORBIDDEN_MODULE_NAMES)
    return loaded_records


def verified_torch_load(record):
    payload, _ = read_verified_bytes(record)
    return torch.load(
        io.BytesIO(payload), map_location="cpu", weights_only=True)


def finite_tensor(value):
    return isinstance(value, torch.Tensor) and \
        torch.isfinite(value).all().item()


def event_key(row):
    return (str(row["sequence"]), int(row["event_id"]),
            int(row["trigger_frame"]))


def load_fixture_targets(spec):
    selected = {event_key(row): row for row in spec["fixture"]["events"]}
    groups = defaultdict(list)
    nontraining_numeric_rows = 0
    record = spec["frozen_inputs"]["m18_0_training_targets"]
    payload, _ = read_verified_bytes(record)
    compressed = gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb")
    with compressed, io.TextIOWrapper(
            compressed, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("record_type") != "action_target":
                continue
            if row.get("partition") != "training":
                nontraining_numeric_rows += 1
                continue
            key = event_key(row)
            if key in selected:
                groups[key].append(row)
    if nontraining_numeric_rows != 0 or set(groups) != set(selected):
        raise ContractError("M18a fixture target partition drifted")
    output = {}
    for key, rows in groups.items():
        rows.sort(key=lambda row: int(row["candidate_role_id"]))
        if (len(rows) != 6 or tuple(row["branch_id"] for row in rows) !=
                BRANCH_ORDER or any(
                    row["strict_event_class"] !=
                    selected[key]["event_class"] for row in rows)):
            raise ContractError("M18a fixture target axis drifted")
        for row in rows:
            if set(row["targets"]) != {"3", "5", "10"}:
                raise ContractError("M18a target horizon drifted")
            for horizon in ("3", "5", "10"):
                target = row["targets"][horizon]
                if set(target) != set(TARGET_METRICS) or any(
                        not finite_number(value) for value in target.values()):
                    raise ContractError("M18a target metric drifted")
        output[key] = rows
    return output


def load_fixture_relations(spec):
    feature_batches = {name: [] for name in FEATURE_SHAPES}
    clip_initial, clip_text, native_rgb, native_depth = [], [], [], []
    for event in spec["fixture"]["events"]:
        feature = verified_torch_load(event["feature"])
        if set(feature) != set(FEATURE_SHAPES):
            raise ContractError("M18a feature key set drifted")
        for name, shape in FEATURE_SHAPES.items():
            if tuple(feature[name].shape) != shape or not finite_tensor(
                    feature[name]):
                raise ContractError("M18a feature tensor drifted")
            feature_batches[name].append(feature[name])
        clip = verified_torch_load(event["clip_anchor"])
        native = verified_torch_load(event["native_anchor"])
        if (set(clip) != {"initial_image", "identity_text"} or
                tuple(clip["initial_image"].shape) != (1, 768) or
                tuple(clip["identity_text"].shape) != (1, 768)):
            raise ContractError("M18a CLIP anchor drifted")
        if (tuple(native["native_template_rgb_tokens"].shape) != (64, 768) or
                tuple(native["native_template_depth_tokens"].shape) !=
                (64, 768)):
            raise ContractError("M18a native anchor drifted")
        clip_initial.append(clip["initial_image"])
        clip_text.append(clip["identity_text"])
        native_rgb.append(native["native_template_rgb_tokens"])
        native_depth.append(native["native_template_depth_tokens"])
    builder = spec["architecture"]["relation_builder_parameters"]
    return build_detached_roi_differences(
        {name: torch.stack(values) for name, values in feature_batches.items()},
        torch.stack(clip_initial), torch.stack(clip_text),
        torch.stack(native_rgb), torch.stack(native_depth),
        ema_alpha=float(builder["ema_alpha"]),
        epsilon=float(builder["l2_epsilon"]),
        soft_distractor_scale=float(builder["soft_distractor_scale"]),
        native_anchor_top_k=int(builder["native_anchor_top_k"]),
        depth_missing_floor=float(builder["depth_missing_floor"]),
    )


def target_tensors(spec, target_groups):
    gains, branch_means, risks, catastrophes = [], [], [], []
    for event in spec["fixture"]["events"]:
        rows = target_groups[event_key(event)]
        gains.append([float(row["targets"]["10"]["gain"])
                      for row in rows])
        branch_means.append([float(row["targets"]["10"][
            "branch_mean_iou"]) for row in rows])
        risks.append([[float(row["targets"][str(horizon)][
            "max_consecutive_low_overlap_run_fraction"])
                       for horizon in HORIZONS] for row in rows])
        catastrophes.append([
            row["strict_label"] == "catastrophic" for row in rows])
    return {
        "gain": torch.tensor(gains, dtype=torch.float32),
        "branch_mean": torch.tensor(branch_means, dtype=torch.float32),
        "risk": torch.tensor(risks, dtype=torch.float32),
        "catastrophe": torch.tensor(catastrophes, dtype=torch.float32),
    }


def pinball_loss(prediction, target, quantile):
    error = target - prediction
    return torch.maximum(
        float(quantile) * error, (float(quantile) - 1.0) * error).mean()


def pairwise_rank_loss(prediction, target):
    losses = []
    for left in range(6):
        for right in range(left + 1, 6):
            difference = target[:, left] - target[:, right]
            mask = difference.abs() > 1e-12
            if mask.any().item():
                signed_prediction = difference[mask].sign() * (
                    prediction[mask, left] - prediction[mask, right])
                losses.append(torch.nn.functional.softplus(
                    -signed_prediction).mean())
    if not losses:
        raise ContractError("M18a pairwise target set is empty")
    return torch.stack(losses).mean()


def compute_losses(outputs, targets):
    return {
        "candidate_pairwise_rank": pairwise_rank_loss(
            outputs["gain_q10_lcb"], targets["gain"]),
        "gain_q10_pinball": pinball_loss(
            outputs["gain_q10_lcb"], targets["gain"], 0.1),
        "branch_mean_q10_pinball": pinball_loss(
            outputs["branch_mean_q10_lcb"], targets["branch_mean"], 0.1),
        "risk_q90_pinball": pinball_loss(
            outputs["risk_q90_ucb"], targets["risk"], 0.9),
        "catastrophe_bce": torch.nn.functional.binary_cross_entropy(
            outputs["catastrophe_probability"], targets["catastrophe"]),
        "monotone_survival_smooth_l1":
            torch.nn.functional.smooth_l1_loss(
                outputs["survival"], 1.0 - targets["risk"]),
    }


def gradient_summary(parameters):
    nonzero = 0
    finite = True
    absolute_sum = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        finite = finite and torch.isfinite(gradient).all().item()
        count = torch.count_nonzero(gradient).item()
        nonzero += int(count > 0)
        absolute_sum += float(gradient.abs().sum())
    return {"nonzero_parameter_gradients": nonzero,
            "finite": bool(finite), "absolute_sum": absolute_sum}


def candidate_gather(value, indices):
    view = [indices.shape[0], indices.shape[1]] + [1] * (value.ndim - 2)
    expand = list(value.shape)
    return torch.gather(value, 1, indices.reshape(view).expand(expand))


def output_shapes_and_finiteness(outputs, batch_size):
    expected = {
        "gain_q10_lcb": (batch_size, 6),
        "branch_mean_q10_lcb": (batch_size, 6),
        "hazard_increments": (batch_size, 6, 3),
        "survival": (batch_size, 6, 3),
        "risk_q90_ucb": (batch_size, 6, 3),
        "catastrophe_probability": (batch_size, 6),
        "dominance_score": (batch_size, 6),
    }
    return (set(outputs) == set(expected) and
            all(tuple(outputs[name].shape) == shape and
                torch.isfinite(outputs[name]).all().item()
                for name, shape in expected.items()))


def state_is_exact(before, model):
    after = model.state_dict()
    return set(before) == set(after) and all(
        torch.equal(value, after[name]) for name, value in before.items())


def run_smoke(spec):
    smoke = spec["smoke"]
    torch.set_num_threads(int(smoke["torch_threads"]))
    torch.manual_seed(int(smoke["seed"]))
    targets_by_event = load_fixture_targets(spec)
    differences, gates, scalar = load_fixture_relations(spec)
    targets = target_tensors(spec, targets_by_event)
    batch_size = int(spec["fixture"]["event_count"])
    candidate_valid = torch.ones(batch_size, 6, dtype=torch.bool)
    canonical_roles = torch.arange(6, dtype=torch.int64).expand(
        batch_size, -1)
    architecture = spec["architecture"]
    model = CausalQuantileSurvivalRouter(
        utility_projection_seed=int(architecture["utility_projection_seed"]),
        safety_projection_seed=int(architecture["safety_projection_seed"]),
        residual_scale=0.1).float().cpu()
    state_before = {name: value.detach().clone()
                    for name, value in model.state_dict().items()}
    parameter_count = trainable_parameter_count(model)
    utility_ids, safety_ids = model.parameter_partition()
    forbidden_modules = sorted(
        module.__class__.__name__ for module in model.modules()
        if module.__class__.__name__ in FORBIDDEN_MODULE_NAMES)
    outputs = model(
        differences, gates, scalar, candidate_valid, canonical_roles)
    losses = compute_losses(outputs, targets)

    loss_towers = {
        "candidate_pairwise_rank": "utility",
        "gain_q10_pinball": "utility",
        "branch_mean_q10_pinball": "utility",
        "risk_q90_pinball": "safety",
        "catastrophe_bce": "safety",
        "monotone_survival_smooth_l1": "safety",
    }
    gradient_probes = {}
    for name, tower in loss_towers.items():
        model.zero_grad(set_to_none=True)
        losses[name].backward(retain_graph=True)
        utility_summary = gradient_summary(model.utility_parameters())
        safety_summary = gradient_summary(model.safety_parameters())
        expected = utility_summary if tower == "utility" else safety_summary
        other = safety_summary if tower == "utility" else utility_summary
        gradient_probes[name] = {
            "tower": tower,
            "loss": float(losses[name].detach()),
            "expected": expected,
            "other": other,
            "pass": (
                math.isfinite(float(losses[name].detach())) and
                expected["finite"] and
                expected["nonzero_parameter_gradients"] > 0 and
                expected["absolute_sum"] > 0.0 and
                other["absolute_sum"] == 0.0),
        }

    with torch.no_grad():
        baseline = model(
            differences, gates, scalar, candidate_valid, canonical_roles)
        invalid_candidate_valid = candidate_valid.clone()
        invalid_candidate_valid[:, -1] = False
        invalid_outputs = model(
            differences, gates, scalar, invalid_candidate_valid,
            canonical_roles)
        invalid_candidate_excluded = bool(
            (invalid_outputs["dominance_score"][:, -1] <= -1.0e8).all().item()
            and (invalid_outputs["gain_q10_lcb"][:, -1] == -1.0).all().item()
            and (invalid_outputs["branch_mean_q10_lcb"][:, -1] == 0.0
                 ).all().item()
            and (invalid_outputs["risk_q90_ucb"][:, -1] >= 1.0 - 1.0e-6
                 ).all().item()
            and (invalid_outputs["catastrophe_probability"][:, -1] == 1.0
                 ).all().item())
        permutation_parity = []
        for shift in range(6):
            permutation = torch.tensor(
                [(index + shift) % 6 for index in range(6)],
                dtype=torch.int64).expand(batch_size, -1)
            permuted_outputs = model(
                differences[:, :, permutation[0]],
                gates[:, :, permutation[0]],
                scalar[:, :, permutation[0]],
                candidate_valid[:, permutation[0]], permutation)
            canonical_to_input = torch.argsort(permutation, dim=1)
            restored = {
                name: candidate_gather(value, canonical_to_input)
                for name, value in permuted_outputs.items()
            }
            permutation_parity.append({
                "shift": shift,
                "exact": all(torch.equal(baseline[name], restored[name])
                             for name in baseline),
            })

    model.zero_grad(set_to_none=True)
    combined_outputs = model(
        differences, gates, scalar, candidate_valid, canonical_roles)
    combined_losses = compute_losses(combined_outputs, targets)
    weights = {name: float(value["weight"])
               for name, value in smoke["losses"].items()}
    combined_loss = sum(weights[name] * value
                        for name, value in combined_losses.items())
    combined_loss.backward()
    preclip = float(torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(smoke["gradient_clip_l2"])))
    postclip = math.sqrt(sum(float(parameter.grad.detach().pow(2).sum())
                             for parameter in model.parameters()
                             if parameter.grad is not None))

    survival = outputs["survival"]
    monotone = bool(((survival[:, :, 0] >= survival[:, :, 1]) &
                     (survival[:, :, 1] >= survival[:, :, 2])).all().item())
    gates_result = {
        "parameter_count_exact": parameter_count == int(
            architecture["expected_trainable_parameters"]),
        "parameter_count_bounded": parameter_count <= int(
            architecture["maximum_trainable_parameters"]),
        "forbidden_modules_absent": not forbidden_modules,
        "utility_safety_parameter_overlap_zero": not (utility_ids & safety_ids),
        "all_parameters_partitioned": len(utility_ids | safety_ids) == len({
            id(parameter) for parameter in model.parameters()}),
        "shape_and_finiteness": output_shapes_and_finiteness(
            outputs, batch_size),
        "invalid_candidate_fail_closed": invalid_candidate_excluded,
        "six_permutation_exact_parity": all(
            row["exact"] for row in permutation_parity),
        "survival_monotone": monotone,
        "individual_gradient_isolation": all(
            row["pass"] for row in gradient_probes.values()),
        "combined_loss_finite": math.isfinite(float(combined_loss.detach())),
        "preclip_l2_in_range": 0.0 < preclip <= 1000.0,
        "postclip_l2_bounded": postclip <= 5.000001,
        "model_state_exact": state_is_exact(state_before, model),
        "optimizer_constructed_false": smoke["optimizer_constructed"] is False,
        "optimizer_steps_zero": int(smoke["optimizer_steps"]) == 0,
        "checkpoint_written_false": smoke["checkpoint_written"] is False,
    }
    return {
        "accepted": all(gates_result.values()),
        "gates": gates_result,
        "parameter_count": parameter_count,
        "utility_parameter_tensors": len(utility_ids),
        "safety_parameter_tensors": len(safety_ids),
        "forbidden_modules": forbidden_modules,
        "losses": {name: float(value.detach())
                   for name, value in combined_losses.items()},
        "gradient_probes": gradient_probes,
        "permutation_parity": permutation_parity,
        "invalid_candidate_fail_closed": invalid_candidate_excluded,
        "preclip_l2": preclip,
        "postclip_l2": postclip,
        "optimizer_steps": 0,
        "checkpoint_written": False,
    }


def journal_inventory(root):
    root = Path(root)
    entries = []

    def visit(directory):
        with os.scandir(directory) as stream:
            children = sorted(list(stream), key=lambda entry: entry.name)
        for child in children:
            path = Path(child.path)
            metadata = os.lstat(path)
            mode = metadata.st_mode
            relative = str(path.relative_to(root)).replace("\\", "/")
            if stat.S_ISREG(mode):
                kind = "regular_file"
            elif stat.S_ISDIR(mode):
                kind = "directory"
            elif stat.S_ISLNK(mode):
                kind = "symlink"
            else:
                kind = "other"
            entries.append({"path": relative, "kind": kind,
                            "link_count": int(metadata.st_nlink)})
            if kind == "directory":
                visit(path)

    visit(root)
    return entries


def exact_regular_root_files(root, expected):
    inventory = journal_inventory(root)
    actual = {entry["path"] for entry in inventory}
    return (actual == set(expected) and all(
        entry["kind"] == "regular_file" and
        entry["link_count"] == 1 and "/" not in entry["path"]
        for entry in inventory)), inventory


def _nofollow_open_flags(directory=False):
    required = ("O_CLOEXEC", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        raise ContractError("descriptor no-follow sealing is unsupported")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if directory:
        if not hasattr(os, "O_DIRECTORY"):
            raise ContractError("descriptor directory sealing is unsupported")
        flags |= os.O_DIRECTORY
    return flags


def _seal_directory_fd(directory_fd, prefix=""):
    records = []
    for name in sorted(os.listdir(directory_fd)):
        metadata = os.stat(
            name, dir_fd=directory_fd, follow_symlinks=False)
        relative = "{}/{}".format(prefix, name) if prefix else name
        if stat.S_ISREG(metadata.st_mode):
            descriptor = os.open(
                name, _nofollow_open_flags(False), dir_fd=directory_fd)
            try:
                opened = os.fstat(descriptor)
                if (not stat.S_ISREG(opened.st_mode) or
                        (opened.st_dev, opened.st_ino) !=
                        (metadata.st_dev, metadata.st_ino) or
                        metadata.st_nlink != 1 or opened.st_nlink != 1):
                    raise ContractError("journal file changed during sealing")
                os.fchmod(
                    descriptor, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
                final_mode = stat.S_IMODE(os.fstat(descriptor).st_mode)
                if final_mode != 0o444:
                    raise ContractError("journal file mode did not seal")
            finally:
                os.close(descriptor)
            records.append({"path": relative, "kind": "regular_file",
                            "mode": "0444"})
        elif stat.S_ISDIR(metadata.st_mode):
            descriptor = os.open(
                name, _nofollow_open_flags(True), dir_fd=directory_fd)
            try:
                opened = os.fstat(descriptor)
                if (not stat.S_ISDIR(opened.st_mode) or
                        (opened.st_dev, opened.st_ino) !=
                        (metadata.st_dev, metadata.st_ino)):
                    raise ContractError(
                        "journal directory changed during sealing")
                records.extend(_seal_directory_fd(descriptor, relative))
                os.fchmod(descriptor, 0o555)
                final_mode = stat.S_IMODE(os.fstat(descriptor).st_mode)
                if final_mode != 0o555:
                    raise ContractError("journal directory mode did not seal")
            finally:
                os.close(descriptor)
            records.append({"path": relative, "kind": "directory",
                            "mode": "0555"})
        elif stat.S_ISLNK(metadata.st_mode):
            records.append({"path": relative, "kind": "symlink",
                            "mode": None})
        else:
            records.append({"path": relative, "kind": "other",
                            "mode": format(stat.S_IMODE(metadata.st_mode),
                                           "04o")})
    return records


def strict_seal_attempt_root(root):
    descriptor = os.open(root, _nofollow_open_flags(True))
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode):
            raise ContractError("attempt root is not a directory")
        records = _seal_directory_fd(descriptor)
        os.fchmod(descriptor, 0o555)
        root_mode = stat.S_IMODE(os.fstat(descriptor).st_mode)
        if root_mode != 0o555:
            raise ContractError("attempt root mode did not seal")
    finally:
        os.close(descriptor)
    return {"root_mode": "0555", "entries": records}


def make_attempt_root_owner_writable(root):
    descriptor = os.open(root, _nofollow_open_flags(True))
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ContractError("attempt root is not a directory")
        os.fchmod(descriptor, 0o755)
    finally:
        os.close(descriptor)


def best_effort_file_record(path):
    try:
        return file_record(path)
    except BaseException as error:
        return {
            "path": str(Path(path).resolve()),
            "unavailable": True,
            "error_type": error.__class__.__name__,
            "error": str(error),
        }


def publish_terminal(attempt_root, start_path, args, spec, binding,
                     control_records, status, smoke_result,
                     runtime_observation, exception_record):
    expected = {"start.json", "terminal.json", "manifest.json"}
    before_exact, before_inventory = exact_regular_root_files(
        attempt_root, {"start.json"})
    unexpected_before = [] if before_exact else before_inventory
    effective_status = status
    if unexpected_before:
        effective_status = "journal_failure"
        if exception_record is None:
            exception_record = {
                "type": "ContractError",
                "message": "unexpected journal files before publication",
                "traceback": None,
            }
    terminal = {
        "schema": "sttrack-lachtt-m18a-attempt-terminal/v1",
        "complete": True,
        "status": effective_status,
        "accepted": bool(effective_status == "success" and smoke_result and
                         smoke_result.get("accepted")),
        "claim_ceiling": (
            "Eight-event engineering smoke only; zero optimizer steps and "
            "no tracking or benchmark claim."),
        "smoke": smoke_result,
        "runtime_observation": runtime_observation,
        "exception": exception_record,
        "scientific_output_exists": args.scientific_output.exists(),
        "publication": {
            "unexpected_files_before_publication": unexpected_before,
            "expected_files": sorted(expected),
        },
        "authorization": {
            "independent_m18a_result_audit": effective_status == "success",
            "m18b_implementation": False,
            "m18b_training": False,
            "checkpoint": False,
            "public_evaluation": False,
            "automatic_next_stage": False,
        },
    }
    terminal_path = attempt_root / "terminal.json"
    write_json(terminal_path, terminal)
    before_manifest_exact, before_manifest_inventory = exact_regular_root_files(
        attempt_root, {"start.json", "terminal.json"})
    if not before_manifest_exact:
        effective_status = "journal_failure"
        terminal["status"] = effective_status
        terminal["accepted"] = False
        terminal["authorization"]["independent_m18a_result_audit"] = False
        terminal["publication"]["entries_before_manifest"] = (
            before_manifest_inventory)
        write_json(terminal_path, terminal)
    fallback_inputs = {
        "spec": best_effort_file_record(args.spec),
        "binding": best_effort_file_record(args.binding),
        "preexecution_audit": best_effort_file_record(
            binding["preexecution_audit"]["path"]
            if isinstance(binding, dict) and "preexecution_audit" in binding
            else args.binding),
        "runner": best_effort_file_record(Path(__file__).resolve()),
        "model": best_effort_file_record(
            binding["model"]["path"]
            if isinstance(binding, dict) and "model" in binding
            else Path(__file__).resolve()),
    }
    manifest = {
        "schema": "sttrack-lachtt-m18a-attempt-manifest/v1",
        "complete": True,
        "status": effective_status,
        "inputs": {name: control_records.get(name, record)
                   for name, record in fallback_inputs.items()},
        "journal": {
            "start.json": file_record(start_path),
            "terminal.json": file_record(terminal_path),
        },
        "scientific_output_absent": not args.scientific_output.exists(),
        "optimizer_steps": 0,
        "checkpoint_written": False,
        "expected_file_set": sorted(expected),
    }
    manifest_path = attempt_root / "manifest.json"
    write_json(manifest_path, manifest)
    final_exact, final_inventory = exact_regular_root_files(
        attempt_root, expected)
    if not final_exact:
        effective_status = "journal_failure"
        terminal["status"] = effective_status
        terminal["accepted"] = False
        terminal["authorization"]["independent_m18a_result_audit"] = False
        terminal["publication"]["entries_after_manifest"] = final_inventory
        write_json(terminal_path, terminal)
        manifest["status"] = effective_status
        manifest["journal"]["terminal.json"] = file_record(terminal_path)
        manifest["actual_entries"] = final_inventory
        write_json(manifest_path, manifest)
    return effective_status


def main():
    args = parse_args()
    args.spec = args.spec.resolve()
    args.binding = args.binding.resolve()
    args.attempt_root = args.attempt_root.resolve()
    args.scientific_output = args.scientific_output.resolve()
    if EXPECTED_ATTEMPT_ROOT.exists():
        raise ContractError("M18a attempt root already exists")
    EXPECTED_ATTEMPT_ROOT.mkdir(mode=0o755, parents=False, exist_ok=False)
    attempt_root = EXPECTED_ATTEMPT_ROOT
    start = {
        "schema": "sttrack-lachtt-m18a-attempt-start/v1",
        "complete": True,
        "phase": "bootstrap_preflight",
        "argv": list(sys.argv),
        "pid": os.getpid(),
        "requested_paths": {
            "spec": str(args.spec),
            "binding": str(args.binding),
            "attempt_root": str(args.attempt_root),
            "scientific_output": str(args.scientific_output),
        },
        "scientific_output_absent": not args.scientific_output.exists(),
        "torch_imported": False,
        "project_imported": False,
        "optimizer_steps": 0,
    }
    start_path = attempt_root / "start.json"
    write_json(start_path, start)

    observer = RuntimeAuditObserver(attempt_root)
    sys.addaudithook(observer.hook)
    spec = None
    binding = None
    control_records = {}
    smoke_result = None
    exception_record = None
    observation = observer.snapshot()
    status = "failed"
    exit_code = 1
    try:
        if (args.spec != EXPECTED_SPEC_PATH or
                args.binding != EXPECTED_BINDING_PATH or
                args.attempt_root != EXPECTED_ATTEMPT_ROOT or
                args.scientific_output != EXPECTED_SCIENTIFIC_OUTPUT):
            raise ContractError("M18a invocation path drifted")
        spec, spec_actual_record = load_json_record(args.spec)
        binding, binding_actual_record = load_json_record(args.binding)
        control_records.update({
            "spec": spec_actual_record,
            "binding": binding_actual_record,
        })
        if (spec.get("schema") !=
                "sttrack-lachtt-m18a-causal-survival-architecture-journal-"
                "smoke-spec/v1" or spec.get("complete") is not True):
            raise ContractError("M18a spec contract drifted")
        audit, repo, repo_identity, audit_actual_record = validate_binding(
            args, spec, binding)
        control_records["preexecution_audit"] = audit_actual_record
        runner_payload, runner_actual_record = read_verified_bytes(
            binding["runner"])
        del runner_payload
        model_payload, model_actual_record = read_verified_bytes(
            binding["model"])
        del model_payload
        control_records.update({
            "runner": runner_actual_record,
            "model": model_actual_record,
        })
        checked_sources = verify_hashed_records(spec)
        if (not checked_sources or
                not all(row["match"] for row in checked_sources)):
            raise ContractError("M18a frozen source identity drifted")
        bootstrap_exact, _ = exact_regular_root_files(
            attempt_root, {"start.json"})
        if not bootstrap_exact:
            raise ContractError("M18a bootstrap journal file set drifted")
        start.update({
            "phase": "preflight_complete_before_torch_import",
            "repository": repo_identity,
            "spec": spec_actual_record,
            "binding": binding_actual_record,
            "runner": runner_actual_record,
            "model": model_actual_record,
            "preexecution_audit": audit_actual_record,
            "frozen_source_records_checked": len(checked_sources),
        })
        write_json(start_path, start)
        loaded_source_records = load_project_components(repo, spec, binding)
        if model_actual_record not in loaded_source_records:
            raise ContractError("executed model source identity drifted")
        smoke_result = run_smoke(spec)
        final_repo_identity = git_identity(repo)
        controls_still_exact = all(
            best_effort_file_record(record["path"]) == record
            for record in control_records.values())
        smoke_result["gates"]["repository_identity_still_exact"] = (
            final_repo_identity == repo_identity)
        smoke_result["gates"]["control_identity_still_exact"] = (
            controls_still_exact)
        observation = observer.snapshot()
        runtime_clean = (
            not observation["forbidden_write_paths"] and
            not observation["unresolved_write_targets"] and
            not observation["forbidden_mutation_paths"] and
            not observation["unresolved_mutation_targets"] and
            not observation["subprocess_events"] and
            not observation["network_events"] and
            not observation["forbidden_modules"])
        scientific_absent = not args.scientific_output.exists()
        smoke_result["gates"]["runtime_side_effects_clean"] = runtime_clean
        smoke_result["gates"]["scientific_output_absent"] = scientific_absent
        smoke_result["accepted"] = all(smoke_result["gates"].values())
        status = "success" if smoke_result["accepted"] else "gate_failure"
        exit_code = 0 if smoke_result["accepted"] else 2
    except BaseException as error:
        exception_record = {
            "type": error.__class__.__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        observation = observer.snapshot()
        status = "exception"
        exit_code = 1
    try:
        status = publish_terminal(
            attempt_root, start_path, args, spec, binding, control_records,
            status, smoke_result, observation, exception_record)
        if status != "success":
            exit_code = 1 if status in ("exception", "journal_failure") else 2
    except BaseException as publication_error:
        status = "publication_exception"
        exit_code = 1
        emergency = {
            "schema": "sttrack-lachtt-m18a-attempt-terminal/v1",
            "complete": False,
            "status": status,
            "accepted": False,
            "exception": {
                "type": publication_error.__class__.__name__,
                "message": str(publication_error),
                "traceback": traceback.format_exc(),
            },
            "authorization": {
                "independent_m18a_result_audit": False,
                "automatic_next_stage": False,
            },
        }
        try:
            write_json(attempt_root / "terminal.json", emergency)
            write_json(attempt_root / "manifest.json", {
                "schema": "sttrack-lachtt-m18a-attempt-manifest/v1",
                "complete": False,
                "status": status,
                "journal": {
                    "start.json": best_effort_file_record(start_path),
                    "terminal.json": best_effort_file_record(
                        attempt_root / "terminal.json"),
                },
                "optimizer_steps": 0,
                "checkpoint_written": False,
            })
        except BaseException:
            pass
    seal_error = None
    try:
        strict_seal_attempt_root(attempt_root)
    except BaseException as error:
        seal_error = {
            "type": error.__class__.__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        status = "seal_failure"
        exit_code = 1
        try:
            make_attempt_root_owner_writable(attempt_root)
            terminal_path = attempt_root / "terminal.json"
            manifest_path = attempt_root / "manifest.json"
            terminal = (load_json(terminal_path)
                        if terminal_path.is_file() else {})
            terminal.update({
                "schema": "sttrack-lachtt-m18a-attempt-terminal/v1",
                "complete": False,
                "status": status,
                "accepted": False,
                "seal_failure": seal_error,
            })
            authorization = terminal.setdefault("authorization", {})
            authorization["independent_m18a_result_audit"] = False
            authorization["automatic_next_stage"] = False
            write_json(terminal_path, terminal)
            manifest = (load_json(manifest_path)
                        if manifest_path.is_file() else {})
            manifest.update({
                "schema": "sttrack-lachtt-m18a-attempt-manifest/v1",
                "complete": False,
                "status": status,
                "seal_failure": seal_error,
            })
            journal = manifest.setdefault("journal", {})
            journal["start.json"] = best_effort_file_record(start_path)
            journal["terminal.json"] = best_effort_file_record(terminal_path)
            write_json(manifest_path, manifest)
        except BaseException as publication_error:
            seal_error["failure_receipt_error"] = {
                "type": publication_error.__class__.__name__,
                "message": str(publication_error),
            }
        try:
            strict_seal_attempt_root(attempt_root)
        except BaseException as retry_error:
            seal_error["retry_error"] = {
                "type": retry_error.__class__.__name__,
                "message": str(retry_error),
            }
    print(json.dumps({
        "status": status,
        "accepted": bool(status == "success" and smoke_result and
                         smoke_result.get("accepted")),
        "attempt_root": str(attempt_root),
        "scientific_output_absent": not args.scientific_output.exists(),
        "seal_error": seal_error,
        "exit_code": exit_code,
    }, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
