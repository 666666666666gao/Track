#!/usr/bin/env python3
"""Attribute bootstrap-only runtime side effects without model execution."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import struct
import sys
import time
import traceback
import types
import zlib


REPOSITORY_ROOT = Path(
    "/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1").resolve()
EXPECTED_SPEC_PATH = Path(
    "/home/SUTrack_RGBD_L/refine-logs/"
    "STTRACK_LACHTT_M19A_BOOTSTRAP_ATTRIBUTED_RUNTIME_PROVENANCE_"
    "SPEC_20260901.json").resolve()
EXPECTED_BINDING_PATH = Path(
    "/home/SUTrack_RGBD_L/refine-logs/"
    "STTRACK_LACHTT_M19A_BOOTSTRAP_ATTRIBUTED_RUNTIME_PROVENANCE_"
    "BINDING_20260901.json").resolve()
EXPECTED_ATTEMPT_ROOT = Path(
    "/root/autodl-tmp/"
    "sttrack_lachtt_m19a_bootstrap_attribution_attempt_v1_20260901"
).resolve()
EXPECTED_PLAN_PATH = Path(
    "/home/SUTrack_RGBD_L/refine-logs/"
    "EXPERIMENT_PLAN_20260901_163759.md").resolve()
EXPECTED_PYTHON = Path(
    "/root/autodl-tmp/envs/sttrack/bin/python3.8").resolve()
PROJECT_MODULE_ORDER = (
    ("lib.models.sttrack.lachtt_target_distractor_memory",
     "lachtt_target_distractor_memory.py"),
    ("lib.models.sttrack.lachtt_rich_roi_relation",
     "lachtt_rich_roi_relation.py"),
    ("lib.models.sttrack.lachtt_learned_bounded_roi_association",
     "lachtt_learned_bounded_roi_association.py"),
    ("lib.models.sttrack.lachtt_causal_quantile_survival",
     "lachtt_causal_quantile_survival.py"),
)
AUTHORIZED_ACTION = (
    "run exactly one M19a import-only bootstrap attribution if the binding "
    "passes its own preflight"
)


class ContractError(RuntimeError):
    """Raised when a frozen M19 contract is not exact."""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--attempt-root", required=True, type=Path)
    return parser.parse_args()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


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
        raise ContractError("input is not an independent regular file: {}".
                            format(path))
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
            raise ContractError("bound file identity drifted: {}".format(path))
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


def read_git_head(repo):
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
        raise ContractError("repository git directory missing")
    common_dir = git_dir
    common_pointer = git_dir / "commondir"
    if common_pointer.is_file():
        common_dir = (git_dir / common_pointer.read_text(
            encoding="utf-8").strip()).resolve()
    head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
    if head.startswith("ref: "):
        ref_name = head[5:]
        ref_path = common_dir / ref_name
        if ref_path.is_file():
            commit = ref_path.read_text(encoding="ascii").strip()
        else:
            commit = None
            packed_refs = common_dir / "packed-refs"
            if packed_refs.is_file():
                for line in packed_refs.read_text(
                        encoding="ascii").splitlines():
                    if not line or line.startswith(("#", "^")):
                        continue
                    object_id, packed_name = line.split(" ", 1)
                    if packed_name == ref_name:
                        commit = object_id
                        break
            if commit is None:
                raise ContractError("git HEAD ref is unresolved")
        prefix = "refs/heads/"
        branch = ref_name[len(prefix):] if ref_name.startswith(prefix) else ref_name
    else:
        commit = head
        branch = "DETACHED"
    if len(commit) != 40 or any(token not in "0123456789abcdef"
                                for token in commit.lower()):
        raise ContractError("git commit identity is invalid")
    return {"path": str(repo), "commit": commit, "branch": branch}


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
        branch = (reference[len(prefix):]
                  if reference.startswith(prefix) else reference)
    else:
        commit = head
        branch = "HEAD"
    if (len(commit) != 40 or
            any(value not in "0123456789abcdef" for value in commit)):
        raise ContractError("git HEAD commit drifted")
    return commit, branch


def _git_index_entries(git_dir):
    payload = (git_dir / "index").read_bytes()
    if (len(payload) < 32 or
            hashlib.sha1(payload[:-20]).digest() != payload[-20:]):
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
        path = payload[offset:terminator].decode(
            "utf-8", "surrogateescape")
        offset = terminator + 1
        while (offset - start) % 8:
            offset += 1
        if ((flags >> 12) & 0x3) != 0:
            raise ContractError("git index contains an unmerged entry")
        entries.append({"path": path, "mode": mode,
                        "object_id": object_id})
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
                submodule_identity = direct_git_identity(path)
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
                if (entry is None or
                        entry["mode"] & 0o170000 != 0o120000):
                    return False
                directories.remove(name)
        directories[:] = [name for name in directories if
                          str((relative_root / name)).replace("\\", "/")
                          not in submodules]
        for name in files:
            relative = str((relative_root / name)).replace("\\", "/")
            if relative.startswith("./"):
                relative = relative[2:]
            if relative == ".git":
                continue
            if relative not in tracked:
                return False
    return True


def direct_git_identity(repo):
    repo = Path(repo).resolve()
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
    entries = _git_index_entries(git_dir)
    clean = (_git_index_tree(entries) == tree_line[5:] and
             _git_worktree_clean(repo, entries))
    return {"path": str(repo), "commit": commit,
            "branch": branch, "clean": clean}


def resolve_audit_target(value):
    if isinstance(value, int):
        try:
            return Path(os.readlink("/proc/self/fd/{}".format(value))).resolve()
        except (OSError, TypeError, ValueError):
            return None
    if not isinstance(value, (str, bytes, os.PathLike)):
        return None
    candidate = Path(os.fsdecode(value))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        return candidate.resolve()
    except OSError:
        return candidate.absolute()


def stack_record(limit=80):
    rows = []
    for frame in traceback.extract_stack(limit=limit)[:-2]:
        rows.append({
            "file": str(Path(frame.filename).absolute()),
            "line": int(frame.lineno),
            "function": str(frame.name),
        })
    return rows


def stable_audit_value(value, limit=2048):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "bytes": len(value),
                "sha256": sha256_bytes(value)}
    if isinstance(value, (str, os.PathLike)):
        text = os.fsdecode(value)
        return text if len(text) <= limit else {
            "type": "long_string", "chars": len(text),
            "sha256": sha256_bytes(text.encode("utf-8", "surrogateescape")),
        }
    if isinstance(value, (list, tuple)):
        return [stable_audit_value(item, limit=limit) for item in value]
    return {"type": type(value).__name__}


def environment_record(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        return {"type": type(value).__name__}
    keys = sorted(str(key) for key in value)
    encoded = "\0".join(keys).encode("utf-8", "surrogateescape")
    return {"key_count": len(keys), "keys": keys,
            "keys_sha256": sha256_bytes(encoded)}


def stdio_record(value, subprocess_module):
    if value is None:
        return "inherit"
    if value == subprocess_module.PIPE:
        return "PIPE"
    if value == subprocess_module.STDOUT:
        return "STDOUT"
    if value == subprocess_module.DEVNULL:
        return "DEVNULL"
    if isinstance(value, int):
        return "fd:{}".format(value)
    return "object:{}".format(type(value).__name__)


class ProvenanceObserver:
    MUTATION_ARGUMENTS = {
        "os.mkdir": (0,), "os.rename": (0, 1), "os.remove": (0,),
        "os.rmdir": (0,), "os.symlink": (1,), "os.link": (0, 1),
        "os.chdir": (0,), "os.chmod": (0,), "os.chown": (0,),
        "os.utime": (0,), "os.truncate": (0,), "os.setxattr": (0,),
        "os.removexattr": (0,),
    }
    WEIGHT_SUFFIXES = {
        ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".npz",
    }
    DATA_PARTS = {
        "depthtrack", "vot", "votrgbd2022", "cdtb", "dataset",
        "datasets", "groundtruth", "ground_truth", "annotations",
        "target", "targets", "prediction", "predictions", "cache",
        "caches", "checkpoint", "checkpoints",
    }
    DATA_SUFFIXES = {".h5", ".hdf5", ".npy", ".pkl", ".pickle"}

    def __init__(self, attempt_root, runner_path):
        self.attempt_root = Path(attempt_root).resolve()
        self.runner_path = Path(runner_path).resolve()
        self.allowed_source_paths = set()
        self.phase = "observer_installed"
        self.capture = True
        self._inside_hook = False
        self._next_event_id = 1
        self._next_correlation_id = 1
        self._active_correlations = []
        self.imports = set()
        self.write_events = []
        self.mutation_events = []
        self.subprocess_audit_events = []
        self.popen_calls = []
        self.network_events = []
        self.sensitive_read_events = []
        self.unresolved_write_events = []

    def set_phase(self, value):
        self.phase = str(value)

    def _event_id(self):
        value = self._next_event_id
        self._next_event_id += 1
        return value

    def _correlation(self):
        return self._active_correlations[-1] if self._active_correlations else None

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

    def allow_source_paths(self, paths):
        self.allowed_source_paths = {
            str(Path(path).resolve()) for path in paths}

    def _sensitive_read(self, path):
        if str(path.resolve()) in self.allowed_source_paths:
            return False
        if path.suffix.lower() in self.WEIGHT_SUFFIXES.union(
                self.DATA_SUFFIXES):
            return True
        parts = {part.lower() for part in path.parts if part != "__pycache__"}
        if parts.intersection(self.DATA_PARTS):
            return True
        lower_name = path.name.lower()
        return ("cached_gate" in lower_name or "candidate_cache" in lower_name or
                "prediction" in lower_name or "checkpoint" in lower_name or
                lower_name.startswith("target") or
                lower_name in {"groundtruth.txt", "groundtruth_rect.txt"})

    def begin_popen(self, popen_args, popen_kwargs, subprocess_module):
        correlation_id = "popen-{:04d}".format(self._next_correlation_id)
        self._next_correlation_id += 1
        self._active_correlations.append(correlation_id)
        command = popen_args[0] if popen_args else popen_kwargs.get("args")

        def positional_or_keyword(position, key):
            return (popen_args[position] if len(popen_args) > position else
                    popen_kwargs.get(key))

        requested_executable = positional_or_keyword(2, "executable")
        if requested_executable is not None:
            effective_executable = requested_executable
        elif isinstance(command, (list, tuple)) and command:
            effective_executable = command[0]
        else:
            effective_executable = command
        requested_cwd = positional_or_keyword(9, "cwd")
        effective_cwd = os.getcwd() if requested_cwd is None else requested_cwd
        requested_environment = positional_or_keyword(10, "env")

        self.popen_calls.append({
            "event_id": self._event_id(),
            "correlation_id": correlation_id,
            "phase": self.phase,
            "monotonic_ns": time.monotonic_ns(),
            "command": stable_audit_value(command),
            "requested_executable": stable_audit_value(requested_executable),
            "effective_executable": stable_audit_value(effective_executable),
            "requested_cwd": stable_audit_value(requested_cwd),
            "effective_cwd": stable_audit_value(effective_cwd),
            "stdin": stdio_record(
                positional_or_keyword(3, "stdin"), subprocess_module),
            "stdout": stdio_record(
                positional_or_keyword(4, "stdout"), subprocess_module),
            "stderr": stdio_record(
                positional_or_keyword(5, "stderr"), subprocess_module),
            "environment": environment_record(requested_environment),
            "stack": stack_record(),
        })
        return correlation_id

    def end_popen(self, correlation_id):
        if (not self._active_correlations or
                self._active_correlations[-1] != correlation_id):
            raise ContractError("Popen correlation stack drifted")
        self._active_correlations.pop()

    def hook(self, event, arguments):
        if not self.capture or self._inside_hook:
            return
        self._inside_hook = True
        try:
            if event == "import" and arguments:
                self.imports.add(str(arguments[0]))
            elif event == "open" and arguments:
                value = arguments[0]
                mode = arguments[1] if len(arguments) > 1 else None
                flags = arguments[2] if len(arguments) > 2 else None
                resolved = resolve_audit_target(value)
                if self._write_open(mode, flags):
                    row = {
                        "event_id": self._event_id(),
                        "phase": self.phase,
                        "monotonic_ns": time.monotonic_ns(),
                        "path": str(resolved) if resolved is not None else None,
                        "raw_target": stable_audit_value(value),
                        "mode": stable_audit_value(mode),
                        "mode_semantics": (
                            "flags_only" if mode is None else str(mode)),
                        "flags": stable_audit_value(flags),
                        "correlation_id": self._correlation(),
                        "stack": stack_record(),
                    }
                    if resolved is None:
                        self.unresolved_write_events.append(row)
                    else:
                        self.write_events.append(row)
                elif resolved is not None and self._sensitive_read(resolved):
                    self.sensitive_read_events.append({
                        "event_id": self._event_id(),
                        "phase": self.phase,
                        "path": str(resolved),
                        "stack": stack_record(),
                    })
            elif event in self.MUTATION_ARGUMENTS:
                for index in self.MUTATION_ARGUMENTS[event]:
                    value = arguments[index] if index < len(arguments) else None
                    resolved = resolve_audit_target(value)
                    self.mutation_events.append({
                        "event_id": self._event_id(),
                        "phase": self.phase,
                        "event": event,
                        "path": str(resolved) if resolved is not None else None,
                        "raw_target": stable_audit_value(value),
                        "correlation_id": self._correlation(),
                        "stack": stack_record(),
                    })
            elif event.startswith("subprocess"):
                executable = arguments[0] if len(arguments) > 0 else None
                command = arguments[1] if len(arguments) > 1 else None
                cwd = arguments[2] if len(arguments) > 2 else None
                environment = arguments[3] if len(arguments) > 3 else None
                self.subprocess_audit_events.append({
                    "event_id": self._event_id(),
                    "phase": self.phase,
                    "monotonic_ns": time.monotonic_ns(),
                    "event": event,
                    "correlation_id": self._correlation(),
                    "executable": stable_audit_value(executable),
                    "command": stable_audit_value(command),
                    "cwd": stable_audit_value(cwd),
                    "environment": environment_record(environment),
                    "stack": stack_record(),
                })
            elif (event.startswith("socket.connect") or
                  event.startswith("socket.getaddrinfo")):
                self.network_events.append({
                    "event_id": self._event_id(),
                    "phase": self.phase,
                    "event": event,
                    "arguments": stable_audit_value(arguments),
                    "stack": stack_record(),
                })
        finally:
            self._inside_hook = False

    def snapshot(self):
        forbidden_writes = []
        devnull_writes = []
        for row in self.write_events:
            path = Path(row["path"])
            if path == Path("/dev/null"):
                devnull_writes.append(row)
                continue
            try:
                path.relative_to(self.attempt_root)
            except ValueError:
                forbidden_writes.append(row)
        forbidden_mutations = []
        for row in self.mutation_events:
            if row["path"] is None:
                forbidden_mutations.append(row)
                continue
            try:
                Path(row["path"]).relative_to(self.attempt_root)
            except ValueError:
                forbidden_mutations.append(row)
        forbidden_modules = sorted(name for name in self.imports if
            "qwen" in name.lower() or name == "vot" or
            name.startswith("vot.") or name.startswith("lib.test"))
        return {
            "phase_at_snapshot": self.phase,
            "imports": sorted(self.imports),
            "popen_calls": self.popen_calls,
            "subprocess_audit_events": self.subprocess_audit_events,
            "devnull_write_events": devnull_writes,
            "all_write_events": self.write_events,
            "forbidden_write_events": forbidden_writes,
            "unresolved_write_events": self.unresolved_write_events,
            "all_mutation_events": self.mutation_events,
            "forbidden_mutation_events": forbidden_mutations,
            "network_events": self.network_events,
            "sensitive_read_events": self.sensitive_read_events,
            "forbidden_modules": forbidden_modules,
        }


def install_traced_popen(observer):
    subprocess_module = importlib.import_module("subprocess")
    original_popen = subprocess_module.Popen

    class TracedPopen(original_popen):
        def __init__(self, *args, **kwargs):
            correlation_id = observer.begin_popen(
                args, kwargs, subprocess_module)
            try:
                super().__init__(*args, **kwargs)
            finally:
                observer.end_popen(correlation_id)

    TracedPopen.__name__ = "M19TracedPopen"
    TracedPopen.__qualname__ = "M19TracedPopen"
    subprocess_module.Popen = TracedPopen
    return subprocess_module, original_popen


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


def load_verified_module(name, record):
    payload, actual = read_verified_bytes(record)
    module = types.ModuleType(name)
    module.__file__ = actual["path"]
    module.__package__ = name.rsplit(".", 1)[0]
    sys.modules[name] = module
    try:
        code = compile(payload, actual["path"], "exec")
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return actual


class ProjectExecutionInstrumentation:
    """Measure model/tensor/optimizer/checkpoint activity during source exec."""

    def __init__(self, torch_module):
        self.torch = torch_module
        self.counts = {
            "model_instantiations": 0,
            "forward_call_entries": 0,
            "tensor_dispatch_ops": 0,
            "optimizer_constructions": 0,
            "optimizer_step_entries": 0,
            "checkpoint_write_entries": 0,
        }
        self._originals = {}
        self._dispatch_mode = None

    def install(self):
        torch_module = self.torch
        counts = self.counts
        from torch.utils._python_dispatch import TorchDispatchMode

        class CountingDispatchMode(TorchDispatchMode):
            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                counts["tensor_dispatch_ops"] += 1
                return function(*args, **({} if kwargs is None else kwargs))

        self._originals = {
            "module_init": torch_module.nn.Module.__init__,
            "module_call": torch_module.nn.Module.__call__,
            "optimizer_init": torch_module.optim.Optimizer.__init__,
            "optimizer_step": torch_module.optim.Optimizer.step,
            "torch_save": torch_module.save,
        }

        def traced_module_init(module, *args, **kwargs):
            counts["model_instantiations"] += 1
            return self._originals["module_init"](module, *args, **kwargs)

        def traced_module_call(module, *args, **kwargs):
            counts["forward_call_entries"] += 1
            return self._originals["module_call"](module, *args, **kwargs)

        def traced_optimizer_init(optimizer, *args, **kwargs):
            counts["optimizer_constructions"] += 1
            return self._originals["optimizer_init"](
                optimizer, *args, **kwargs)

        def traced_optimizer_step(optimizer, *args, **kwargs):
            counts["optimizer_step_entries"] += 1
            return self._originals["optimizer_step"](
                optimizer, *args, **kwargs)

        def traced_torch_save(*args, **kwargs):
            counts["checkpoint_write_entries"] += 1
            return self._originals["torch_save"](*args, **kwargs)

        torch_module.nn.Module.__init__ = traced_module_init
        torch_module.nn.Module.__call__ = traced_module_call
        torch_module.optim.Optimizer.__init__ = traced_optimizer_init
        torch_module.optim.Optimizer.step = traced_optimizer_step
        torch_module.save = traced_torch_save
        self._dispatch_mode = CountingDispatchMode()
        self._dispatch_mode.__enter__()

    def restore(self):
        if self._dispatch_mode is not None:
            self._dispatch_mode.__exit__(None, None, None)
            self._dispatch_mode = None
        if self._originals:
            self.torch.nn.Module.__init__ = self._originals["module_init"]
            self.torch.nn.Module.__call__ = self._originals["module_call"]
            self.torch.optim.Optimizer.__init__ = self._originals[
                "optimizer_init"]
            self.torch.optim.Optimizer.step = self._originals[
                "optimizer_step"]
            self.torch.save = self._originals["torch_save"]


def exact_root_file_set(root, expected):
    observed = set()
    for path in root.iterdir():
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            return False, sorted(item.name for item in root.iterdir())
        observed.add(path.name)
    return observed == set(expected), sorted(observed)


def validate_control_contract(args, spec, binding):
    if (args.spec != EXPECTED_SPEC_PATH or
            args.binding != EXPECTED_BINDING_PATH or
            args.attempt_root != EXPECTED_ATTEMPT_ROOT):
        raise ContractError("M19a invocation path drifted")
    if (spec.get("schema") !=
            "sttrack-lachtt-m19a-bootstrap-attributed-runtime-provenance-"
            "spec/v1" or spec.get("complete") is not True):
        raise ContractError("M19a spec schema drifted")
    if (binding.get("schema") !=
            "sttrack-lachtt-m19a-bootstrap-attribution-binding/v1" or
            binding.get("complete") is not True):
        raise ContractError("M19a binding schema drifted")
    if binding.get("binding_path") != str(args.binding):
        raise ContractError("M19a binding path drifted")
    for name, actual_path in (
            ("spec", args.spec), ("runner", Path(__file__).resolve()),
            ("plan", EXPECTED_PLAN_PATH)):
        record = binding[name]
        if Path(record["path"]).resolve() != actual_path:
            raise ContractError("{} path drifted".format(name))
        read_verified_bytes(record)
        if name != "spec" and spec[name] != record:
            raise ContractError("spec/binding {} identity mismatch".format(
                name))
    audit, audit_actual = load_verified_json(binding["preexecution_audit"])
    expected_identity = {
        "spec_sha256": binding["spec"]["sha256"],
        "runner_sha256": binding["runner"]["sha256"],
        "plan_sha256": binding["plan"]["sha256"],
        "repository_commit": binding["repository"]["commit"],
        "python_sha256": binding["python"]["sha256"],
        "attempt_root": str(args.attempt_root),
    }
    allowed = audit.get("authorization_boundary", {}).get(
        "authorized_next_actions_after_pass", [])
    if (audit.get("overall_verdict") != "PASS" or
            AUTHORIZED_ACTION not in allowed or
            any(audit.get("audited_identity", {}).get(key) != value
                for key, value in expected_identity.items())):
        raise ContractError("M19a preexecution authorization drifted")
    if spec["repository"] != binding["repository"]:
        raise ContractError("spec/binding repository identity mismatch")
    head_identity = read_git_head(REPOSITORY_ROOT)
    if (head_identity["commit"] != binding["repository"]["commit"] or
            head_identity["branch"] != binding["repository"]["branch"]):
        raise ContractError("repository identity drifted")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON:
        raise ContractError("Python executable path drifted")
    python_actual = regular_file_record(EXPECTED_PYTHON)
    if any(python_actual[key] != binding["python"][key]
           for key in ("path", "bytes", "sha256", "mode")):
        raise ContractError("Python executable identity drifted")
    if spec["python"] != binding["python"]:
        raise ContractError("spec/binding Python identity mismatch")
    source_records = binding["project_sources"]
    expected_names = {filename for _, filename in PROJECT_MODULE_ORDER}
    if {Path(record["path"]).name for record in source_records} != expected_names:
        raise ContractError("project source closure drifted")
    verified_sources = {}
    for record in source_records:
        _, actual = read_verified_bytes(record)
        verified_sources[Path(actual["path"]).name] = actual
    if spec["project_sources"] != source_records:
        raise ContractError("spec/binding source closure mismatch")
    if spec["prior_observation"] != binding["prior_observation"]:
        raise ContractError("spec/binding prior observation mismatch")
    prior_records = []
    for record in binding["prior_observation"]:
        _, actual = read_verified_bytes(record)
        prior_records.append(actual)
    repo_identity = direct_git_identity(REPOSITORY_ROOT)
    if (repo_identity["commit"] != binding["repository"]["commit"] or
            repo_identity["branch"] != binding["repository"]["branch"] or
            repo_identity["clean"] is not True or
            binding["repository"].get("clean_required") is not True or
            binding["repository"].get(
                "clean_verified_at_closure") is not True):
        raise ContractError("full repository identity/clean state drifted")
    if spec["runtime"]["attempt_root"] != str(args.attempt_root):
        raise ContractError("attempt root spec drifted")
    if spec["execution"]["model_instantiations"] != 0 or \
            spec["execution"]["forward_calls"] != 0 or \
            spec["execution"]["optimizer_steps"] != 0:
        raise ContractError("zero-model execution contract drifted")
    return {
        "audit": audit_actual,
        "repository": repo_identity,
        "python": python_actual,
        "sources": verified_sources,
        "prior_observation": prior_records,
    }


def first_attribution_frame(stack, runner_path):
    runner_path = str(Path(runner_path).resolve())
    for row in reversed(stack):
        frame_path = row["file"]
        normalized = frame_path.replace("\\", "/")
        if frame_path == runner_path or frame_path.startswith("<"):
            continue
        if ("/lib/python3." in normalized and
                "/site-packages/" not in normalized):
            continue
        return row
    return None


def event_analysis(observation, runner_path):
    popen_by_id = {
        row["correlation_id"]: row for row in observation["popen_calls"]}
    subprocess_rows = observation["subprocess_audit_events"]
    devnull_rows = observation["devnull_write_events"]
    relevant = subprocess_rows + devnull_rows
    audit_correlations = {
        row.get("correlation_id") for row in subprocess_rows}
    wrapper_correlations = set(popen_by_id)
    correlations_complete = (
        bool(relevant) and audit_correlations == wrapper_correlations and
        None not in audit_correlations and all(
            row.get("correlation_id") in popen_by_id for row in devnull_rows))
    stacks_complete = (
        bool(relevant) and all(bool(row.get("stack")) for row in relevant) and
        all(bool(row.get("stack")) for row in observation["popen_calls"]))
    audit_parameters_complete = bool(subprocess_rows) and all(
        row.get("command") is not None and row.get("executable") is not None
        for row in subprocess_rows)
    wrapper_parameters_complete = bool(observation["popen_calls"]) and all(
        row.get("command") is not None and
        row.get("effective_executable") is not None and
        row.get("effective_cwd") is not None and
        all(key in row for key in ("stdin", "stdout", "stderr"))
        for row in observation["popen_calls"])
    devnull_parameters_complete = bool(devnull_rows) and all(
        row.get("path") == "/dev/null" and
        row.get("raw_target") is not None and
        bool(row.get("mode_semantics")) and
        row.get("flags") is not None
        for row in devnull_rows)
    parameters_complete = (
        audit_parameters_complete and wrapper_parameters_complete and
        devnull_parameters_complete)
    allowed_phases = {
        "observer_installed", "subprocess_wrapper_installed", "torch_import",
    }
    pure_bootstrap_phase = bool(relevant) and all(
        row.get("phase") in allowed_phases for row in relevant)
    attribution_callers_complete = bool(relevant) and all(
        first_attribution_frame(row.get("stack", []), runner_path) is not None
        for row in relevant)
    linked_pairs = []
    for row in devnull_rows:
        correlation_id = row.get("correlation_id")
        linked_pairs.append({
            "devnull_event_id": row["event_id"],
            "correlation_id": correlation_id,
            "popen_call_event_id": (
                popen_by_id[correlation_id]["event_id"]
                if correlation_id in popen_by_id else None),
            "first_attribution_caller": first_attribution_frame(
                row.get("stack", []), runner_path),
        })
    return {
        "popen_audit_count": len(subprocess_rows),
        "popen_wrapper_count": len(observation["popen_calls"]),
        "devnull_write_count": len(devnull_rows),
        "correlations_complete": correlations_complete,
        "stacks_complete": stacks_complete,
        "parameters_complete": parameters_complete,
        "pure_bootstrap_phase": pure_bootstrap_phase,
        "attribution_callers_complete": attribution_callers_complete,
        "linked_pairs": linked_pairs,
        "subprocess_first_attribution_callers": [
            first_attribution_frame(row.get("stack", []), runner_path)
            for row in subprocess_rows
        ],
    }


def seal_journal(root, start_path, terminal_path, manifest_path,
                 status, exit_code, observer):
    os.chmod(start_path, 0o444)
    os.chmod(terminal_path, 0o444)
    start_record = regular_file_record(start_path)
    terminal_record = regular_file_record(terminal_path)
    post_terminal_observation = observer.snapshot()
    observer.capture = False
    manifest = {
        "schema": "sttrack-lachtt-m19a-bootstrap-attribution-manifest/v1",
        "complete": True,
        "status": status,
        "exit_code": exit_code,
        "expected_file_set": ["manifest.json", "start.json", "terminal.json"],
        "files": {"start.json": start_record,
                   "terminal.json": terminal_record},
        "post_terminal_runtime_observation": post_terminal_observation,
        "journal_write_contract": {
            "classification": "attempt_root_allowed_journal_publication",
            "start": {"path": str(start_path),
                      "temporary": str(start_path) + ".tmp"},
            "terminal": {"path": str(terminal_path),
                         "temporary": str(terminal_path) + ".tmp"},
            "manifest": {"path": str(manifest_path),
                         "temporary": str(manifest_path) + ".tmp",
                         "self_write_evidence":
                             "manifest file existence, mode and external hash"},
        },
    }
    write_json_atomic(manifest_path, manifest)
    os.chmod(manifest_path, 0o444)
    exact, observed = exact_root_file_set(
        root, {"manifest.json", "start.json", "terminal.json"})
    if not exact:
        raise ContractError("journal file set drifted: {}".format(observed))
    os.chmod(root, 0o555)


def recover_journal_failure(root, start_path, terminal_path, manifest_path,
                            terminal, seal_error):
    """Best-effort immutable failure receipt; never reports acceptance."""
    try:
        os.chmod(root, 0o755)
    except OSError:
        pass
    for path in (start_path, terminal_path, manifest_path):
        if path.exists() and not path.is_symlink():
            try:
                os.chmod(path, 0o644)
            except OSError:
                pass
    terminal.update({
        "status": "journal_failure",
        "exit_code": 3,
        "accepted": False,
        "seal_error": {
            "type": seal_error.__class__.__name__,
            "message": str(seal_error),
            "traceback": traceback.format_exc(),
        },
    })
    write_json_atomic(terminal_path, terminal)
    os.chmod(start_path, 0o444)
    os.chmod(terminal_path, 0o444)
    inventory = sorted(path.name for path in root.iterdir())
    manifest = {
        "schema": "sttrack-lachtt-m19a-bootstrap-attribution-manifest/v1",
        "complete": True,
        "status": "journal_failure",
        "exit_code": 3,
        "expected_file_set": ["manifest.json", "start.json", "terminal.json"],
        "observed_file_set": inventory,
        "files": {
            "start.json": regular_file_record(start_path),
            "terminal.json": regular_file_record(terminal_path),
        },
    }
    write_json_atomic(manifest_path, manifest)
    os.chmod(manifest_path, 0o444)
    os.chmod(root, 0o555)


def main():
    args = parse_args()
    args.spec = args.spec.resolve()
    args.binding = args.binding.resolve()
    args.attempt_root = args.attempt_root.resolve()
    if EXPECTED_ATTEMPT_ROOT.exists():
        raise ContractError("M19a attempt root already exists")

    # Authorization is a true precondition: invalid controls must not consume
    # the unique attempt root. This preflight uses stdlib-only file reads and
    # performs no torch or project import.
    spec_record = regular_file_record(args.spec)
    binding_record = regular_file_record(args.binding)
    spec, _ = load_verified_json(spec_record)
    binding, _ = load_verified_json(binding_record)
    controls = validate_control_contract(args, spec, binding)
    if EXPECTED_ATTEMPT_ROOT.exists():
        raise ContractError("M19a attempt root appeared during preflight")

    EXPECTED_ATTEMPT_ROOT.mkdir(mode=0o755, parents=False, exist_ok=False)
    attempt_root = EXPECTED_ATTEMPT_ROOT
    start_path = attempt_root / "start.json"
    terminal_path = attempt_root / "terminal.json"
    manifest_path = attempt_root / "manifest.json"
    observer = ProvenanceObserver(attempt_root, Path(__file__).resolve())
    sys.addaudithook(observer.hook)
    start = {
        "schema": "sttrack-lachtt-m19a-bootstrap-attribution-start/v1",
        "complete": True,
        "phase": "observer_installed",
        "argv": list(sys.argv),
        "pid": os.getpid(),
        "python": str(Path(sys.executable).resolve()),
        "requested_paths": {
            "spec": str(args.spec), "binding": str(args.binding),
            "attempt_root": str(args.attempt_root),
        },
        "execution_contract": {
            "model_instantiations": 0,
            "forward_call_entries": 0,
            "tensor_dispatch_ops": 0,
            "optimizer_constructions": 0,
            "optimizer_step_entries": 0,
            "checkpoint_write_entries": 0,
        },
        "spec": spec_record,
        "binding": binding_record,
        "repository": controls["repository"],
        "python_identity": controls["python"],
    }
    write_json_atomic(start_path, start)
    status = "failed"
    exit_code = 1
    source_load_records = []
    execution_counts = {
        "model_instantiations": 0,
        "forward_call_entries": 0,
        "tensor_dispatch_ops": 0,
        "optimizer_constructions": 0,
        "optimizer_step_entries": 0,
        "checkpoint_write_entries": 0,
        "benchmark_frames": 0,
    }
    exception_record = None
    observation = observer.snapshot()
    analysis = None
    gates = {}
    try:
        observer.allow_source_paths(
            record["path"] for record in controls["sources"].values())

        observer.set_phase("subprocess_wrapper_installed")
        subprocess_module, original_popen = install_traced_popen(observer)
        if subprocess_module.Popen is original_popen:
            raise ContractError("Popen wrapper installation failed")

        sys.dont_write_bytecode = True
        observer.set_phase("torch_import")
        torch_module = importlib.import_module("torch")
        torch_identity = {
            "version": str(torch_module.__version__),
            "module_path": str(Path(torch_module.__file__).resolve()),
            "cuda_version": str(torch_module.version.cuda),
        }
        if torch_identity != spec["torch"]:
            raise ContractError("torch identity drifted")

        instrumentation = ProjectExecutionInstrumentation(torch_module)
        try:
            instrumentation.install()
            install_stub_packages(REPOSITORY_ROOT)
            for module_name, filename in PROJECT_MODULE_ORDER[:-1]:
                observer.set_phase("project_relation_imports")
                source_load_records.append(load_verified_module(
                    module_name, controls["sources"][filename]))
            observer.set_phase("causal_model_import")
            module_name, filename = PROJECT_MODULE_ORDER[-1]
            source_load_records.append(load_verified_module(
                module_name, controls["sources"][filename]))
        finally:
            instrumentation.restore()
            execution_counts.update(instrumentation.counts)

        observer.set_phase("terminal_snapshot")
        observation = observer.snapshot()
        analysis = event_analysis(observation, Path(__file__).resolve())
        final_head_identity = read_git_head(REPOSITORY_ROOT)
        expected_head_identity = {
            "path": controls["repository"]["path"],
            "commit": controls["repository"]["commit"],
            "branch": controls["repository"]["branch"],
        }
        gates = {
            "repository_clean_preflight_exact": (
                controls["repository"]["clean"] is True),
            "repository_head_still_exact": (
                final_head_identity == expected_head_identity),
            "python_identity_exact": (
                regular_file_record(EXPECTED_PYTHON) == controls["python"]),
            "project_sources_loaded_exactly_once": (
                len(source_load_records) == len(PROJECT_MODULE_ORDER) and
                {row["sha256"] for row in source_load_records} ==
                {row["sha256"] for row in controls["sources"].values()}),
            "popen_reproduced": analysis["popen_audit_count"] >= 1,
            "devnull_write_reproduced": analysis["devnull_write_count"] >= 1,
            "event_correlations_complete": analysis["correlations_complete"],
            "event_stacks_complete": analysis["stacks_complete"],
            "event_parameters_complete": analysis["parameters_complete"],
            "event_attribution_callers_complete": analysis[
                "attribution_callers_complete"],
            "events_only_in_pure_bootstrap_phase": analysis[
                "pure_bootstrap_phase"],
            "no_other_forbidden_writes": not observation[
                "forbidden_write_events"],
            "no_unresolved_writes": not observation[
                "unresolved_write_events"],
            "no_forbidden_mutations": not observation[
                "forbidden_mutation_events"],
            "no_network": not observation["network_events"],
            "no_sensitive_data_reads": not observation[
                "sensitive_read_events"],
            "no_forbidden_modules": not observation["forbidden_modules"],
            "zero_model_instantiations": (
                execution_counts["model_instantiations"] == 0),
            "zero_forward_call_entries": (
                execution_counts["forward_call_entries"] == 0),
            "zero_tensor_dispatch_ops": (
                execution_counts["tensor_dispatch_ops"] == 0),
            "zero_optimizer_constructions": (
                execution_counts["optimizer_constructions"] == 0),
            "zero_optimizer_steps": (
                execution_counts["optimizer_step_entries"] == 0),
            "zero_checkpoint_writes": (
                execution_counts["checkpoint_write_entries"] == 0),
        }
        accepted = all(gates.values())
        status = "success" if accepted else "gate_failure"
        exit_code = 0 if accepted else 2
    except BaseException as error:
        exception_record = {
            "type": error.__class__.__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        observation = observer.snapshot()
        analysis = event_analysis(observation, Path(__file__).resolve())
        status = "exception"
        exit_code = 1
    finally:
        observer.set_phase("journal_publication")
        terminal = {
            "schema": "sttrack-lachtt-m19a-bootstrap-attribution-terminal/v1",
            "complete": True,
            "status": status,
            "exit_code": exit_code,
            "accepted": status == "success",
            "claim_ceiling": "import-only runtime provenance; no model or benchmark claim",
            "controls": controls,
            "loaded_project_sources": source_load_records,
            "observation": observation,
            "analysis": analysis,
            "gates": gates,
            "exception": exception_record,
            "execution_counts": execution_counts,
            "authorization": {
                "m19b_receipt": False,
                "model_smoke": False,
                "training": False,
                "public_evaluation": False,
            },
        }
        try:
            write_json_atomic(terminal_path, terminal)
            seal_journal(attempt_root, start_path, terminal_path, manifest_path,
                         status, exit_code, observer)
        except BaseException as seal_error:
            exit_code = 3
            observer.capture = False
            recover_journal_failure(
                attempt_root, start_path, terminal_path, manifest_path,
                terminal, seal_error)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
