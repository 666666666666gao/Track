"""Run the pure CADTrack fine-tune and preserve its process receipts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a child command is required after --")

    run_root = args.run_root.resolve()
    control_root = args.control_root.resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError("Run directory is not empty: " + str(run_root))
    control_root.mkdir(parents=True, exist_ok=False)
    run_root.mkdir(parents=True, exist_ok=True)
    temporary_root = control_root / "tmp"
    temporary_root.mkdir()

    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONOPTIMIZE="0",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        TMPDIR=str(temporary_root),
    )
    started_at = datetime.now(timezone.utc).isoformat()
    identity = {
        "command": command,
        "cwd": str(run_root),
        "started_at": started_at,
        "run_root": str(run_root),
        "control_root": str(control_root),
    }
    (control_root / "supervisor.json").write_text(
        json.dumps(identity, indent=2) + "\n", encoding="utf-8"
    )
    with (control_root / "train.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(
            command,
            cwd=run_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        identity.update(supervisor_pid=os.getpid(), child_pid=child.pid)
        (control_root / "child.json").write_text(
            json.dumps(identity, indent=2) + "\n", encoding="utf-8"
        )
        return_code = child.wait()

    (control_root / "train.exit").write_text(
        str(return_code) + "\n", encoding="utf-8"
    )
    terminal = dict(
        identity,
        exit_code=return_code,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    (control_root / "terminal.json").write_text(
        json.dumps(terminal, indent=2) + "\n", encoding="utf-8"
    )
    return return_code if return_code >= 0 else 128 - return_code


if __name__ == "__main__":
    sys.exit(main())
