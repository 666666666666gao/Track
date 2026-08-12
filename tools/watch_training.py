#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime


CHECKPOINT_RE = re.compile(r"MPLTTrack_(ep\d{4}(?:_it\d+)?)\.pth\.tar$")


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def list_checkpoints(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return []
    checkpoints = []
    for name in os.listdir(checkpoint_dir):
        match = CHECKPOINT_RE.match(name)
        if match is None:
            continue
        path = os.path.join(checkpoint_dir, name)
        try:
            mtime = os.path.getmtime(path)
            size = os.path.getsize(path)
        except OSError:
            continue
        checkpoints.append((mtime, match.group(1), path, size))
    checkpoints.sort()
    return checkpoints


def run_diagnosis(log_path, last):
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "diagnose_training_log.py"),
        log_path,
        "--last",
        str(last),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, check=False)
    return result.returncode, result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Periodically summarize DaLaTrack/RGB-D-L training health.")
    parser.add_argument("--log", default="output/depthtrack_roberta/train_roberta_moe.log")
    parser.add_argument(
        "--checkpoint_dir",
        default=("output/depthtrack_roberta/checkpoints/train/mplt_track/"
                 "vitb_256_mplt_32x1_1e4_depthtrack_15ep_roberta"))
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument("--last", type=int, default=200)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    seen = set()
    while True:
        print("=" * 88, flush=True)
        print("[{}] training health check".format(timestamp()), flush=True)

        if os.path.isfile(args.log):
            code, output = run_diagnosis(args.log, args.last)
            print(output, flush=True)
            if code != 0:
                print("diagnosis command failed with return code {}".format(code), flush=True)
        else:
            print("log not found: {}".format(args.log), flush=True)

        checkpoints = list_checkpoints(args.checkpoint_dir)
        if checkpoints:
            latest_mtime, latest_epoch, latest_path, latest_size = checkpoints[-1]
            print("latest checkpoint: {}  size={:.2f}GB  mtime={}".format(
                latest_epoch, latest_size / (1024 ** 3),
                datetime.fromtimestamp(latest_mtime).strftime("%Y-%m-%d %H:%M:%S")),
                flush=True)
            for _, epoch, path, _ in checkpoints:
                if path not in seen:
                    print("new checkpoint detected: {}".format(epoch), flush=True)
                    seen.add(path)
        else:
            print("no checkpoints found in {}".format(args.checkpoint_dir), flush=True)

        if args.once:
            break
        time.sleep(max(args.interval, 30))


if __name__ == "__main__":
    main()
