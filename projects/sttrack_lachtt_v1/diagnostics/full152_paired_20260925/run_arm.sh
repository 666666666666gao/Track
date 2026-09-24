#!/usr/bin/env bash
set -u

root=/root/autodl-tmp/sttrack_full152_paired_20260925
name="$1"
gpu="$2"
case "$name" in
  M67) arm=control ;;
  M82) arm=category ;;
  *) exit 2 ;;
esac

CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
  /root/autodl-tmp/envs/sttrack/bin/python "$root/$name/train_causal.py" --arm "$arm" \
  > "$root/$name/training/train.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/$name/training/train.exit"
exit "$status"
