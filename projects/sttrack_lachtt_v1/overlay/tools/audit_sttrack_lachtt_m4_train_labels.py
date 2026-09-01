#!/usr/bin/env python3
"""Audit M4 outcome frequencies using training folds only."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lib.config.sttrack.config import cfg, update_config_from_file
from lib.models.sttrack import build_sttrack
from lib.models.sttrack.lachtt_rollout_association import LanguageAnchoredDenseAssociation
from lib.models.sttrack.lachtt_setwise_association import SetwiseCandidateAssociation
from lib.test.tracker.data_utils import PreprocessorMM
from lib.test.utils.hann import hann2d
from tools.run_sttrack_lachtt_recursive_pilot import (
    atomic_json, build_context, load_schedule, sha256_file, stable_fold,
    valid_window,
)
from tools.run_sttrack_lachtt_setwise_pilot import run_event


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--heads", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.spec = args.spec.resolve(); args.heads = args.heads.resolve()
    args.output = args.output.resolve()
    if args.output.exists(): raise FileExistsError(args.output)
    started = time.time(); spec = json.loads(args.spec.read_text())
    config = Path(spec["base_model"]["config"]["path"])
    checkpoint = Path(spec["base_model"]["checkpoint"]["path"])
    update_config_from_file(str(config))
    network = build_sttrack(cfg, training=False)
    network.load_state_dict(torch.load(str(checkpoint), map_location="cpu")["net"], strict=True)
    network = network.cuda().eval()
    for parameter in network.parameters(): parameter.requires_grad_(False)
    dense = LanguageAnchoredDenseAssociation().cuda().eval()
    setwise = SetwiseCandidateAssociation().cuda().eval()
    saved = torch.load(str(args.heads), map_location="cpu")
    dense.load_state_dict(saved["dense"], strict=True)
    setwise.load_state_dict(saved["setwise"], strict=True)
    preprocessor = PreprocessorMM(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    keep_rate = [value for value in torch.linspace(0.7, 1.0, 3)][::-1]
    window = hann2d(torch.tensor([16, 16]).long(), centered=True).cuda()
    trace_paths = [Path(row["path"]) for row in spec["data"]["protected_trace_shards"]]
    rows_by_sequence, events = load_schedule(trace_paths, 4)
    folds = int(spec["data"]["outer_fold_count"])
    heldout = int(spec["data"]["evaluation_outer_fold"])
    events = [event for event in events if stable_fold(
        event["sequence"], "sttrack-lachtt-outer-v1", folds) != heldout]
    events = events[:int(spec["data"]["training_event_limit"])]
    contexts = {}; dataset_root = Path(spec["data"]["dataset_root"])
    language = Path(spec["inputs"]["language_manifest"]["path"])
    clip_model = Path(spec["inputs"]["clip"]["path"])
    def context(sequence):
        if sequence not in contexts:
            contexts[sequence] = build_context(
                sequence, dataset_root, rows_by_sequence[sequence], network,
                preprocessor, keep_rate, language, clip_model)
        return contexts[sequence]
    identity = torch.arange(6, device="cuda")
    valid_events = 0; unavailable = 0; beneficial_events = 0
    beneficial_candidates = 0; catastrophic_candidates = 0; neutral_candidates = 0
    per_sequence = {}
    for event in events:
        item = context(event["sequence"]); frame = event["trigger_frame"]
        if not valid_window(item.gt, frame, 4): unavailable += 1; continue
        trace = run_event(network, dense, setwise, None, preprocessor, item,
                          frame, 4, keep_rate, window, identity, False)
        valid_events += 1
        labels = ["catastrophic" if action["actual_catastrophic"] else
                  "beneficial" if action["actual_beneficial"] else "neutral"
                  for action in trace["actions"]]
        beneficial_candidates += labels.count("beneficial")
        catastrophic_candidates += labels.count("catastrophic")
        neutral_candidates += labels.count("neutral")
        has_beneficial = "beneficial" in labels
        beneficial_events += int(has_beneficial)
        row = per_sequence.setdefault(event["sequence"], {"events":0,"beneficial_events":0})
        row["events"] += 1; row["beneficial_events"] += int(has_beneficial)
    candidate_count = valid_events * 6
    result = {
        "schema":"sttrack-lachtt-m4-training-fold-label-audit/v1",
        "complete":True,
        "train_only":True,
        "heldout_fold_opened":False,
        "repository_commit":subprocess.check_output(
            ["git","-C",str(REPOSITORY_ROOT),"rev-parse","HEAD"],text=True).strip(),
        "runner_sha256":sha256_file(Path(__file__).resolve()),
        "spec_sha256":sha256_file(args.spec),
        "heads_sha256":sha256_file(args.heads),
        "scheduled_events":len(events),"valid_events":valid_events,
        "unavailable_events":unavailable,
        "beneficial_events":beneficial_events,
        "no_beneficial_events":valid_events-beneficial_events,
        "beneficial_event_fraction":beneficial_events/max(1,valid_events),
        "candidate_count":candidate_count,
        "beneficial_candidates":beneficial_candidates,
        "catastrophic_candidates":catastrophic_candidates,
        "neutral_candidates":neutral_candidates,
        "beneficial_candidate_fraction":beneficial_candidates/max(1,candidate_count),
        "catastrophic_candidate_fraction":catastrophic_candidates/max(1,candidate_count),
        "suggested_inverse_event_weight":(valid_events-beneficial_events)/max(1,beneficial_events),
        "suggested_inverse_beneficial_candidate_weight":(candidate_count-beneficial_candidates)/max(1,beneficial_candidates),
        "suggested_inverse_catastrophic_candidate_weight":(candidate_count-catastrophic_candidates)/max(1,catastrophic_candidates),
        "per_sequence":per_sequence,"elapsed_seconds":time.time()-started,
        "checkpoint_written":False,"qwen_used":False,"vot_run":False}
    atomic_json(args.output,result); args.output.chmod(0o444)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == "__main__": main()
