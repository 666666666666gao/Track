#!/usr/bin/env python3
import argparse
import math
import re
from statistics import mean


METRIC_RE = re.compile(r"([A-Za-z0-9_/-]+):\s*(-?(?:\d+(?:\.\d*)?|\.\d+))")
STEP_RE = re.compile(r"\[(\w+):\s*(\d+),\s*(\d+)\s*/\s*(\d+)\]")


def parse_line(line):
    step_match = STEP_RE.search(line)
    if step_match is None:
        return None
    record = {
        "loader": step_match.group(1),
        "epoch": int(step_match.group(2)),
        "iter": int(step_match.group(3)),
        "total_iter": int(step_match.group(4)),
    }
    for key, value in METRIC_RE.findall(line):
        try:
            record[key] = float(value)
        except ValueError:
            continue
    return record


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is not None:
                records.append(parsed)
    return records


def avg(records, key):
    vals = [r[key] for r in records if key in r and math.isfinite(r[key])]
    return mean(vals) if vals else None


def fmt(value):
    return "n/a" if value is None else "{:.5f}".format(value)


def add_warning(warnings, condition, text):
    if condition:
        warnings.append(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", default="output/depthtrack_roberta/train_roberta_moe.log")
    parser.add_argument("--last", type=int, default=20, help="Number of latest training log records to summarize.")
    args = parser.parse_args()

    records = load_records(args.log)
    if not records:
        raise SystemExit("No train records found in {}".format(args.log))
    latest = records[-max(args.last, 1):]
    last = latest[-1]

    keys = [
        "Loss/total",
        "Loss/lang_route",
        "Loss/lang_route_entropy",
        "Loss/lang_route_sup",
        "Loss/lang_match_scale",
        "IoU",
        "Lang/targetness_gap",
        "Lang/targetness_mean",
        "Lang/targetness_min",
        "Lang/targetness_max",
        "Lang/target_match_gap",
        "Lang/context_bg_minus_fg",
        "Lang/targetness_logit_std",
        "Lang/candidate_keep_ratio",
        "Lang/candidate_gate_min",
        "Lang/candidate_gate_max",
        "Lang/match_scale",
        "Lang/train_gate",
        "MoE/w_rgb",
        "MoE/w_depth",
        "MoE/w_lang",
        "MoE/w_shared",
        "MoE/target_w_rgb",
        "MoE/target_w_depth",
        "MoE/target_w_lang",
        "MoE/target_w_shared",
        "MoE/route_target_l1",
        "MoE/reliable_w_depth",
        "MoE/reliable_w_rgb",
        "MoE/unreliable_w_depth",
        "MoE/unreliable_w_rgb",
        "MoE/raw_w_lang",
        "MoE/entropy",
        "MoE/route_max",
        "MoE/delta_norm_ratio",
        "MoE/residual_gate",
        "MoE/residual_weight",
        "MoE/matching_strength",
        "MoE/token_keep_ratio_cfg",
        "MoE/token_mask_strength",
        "MoE/effective_token_mask_strength",
        "Input/depth_std",
        "Input/depth_bbox_contrast",
        "Input/depth_bbox_target_std",
        "Input/depth_global_std_score",
        "Input/depth_image_reliability",
    ]

    print("log: {}".format(args.log))
    print("records: {}  latest: epoch {} iter {}/{}".format(
        len(records), last["epoch"], last["iter"], last["total_iter"]))
    print("window: last {} records".format(len(latest)))
    for key in keys:
        print("{:<28} latest={}  avg={}".format(key, fmt(last.get(key)), fmt(avg(latest, key))))

    warnings = []
    target_gap = avg(latest, "Lang/targetness_gap")
    target_match_gap = avg(latest, "Lang/target_match_gap")
    context_gap = avg(latest, "Lang/context_bg_minus_fg")
    targetness_mean = avg(latest, "Lang/targetness_mean")
    targetness_logit_std = avg(latest, "Lang/targetness_logit_std")
    match_scale = avg(latest, "Lang/match_scale")
    route_max = avg(latest, "MoE/route_max")
    route_target_l1 = avg(latest, "MoE/route_target_l1")
    w_lang = avg(latest, "MoE/w_lang")
    target_w_lang = avg(latest, "MoE/target_w_lang")
    raw_w_lang = avg(latest, "MoE/raw_w_lang")
    reliable_w_depth = avg(latest, "MoE/reliable_w_depth")
    reliable_w_rgb = avg(latest, "MoE/reliable_w_rgb")
    unreliable_w_depth = avg(latest, "MoE/unreliable_w_depth")
    unreliable_w_rgb = avg(latest, "MoE/unreliable_w_rgb")
    delta_norm = avg(latest, "MoE/delta_norm_ratio")
    candidate_keep = avg(latest, "Lang/candidate_keep_ratio")
    candidate_gate_min = avg(latest, "Lang/candidate_gate_min")
    candidate_gate_max = avg(latest, "Lang/candidate_gate_max")
    depth_std = avg(latest, "Input/depth_std")
    depth_bbox_contrast = avg(latest, "Input/depth_bbox_contrast")
    depth_image_reliability = avg(latest, "Input/depth_image_reliability")
    loss_total = avg(latest, "Loss/total")
    iou = avg(latest, "IoU")
    residual_weight = avg(latest, "MoE/residual_weight")
    matching_strength = avg(latest, "MoE/matching_strength")
    token_mask_strength = avg(latest, "MoE/token_mask_strength")
    effective_token_mask_strength = avg(latest, "MoE/effective_token_mask_strength")
    train_gate = avg(latest, "Lang/train_gate")
    language_inactive = (
        residual_weight is not None and matching_strength is not None
        and abs(residual_weight) < 1e-8 and abs(matching_strength) < 1e-8
        and (token_mask_strength is None or abs(token_mask_strength) < 1e-8)
    )

    add_warning(warnings, loss_total is not None and (not math.isfinite(loss_total) or loss_total > 10.0),
                "loss is abnormal")
    add_warning(warnings, iou is not None and iou < 0.45,
                "training IoU is low")
    add_warning(warnings, (not language_inactive) and target_gap is not None and last["iter"] >= 500 and target_gap < 0.10,
                "targetness foreground-background gap is weak")
    add_warning(warnings, (not language_inactive) and target_match_gap is not None and last["iter"] >= 500 and target_match_gap < 0.05,
                "target text matching gap is weak")
    add_warning(warnings, (not language_inactive) and context_gap is not None and last["iter"] >= 500 and context_gap < 0.03,
                "context suppression gap is weak or inverted")
    add_warning(warnings, (not language_inactive) and targetness_mean is not None
                and targetness_mean < 0.03 and (target_gap is None or target_gap < 0.20),
                "targetness map is extremely sparse without a useful foreground-background gap")
    add_warning(warnings, (not language_inactive) and targetness_mean is not None and last["iter"] >= 500
                and targetness_mean < 0.06 and (target_gap is None or target_gap < 0.20),
                "targetness map is becoming over-sparse without a useful foreground-background gap")
    add_warning(warnings, (not language_inactive) and targetness_mean is not None and targetness_logit_std is not None
                and targetness_mean < 0.03 and targetness_logit_std > 1.60,
                "targetness is sparse and high-contrast; verify tracking eval before continuing")
    add_warning(warnings, (not language_inactive) and targetness_logit_std is not None and targetness_logit_std > 3.0,
                "targetness logits may be over-saturated")
    add_warning(warnings, (not language_inactive) and match_scale is not None and match_scale > 12.0,
                "language matching scale is too high")
    add_warning(warnings, (not language_inactive) and route_max is not None and route_max > 0.75,
                "MoE route is collapsing to one expert")
    add_warning(warnings, (not language_inactive) and route_target_l1 is not None
                and last["iter"] >= 500 and route_target_l1 > 0.18,
                "MoE route is far from the reliability-aware target")
    add_warning(warnings, (not language_inactive) and raw_w_lang is not None and w_lang is not None and raw_w_lang - w_lang > 0.20,
                "language route cap is heavily clipping the language expert")
    add_warning(warnings, (not language_inactive) and target_w_lang is not None and w_lang is not None
                and last["iter"] >= 500 and abs(w_lang - target_w_lang) > 0.05,
                "language expert route deviates from the supervised target")
    add_warning(warnings, (not language_inactive) and reliable_w_depth is not None and reliable_w_rgb is not None
                and reliable_w_depth + 0.05 < reliable_w_rgb,
                "reliable-depth samples still route more to RGB than depth")
    add_warning(warnings, (not language_inactive) and unreliable_w_depth is not None and unreliable_w_rgb is not None
                and unreliable_w_depth > unreliable_w_rgb,
                "unreliable-depth samples route more to depth than RGB")
    add_warning(warnings, delta_norm is not None and delta_norm > 0.35,
                "MoE residual is large relative to the RGB-D fused feature")
    add_warning(warnings, (not language_inactive) and candidate_keep is not None and candidate_keep < 0.50,
                "language candidate selection may be too sparse")
    add_warning(warnings, (not language_inactive) and candidate_gate_min is not None and candidate_gate_min < 0.75,
                "language candidate gate suppresses some tokens too strongly")
    add_warning(warnings, (not language_inactive) and candidate_gate_max is not None and candidate_gate_max > 1.25,
                "language candidate gate amplifies some tokens too strongly")
    add_warning(warnings, (not language_inactive) and w_lang is not None and last["iter"] >= 1000 and w_lang < 0.04,
                "language expert route is nearly unused")
    add_warning(warnings, train_gate is not None and last["epoch"] >= 2 and train_gate < 0.95,
                "language train gate is still below full strength after epoch 1")
    add_warning(warnings, (not language_inactive) and token_mask_strength is not None
                and effective_token_mask_strength is not None and train_gate is not None
                and train_gate < 0.95 and effective_token_mask_strength >= token_mask_strength * 0.95,
                "effective token mask is not following the language train gate")
    add_warning(warnings, depth_std is not None and depth_std < 0.10,
                "depth input variance is near zero")
    add_warning(warnings, depth_bbox_contrast is not None and depth_bbox_contrast < 0.05,
                "target-context depth contrast is weak")
    add_warning(warnings, depth_image_reliability is not None and depth_image_reliability < 0.20,
                "image-derived depth reliability is very low")

    if warnings:
        print("warnings:")
        for warning in warnings:
            print("- {}".format(warning))
    else:
        print("warnings: none")
    if language_inactive:
        print("note: language/MoE influence is inactive; language targetness diagnostics are informational only")


if __name__ == "__main__":
    main()
