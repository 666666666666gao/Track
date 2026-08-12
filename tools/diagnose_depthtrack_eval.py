#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.test.analysis.depthtrack_prre import evaluate_depthtrack_prre, format_depthtrack_prre
from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
from lib.test.utils.load_text import load_text


def _result_base(tracker, sequence):
    return os.path.join(tracker.results_dir, sequence.name)


def _read_times(path):
    if not os.path.isfile(path):
        return None
    try:
        values = np.asarray(load_text(path, delimiter=('\t', ',', ' '), dtype=np.float64)).reshape(-1)
    except Exception:
        return None
    values = values[np.isfinite(values) & (values >= 0.0)]
    return values if values.size else None


def _read_array(path, dtype=np.float64):
    if not os.path.isfile(path):
        return None
    try:
        arr = np.asarray(load_text(path, delimiter=('\t', ',', ' '), dtype=dtype))
    except Exception:
        return None
    return arr if arr.size else None


def _bbox_iou_xywh(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    n = min(len(a), len(b))
    if n == 0:
        return np.asarray([], dtype=np.float64)
    a = a[:n]
    b = b[:n]
    ax1, ay1 = a[:, 0], a[:, 1]
    ax2, ay2 = ax1 + np.maximum(a[:, 2], 0.0), ay1 + np.maximum(a[:, 3], 0.0)
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = bx1 + np.maximum(b[:, 2], 0.0), by1 + np.maximum(b[:, 3], 0.0)
    ix1, iy1 = np.maximum(ax1, bx1), np.maximum(ay1, by1)
    ix2, iy2 = np.minimum(ax2, bx2), np.minimum(ay2, by2)
    inter = np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)
    area_a = np.maximum(ax2 - ax1, 0.0) * np.maximum(ay2 - ay1, 0.0)
    area_b = np.maximum(bx2 - bx1, 0.0) * np.maximum(by2 - by1, 0.0)
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


def _first_failure_frame(iou, threshold=0.1, consecutive=10):
    if iou.size < consecutive:
        return None
    below = iou < threshold
    run = 0
    for idx, is_bad in enumerate(below):
        run = run + 1 if is_bad else 0
        if run >= consecutive:
            return idx - consecutive + 1
    return None


def _sequence_drift_stats(tracker, seq):
    base = _result_base(tracker, seq)
    pred = _read_array('{}.txt'.format(base))
    if pred is None:
        return None
    pred = pred.reshape(-1, 4)
    gt = np.asarray(seq.ground_truth_rect, dtype=np.float64).reshape(-1, 4)
    iou = _bbox_iou_xywh(pred, gt)
    scores = _read_array('{}_best_score.txt'.format(base))
    if scores is not None:
        scores = scores.reshape(-1)[:iou.size]
    consistency = _read_array('{}_template_consistency.txt'.format(base))
    if consistency is not None:
        consistency = consistency.reshape(-1)[:iou.size]
    rgb_consistency = _read_array('{}_template_rgb_consistency.txt'.format(base))
    if rgb_consistency is not None:
        rgb_consistency = rgb_consistency.reshape(-1)[:iou.size]
    depth_consistency = _read_array('{}_template_depth_consistency.txt'.format(base))
    if depth_consistency is not None:
        depth_consistency = depth_consistency.reshape(-1)[:iou.size]
    update_ratio = _read_array('{}_state_update_ratio.txt'.format(base))
    if update_ratio is not None:
        update_ratio = update_ratio.reshape(-1)[:iou.size]
    score_ema = _read_array('{}_score_ema.txt'.format(base))
    if score_ema is not None:
        score_ema = score_ema.reshape(-1)[:iou.size]
    lost_thr = _read_array('{}_effective_lost_thr.txt'.format(base))
    if lost_thr is not None:
        lost_thr = lost_thr.reshape(-1)[:iou.size]
    damping_thr = _read_array('{}_effective_damping_thr.txt'.format(base))
    if damping_thr is not None:
        damping_thr = damping_thr.reshape(-1)[:iou.size]
    response_entropy = _read_array('{}_response_entropy.txt'.format(base))
    if response_entropy is not None:
        response_entropy = response_entropy.reshape(-1)[:iou.size]
    response_peak_ratio = _read_array('{}_response_peak_ratio.txt'.format(base))
    if response_peak_ratio is not None:
        response_peak_ratio = response_peak_ratio.reshape(-1)[:iou.size]
    response_center_distance = _read_array('{}_response_center_distance.txt'.format(base))
    if response_center_distance is not None:
        response_center_distance = response_center_distance.reshape(-1)[:iou.size]
    response_guard = _read_array('{}_response_guard.txt'.format(base))
    if response_guard is not None:
        response_guard = response_guard.reshape(-1)[:iou.size]
    center_follow = _read_array('{}_center_follow.txt'.format(base))
    if center_follow is not None:
        center_follow = center_follow.reshape(-1)[:iou.size]
    depth_guard = _read_array('{}_depth_guard.txt'.format(base))
    if depth_guard is not None:
        depth_guard = depth_guard.reshape(-1)[:iou.size]
    confident_mismatch_search = _read_array('{}_confident_mismatch_search.txt'.format(base))
    if confident_mismatch_search is not None:
        confident_mismatch_search = confident_mismatch_search.reshape(-1)[:iou.size]
    candidate_rerank = _read_array('{}_candidate_rerank.txt'.format(base))
    if candidate_rerank is not None:
        candidate_rerank = candidate_rerank.reshape(-1)[:iou.size]
    candidate_rank = _read_array('{}_candidate_rank.txt'.format(base))
    if candidate_rank is not None:
        candidate_rank = candidate_rank.reshape(-1)[:iou.size]
    candidate_score_gain = _read_array('{}_candidate_score_gain.txt'.format(base))
    if candidate_score_gain is not None:
        candidate_score_gain = candidate_score_gain.reshape(-1)[:iou.size]
    candidate_consistency_gain = _read_array('{}_candidate_consistency_gain.txt'.format(base))
    if candidate_consistency_gain is not None:
        candidate_consistency_gain = candidate_consistency_gain.reshape(-1)[:iou.size]
    candidate_selected_iou = _read_array('{}_candidate_selected_iou.txt'.format(base))
    if candidate_selected_iou is not None:
        candidate_selected_iou = candidate_selected_iou.reshape(-1)[:iou.size]
    candidate_oracle_iou = _read_array('{}_candidate_oracle_iou.txt'.format(base))
    if candidate_oracle_iou is not None:
        candidate_oracle_iou = candidate_oracle_iou.reshape(-1)[:iou.size]
    candidate_oracle_rank = _read_array('{}_candidate_oracle_rank.txt'.format(base))
    if candidate_oracle_rank is not None:
        candidate_oracle_rank = candidate_oracle_rank.reshape(-1)[:iou.size]
    candidate_oracle_score_ratio = _read_array('{}_candidate_oracle_score_ratio.txt'.format(base))
    if candidate_oracle_score_ratio is not None:
        candidate_oracle_score_ratio = candidate_oracle_score_ratio.reshape(-1)[:iou.size]
    candidate_oracle_rgb_consistency = _read_array('{}_candidate_oracle_rgb_consistency.txt'.format(base))
    if candidate_oracle_rgb_consistency is not None:
        candidate_oracle_rgb_consistency = candidate_oracle_rgb_consistency.reshape(-1)[:iou.size]
    candidate_oracle_depth_consistency = _read_array('{}_candidate_oracle_depth_consistency.txt'.format(base))
    if candidate_oracle_depth_consistency is not None:
        candidate_oracle_depth_consistency = candidate_oracle_depth_consistency.reshape(-1)[:iou.size]
    candidate_oracle_consistency = _read_array('{}_candidate_oracle_consistency.txt'.format(base))
    if candidate_oracle_consistency is not None:
        candidate_oracle_consistency = candidate_oracle_consistency.reshape(-1)[:iou.size]
    candidate_oracle_motion_score = _read_array('{}_candidate_oracle_motion_score.txt'.format(base))
    if candidate_oracle_motion_score is not None:
        candidate_oracle_motion_score = candidate_oracle_motion_score.reshape(-1)[:iou.size]
    candidate_probation = _read_array('{}_candidate_probation.txt'.format(base))
    if candidate_probation is not None:
        candidate_probation = candidate_probation.reshape(-1)[:iou.size]
    candidate_recovery_trigger = _read_array('{}_candidate_recovery_trigger.txt'.format(base))
    if candidate_recovery_trigger is not None:
        candidate_recovery_trigger = candidate_recovery_trigger.reshape(-1)[:iou.size]
    candidate_stable_block = _read_array('{}_candidate_stable_block.txt'.format(base))
    if candidate_stable_block is not None:
        candidate_stable_block = candidate_stable_block.reshape(-1)[:iou.size]
    candidate_ordinary_trigger = _read_array('{}_candidate_ordinary_trigger.txt'.format(base))
    if candidate_ordinary_trigger is not None:
        candidate_ordinary_trigger = candidate_ordinary_trigger.reshape(-1)[:iou.size]
    language_candidate_rerank = _read_array('{}_language_candidate_rerank.txt'.format(base))
    if language_candidate_rerank is not None:
        language_candidate_rerank = language_candidate_rerank.reshape(-1)[:iou.size]
    language_candidate_gain = _read_array('{}_language_candidate_gain.txt'.format(base))
    if language_candidate_gain is not None:
        language_candidate_gain = language_candidate_gain.reshape(-1)[:iou.size]
    template_update = _read_array('{}_template_update.txt'.format(base))
    if template_update is not None:
        template_update = template_update.reshape(-1)[:iou.size]
    language_runtime_gate = _read_array('{}_language_runtime_gate.txt'.format(base))
    if language_runtime_gate is not None:
        language_runtime_gate = language_runtime_gate.reshape(-1)[:iou.size]
    valid_iou = iou[np.isfinite(iou)]
    out = {
        'mean_iou': float(valid_iou.mean()) if valid_iou.size else 0.0,
        'median_iou': float(np.median(valid_iou)) if valid_iou.size else 0.0,
        'frames': int(iou.size),
        'iou_lt_010': float((valid_iou < 0.10).mean()) if valid_iou.size else 1.0,
        'iou_lt_050': float((valid_iou < 0.50).mean()) if valid_iou.size else 1.0,
        'first_fail': _first_failure_frame(valid_iou, threshold=0.1, consecutive=10),
    }
    if scores is not None and scores.size:
        scores = scores[np.isfinite(scores)]
        if scores.size:
            out.update({
                'score_mean': float(scores.mean()),
                'score_p10': float(np.percentile(scores, 10)),
                'score_p50': float(np.percentile(scores, 50)),
                'score_p90': float(np.percentile(scores, 90)),
            })
            paired_len = min(scores.size, iou.size)
            if paired_len:
                score_high = scores[:paired_len] > 0.50
                iou_pair = iou[:paired_len]
                if score_high.any():
                    out.update({
                        'high_score_iou_lt_10': float((iou_pair[score_high] < 0.10).mean()),
                        'high_score_iou_lt_50': float((iou_pair[score_high] < 0.50).mean()),
                    })
    if consistency is not None and consistency.size:
        consistency = consistency[np.isfinite(consistency)]
        if consistency.size:
            out.update({
                'cons_mean': float(consistency.mean()),
                'cons_p10': float(np.percentile(consistency, 10)),
                'cons_p50': float(np.percentile(consistency, 50)),
            })
    if rgb_consistency is not None and rgb_consistency.size:
        rgb_consistency = rgb_consistency[np.isfinite(rgb_consistency)]
        if rgb_consistency.size:
            out.update({
                'rgb_cons_mean': float(rgb_consistency.mean()),
                'rgb_cons_p10': float(np.percentile(rgb_consistency, 10)),
            })
    if depth_consistency is not None and depth_consistency.size:
        depth_consistency = depth_consistency[np.isfinite(depth_consistency)]
        if depth_consistency.size:
            out.update({
                'depth_cons_mean': float(depth_consistency.mean()),
                'depth_cons_p10': float(np.percentile(depth_consistency, 10)),
            })
    if update_ratio is not None and update_ratio.size:
        update_ratio = update_ratio[np.isfinite(update_ratio)]
        if update_ratio.size:
            out.update({
                'upd_mean': float(update_ratio.mean()),
                'upd_p10': float(np.percentile(update_ratio, 10)),
                'upd_p50': float(np.percentile(update_ratio, 50)),
            })
    if score_ema is not None and score_ema.size:
        score_ema = score_ema[np.isfinite(score_ema)]
        if score_ema.size:
            out.update({
                'ema_mean': float(score_ema.mean()),
                'ema_p50': float(np.percentile(score_ema, 50)),
            })
    if lost_thr is not None and lost_thr.size:
        lost_thr = lost_thr[np.isfinite(lost_thr)]
        if lost_thr.size:
            out.update({
                'lost_thr_mean': float(lost_thr.mean()),
                'lost_thr_p50': float(np.percentile(lost_thr, 50)),
            })
    if damping_thr is not None and damping_thr.size:
        damping_thr = damping_thr[np.isfinite(damping_thr)]
        if damping_thr.size:
            out.update({
                'damp_thr_mean': float(damping_thr.mean()),
                'damp_thr_p50': float(np.percentile(damping_thr, 50)),
            })
    if response_entropy is not None and response_entropy.size:
        response_entropy = response_entropy[np.isfinite(response_entropy)]
        if response_entropy.size:
            out.update({
                'resp_ent_mean': float(response_entropy.mean()),
                'resp_ent_p90': float(np.percentile(response_entropy, 90)),
            })
    if response_peak_ratio is not None and response_peak_ratio.size:
        response_peak_ratio = response_peak_ratio[np.isfinite(response_peak_ratio)]
        if response_peak_ratio.size:
            out.update({
                'resp_peak_ratio_mean': float(response_peak_ratio.mean()),
                'resp_peak_ratio_p90': float(np.percentile(response_peak_ratio, 90)),
            })
    if response_center_distance is not None and response_center_distance.size:
        response_center_distance = response_center_distance[np.isfinite(response_center_distance)]
        if response_center_distance.size:
            out.update({
                'resp_center_mean': float(response_center_distance.mean()),
                'resp_center_p90': float(np.percentile(response_center_distance, 90)),
            })
    if response_guard is not None and response_guard.size:
        response_guard = response_guard[np.isfinite(response_guard)]
        if response_guard.size:
            out.update({'resp_guard_rate': float(response_guard.mean())})
    if center_follow is not None and center_follow.size:
        valid_center_follow = center_follow[np.isfinite(center_follow)]
        if valid_center_follow.size:
            out.update({'center_follow_rate': float(valid_center_follow.mean())})
            paired_len = min(center_follow.size, iou.size)
            if paired_len:
                cf = center_follow[:paired_len] > 0.5
                iou_pair = iou[:paired_len]
                if cf.any():
                    out.update({
                        'center_follow_iou_lt_10': float((iou_pair[cf] < 0.10).mean()),
                        'center_follow_iou_lt_50': float((iou_pair[cf] < 0.50).mean()),
                    })
    if depth_guard is not None and depth_guard.size:
        depth_guard = depth_guard[np.isfinite(depth_guard)]
        if depth_guard.size:
            out.update({'depth_guard_rate': float(depth_guard.mean())})
    if confident_mismatch_search is not None and confident_mismatch_search.size:
        valid_conf_mismatch = confident_mismatch_search[np.isfinite(confident_mismatch_search)]
        if valid_conf_mismatch.size:
            out.update({'conf_mismatch_search_rate': float(valid_conf_mismatch.mean())})
            paired_len = min(confident_mismatch_search.size, iou.size)
            if paired_len:
                triggered = confident_mismatch_search[:paired_len] > 0.5
                iou_pair = iou[:paired_len]
                if triggered.any():
                    out.update({
                        'conf_mismatch_iou_lt_10': float((iou_pair[triggered] < 0.10).mean()),
                        'conf_mismatch_iou_lt_50': float((iou_pair[triggered] < 0.50).mean()),
                    })
    if candidate_rerank is not None and candidate_rerank.size:
        candidate_rerank = candidate_rerank[np.isfinite(candidate_rerank)]
        if candidate_rerank.size:
            out.update({'cand_rerank_rate': float(candidate_rerank.mean())})
    if candidate_rank is not None and candidate_rank.size:
        candidate_rank = candidate_rank[np.isfinite(candidate_rank)]
        if candidate_rank.size:
            out.update({'cand_rank_mean': float(candidate_rank.mean())})
    if candidate_score_gain is not None and candidate_score_gain.size:
        candidate_score_gain = candidate_score_gain[np.isfinite(candidate_score_gain)]
        if candidate_score_gain.size:
            out.update({'cand_score_gain_mean': float(candidate_score_gain.mean())})
    if candidate_consistency_gain is not None and candidate_consistency_gain.size:
        candidate_consistency_gain = candidate_consistency_gain[np.isfinite(candidate_consistency_gain)]
        if candidate_consistency_gain.size:
            out.update({'cand_cons_gain_mean': float(candidate_consistency_gain.mean())})
    if candidate_selected_iou is not None and candidate_selected_iou.size:
        candidate_selected_iou = candidate_selected_iou[np.isfinite(candidate_selected_iou) & (candidate_selected_iou >= 0.0)]
        if candidate_selected_iou.size:
            out.update({'cand_selected_iou_mean': float(candidate_selected_iou.mean())})
    if candidate_oracle_iou is not None and candidate_oracle_iou.size:
        candidate_oracle_iou = candidate_oracle_iou[np.isfinite(candidate_oracle_iou) & (candidate_oracle_iou >= 0.0)]
        if candidate_oracle_iou.size:
            out.update({'cand_oracle_iou_mean': float(candidate_oracle_iou.mean())})
            if 'cand_selected_iou_mean' in out:
                out.update({'cand_oracle_gap_mean': float(candidate_oracle_iou.mean() - out['cand_selected_iou_mean'])})
    if candidate_oracle_rank is not None and candidate_oracle_rank.size:
        candidate_oracle_rank = candidate_oracle_rank[np.isfinite(candidate_oracle_rank) & (candidate_oracle_rank >= 0.0)]
        if candidate_oracle_rank.size:
            out.update({'cand_oracle_rank_mean': float(candidate_oracle_rank.mean())})
    if candidate_oracle_score_ratio is not None and candidate_oracle_score_ratio.size:
        candidate_oracle_score_ratio = candidate_oracle_score_ratio[
            np.isfinite(candidate_oracle_score_ratio) & (candidate_oracle_score_ratio >= 0.0)]
        if candidate_oracle_score_ratio.size:
            out.update({'cand_oracle_score_ratio_mean': float(candidate_oracle_score_ratio.mean())})
    if candidate_oracle_rgb_consistency is not None and candidate_oracle_rgb_consistency.size:
        candidate_oracle_rgb_consistency = candidate_oracle_rgb_consistency[
            np.isfinite(candidate_oracle_rgb_consistency) & (candidate_oracle_rgb_consistency >= 0.0)]
        if candidate_oracle_rgb_consistency.size:
            out.update({'cand_oracle_rgb_mean': float(candidate_oracle_rgb_consistency.mean())})
    if candidate_oracle_depth_consistency is not None and candidate_oracle_depth_consistency.size:
        candidate_oracle_depth_consistency = candidate_oracle_depth_consistency[
            np.isfinite(candidate_oracle_depth_consistency) & (candidate_oracle_depth_consistency >= 0.0)]
        if candidate_oracle_depth_consistency.size:
            out.update({'cand_oracle_depth_mean': float(candidate_oracle_depth_consistency.mean())})
    if candidate_oracle_consistency is not None and candidate_oracle_consistency.size:
        candidate_oracle_consistency = candidate_oracle_consistency[
            np.isfinite(candidate_oracle_consistency) & (candidate_oracle_consistency >= 0.0)]
        if candidate_oracle_consistency.size:
            out.update({'cand_oracle_cons_mean': float(candidate_oracle_consistency.mean())})
    if candidate_oracle_motion_score is not None and candidate_oracle_motion_score.size:
        candidate_oracle_motion_score = candidate_oracle_motion_score[
            np.isfinite(candidate_oracle_motion_score) & (candidate_oracle_motion_score >= 0.0)]
        if candidate_oracle_motion_score.size:
            out.update({'cand_oracle_motion_mean': float(candidate_oracle_motion_score.mean())})
    if candidate_probation is not None and candidate_probation.size:
        candidate_probation = candidate_probation[np.isfinite(candidate_probation)]
        if candidate_probation.size:
            out.update({'cand_probation_rate': float((candidate_probation > 0.0).mean())})
    if candidate_recovery_trigger is not None and candidate_recovery_trigger.size:
        candidate_recovery_trigger = candidate_recovery_trigger[np.isfinite(candidate_recovery_trigger)]
        if candidate_recovery_trigger.size:
            out.update({'cand_recovery_trigger_rate': float(candidate_recovery_trigger.mean())})
    if candidate_stable_block is not None and candidate_stable_block.size:
        candidate_stable_block = candidate_stable_block[np.isfinite(candidate_stable_block)]
        if candidate_stable_block.size:
            out.update({'cand_stable_block_rate': float(candidate_stable_block.mean())})
    if candidate_ordinary_trigger is not None and candidate_ordinary_trigger.size:
        candidate_ordinary_trigger = candidate_ordinary_trigger[np.isfinite(candidate_ordinary_trigger)]
        if candidate_ordinary_trigger.size:
            out.update({'cand_ordinary_trigger_rate': float(candidate_ordinary_trigger.mean())})
    if language_candidate_rerank is not None and language_candidate_rerank.size:
        language_candidate_rerank = language_candidate_rerank[np.isfinite(language_candidate_rerank)]
        if language_candidate_rerank.size:
            out.update({'lang_cand_rerank_rate': float(language_candidate_rerank.mean())})
    if language_candidate_gain is not None and language_candidate_gain.size:
        language_candidate_gain = language_candidate_gain[np.isfinite(language_candidate_gain)]
        if language_candidate_gain.size:
            out.update({
                'lang_cand_gain_mean': float(language_candidate_gain.mean()),
                'lang_cand_gain_p90': float(np.percentile(language_candidate_gain, 90)),
            })
    if template_update is not None and template_update.size:
        template_update = template_update[np.isfinite(template_update)]
        if template_update.size:
            out.update({'template_update_rate': float(template_update.mean())})
    if language_runtime_gate is not None and language_runtime_gate.size:
        language_runtime_gate = language_runtime_gate[np.isfinite(language_runtime_gate)]
        if language_runtime_gate.size:
            out.update({
                'lang_gate_mean': float(language_runtime_gate.mean()),
                'lang_gate_p10': float(np.percentile(language_runtime_gate, 10)),
            })
    return out


def _count_completed(dataset, tracker):
    completed = []
    missing = []
    timing = []
    for seq in dataset:
        base = _result_base(tracker, seq)
        box_path = '{}.txt'.format(base)
        if os.path.isfile(box_path):
            completed.append(seq.name)
            times = _read_times('{}_time.txt'.format(base))
            if times is not None:
                timing.append((seq.name, float(times.sum()), float(times.mean()), int(times.size)))
        else:
            missing.append(seq.name)
    return completed, missing, timing


def main():
    parser = argparse.ArgumentParser(description='Diagnose partial DepthTrack RGB-D-L evaluation results.')
    parser.add_argument('tracker_name', nargs='?', default='mplt_track')
    parser.add_argument('tracker_param', nargs='?', default='vitb_256_mplt_32x1_1e4_depthtrack_15ep_roberta')
    parser.add_argument('--epoch', default='ep0002_it05000')
    parser.add_argument('--dataset_name', default='depthtrack_test')
    parser.add_argument('--checkpoint_root', default='./output/depthtrack_roberta')
    parser.add_argument('--topk', type=int, default=8)
    args = parser.parse_args()

    if args.checkpoint_root:
        os.environ['MPLT_CHECKPOINT_ROOT'] = os.path.abspath(args.checkpoint_root)

    dataset = get_dataset(args.dataset_name)
    tracker = Tracker(args.tracker_name, args.tracker_param, args.dataset_name, args.epoch)
    completed, missing, timing = _count_completed(dataset, tracker)

    print('dataset: {}  total={}  completed={}  missing={}'.format(
        args.dataset_name, len(dataset), len(completed), len(missing)))
    print('results_dir: {}'.format(tracker.results_dir))
    if missing:
        print('next missing: {}'.format(', '.join(missing[:min(args.topk, len(missing))])))

    if completed:
        partial = evaluate_depthtrack_prre([tracker], dataset, skip_missing_seq=True)
        print(format_depthtrack_prre(partial))
        per_seq = partial[0].get('per_sequence', [])
        if per_seq:
            low = sorted(per_seq, key=lambda x: x.get('F-score', 0.0))[:args.topk]
            print('lowest F-score sequences:')
            for item in low:
                print('- {sequence:<28} Pr={Pr:6.2f} Re={Re:6.2f} F={F-score:6.2f}'.format(**item))
            seq_by_name = {seq.name: seq for seq in dataset}
            print('drift/score diagnostics for lowest sequences:')
            for item in low:
                seq = seq_by_name.get(item['sequence'])
                if seq is None:
                    continue
                stats = _sequence_drift_stats(tracker, seq)
                if not stats:
                    continue
                fail = stats['first_fail']
                fail_text = 'none' if fail is None else str(fail)
                score_text = ''
                if 'score_mean' in stats:
                    score_text = ' score(mean/p10/p50/p90)={:.3f}/{:.3f}/{:.3f}/{:.3f}'.format(
                        stats['score_mean'], stats['score_p10'], stats['score_p50'], stats['score_p90'])
                if 'high_score_iou_lt_10' in stats:
                    score_text += ' hiScoreDrift(<.1/<.5)={:.1f}%/{:.1f}%'.format(
                        100.0 * stats['high_score_iou_lt_10'],
                        100.0 * stats['high_score_iou_lt_50'])
                if 'cons_mean' in stats:
                    score_text += ' cons(mean/p10/p50)={:.3f}/{:.3f}/{:.3f}'.format(
                        stats['cons_mean'], stats['cons_p10'], stats['cons_p50'])
                if 'rgb_cons_mean' in stats:
                    score_text += ' rgbCons(mean/p10)={:.3f}/{:.3f}'.format(
                        stats['rgb_cons_mean'], stats['rgb_cons_p10'])
                if 'depth_cons_mean' in stats:
                    score_text += ' depCons(mean/p10)={:.3f}/{:.3f}'.format(
                        stats['depth_cons_mean'], stats['depth_cons_p10'])
                if 'upd_mean' in stats:
                    score_text += ' upd(mean/p10/p50)={:.3f}/{:.3f}/{:.3f}'.format(
                        stats['upd_mean'], stats['upd_p10'], stats['upd_p50'])
                if 'ema_mean' in stats:
                    score_text += ' ema(mean/p50)={:.3f}/{:.3f}'.format(
                        stats['ema_mean'], stats['ema_p50'])
                if 'lost_thr_mean' in stats:
                    score_text += ' lostThr(mean/p50)={:.3f}/{:.3f}'.format(
                        stats['lost_thr_mean'], stats['lost_thr_p50'])
                if 'damp_thr_mean' in stats:
                    score_text += ' dampThr(mean/p50)={:.3f}/{:.3f}'.format(
                        stats['damp_thr_mean'], stats['damp_thr_p50'])
                if 'resp_ent_mean' in stats:
                    score_text += ' respEnt(mean/p90)={:.3f}/{:.3f}'.format(
                        stats['resp_ent_mean'], stats['resp_ent_p90'])
                if 'resp_peak_ratio_mean' in stats:
                    score_text += ' peak2/1(mean/p90)={:.3f}/{:.3f}'.format(
                        stats['resp_peak_ratio_mean'], stats['resp_peak_ratio_p90'])
                if 'resp_center_mean' in stats:
                    score_text += ' peakDist(mean/p90)={:.3f}/{:.3f}'.format(
                        stats['resp_center_mean'], stats['resp_center_p90'])
                if 'resp_guard_rate' in stats:
                    score_text += ' respGuard={:.1f}%'.format(100.0 * stats['resp_guard_rate'])
                if 'center_follow_rate' in stats:
                    score_text += ' centerFollow={:.1f}%'.format(100.0 * stats['center_follow_rate'])
                    if 'center_follow_iou_lt_10' in stats:
                        score_text += ' cfDrift(<.1/<.5)={:.1f}%/{:.1f}%'.format(
                            100.0 * stats['center_follow_iou_lt_10'],
                            100.0 * stats['center_follow_iou_lt_50'])
                if 'depth_guard_rate' in stats:
                    score_text += ' depthGuard={:.1f}%'.format(100.0 * stats['depth_guard_rate'])
                if 'cand_rerank_rate' in stats:
                    score_text += ' candRerank={:.1f}% rankMean={:.2f} gain(score/cons)={:.3f}/{:.3f}'.format(
                        100.0 * stats['cand_rerank_rate'],
                        stats.get('cand_rank_mean', 0.0),
                        stats.get('cand_score_gain_mean', 0.0),
                        stats.get('cand_cons_gain_mean', 0.0))
                if 'template_update_rate' in stats:
                    score_text += ' tmplUpd={:.1f}%'.format(100.0 * stats['template_update_rate'])
                if 'lang_gate_mean' in stats:
                    score_text += ' langGate(mean/p10)={:.3f}/{:.3f}'.format(
                        stats['lang_gate_mean'], stats['lang_gate_p10'])
                if 'cand_oracle_iou_mean' in stats:
                    score_text += ' candIoU(sel/oracle/gap/rank/scoreR)={:.3f}/{:.3f}/{:.3f}/{:.2f}/{:.3f}'.format(
                        stats.get('cand_selected_iou_mean', -1.0),
                        stats['cand_oracle_iou_mean'],
                        stats.get('cand_oracle_gap_mean', 0.0),
                        stats.get('cand_oracle_rank_mean', -1.0),
                        stats.get('cand_oracle_score_ratio_mean', 0.0))
                    if 'cand_oracle_cons_mean' in stats:
                        score_text += ' oracleFeat(cons/rgb/dep/mot)={:.3f}/{:.3f}/{:.3f}/{:.3f}'.format(
                            stats.get('cand_oracle_cons_mean', 0.0),
                            stats.get('cand_oracle_rgb_mean', 0.0),
                            stats.get('cand_oracle_depth_mean', 0.0),
                            stats.get('cand_oracle_motion_mean', 0.0))
                if 'cand_recovery_trigger_rate' in stats:
                    score_text += ' candTrig(ord/rec/stable/prob)={:.1f}%/{:.1f}%/{:.1f}%/{:.1f}%'.format(
                        100.0 * stats.get('cand_ordinary_trigger_rate', 0.0),
                        100.0 * stats.get('cand_recovery_trigger_rate', 0.0),
                        100.0 * stats.get('cand_stable_block_rate', 0.0),
                        100.0 * stats.get('cand_probation_rate', 0.0))
                if 'lang_cand_rerank_rate' in stats:
                    score_text += ' langCand={:.1f}% gain(mean/p90)={:.3f}/{:.3f}'.format(
                        100.0 * stats['lang_cand_rerank_rate'],
                        stats.get('lang_cand_gain_mean', 0.0),
                        stats.get('lang_cand_gain_p90', 0.0))
                print('- {:<28} meanIoU={:5.3f} medIoU={:5.3f} IoU<.1={:5.1f}% IoU<.5={:5.1f}% firstFail={}{}'
                      .format(item['sequence'], stats['mean_iou'], stats['median_iou'],
                              100.0 * stats['iou_lt_010'], 100.0 * stats['iou_lt_050'],
                              fail_text, score_text))

    if timing:
        print('slowest completed sequences:')
        for name, total_time, mean_time, frames in sorted(timing, key=lambda x: x[1], reverse=True)[:args.topk]:
            fps = frames / total_time if total_time > 0 else 0.0
            print('- {:<28} total={:8.2f}s mean={:7.4f}s frames={:<5d} fps={:6.2f}'.format(
                name, total_time, mean_time, frames, fps))


if __name__ == '__main__':
    main()
