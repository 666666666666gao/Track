"""Derive sequence-level diagnostics from the sealed completed CPU-only summary."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def analyze(source):
    data = json.loads(source.read_text(encoding='utf-8'))
    assert data['status'] == 'actual_complete_CDTB_saved_trajectory_recall_diagnosis'
    assert data['sequences'] == 80 and data['frames'] == 101956
    assert data['NN_calls'] == data['GPU_queries'] == data['training_progress_queries'] == 0
    current = [r for r in data['per_sequence'] if r['model'] == 'CurrentC']
    previous = {r['sequence']: r for r in data['per_sequence'] if r['model'] == 'P1'}
    assert len(current) == len(previous) == 80
    assert {r['sequence'] for r in current} == set(previous)
    assert sum(r['frames'] for r in current) == 101956
    total = data['aggregate']['CurrentC']
    for key, field in [('P', 'P_percent'), ('R', 'R_percent'), ('all_saved_boxes_R', 'all_saved_boxes_R_percent')]:
        assert math.isclose(mean(r[field] for r in current), total[key], abs_tol=1e-10)
    assert math.isclose(2 * total['P'] * total['R'] / (total['P'] + total['R']), total['F'], abs_tol=1e-10)
    hashes = data['inputs_sha256']
    current_prefix = '/root/autodl-tmp/m122_current_quality_C_full152_20261010/evaluation/cdtb/predictions/'
    previous_prefix = '/root/autodl-tmp/sttrack_m122_official_evaluation_20261007/precision1/cdtb/predictions/'
    rows = []
    for row in current:
        name = row['sequence']
        prior = previous[name]
        same_box = hashes[current_prefix + name + '.txt'] == hashes[previous_prefix + name + '.txt']
        same_score = hashes[current_prefix + name + '_all_scores.txt'] == hashes[previous_prefix + name + '_all_scores.txt']
        rows.append(dict(sequence=name, frames=row['frames'], valid_GT_frames=row['valid_GT_frames'],
            CurrentC_P_percent=row['P_percent'], CurrentC_R_percent=row['R_percent'],
            CurrentC_all_boxes_R_percent=row['all_saved_boxes_R_percent'],
            P1_R_percent=prior['R_percent'], P1_all_boxes_R_percent=prior['all_saved_boxes_R_percent'],
            delta_R_pp=row['R_percent'] - prior['R_percent'],
            delta_all_boxes_R_pp=row['all_saved_boxes_R_percent'] - prior['all_saved_boxes_R_percent'],
            confidence_R_loss_pp=row['confidence_R_loss_pp'],
            target75_6_signed_gap_contribution_pp=(75.6 - row['R_percent']) / 80,
            saved_bbox_files_exact=same_box, saved_score_files_exact=same_score,
            serious_low_IoU_frames=row['serious_low_IoU_frames'],
            serious_low_IoU_crop_inside_frames=row['serious_low_IoU_crop_inside_frames'],
            serious_low_IoU_crop_outside_frames=row['serious_low_IoU_crop_outside_frames'],
            long_low_IoU_segments=len(row['long_low_IoU_segments']),
            long_low_IoU_segment_frames=sum(s['length'] for s in row['long_low_IoU_segments']),
            valid_GT_reappearance_events=row['valid_GT_reappearance_count'],
            valid_GT_reappearance_center_outside_events=row['reappearance_center_outside_count']))
    rows.sort(key=lambda r: r['CurrentC_R_percent'])
    segments = [s for r in current for s in r['long_low_IoU_segments']]
    assert len(segments) == total['long_low_IoU_segment_count'] == 98
    long_frames = sum(s['length'] for s in segments)
    assert 0 <= long_frames <= total['serious_low_IoU_frames']
    changed_all_R = [r for r in rows if abs(r['delta_all_boxes_R_pp']) > 1e-12]
    summary = dict(status='completed_local_saved_CDTB_summary_analysis', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        formal_metrics=data['aggregate'], sequences=80, frames=101956,
        identical_saved_bbox_files=sum(r['saved_bbox_files_exact'] for r in rows),
        identical_saved_score_files=sum(r['saved_score_files_exact'] for r in rows),
        changed_saved_bbox_sequences=[r['sequence'] for r in rows if not r['saved_bbox_files_exact']],
        changed_saved_score_sequences=[r['sequence'] for r in rows if not r['saved_score_files_exact']],
        changed_all_boxes_R_sequences=[dict(sequence=r['sequence'], delta_all_boxes_R_pp=r['delta_all_boxes_R_pp']) for r in changed_all_R],
        macro_all_boxes_R_delta_pp=mean(r['delta_all_boxes_R_pp'] for r in rows),
        long_low_IoU_segment_count=len(segments), long_low_IoU_segment_frames=long_frames,
        long_segments_share_of_serious_low_frames_percent=long_frames / total['serious_low_IoU_frames'] * 100,
        segment_start_center_inside_count=sum(s['start_center_in_crop'] for s in segments),
        segment_start_center_outside_count=sum(not s['start_center_in_crop'] for s in segments),
        started_inside_then_any_center_outside_count=sum(s['start_center_in_crop'] and s['center_outside_frames'] > 0 for s in segments),
        all_segment_centers_inside_count=sum(s['center_outside_frames'] == 0 for s in segments),
        valid_GT_reappearance_events=sum(r['valid_GT_reappearance_count'] for r in current),
        valid_GT_reappearance_center_outside_events=sum(r['reappearance_center_outside_count'] for r in current),
        bottom5_target75_6_signed_gap_contribution_pp=sum(r['target75_6_signed_gap_contribution_pp'] for r in rows[:5]),
        formal_R_gap_to_target_pp=75.6 - total['R'],
        NN_calls=0, GPU_queries=0, server_queries=0, training_progress_queries=0,
        scope='Existing completed CurrentC versus P1 saved-file hashes and summary only; no new model performance, checkpoint selection, trajectory replay or raw-NN verification.',
        limits='GT validity transitions do not prove physical absence. Segment onset is not necessarily the earliest causal divergence. File identity is saved six-decimal output identity, not hidden-state identity. Equal recall does not imply equal boxes. Signed per-sequence target gaps are arithmetic decomposition, not causal recovery or sequence-specific targets.')
    assert math.isclose(sum(r['target75_6_signed_gap_contribution_pp'] for r in rows), summary['formal_R_gap_to_target_pp'], abs_tol=1e-10)
    assert math.isclose(summary['macro_all_boxes_R_delta_pp'], data['aggregate']['CurrentC']['all_saved_boxes_R'] - data['aggregate']['P1']['all_saved_boxes_R'], abs_tol=1e-10)
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary, rows = analyze(args.input)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (args.output / 'all80_sequence_diagnostics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = '\n'.join('| {} | {:.3f} | {:.3f} | {} | {} |'.format(r['sequence'], r['CurrentC_R_percent'],
        r['CurrentC_all_boxes_R_percent'], r['serious_low_IoU_frames'], r['serious_low_IoU_crop_outside_frames']) for r in rows[:10])
    report = '''# CDTB已保存结果的召回与长失锁分解

本报告只读取此前完成的CurrentC/P1 CPU诊断JSON（80条、101,956帧及400项输入SHA），未读取服务器进度、未调用神经网络、未改变训练。完整80条原始汇总值及派生量见all80_sequence_diagnostics.csv；原始汇总JSON SHA为`{sha}`。

| CurrentC召回最低的10条 | 正式R（%） | 固定全框R（%） | 有效GT严重低重叠帧 | 其中中心在crop外 |
|---|---:|---:|---:|---:|
{table}

1. CurrentC与P1的80条中，78条保存框文件与分数文件SHA均完全相同；只有mug_gs、two_mugs两条文件改变。固定全框R只有mug_gs改变，增加8.42047654820297个百分点，贡献宏平均+0.10525595685254个百分点。two_mugs文件不同但全框R相同；不能将相同R当成框相同，也不能由保存文件相同推断所有内部状态相同。
2. 10,509个严重低重叠帧中，10,203帧属于至少10帧的连续低重叠段，占{long_share:.4f}%。主要损失集中于长失锁，而不是仅几个瞬时偏框。98段中45段起点目标中心仍在crop内，53段在外；45段中24段之后曾出crop，21段全程中心仍在内。该段起点不是必然的最早因果分叉；长段统计与逐帧crop外比例不能互相替代。
3. 全80条有159个GT从无效变有效的事件，其中78个事件的目标中心在重建crop外。它说明部分重新可定位时刻缺少局部观察；GT状态转换不单独证明真实物理遮挡或出画。
4. 若仅将R最低的5条提高到75.6%，其他75条保持原数值，宏平均R算术增加3.44748346295610个百分点；总体仍低于目标。这只是集中性敏感分析，不是恢复实验，不将75.6规定为逐序列目标，不据此写测试序列专用规则。

CurrentC完整正式P/R/F仍为74.59362462526585/69.88623866187693/72.1632445464765%；P1为74.47906262771164/69.7965033243733/72.06179519015306%。两者在各自全数据集F最优阈值报告正式PR；全框R不依赖报告阈值。CurrentC增加C后有局部收益，但本次保存轨迹证据显示其影响范围有限，不能声称已普遍解决恢复。

对后续的含义：先完成当前M124全Train152联合逐轮训练，再比较是否改善这些通用失败类型；候选存在却选错与候选不存在应分别检查。准确类别不保证视觉位置和框范围；现有C主要控制模板，不会自动增加新观察。后续机制规则和监督只从DepthTrack Train建立，公开CDTB案例仅用于解释。

这是本地只读派生统计，不是新训练指标、完整因果消融或外部独立重执行。
'''.format(sha=summary['source_sha256'], table=table, long_share=summary['long_segments_share_of_serious_low_frames_percent'])
    (args.output / '召回分解说明.md').write_text(report, encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
