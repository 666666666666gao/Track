"""Publish only the complete M67 development result after the frozen artifact/gate audit."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

BASE = Path('/root/autodl-tmp')
ROOT = BASE / 'sttrack_m67_supervised_semantic_support_20260907'
OUT = ROOT / 'completed_development_publication'
QUEUE = BASE / 'sttrack_post_m67_diagnostic_queue_20260907'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())
assert sha(BASE / 'audit_m67_completed_20260907.py') == '1773887deb3be27aa11a4409aa77bd7581f32b8d891ecab2a03454d277e769bf'
assert sha(ROOT / 'training_spec.json') == '2027b1893343e5e1520c447bd82a5c45468542cecaab0f59b3ec72a345e48b2e'
assert sha(ROOT / 'recursive_spec.json') == 'd4dd7ca74b8f4b318f8e32f961bc21a10b95aa3802137dd293b49948d83411c5'
for name in ['controller.exit', 'training_control.exit', 'training_support.exit', 'control_recursive.exit',
    'support_recursive.exit', 'recursive_analysis.exit']:
    assert (ROOT / name).read_text().strip() == '0', name
audit = read(ROOT / 'completed_evidence_audit.json')
result = read(ROOT / 'recursive_result.json')
assert audit['status'] == 'completed_M67_artifacts_and_development_audited'
assert audit['auditor_sha256'] == sha(BASE / 'audit_m67_completed_20260907.py')
assert audit['result_sha256'] == sha(ROOT / 'recursive_result.json')
assert audit['recomputed_frozen_gates'] == result['gates']
assert audit['paired_development_gate_pass'] == result['primary_pass'] == all(result['gates'].values())
assert audit['content_counterfactuals_allowed'] == result['primary_pass']
assert not audit['public_evaluation_allowed'] and not audit['formal_three_dataset_metrics_exist']
assert set(result['per_sequence']['control']) == set(result['per_sequence']['support']) == set(result['per_sequence']['native'])
assert len(result['per_sequence']['support']) == 22 and len(result['gates']) == 11
for arm in ['control', 'support']:
    assert sha(ROOT / 'training' / arm / 'final.pth') == audit['training'][arm]['final_checkpoint_sha256']
    assert sha(ROOT / (arm + '_recursive_receipt.json')) == result['receipts'][arm]
    receipt = read(ROOT / (arm + '_recursive_receipt.json'))
    assert receipt['total_frames'] == 33130 and len(receipt['sequences']) == 22
    for sequence in receipt['sequences']:
        assert sha(ROOT / 'recursive' / arm / (sequence['sequence'] + '.json')) == sequence['sha256']

aggregates = audit['recomputed_aggregates']
differences = {}
for reference in ['native', 'control']:
    differences[reference] = dict(
        mean_iou_percentage_points=100 * (aggregates['support']['mean_iou'] - aggregates[reference]['mean_iou']),
        sequence_equal_iou_percentage_points=100 * (aggregates['support']['macro_sequence_mean_iou'] - aggregates[reference]['macro_sequence_mean_iou']),
        low_overlap_frames=aggregates['support']['low_iou_frames'] - aggregates[reference]['low_iou_frames'],
        H10_episodes=aggregates['support']['failure_episodes'] - aggregates[reference]['failure_episodes'])
identity = read(QUEUE / 'running_identity.json')['identity']
proc = Path('/proc') / str(identity['pid'])
fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
assert fields[0] != 'Z' and fields[19] == identity['start_ticks']
assert str((proc / 'cwd').resolve()) == identity['cwd']
assert [s.decode() for s in (proc / 'cmdline').read_bytes().split(b'\0') if s] == identity['argv']
summary = dict(status='completed_M67_development_report_bound_to_frozen_audit', observed_utc=datetime.now(timezone.utc).isoformat(),
    source_sha256=sha(__file__), result_sha256=sha(ROOT / 'recursive_result.json'), audit_sha256=sha(ROOT / 'completed_evidence_audit.json'),
    aggregates=aggregates, support_minus_reference=differences, gates=result['gates'], passed_gates=sum(result['gates'].values()),
    gate_count=11, primary_pass=result['primary_pass'], broken_success_sequences=audit['broken_success_sequences'],
    sustained_support_H10_with_reference_correct_every_frame=audit['sustained_support_H10_with_reference_correct_every_frame'],
    live_queue_identity=identity, queue_latest=read(QUEUE / 'latest.json'), disk_free_bytes=shutil.disk_usage(BASE).free,
    new_tracking_calls=0, new_optimizer_steps=0, new_formal_metrics=False, public_evaluation_allowed=False,
    independent_model_review_pass=False)

gate_labels = {
    'prior_control_success_protection': '保住历史M65 Control零H10序列',
    'mean_vs_native': '按帧均值达到原生+0.002',
    'mean_vs_control': '按帧均值达到同预算Control+0.001',
    'macro_vs_native': '序列等权均值不低于原生',
    'macro_vs_control': '序列等权均值不低于Control',
    'low_frames_vs_native': '低重叠帧不多于原生',
    'low_frames_vs_control': '低重叠帧不多于Control',
    'H10_vs_native': 'H10段不多于原生',
    'H10_vs_control': 'H10段不多于Control',
    'native_success_protection': '保住原生零H10序列',
    'control_success_protection': '保住本轮Control零H10序列',
}
assert set(gate_labels) == set(result['gates'])
lines = ['# M67配对完整递归结果与原冻结门审计', '',
    '本报告只包含DepthTrack Train反复使用的开发22。两组最终权重均完成130条拟合序列、186,694次训练跟踪调用和5,798次优化。随后各自完成33,108次开发跟踪调用，含初始化共33,130帧、28,897个有效评价帧。不是DepthTrack Test、CDTB或正式VOT结果。', '',
    '## 同一开发22完整结果', '',
    '| 配置 | 按有效帧平均IoU | 序列等权平均IoU | IoU≤0.1帧 | H10段 |',
    '| --- | ---: | ---: | ---: | ---: |']
for arm, label in [('native', '原生STTrack'), ('control', 'M67 Control'), ('support', 'M67 Support')]:
    row = aggregates[arm]
    lines.append('| {} | {:.6f} | {:.6f} | {} | {} |'.format(label, row['mean_iou'], row['macro_sequence_mean_iou'], row['low_iou_frames'], row['failure_episodes']))
lines += ['', '## 原冻结门', '',
    '11项原条件通过{}项。门槛与保护集合未改变。'.format(summary['passed_gates']), '',
    '| 条件 | 结果 |', '| --- | --- |']
for key, label in gate_labels.items(): lines.append('| {} | {} |'.format(label, '通过' if result['gates'][key] else '未通过'))
lines += ['', '破坏零H10保护的序列：', '']
for family, names in audit['broken_success_sequences'].items():
    lines.append('- {}：{}'.format(family, '、'.join(names) if names else '无'))
lines += ['', '## 逐序列结果', '',
    '| 序列 | 原生IoU / H10 | Control IoU / H10 | Support IoU / H10 |',
    '| --- | ---: | ---: | ---: |']
for seq in result['per_sequence']['support']:
    values = [result['per_sequence'][arm][seq] for arm in ['native', 'control', 'support']]
    lines.append('| {} | {} |'.format(seq, ' | '.join('{:.6f} / {}'.format(v['mean_iou'], v['failure_episodes']) for v in values)))
lines += ['', '## 持续损害核查', '',
    '以下为Support出现完整H10区间，且同一区间参考轨迹每一帧IoU均≥0.5的情况。帧号为零基，结束帧不包含在区间内；不能把这些区间仅解释成长失败被分碎。', '',
    '| 序列 | 参考 | 起点 | 结束，不含 | 帧数 |', '| --- | --- | ---: | ---: | ---: |']
harms = audit['sustained_support_H10_with_reference_correct_every_frame']
for harm in harms:
    lines.append('| {} | {} | {} | {} | {} |'.format(harm['sequence'], harm['reference'], harm['start'], harm['end_exclusive'], harm['frames']))
if not harms: lines.append('| 无符合上述严格条件的区间 | — | — | — | — |')
lines += ['', '## 当前结论边界及接续', '',
    ('完整递归通过原门，可以继续既定同权重类别、空文本与替换类别内容对照。此时仍没有直接进入低22或正式全量评测的资格。'
     if result['primary_pass'] else
     '完整递归未通过原门。本轮版本停止晋升，原M67内容晋升阶段应按既定队列跳过；不放宽阈值，不改选早期权重。'), '',
    'M69是针对M65最终权重的独立内容诊断；M68检查M67两组固定权重的分数与支持量；M70/M71检查M65同状态跨区域候选容量及等细窗口数量的内容影响。这些诊断不改变各自原失败结论，也不会自动赋予公开评测资格。', '',
    '支持损失以目标框位置构造监督，没有直接监督caption真伪。即使位置支持或聚合跟踪改善，也不能据此声称模型学会拒绝错误文字；反之，本轮未晋升也不能直接否定所有文本交互。具体机制需要后续固定权重诊断。', '',
    '本报告由实际完整结果及原冻结审计器生成。原生三数据集正式指标不变；没有新正式指标、独立审阅通过或多seed泛化结论。', '',
    '结果SHA：`{}`。审计SHA：`{}`。'.format(summary['result_sha256'], summary['audit_sha256']), '']
OUT.mkdir()
for name in ['recursive_result.json', 'completed_evidence_audit.json', 'control_recursive_receipt.json', 'support_recursive_receipt.json']:
    shutil.copyfile(ROOT / name, OUT / name)
shutil.copyfile(__file__, OUT / Path(__file__).name)
(OUT / 'report_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
(OUT / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
(OUT / 'evidence_manifest.json').write_text(json.dumps([dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p))
    for p in sorted(OUT.iterdir())], indent=2) + '\n')
archive = ROOT / 'completed_development_evidence.tar.gz'
with tarfile.open(str(archive), 'w:gz') as tar:
    for path in sorted(OUT.iterdir()): tar.add(str(path), arcname=path.name)
print(json.dumps(dict(archive_sha256=sha(archive), archive_bytes=archive.stat().st_size, summary=summary), indent=2))
