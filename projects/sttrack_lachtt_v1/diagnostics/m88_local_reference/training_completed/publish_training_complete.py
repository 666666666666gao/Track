import hashlib
import json
import shutil
from pathlib import Path

work = Path(__file__).resolve().parent
repo = Path('C:/Users/gb/.codex_track_publish_m29_20260902')
master = repo / 'docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md'
desktop = Path('C:/Users/gb/Desktop/document/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md')
public = repo / 'projects/sttrack_lachtt_v1/diagnostics/m88_local_reference/training_completed'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert sha(master) == sha(desktop) == '011122090f8b148595e3aeeacd6ce4c51071c2cd59316f8fca2f55f711ce295f'
result = json.loads((work / 'result.json').read_text(encoding='utf-8'))
assert sha(work / 'result.json') == 'cea9691518899b2f480806569e357a120ec2b6f03aca2ae031f655591951dbc4'
assert sha(work / 'final.pth') == result['final_checkpoint_sha256']
assert sha(work / 'sequence_log.jsonl') == result['sequence_log_sha256']
rows = [json.loads(line) for line in (work / 'sequence_log.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(rows) == len({row['sequence'] for row in rows}) == result['sequences'] == 130
assert rows[-1]['total_track_calls'] == result['total_track_calls'] == 186694
assert rows[-1]['total_optimizer_steps'] == result['optimizer_steps'] == 5798
assert result['status'] == 'one_full_causal_fit_pass_complete'
assert result['evaluation_metrics_computed'] is False
assert result['training_spec_sha256'] == '0d82b358e88525b17474e50215afd84f39898f40c7ff1b0df13b44bc92da4428'
receipt = {
    'scope': 'training completion and saved artifact checks; not performance audit',
    'remote_observed_utc': '2026-09-21T04:33:11.493244+00:00',
    'training_exit': 0,
    'controller_pid': 48174,
    'category_evaluation_pid': 62655,
    'empty_evaluation_pid': 62656,
    'collector_pid': 49000,
    'evaluation_processes_observed_live': True,
    'training_sequences': len(rows),
    'track_calls': result['total_track_calls'],
    'optimizer_steps': result['optimizer_steps'],
    'result_sha256': sha(work / 'result.json'),
    'final_checkpoint_sha256': sha(work / 'final.pth'),
    'sequence_log_sha256': sha(work / 'sequence_log.jsonl'),
    'training_report_base_unchanged': result['base_parameters_and_buffers_unchanged'],
    'independent_checkpoint_tensor_audit_completed': False,
    'complete_development_metrics_available': False,
}
(work / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
public.mkdir(parents=True, exist_ok=True)
for name in ['result.json', 'sequence_log.jsonl', 'verification.json', 'publish_training_complete.py']:
    shutil.copyfile(work / name, public / name)
section = '''

### 5.189 M88完整训练结束，最终权重核验与开发评测接续（2026-09-21）

M88训练完成记录时间为`2026-09-21T04:28:34.940824+00:00`（北京时间12:28:34），训练退出码0。沿用§5.186的架构、旧文字bank、损失与单seed2027计划，没有增加训练臂或挑选中途权重。

| 核验项目 | 完成态证据 |
|---|---|
| 拟合序列 | 130条，序列日志名称不重复 |
| 跟踪调用 | 186694，日志末行与训练result一致 |
| 优化次数 | 5798，日志末行与训练result一致 |
| 完整训练耗时 | 15327.892779秒，约4小时15分28秒 |
| 可训练参数 | 289154，冻结底座参数和buffer未变由训练result报告 |
| 最终权重 | 完整遍历保存；远端与本地下载SHA256一致 |
| 开发成绩 | 尚无三组完整完成态结果，不填写IoU、P/R/F或VOT指标 |

`2026-09-21T04:33:11.493244+00:00`实测原控制器PID48174、Category开发PID62655、Empty开发PID62656和收集器PID49000均存活。原队列在Category/Empty成功后运行Swapped，全部三组封存后才分析14项预定条件；本轮没有重启训练或改变运行策略。这里的训练结果及文件核验不替代后续独立tensor/指标/完整轨迹审计，也不证明Empty全程一致或语义性能通过。

最终权重SHA256 `a6705a512444f0d715e50d2655edb1611dcbb7801be83329ce7bff2d13611c78`；训练result `cea9691518899b2f480806569e357a120ec2b6f03aca2ae031f655591951dbc4`；130条序列日志 `79cd01ad017d2edf56064642b01bd4ed45cf0005fc8bb69abb691f1892a5bfac`。训练spec仍为`0d82b358e88525b17474e50215afd84f39898f40c7ff1b0df13b44bc92da4428`。公开训练result、序列日志和核验记录见`projects/sttrack_lachtt_v1/diagnostics/m88_local_reference/training_completed/`，最终权重仅保存在远端和D盘临时目录，不上传GitHub。

下一步完成三组开发、既定收集器封存与本地逐项哈希核验，再使用§5.187已验证的分析脚本及独立审查复算指标、14项条件、内容干预和持续损害；依据完整结果决定后续优化，不自动进入正式外部评测。建议续接技能为monitor-experiment、analyze-results和experiment-audit。两套Qwen继续保留，不做多seed；同一模型三数据集目标仍未完成。
'''
with master.open('a', encoding='utf-8', newline='\n') as stream:
    stream.write(section)
shutil.copyfile(master, desktop)
assert sha(master) == sha(desktop)
print(json.dumps({'master_sha256': sha(master), 'public_files': 4, 'training_verification': receipt}))
