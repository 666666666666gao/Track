from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil, tarfile

B = Path('/root/autodl-tmp')
R = B / 'sttrack_m81_multistart_20260908'
O = R / 'training_completed_publication'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
master = Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md')
old = master.read_bytes()
assert sha(master) == '9c5c17af1f0f0c8f0ea39fa8cf5fdadd7ddef4bc6b44dea5fda7427ef588d10a'
assert b'\n## 5.167 ' not in old and not O.exists()
audit = read(R / 'saved_training_audit.json')
assert audit['status'] == 'completed_saved_training_artifacts_verified'
assert audit['formal_training_completion_verified'] and audit['initializations'] == 426
assert audit['auditor_sha256'] == sha(B / 'audit_m80_m81_training_20260908.py')
assert (B / 'sttrack_training_audit_watch_20260908/m81_audit.exit').read_text().strip() == '0'
training = {}
for arm in ['category', 'empty']:
    folder = R / 'training' / arm
    assert (R / ('training_' + arm + '.exit')).read_text().strip() == '0'
    result = read(folder / 'result.json')
    assert result['status'] == 'one_full_causal_fit_pass_complete'
    assert result['total_track_calls'] == 186694 and result['optimizer_steps'] == 5798
    assert result['initializations'] == 426
    assert sha(folder / 'final.pth') == result['final_checkpoint_sha256'] == audit['arms'][arm]['final_checkpoint_sha256']
    assert audit['arms'][arm]['sequence_rows'] == 130
    training[arm] = result
assert sha(R / 'training_spec.json') == 'c62574e32c051f183f562a3417c5e2ed27ed28ef6e3101dbf746de8bdf40d06f'
assert sha(R / 'evaluation_spec.json') == 'dee5f2d118a3fca1f41724104f2c39c2aba2e48d9f5430751d0a3a44777627c8'
assert not (R / 'result.json').exists()
processes = []
for p in Path('/proc').iterdir():
    if p.name.isdigit() and (p / 'cmdline').exists():
        cmd = (p / 'cmdline').read_bytes().replace(b'\0', b' ')
        if b'evaluate.py' in cmd and (p / 'cwd').resolve() == R:
            processes.append(dict(pid=int(p.name), command=cmd.decode()))
assert processes
status = dict(observed_utc=datetime.now(timezone.utc).isoformat(), status='paired_training_verified_development_evaluation_running', processes=processes, free_disk_bytes=shutil.disk_usage(R).free)
rows = '\n'.join('| {} | {} | {} | {} |'.format(arm, audit['arms'][arm]['sequence_rows'], audit['arms'][arm]['sampled_trace_rows'], training[arm]['final_checkpoint_sha256']) for arm in ['category', 'empty'])
note = '''

## 5.167 M81多起点配对训练完成，保存产物核验通过

两组均完成固定seed2027、130条DepthTrack Train拟合序列、426次清单初始化、186694次跟踪调用和5798次优化，training_category.exit与training_empty.exit均为0。沿用§5.162冻结的M78 Raw损失、适配器、模板及推理规则，没有新增反向训练、跨初始化一致性损失或在线文字更新。固定边界初始化不由失败触发；额外296次初始化的前向计算开销不计入相同跟踪调用预算，不能称为严格等总计算量。

| 训练组 | 逐序列日志 | 抽样状态记录 | 最终head SHA256 |
| --- | ---: | ---: | --- |
''' + rows + '''

§5.163的CPU观察器已运行完成态独立核验：初始化位置、跟踪及优化次数、全部计划抽样位置、原生crop监督坐标、模板相对时钟与最终有限权重通过检查，m81_audit.exit=0。每组只训练289154个适配器参数。底座未变的证据限定为完成态训练器断言、记录的张量指纹和原checkpoint文件SHA；没有另存训练后完整底座，不将确定性核验写成独立模型审阅通过。

首帧及多起点开发评测已按原队列接续，实际进程命令与工作目录已核对。计划为四组首帧条件、五组多起点条件，每组33108次跟踪调用，共458条episode、298430个输出位置。全部九组轨迹封存后才计算GT指标与32项冻结条件，随后运行§5.166独立开发核验。当前没有M81开发完成态指标，也没有新的正式三数据集成绩，不依据训练完成晋升模型。

训练结果、保存产物核验、两组逐序列日志与实际评测进程状态发布于diagnostics/m81_multistart/training_completed/。完整抽样轨迹和最终权重保留远端，报告绑定其SHA256。
'''
O.mkdir()
F = O / 'files'
F.mkdir()
sources = [(R / 'saved_training_audit.json', 'saved_training_audit.json'), (Path(__file__), Path(__file__).name)]
for arm in ['category', 'empty']:
    sources.extend([(R / 'training' / arm / 'result.json', arm + '_training_result.json'), (R / 'training' / arm / 'sequence_log.jsonl', arm + '_sequence_log.jsonl')])
for src, name in sources:
    shutil.copyfile(src, F / name)
(F / 'verified_status.json').write_text(json.dumps(status, indent=2) + '\n')
(F / 'handoff_append.md').write_text(note)
manifest = [dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(F.iterdir())]
(F / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
archive = O / 'published.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    for p in sorted(F.iterdir()):
        tar.add(p, arcname=p.name)
master.write_bytes(old + note.encode('utf-8'))
record = dict(archive_sha256=sha(archive), master_sha256=sha(master), master_bytes=master.stat().st_size, files=len(manifest))
(O / 'publication_record.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
