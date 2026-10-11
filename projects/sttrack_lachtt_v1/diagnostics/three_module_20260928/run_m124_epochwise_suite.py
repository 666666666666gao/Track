"""Full Train152 each epoch, two complete OPEs, patience two, best-only VOT."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import sys

from bind_m122_official_initializations import sha
from prepare_m122_evaluation_suite import read, write
from run_m123_joint_full_suite import run
from m124_epochwise_selection import next_decision


def main():
    root = Path(sys.argv[1]); gate = read(root / 'gate.json')
    assert gate['status'] == 'reviewed_M124_epochwise_joint_full152_source_gate'
    assert gate['maximum_epochs'] == 20 and gate['early_stop_patience'] == 2
    for path, digest in gate['source_sha256'].items():
        assert sha(path) == digest, path
    for path, digest in gate['input_sha256'].items():
        assert sha(path) == digest, path
    code = Path(gate['code_root']); st = gate['python']; metric = gate['metric_python']
    resume = Path(gate['handoff_checkpoint']); previous = gate['initial_OPE_F_percent']
    streak = 0; best = None; history = []; stages = []
    epochs = root / 'epochs'; epochs.mkdir()
    for epoch in range(1, 21):
        folder = epochs / ('epoch-%02d' % epoch); folder.mkdir()
        fit = folder / 'fit'; evaluation = folder / 'evaluation'
        arguments = [str(code / 'train_m124_joint_epoch.py')]
        for name in ['spec', 'repository', 'checkpoint', 'clip_weight', 'bank', 'labels', 'warm_final', 'warm_result']:
            arguments += ['--' + name.replace('_', '-'), gate[name]]
        arguments += ['--output', str(fit), '--resume', str(resume), '--epoch', str(epoch)]
        stages.append(run(folder, gate, 'train', st, arguments))
        trained = read(fit / 'result.json')
        assert trained['epochs'] == epoch and trained['track_calls'] == epoch * 219802
        assert trained['sequence_runs'] == epoch * 152 and trained['no_development_split']
        stages.append(run(folder, gate, 'prepare_ope', st, [str(code / 'prepare_m124_epochwise_evaluation.py'),
            '--training', str(fit), '--base-evaluation', gate['base_evaluation'],
            '--source-gate', str(root / 'gate.json'), '--output', str(evaluation)]))
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(run, folder, gate, dataset + '_track', st,
                    [str(code / 'run_m124_epochwise_official.py'), '--mode', 'ope_track',
                     '--plan', str(evaluation / dataset / 'plan.json')], gpu)
                    for gpu, dataset in enumerate(['depthtrack_test', 'cdtb'])]
            stages.extend(job.result() for job in jobs)
        current = []; results = {}
        for dataset in ['depthtrack_test', 'cdtb']:
            stages.append(run(folder, gate, dataset + '_analyze', metric,
                          [str(code / 'run_m124_epochwise_official.py'), '--mode', 'ope_analyze',
                           '--plan', str(evaluation / dataset / 'plan.json')]))
            path = evaluation / dataset / 'predictions/metrics.json'; value = read(path)
            assert value['status'] == 'complete' and value['final_sha256'] == trained['final_sha256']
            assert value['metrics']['sequences'] == {'depthtrack_test': 50, 'cdtb': 80}[dataset]
            assert value['metrics']['frames'] == {'depthtrack_test': 76373, 'cdtb': 101956}[dataset]
            current.append(value['metrics']['f_score_percent'])
            results[dataset] = dict(path=str(path), sha256=sha(path), metrics=value['metrics'])
        decision = next_decision(previous, current, streak, None if best is None else best['score'])
        row = dict(epoch=epoch, final_sha256=trained['final_sha256'], metrics=results,
                   previous_F_percent=previous, F_percent=current, **decision)
        history.append(row); write(root / 'epoch_history.json', history)
        if decision['replace_best']:
            shutil.copyfile(fit / 'final.pt', root / 'best.pt')
            assert sha(root / 'best.pt') == trained['final_sha256']
            best = dict(epoch=epoch, score=decision['score'], F_percent=current,
                        final_sha256=trained['final_sha256'], fit=str(fit), evaluation=str(evaluation),
                        criterion='equal mean of DepthTrack Test and CDTB F; earliest epoch on ties',
                        external_metric_checkpoint_selection=True)
            write(root / 'best.json', best)
        streak = decision['consecutive_dual_declines']; previous = current
        resume = fit / 'latest.pt'
        write(root / 'current_training_state.json', dict(epoch_completed=epoch, best=best,
              early_stop=decision['stop'], maximum_epochs=20, next_resume_path=str(resume),
              consecutive_dual_declines=streak, completed_at=datetime.now(timezone.utc).isoformat()))
        if decision['stop']:
            break
    assert best is not None and sha(root / 'best.pt') == best['final_sha256']
    evaluation = Path(best['evaluation']); fit = Path(best['fit'])
    stages.append(run(root, gate, 'prepare_best_vot', st, [str(code / 'prepare_m124_epochwise_evaluation.py'),
        '--training', str(fit), '--base-evaluation', gate['base_evaluation'],
        '--source-gate', str(root / 'gate.json'), '--output', str(evaluation), '--vot-only']))
    vot = evaluation / 'vot'
    stages.append(run(root, gate, 'best_vot_track', st,
                      [str(code / 'run_m123_joint_vot_shards.py'), '--root', str(vot / 'run')]))
    (evaluation / 'vot.exit').write_text('0\n')
    manifest = read(vot / 'run/shard_manifest.json'); selection = read(evaluation / 'selection.json')
    model_name = selection['models'][0]['name']
    stages.append(run(root, gate, 'best_vot_official_analysis', metric,
                  ['-m', 'vot', 'analysis', '--workspace', str(vot / 'run/master'), '--format', 'json',
                   '--name', model_name + '_full127', manifest['tracker']]))
    stages.append(run(root, gate, 'best_vot_seal', metric,
                  [str(code / 'analyze_m124_epochwise_vot.py'), '--root', str(evaluation)]))
    stages.append(run(root, gate, 'collect_best', metric,
                  [str(code / 'collect_m124_epochwise_results.py'), '--root', str(evaluation)]))
    result = read(evaluation / 'all_results.json')
    assert result['status'] == 'three_M124_selected_epoch_full_evaluations_complete'
    write(root / 'result.json', dict(status='complete_M124_epochwise_training_best_and_full_VOT',
          epochs_completed=len(history), maximum_epochs=20, early_stopped=history[-1]['stop'],
          best=best, stages=stages, any_joint_pass=result['any_joint_pass'], automatic_retries=0,
          formal_nine_metrics=True, external_metric_checkpoint_selection=True,
          Test50_CDTB80_used_for_checkpoint_selection=True, external_optimizer_steps=0,
          finished_at=datetime.now(timezone.utc).isoformat()))
    (root / 'controller.exit').write_text('0\n')


if __name__ == '__main__':
    main()
