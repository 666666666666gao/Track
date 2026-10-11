"""Local-only epoch ledger from an already collected M124 hourly observation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


DATASETS = {'depthtrack_test': (50, 76373), 'cdtb': (80, 101956)}


def summarize(path):
    raw = path.read_bytes()
    observed = json.loads(raw)
    assert observed['root'] == '/root/autodl-tmp/m124_ABC_joint_epoch20_20261011'
    assert observed['GPU_queries'] == observed['NN_progress_queries'] == 1
    assert observed['NN_forward_calls'] == 0
    files = observed['files']
    history = json.loads(files['epoch_history.json']) if 'epoch_history.json' in files else []
    best = json.loads(files['best.json']) if 'best.json' in files else None
    epoch_names = sorted({name.split('/')[1] for name in files if name.startswith('epochs/')})
    rows = []
    for name in epoch_names:
        epoch = int(name.removeprefix('epoch-'))
        assert 1 <= epoch <= 20
        prefix = 'epochs/' + name + '/'
        log = files.get(prefix + 'fit/sequence_log.jsonl', '')
        records = [json.loads(line) for line in log.splitlines()]
        current = [r for r in records if r['epoch'] == epoch]
        last = records[-1] if records else None
        train_exit = int(files[prefix + 'train.exit']) if prefix + 'train.exit' in files else None
        training = json.loads(files[prefix + 'fit/result.json']) if prefix + 'fit/result.json' in files else None
        final_sha = None
        if training is not None:
            assert training['status'] == 'complete_M124_joint_Train152_epoch'
            assert training['epochs'] == epoch and training['training_sequences'] == 152
            assert training['sequence_runs'] == epoch * 152 and training['track_calls'] == epoch * 219802
            assert training['seed'] == 2027 and training['no_development_split']
            assert training['frozen_before_after_exact'] and training['final_state_roundtrip_exact']
            assert training['GT_reinitializations_after_first_frame'] == 0
            final_sha = training['final_sha256']
        row = dict(epoch=epoch, last_saved_record_epoch=last['epoch'] if last else None,
            saved_complete_sequences_this_epoch=current[-1]['sequence_index'] + 1 if current else (0 if records else None),
            saved_total_track_calls=last['total_track_calls'] if last else None,
            saved_total_optimizer_steps=last['total_optimizer_steps'] if last else None,
            training_exit=train_exit, training_complete_observed=training is not None and train_exit == 0,
            final_sha256=final_sha, paired_OPE_complete_observed=False,
            dual_decline=None, consecutive_dual_declines=None, selected_best_epoch_observed=best['epoch'] if best else None)
        for dataset, (count, frames) in DATASETS.items():
            for metric in ['precision', 'recall', 'f_score']:
                row[dataset + '_' + metric + '_percent'] = None
            key = prefix + 'evaluation/' + dataset + '/predictions/metrics.json'
            if key in files:
                result = json.loads(files[key])
                assert result['status'] == 'complete' and row['training_complete_observed']
                assert result['final_sha256'] == final_sha
                assert result['metrics']['sequences'] == count and result['metrics']['frames'] == frames
                for metric in ['precision', 'recall', 'f_score']:
                    row[dataset + '_' + metric + '_percent'] = result['metrics'][metric + '_percent']
        row['paired_OPE_complete_observed'] = all(row[d + '_f_score_percent'] is not None for d in DATASETS)
        matched = [r for r in history if r['epoch'] == epoch]
        assert len(matched) <= 1
        if matched:
            decision = matched[0]
            assert row['paired_OPE_complete_observed'] and decision['final_sha256'] == final_sha
            assert decision['F_percent'] == [row[d + '_f_score_percent'] for d in DATASETS]
            row.update(dual_decline=decision['dual_decline'], consecutive_dual_declines=decision['consecutive_dual_declines'])
        rows.append(row)
    return dict(status='local_M124_existing_observation_ledger', observed_at=observed['observed_at'],
        original_observation_sha256=hashlib.sha256(raw).hexdigest(), rows=rows, observed_best=best,
        completed_train_epochs_observed=sum(r['training_complete_observed'] for r in rows),
        completed_paired_OPE_epochs_observed=sum(r['paired_OPE_complete_observed'] for r in rows),
        controller_result=json.loads(files['result.json']) if 'result.json' in files else None,
        new_GPU_queries=0, new_NN_progress_queries=0, server_queries=0, NN_calls=0,
        goal_completion_assessed=False,
        scope='Parse existing observations only. No raw checkpoint, prediction, GT, or NN re-execution audit. Blank metrics remain uncomputed/unobserved; no partial-sequence PR estimate.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.observation)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'ledger.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (args.output / 'epoch_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result['rows'][0]))
        writer.writeheader()
        writer.writerows(result['rows'])
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
