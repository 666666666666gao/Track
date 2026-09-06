"""Verify completed paired M55 training using its frozen recursive binding."""
import argparse
from collections import Counter
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import tarfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root, args.output
    assert not output.exists()
    spec = importlib.util.spec_from_file_location('m55_frozen_recursion', root / 'run_recursive.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    frozen, training, cases, results = runner.binding(root)
    preparation = json.loads((root / 'preparation.json').read_text())
    names = set(preparation['code_sha256']['control'])
    assert names == set(preparation['code_sha256']['clone'])
    differing_sources = [name for name in sorted(names)
        if preparation['code_sha256']['control'][name] != preparation['code_sha256']['clone'][name]]
    assert differing_sources == ['lib/models/sttrack/sttrack.py']
    control_source = (root / 'code/control/lib/models/sttrack/sttrack.py').read_text()
    clone_source = (root / 'code/clone/lib/models/sttrack/sttrack.py').read_text()
    assert clone_source.replace('temp_x_flip = temp_x.clone()', 'temp_x_flip =  temp_x').replace(
        'temp_r_flip = temp_r.clone()', 'temp_r_flip =  temp_r') == control_source
    summaries = {}
    for arm, result in results.items():
        folder = root / 'training' / arm
        steps = [json.loads(line) for line in (folder / 'steps.jsonl').open()]
        assert [row['optimizer_step'] for row in steps] == list(range(1, 3841))
        assert all(row['microbatches'] == 4 * row['optimizer_step'] for row in steps)
        assert all(row['epoch'] == (row['optimizer_step'] - 1) // 256 + 1 for row in steps)
        assert all(math.isfinite(row[key]) for row in steps
            for key in ['mean_summed_four_frame_loss', 'grad_norm_before_clip'])
        batches = [json.loads(line) for line in (folder / 'batches.jsonl').open()]
        assert len(batches) == 15360
        samples = [sample for row in batches for sample in row['samples']]
        assert len(samples) == 30720
        sample_keys = {(sample['epoch'], sample['index']) for sample in samples}
        assert sample_keys == {(epoch, index) for epoch in range(1, 16) for index in range(2048)}
        assert all(len(s['template_ids']) == 2 and len(s['search_ids']) == 4 for s in samples)
        coverage = Counter(sample['sequence'] for sample in samples)
        assert set(coverage) <= set(training['fit_sequences'])
        assert not set(coverage) & set(training['development_sequences'])
        epochs = json.loads((folder / 'epochs.json').read_text())
        assert epochs == result['epochs']
        assert [row['optimizer_steps'] for row in epochs] == list(range(256, 3841, 256))
        assert all(row['samples'] == 2048 for row in epochs)
        summaries[arm] = dict(finished_utc=result['finished_utc'],
            elapsed_seconds=result['elapsed_seconds'], epochs=len(epochs),
            optimizer_steps=len(steps), microbatches=len(batches), clips=len(samples),
            search_frames=sum(len(s['search_ids']) for s in samples),
            fitting_sequences_observed=len(coverage), fit_clip_counts=dict(sorted(coverage.items())),
            weight_path=result['weight_path'], weight_bytes=Path(result['weight_path']).stat().st_size,
            weight_sha256=result['weight_sha256'], data_stream_sha256=result['data_stream_sha256'],
            result_sha256=sha(folder / 'result.json'), steps_sha256=result['steps_sha256'],
            execution_binding_sha256=result['execution_binding_sha256'],
            initial_state_file_sha256=sha(folder / 'initial_state_sha256.json'),
            reported_changed_state_tensors=result['changed_state_tensors'],
            final_epoch_training_loss=epochs[-1]['mean_summed_four_frame_loss'],
            epoch_rows=epochs, loss_is_tracking_metric=False)
    assert summaries['control']['fit_clip_counts'] == summaries['clone']['fit_clip_counts']
    assert [x['batch_stream_sha256'] for x in results['control']['epochs']] == [
        x['batch_stream_sha256'] for x in results['clone']['epochs']]
    report = dict(status='complete_training_binding_verified',
        observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        exporter_sha256=sha(__file__), frozen_runner_sha256=sha(root / 'run_recursive.py'),
        recursive_spec_sha256=sha(root / 'recursive_spec.json'),
        training_spec_sha256=sha(root / 'training_spec.json'),
        trainer_sha256=sha(root / 'train.py'), preparation_sha256=sha(root / 'preparation.json'),
        code_files_per_arm=len(names), differing_model_sources=differing_sources,
        sole_model_delta='Two .clone() calls for independent TSG direction inputs',
        initial_state_and_complete_data_stream_equal=True,
        checkpoint_internal_metadata_checked_by_frozen_runner=True, arms=summaries,
        recursive_sequences=len(cases), recursive_frames=sum(x['frames'] for x in cases),
        gpu_execution_in_this_export=False, development_metrics_computed=False,
        public_evaluation=False, adopted_weight=False,
        independent_review='Previously reviewed frozen binding source; no new completed independent result review',
        scope='Training completion and artifact binding only. No recursive performance or final-base selection.')
    output.mkdir()
    (output / 'completion_report.json').write_text(json.dumps(report, indent=2) + '\n')
    shutil.copyfile(__file__, output / 'export_completion.py')
    for arm in ['control', 'clone']:
        destination = output / arm
        destination.mkdir()
        for name in ['result.json', 'execution_binding.json', 'initial_state_sha256.json', 'epochs.json', 'steps.jsonl']:
            shutil.copyfile(root / 'training' / arm / name, destination / name)
        for suffix in ['.exit', '_controller.exit']:
            name = 'train_' + arm + suffix
            shutil.copyfile(root / name, output / name)
    manifest = [dict(path=path.relative_to(output).as_posix(), bytes=path.stat().st_size, sha256=sha(path))
        for path in sorted(output.rglob('*')) if path.is_file()]
    (output / 'completion_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    archive = output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                stream.add(path, arcname=path.relative_to(output).as_posix())
    print(json.dumps(dict(status=report['status'], archive=str(archive), archive_bytes=archive.stat().st_size,
        archive_sha256=sha(archive), files=len(manifest)+1,
        report_sha256=sha(output/'completion_report.json'),
        arms={arm:{key:value for key,value in data.items() if key not in ['fit_clip_counts', 'epoch_rows']}
            for arm,data in summaries.items()})), flush=True)


if __name__ == '__main__':
    main()
