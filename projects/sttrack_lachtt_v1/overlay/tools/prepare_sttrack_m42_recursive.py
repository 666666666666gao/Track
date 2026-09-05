"""Freeze the recursive comparison before reading M42 trained outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
    assert not (root/'training_result.json').exists()
    assert not (root/'recursive_spec.json').exists()
    spec=json.loads((root/'spec.json').read_text());repo=Path(spec['repository'])
    smoke=json.loads((root/'recursive_smoke_receipt.json').read_text())
    assert smoke['status']=='complete' and len(smoke['sequences'])==2
    assert smoke['model_source_sha256']==sha(repo/'lib/test/tracker/sttrack_local_spatial.py')
    assert smoke['runner_sha256']==sha(repo/'tools/run_sttrack_m42_recursive.py')
    baseline=Path('/root/autodl-tmp/sttrack_innovation_v1/risk_recovery_full152_v1')
    sources=['lib/test/tracker/sttrack_local_spatial.py','tools/run_sttrack_m42_recursive.py',
             'tools/analyze_sttrack_m42_recursive.py','tools/gate_sttrack_m42_recursive.py','tools/prepare_sttrack_m42_recursive.py']
    result=dict(schema='sttrack_m42_recursive_development_v1',created_unix=time.time(),created_before_training_results=True,
        static_spec_sha256=sha(root/'spec.json'),recursive_smoke_sha256=sha(root/'recursive_smoke_receipt.json'),
        source_sha256={name:sha(repo/name) for name in sources},
        baseline_shard_sha256={f'shard{i}.json':sha(baseline/f'shard{i}.json') for i in [0,1]},
        sequences='all22 existing Train development fold5 sequences, full available image lengths, one normal initialization',
        inference='Each arm owns its predicted crop, query, default template images and native reference bank; no subsequent GT input.',
        template_update='Existing interval50/threshold.75; when reranked, confidence belongs to the actually selected peak. Choice0/NONE exactly preserves default.',
        metrics='Continuous xywh IoU. Exclude initialization and invalid GT. Invalid-GT gaps break failure spans; persistent failure requires10 valid consecutive frames with IoU<=.1.',
        gates=dict(mean_iou='spatial > pooled and spatial > default',fewer_low_iou_frames='spatial < default',
            no_episode_increase='spatial <= default',sequence_coverage='positive mean IoU gain on at least3 sequences',
            successful_sequence_protection='No default-zero-episode sequence acquires a persistent failure'),
        authorization='Only run after static information gate passes. No automatic public evaluation. Passing recursion permits freezing a low22 candidate evaluation.',
        checkpoint_identity='Official STTrack base SHA plus newly trained association-final SHA and frozen runtime source/config; no mixed historical model metrics.',
        poll_seconds=240)
    (root/'recursive_spec.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='frozen_before_training_outcomes',recursive_spec_sha256=sha(root/'recursive_spec.json'))))


if __name__=='__main__':
    main()
