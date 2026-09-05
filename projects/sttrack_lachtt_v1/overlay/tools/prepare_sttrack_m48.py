"""Freeze the single native-continuity admission comparison without training."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    root = p.parse_args().root
    assert not root.exists()
    source = Path('/root/autodl-tmp/sttrack_m44_candidate_set_v1_20260905')
    control = Path('/root/autodl-tmp/sttrack_m45_default_priority_v1_20260905')
    audit = Path('/root/autodl-tmp/sttrack_m48_native_continuity_audit_v1_20260905/native_continuity_audit.json')
    assert sha(audit) == '969a982bf7633b31d25569b2cdd4bcf6e957b45934705fceb808085b3fa24950'
    assert sha(control / 'geometry_final.pth') == '853a25fbc3c9ef12ab54442c30b27bab75f0b1fadcc9bcc82cbec6e8700ed59c'
    assert sha(control / 'recursive_result.json') == 'a3dd821916ce08a8964bb2df85b9da277f9262d41a1e42c32f0fdde85eed3317'
    spec = json.loads((source / 'spec.json').read_text())
    repo = Path(spec['repository'])
    cases = [c for c in json.loads((source / 'inference_inputs.json').read_text()) if c['split'] == 'development']
    assert len(cases) == 22
    shards = [[], []]
    loads = [0, 0]
    for case in sorted(cases, key=lambda c: (-c['frames'], c['sequence'])):
        index = loads.index(min(loads))
        shards[index].append(case['sequence'])
        loads[index] += case['frames']
    for names in shards:
        names.sort()
    names = ['lib/test/tracker/sttrack_candidate_continuity.py', 'tools/prepare_sttrack_m48.py',
        'tools/check_sttrack_m48.py', 'tools/run_sttrack_m48.py', 'tools/launch_sttrack_m48.py']
    plan = dict(schema='sttrack_m48_native_continuity_admission_v1', source_root=str(source),
        source_spec_sha256=sha(source / 'spec.json'), source_recursive_result_sha256=sha(source / 'recursive_result.json'),
        control_root=str(control), control_spec_sha256=sha(control / 'spec.json'),
        control_recursive_result_sha256=sha(control / 'recursive_result.json'),
        control_training_result_sha256=sha(control / 'geometry_result.json'),
        checkpoint_sha256=sha(control / 'geometry_final.pth'), fitting_audit_sha256=sha(audit),
        source_sha256={name: sha(repo / name) for name in names}, primary='native_continuity_admission',
        single_change='Veto a nondefault M45 proposal unless BOTH native RGB and depth RoI cosine similarities to the preceding selected RoI are >= candidate0. Zero margin, no sweep.',
        inference='Exact frozen STTrackCandidateSet state update after admitted action; native default on veto. Language OFF, default templates/query/search/confidence unchanged.',
        training='No new parameters, architecture or optimization. Reuse the exact M45 head trained on DepthTrack Train63sequences/1511pairs,20epochs960steps.',
        parameters=448739, additional_parameters=0, additional_optimizer_steps=0,
        development_sequences=22, performance_gate=spec['recursive_performance_gate'], public_gate=spec['public_gate'],
        shards=shards, shard_frames=loads, public_automatic_launch=False,
        rationale='M45 first wrong choices can have weaker native continuity than default. Fitting audit retains5of26rescues in4sequences and20of96changes, while retaining its only harmful change. This does not establish safety.',
        scope='One complete fixed recursive comparison on repeatedly used DepthTrack Train development sequences. No public GT in inference, no similarity margin tuning or text claim.')
    root.mkdir()
    (root / 'spec.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(status='prepared', spec_sha256=sha(root / 'spec.json'), shard_frames=loads), indent=2))


if __name__ == '__main__':
    main()
