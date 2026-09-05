"""Inspect one causal native/clone-only forward; no current-frame GT or metrics."""
import argparse
import ast
import copy
import datetime
import hashlib
import importlib
import json
from pathlib import Path
import types

import torch

from tools.sttrack_m54_common import check_sources, parameters, sha


def clone_forward(source, module_globals):
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'STTrack')
    forward = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward'))
    count = 0
    for node in ast.walk(forward):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('temp_x_flip', 'temp_r_flip'):
                assert isinstance(node.value, ast.Name)
                node.value = ast.Call(func=ast.Attribute(value=node.value, attr='clone', ctx=ast.Load()), args=[], keywords=[])
                count += 1
    assert count == 2
    tree = ast.Module(body=[forward], type_ignores=[])
    ast.fix_missing_locations(tree)
    namespace = dict(module_globals)
    exec(compile(tree, '<clone-only-forward-probe>', 'exec'), namespace)
    return namespace['forward']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m54-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan, parent, spec = check_sources(args.m54_root)
    assert (args.m54_root / 'controller.exit').read_text().strip() == '0'
    assert (args.m54_root / 'recursive.exit').read_text().strip() == '0'
    assert (args.m54_root / 'analysis.exit').read_text().strip() == '0'
    from lib.test.tracker.sttrack import STTrack
    from lib.train.dataset.depth_utils import get_rgbd_frame
    model_module = importlib.import_module('lib.models.sttrack.sttrack')
    model_path = Path(model_module.__file__)
    expected_source = 'd62cd0b2e6b383fd2049212f22d62334d32ea972150871522b874515e57ecb13'
    assert sha(model_path) == expected_source
    mamba_path = model_path.parents[1] / 'layers/mamba.py'
    assert sha(mamba_path) == 'a11e49551cea9c7a188b9b7783b541b315a1f684b86e1de364bb3ad6c04f00be'
    alternate_forward = clone_forward(model_path.read_bytes(), model_module.__dict__)
    case = next(c for c in json.loads((parent / 'inference_inputs.json').read_text()) if c['sequence'] == 'chair01_indoor' and c['split'] == 'fit')
    folder = Path(spec['dataset_root']) / case['sequence']
    image_paths = [folder / kind / ('%08d.%s' % (index, extension)) for index in (1, 2) for kind, extension in [('color', 'jpg'), ('depth', 'png')]]
    image_hashes = {str(path): sha(path) for path in image_paths}
    images = [get_rgbd_frame(str(folder / 'color' / ('%08d.jpg' % index)), str(folder / 'depth' / ('%08d.png' % index)), dtype='rgbcolormap', depth_clip=True) for index in (1, 2)]
    torch.set_num_threads(1)
    variants = {}
    for variant in ('original', 'clone_only'):
        tracker = STTrack(parameters(spec))
        assert tracker.network.fix_query_window
        assert not tracker.network.training
        if variant == 'clone_only':
            tracker.network.forward = types.MethodType(alternate_forward, tracker.network)
        input_records, output_records, fusion_records = {}, {}, []
        handles = []
        for index, block in enumerate(tracker.network.TSG):
            input_records[index], output_records[index] = [], []

            def pre_hook(module, inputs, index=index):
                input_records[index].append({'values': [x.detach().cpu().clone() for x in inputs], 'storage': [x.data_ptr() for x in inputs]})

            def post_hook(module, inputs, output, index=index):
                output_records[index].append([x.detach().cpu().clone() for x in output])

            handles.extend([block.register_forward_pre_hook(pre_hook), block.register_forward_hook(post_hook)])

        def fusion_hook(module, inputs):
            fusion_records.append([{'shape': list(x.shape), 'spatial_palindrome_exact': bool(torch.equal(x, x.flip(1))),
                'spatial_palindrome_max_abs': float((x - x.flip(1)).abs().max()),
                'spatial_palindrome_mean_abs': float((x - x.flip(1)).abs().mean())} for x in inputs])

        handles.append(tracker.network.MambaFusion.register_forward_pre_hook(fusion_hook))
        tracker.initialize(images[0], dict(init_bbox=list(case['init_bbox'])))
        output = tracker.track(images[1])
        for handle in handles:
            handle.remove()
        assert len(fusion_records) == 1 and len(input_records) == 2
        layers = []
        for index in sorted(input_records):
            incoming, outgoing = input_records[index], output_records[index]
            assert len(incoming) == len(outgoing) == 2
            layers.append(dict(layer=index, input_shapes=[list(x.shape) for x in incoming[0]['values']],
                same_input_storage=[incoming[0]['storage'][m] == incoming[1]['storage'][m] for m in (0, 1)],
                input_call_max_abs=[float((incoming[0]['values'][m] - incoming[1]['values'][m]).abs().max()) for m in (0, 1)],
                output_call_max_abs=[float((outgoing[0][m] - outgoing[1][m]).abs().max()) for m in (0, 1)]))
        variants[variant] = dict(layers=layers, fusion_input=fusion_records[0], bbox=output['target_bbox'], confidence=float(output['best_score']),
            query_lengths=[x.shape[1] for x in tracker.track_query_before])
        del tracker, input_records, output_records
        torch.cuda.empty_cache()
    check_sources(args.m54_root)
    assert sha(model_path) == expected_source
    assert {str(path): sha(path) for path in image_paths} == image_hashes
    result = dict(status='complete_real_forward_diagnostic', observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        probe_sha256=sha(Path(__file__)), source_sha256=expected_source, mamba_source_sha256=sha(mamba_path),
        base_checkpoint_sha256=spec['checkpoint_sha256'], m54_spec_sha256=sha(args.m54_root / 'spec.json'),
        sequence=case['sequence'], split=case['split'], current_frame_zero_based=1, image_sha256=image_hashes,
        variants=variants, actual_tsg_weights_loaded=True, inference_forwards=2, current_or_future_gt_opened=False,
        initialization_box_source='Sealed fitting inference manifest', optimizer_steps=0, runtime_source_changed=False,
        tracking_metrics_computed=False, training_benefit_measured=False,
        scope='Two same-weight one-frame probes from independent initialized states. Clone-only change exists solely on the second in-memory model instance. No trajectory/benchmark comparison.')
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
