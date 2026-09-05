"""CPU semantic probe of the exact STTrack TSG loop, not tracking evaluation."""
import argparse
import ast
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch


def operation(a, b, layer):
    # An explicit order-sensitive probe operator, not a pretrained TSG block.
    return (a.cumsum(1) + (layer + 1) * 0.25 * b,
            b.cumsum(1) - (layer + 1) * 0.125 * a)


class Spy:
    def __init__(self, layer):
        self.layer = layer
        self.inputs = []

    def __call__(self, a, b):
        self.inputs.append((a.detach().clone(), b.detach().clone(), a.data_ptr(), b.data_ptr()))
        return operation(a, b, self.layer)


def compiled_block(loop, clone_inputs):
    body = copy.deepcopy(loop)
    replacements = 0
    if clone_inputs:
        for node in ast.walk(body):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in ('temp_x_flip', 'temp_r_flip'):
                    assert isinstance(node.value, ast.Name)
                    node.value = ast.Call(func=ast.Attribute(value=node.value, attr='clone', ctx=ast.Load()), args=[], keywords=[])
                    replacements += 1
        assert replacements == 2
    function = ast.FunctionDef(name='run', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=x) for x in ['self', 'temp_x', 'temp_r']], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=[body, ast.Return(value=ast.Tuple(elts=[ast.Name(id=x, ctx=ast.Load()) for x in ('temp_x', 'temp_r')], ctx=ast.Load()))], decorator_list=[])
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, '<extracted-STTrack-TSG-loop>', 'exec'), namespace)
    return namespace['run']


def flip_spatial(x, tail=5):
    return torch.cat((x[:, :-tail].flip(1), x[:, -tail:]), dim=1)


def reference(a, b, layers):
    for layer in range(layers):
        forward_a, forward_b = operation(a, b, layer)
        reverse_a, reverse_b = operation(flip_spatial(a), flip_spatial(b), layer)
        a = forward_a + flip_spatial(reverse_a)
        b = forward_b + flip_spatial(reverse_b)
    return a, b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_bytes()
    expected = 'd62cd0b2e6b383fd2049212f22d62334d32ea972150871522b874515e57ecb13'
    assert hashlib.sha256(source).hexdigest() == expected
    module = ast.parse(source)
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'STTrack')
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    loops = [n for n in ast.walk(forward) if isinstance(n, ast.For) and isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Name) and n.iter.func.id == 'range' and len(n.iter.args) == 1 and isinstance(n.iter.args[0], ast.Attribute) and n.iter.args[0].attr == 'TSG_layer']
    assert len(loops) == 1
    original = compiled_block(loops[0], False)
    cloned = compiled_block(loops[0], True)
    torch.set_num_threads(1)
    results = []
    for layers in [1, 2]:
        a = torch.arange(52, dtype=torch.float64).reshape(1, 13, 4) / 7
        b = 3 - a * 0.3
        expected_out = reference(a, b, layers)
        states = {}
        outputs = {}
        for name, fn in [('original', original), ('clone_only', cloned)]:
            spies = [Spy(layer) for layer in range(layers)]
            holder = SimpleNamespace(TSG_layer=layers, track_query_len=1, track_beforequery_len=4, TSG=spies)
            outputs[name] = fn(holder, a.clone(), b.clone())
            states[name] = spies
        original_same = [all(torch.equal(s.inputs[0][m], s.inputs[1][m]) for m in (0, 1)) for s in states['original']]
        original_shared_storage = [all(s.inputs[0][m] == s.inputs[1][m] for m in (2, 3)) for s in states['original']]
        fixed_distinct = [all(not torch.equal(s.inputs[0][m], s.inputs[1][m]) for m in (0, 1)) for s in states['clone_only']]
        original_error = max(float((x - y).abs().max()) for x, y in zip(outputs['original'], expected_out))
        fixed_error = max(float((x - y).abs().max()) for x, y in zip(outputs['clone_only'], expected_out))
        assert all(original_same) and all(original_shared_storage) and all(fixed_distinct)
        assert original_error > 0 and fixed_error == 0
        assert not torch.equal(states['original'][0].inputs[0][0], a)
        assert torch.equal(states['original'][0].inputs[0][0], flip_spatial(a))
        assert torch.equal(states['clone_only'][0].inputs[0][0], a)
        leaf_a, leaf_b = a.clone().requires_grad_(), b.clone().requires_grad_()
        holder = SimpleNamespace(TSG_layer=layers, track_query_len=1, track_beforequery_len=4, TSG=[Spy(i) for i in range(layers)])
        fixed_out = cloned(holder, leaf_a.clone(), leaf_b.clone())
        sum(x.square().mean() for x in fixed_out).backward()
        ref_a, ref_b = a.clone().requires_grad_(), b.clone().requires_grad_()
        sum(x.square().mean() for x in reference(ref_a, ref_b, layers)).backward()
        grad_error = max(float((x.grad - y.grad).abs().max()) for x, y in [(leaf_a, ref_a), (leaf_b, ref_b)])
        assert torch.allclose(leaf_a.grad, ref_a.grad, rtol=1e-12, atol=1e-12)
        assert torch.allclose(leaf_b.grad, ref_b.grad, rtol=1e-12, atol=1e-12)
        results.append(dict(layers=layers, original_inputs_equal=original_same, original_inputs_share_storage=original_shared_storage,
            clone_inputs_distinct=fixed_distinct, original_first_input_is_flipped=True, clone_first_input_is_unmodified=True,
            original_vs_functional_reference_max_error=original_error, clone_vs_functional_reference_max_error=fixed_error,
            clone_vs_functional_reference_max_gradient_error=grad_error))
    assert not torch.cuda.is_initialized()
    result = dict(status='confirmed_source_semantics', observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=expected, probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), torch_version=torch.__version__,
        extracted_lines=[loops[0].lineno, loops[0].end_lineno], input_shape=[1, 13, 4], query_tail=5, device='cpu', results=results,
        scope='Exact extracted assignment/call/addition loop with an explicit order-sensitive probe operator. Not a real TSG/Mamba numerical or tracking evaluation.',
        actual_tsg_weights_loaded=False, images_or_gt_opened=False, optimizer_steps=0, cuda_initialized=False,
        runtime_source_changed=False, inference_benefit_measured=False, training_benefit_measured=False)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
