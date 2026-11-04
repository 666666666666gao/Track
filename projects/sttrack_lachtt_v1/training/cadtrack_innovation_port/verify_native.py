"""Train-only 64-frame native/zero-residual control; never a formal metric."""
import json
import torch

from train import arguments, prepare


@torch.no_grad()
def main():
    args = arguments()
    assert args.mode == 'smoke'
    spec, bank, decoder, writer, actor, tracker = prepare(args)
    from full_dense_tracker import state_digest
    from lib.train.dataset.depth_utils import get_rgbd_frame

    case = spec['sequence_order'][0]
    from pathlib import Path
    folder = Path(spec['dataset_root']) / case['sequence']

    def image(frame):
        return get_rgbd_frame(str(folder / 'color' / ('%08d.jpg' % (frame + 1))),
            str(folder / 'depth' / ('%08d.png' % (frame + 1))),
            dtype='rgbcolormap', depth_clip=True)

    frozen = actor.frozen_digest()
    decoder.eval()
    writer.eval()
    expected = []
    template_snapshots = []
    with torch.cuda.device(1):
        actor.native.initialize(image(0), dict(init_bbox=list(case['first_box'])))
        # The hooks must leave the original network outputs byte exact.
        patch, _, _ = actor.sample_target(image(1), actor.native.state, 4., output_sz=256)
        search = [actor.native.preprocessor.process(patch)]
        hooked = actor.native.network(template=actor.native.z_dict, search=search)[0]
        for hook in actor.hooks:
            hook.remove()
        plain = actor.native.network(template=actor.native.z_dict, search=search)[0]
        for key in ['pred_boxes', 'score_map', 'size_map', 'offset_map']:
            assert torch.equal(hooked[key], plain[key]), key
        actor.hooks = [actor.native.network.backbone.register_forward_hook(actor._backbone_features),
            actor.native.network.decode_fuse_search.register_forward_hook(actor._fused_features)]
        for frame in range(1, 65):
            result = actor.native.track(image(frame))
            expected.append((list(result['target_bbox']), float(result['best_score'])))
            template_snapshots.append([tensor.detach().cpu().clone() for tensor in actor.native.z_dict])
    tracker.initialize(image(0), case['first_box'], bank['sequences'].index(case['sequence']))
    writes = 0
    for frame in range(1, 65):
        out, data, record, _ = tracker.step(image(frame), gate_writes=False)
        assert torch.equal(out['response'], data['native_response'] * .5)
        assert record['bbox'] == expected[frame - 1][0], (frame, record['bbox'], expected[frame - 1][0])
        assert record['native_same_position_response'] == expected[frame - 1][1]
        for actual, wanted in zip(actor.native.z_dict, template_snapshots[frame - 1]):
            assert torch.equal(actual.cpu(), wanted), frame
        writes += int(record['template_write'])
    assert actor.frozen_digest() == frozen
    args.output.mkdir(parents=True)
    result = dict(status='native_parity_pass', sequence=case['sequence'], frames=64,
        hooked_native_outputs_exact=True, zero_residual_boxes_exact=True,
        native_confidence_exact=True, template_history_exact=True, native_writes=writes,
        C_bypassed_for_native_control_only=True, frozen_before_after_exact=True, formal_metrics=False)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
