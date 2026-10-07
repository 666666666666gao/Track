"""Read the actual cached Train regions, human text and localization GT only."""
import argparse, json, time
from pathlib import Path
import torch
from torch.nn import functional as F
from collect_train_states import sha
from region_patch_evidence import region_samples, phrase_scores, coordinate_sanity


def pooled(tokens, valid, boxes, origin):
    values, weights = region_samples(tokens, valid, boxes, origin)
    values = F.normalize(values, dim=-1)
    mean = (values * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1.)
    return F.normalize(mean, dim=-1), weights.mean(1)


def main():
    parser = argparse.ArgumentParser()
    for key in ['cache', 'dense', 'bank', 'labels', 'parent-rows', 'output']:
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--shard', type=int, choices=[0,1], required=True)
    parser.add_argument('--mode', choices=['sanity','full'], required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    torch.set_num_threads(1); torch.manual_seed(2027)
    preparation = json.loads((args.cache/'preparation.json').read_text())
    assert preparation['training_labels_sha256'] == sha(args.cache/'training_labels.json')
    assert preparation['inference_inputs_sha256'] == sha(args.cache/'inference_inputs.json')
    gt = json.loads((args.cache/'training_labels.json').read_text())
    plans = {r['sequence']:r for r in json.loads((args.cache/'inference_inputs.json').read_text())}
    assert (args.dense/'driver.exit').read_text().strip() == '0'
    dense_receipt = json.loads((args.dense/'result.json').read_text())
    assert dense_receipt['status'] == 'complete_M117_region_pair' and dense_receipt['optimizer_steps'] == 0
    assert sum(r['events'] for r in dense_receipt['full']) == 3502
    bank = torch.load(args.bank, map_location='cpu')
    reviewed = json.loads(args.labels.read_text())
    assert bank['human_confirmed'] and reviewed['human_confirmed']
    assert bank['dataset'] == reviewed['dataset'] == 'depthtrack'
    assert bank['labels_sha256'] == sha(args.labels)
    assert bank['sequences'] == [r['sequence'] for r in reviewed['initial']]
    parent_report = json.loads((args.parent_rows/'result.json').read_text())
    assert parent_report['status'] == 'complete_M118_regional_training'
    assert parent_report['parent_sha256'] == '1d187d78c1dd76ebf59e41fd31af16fa69cecc37b4cb6c74fbdfe3ed5c237293'
    parent = {}
    for split, filename in [('fit','fit_events.jsonl'), ('development','empty_development.jsonl')]:
        for line in (args.parent_rows/filename).read_text().splitlines():
            row = json.loads(line); assert row['key'] not in parent
            parent[row['key']] = dict(split=split, scores=row['parent_scores'], selected=row['parent_selected'],
                                      iou=row['candidate_iou'])
    assert len(parent) == 3039
    old_rows = {r['key']:r for r in (json.loads(line) for line in
        (args.dense/('full_shard'+str(args.shard))/'phrase_responses.jsonl').read_text().splitlines())}
    cases = dense_receipt['full'][args.shard]['sequences']
    if args.mode == 'sanity':
        cases = [next(r for r in cases if r['split'] == 'fit')]
    coordinates = coordinate_sanity()
    records = []; events = []; initials = []; started = time.time()
    with torch.no_grad():
        for case in cases:
            name = case['sequence']; plan = plans[name]
            source = args.dense/('full_shard'+str(args.shard))/'features'/(name+'.pt')
            assert sha(source) == case['feature_sha256']
            assert sha(args.cache/'features'/(name+'.pt')) == case['original_feature_sha256']
            dense = torch.load(source, map_location='cpu')
            native = torch.load(args.cache/'features'/(name+'.pt'), map_location='cpu')
            assert not dense['GT_loaded'] and not native['labels_loaded']
            assert dense['sequence'] == name and dense['split'] == plan['split'] == case['split']
            assert dense['event_frames'] == plan['event_frames'] == native['event_frames']
            bank_at = bank['sequences'].index(name)
            assert bank['splits'][bank_at] == case['split']
            text = torch.cat((bank['tokens'][bank_at], bank['generic'][None], bank['empty'][None])).cuda().float()
            init_box = torch.tensor(plan['init_bbox'], device='cuda', dtype=torch.float32)[None]
            init_grid = dense['initial_search_tokens'].cuda().float()
            init_valid = dense['initial_observed_fraction'].cuda().float()
            init_region, init_coverage = phrase_scores(init_grid, init_valid, init_box, dense['crop_origins'][0], text)
            init_ring, _ = phrase_scores(init_grid, init_valid, init_box, dense['crop_origins'][0], text, True)
            reference, _ = pooled(init_grid, init_valid, init_box, dense['crop_origins'][0])
            old = old_rows[name+'@0']
            torch.testing.assert_close(init_region.cpu(), torch.tensor(old['region_cosine']))
            torch.testing.assert_close(init_ring.cpu(), torch.tensor(old['ring_cosine']))
            initials.append(dict(sequence=name, split=case['split'], category=reviewed['initial'][bank_at]['phrases'][0],
                region_cosine=init_region[0].cpu().tolist(), ring_cosine=init_ring[0].cpu().tolist(),
                observed_fraction=float(init_coverage[0]), original_M117_reader_default_close=True))
            frames = dense['event_frames'][:3] if args.mode == 'sanity' else dense['event_frames']
            for at, frame in enumerate(frames):
                key = name+'@'+str(frame); label = gt[key]
                assert label['split'] == dense['split']
                boxes = native['boxes'][at].cuda().float()
                grid = dense['search_tokens'][at].cuda().float()
                valid = dense['observed_fraction'][at].cuda().float()
                origin = dense['crop_origins'][at+1]
                region, coverage = phrase_scores(grid, valid, boxes, origin, text)
                ring, ring_coverage = phrase_scores(grid, valid, boxes, origin, text, True)
                current, _ = pooled(grid, valid, boxes, origin)
                instance = (current*reference).sum(-1)
                old = old_rows[key]
                torch.testing.assert_close(region.cpu(), torch.tensor(old['region_cosine']))
                torch.testing.assert_close(ring.cpu(), torch.tensor(old['ring_cosine']))
                row = dict(key=key, split=dense['split'], strata=label['strata'], category=reviewed['initial'][bank_at]['phrases'][0],
                    phrase_mask=bank['mask'][bank_at].tolist(), candidate_region=region.cpu().tolist(),
                    candidate_ring=ring.cpu().tolist(), candidate_instance=instance.cpu().tolist(),
                    candidate_observed_fraction=coverage.cpu().tolist(), ring_observed_fraction=ring_coverage.cpu().tolist(),
                    gt_valid=label['current'] is not None)
                if label['current'] is not None:
                    prior = parent[key]
                    assert prior['split'] == row['split']
                    from analyze_train_states import overlaps
                    target = torch.tensor(label['current'], device='cuda', dtype=torch.float32)[None]
                    iou = overlaps(boxes.cpu(), target[0].cpu())
                    assert iou.tolist() == prior['iou']
                    target_region, target_coverage = phrase_scores(grid, valid, target, origin, text)
                    target_ring, target_ring_coverage = phrase_scores(grid, valid, target, origin, text, True)
                    target_visual, _ = pooled(grid, valid, target, origin)
                    left, top, side = origin
                    x, y, w, h = label['current']; height, width = native['image_shape'][at].tolist()
                    row.update(candidate_iou=iou.tolist(), parent_selected=prior['selected'], parent_scores=prior['scores'],
                        gt_box=label['current'], crop_origin=origin,
                        gt_center_in_crop=left<=x+.5*w<left+side and top<=y+.5*h<top+side,
                        gt_whole_in_crop=left<=x and top<=y and x+w<=left+side and y+h<=top+side,
                        gt_whole_in_observed_image=max(left,0)<=x and max(top,0)<=y and x+w<=min(left+side,width) and y+h<=min(top+side,height),
                        gt_region=target_region[0].cpu().tolist(), gt_ring=target_ring[0].cpu().tolist(),
                        gt_observed_fraction=float(target_coverage[0]), gt_ring_observed_fraction=float(target_ring_coverage[0]),
                        gt_instance=float((target_visual*reference).sum()))
                else:
                    assert key not in parent
                events.append(row)
            record = dict(sequence=name, split=case['split'], events=len(frames), source_feature_sha256=case['feature_sha256'],
                          original_M117_reader_default_close=True, seconds=time.time()-started)
            records.append(record); print(json.dumps(dict(done=len(records), total=len(cases), **record)), flush=True)
    (args.output/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
    (args.output/'initial.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in initials))
    result = dict(status='complete_M119_'+args.mode+'_shard', mode=args.mode, shard=args.shard, seed=2027,
        sequences=records, events=len(events), valid_gt=sum(r['gt_valid'] for r in events),
        human_confirmed_initialization=True, current_attribute_visibility_truth=False,
        current_GT_loaded_for_readonly_analysis=True, optimizer_steps=0, checkpoints_created=False,
        model_forward_executed=False, tracker_state_committed=False, no_public_evaluation=True,
        coordinates=coordinates, source_sha256=sha(__file__), interface_sha256=sha(Path(__file__).with_name('region_patch_evidence.py')),
        bank_sha256=sha(args.bank), labels_sha256=sha(args.labels), GT_sha256=sha(args.cache/'training_labels.json'),
        dense_receipt_sha256=sha(args.dense/'result.json'), parent_result_sha256=sha(args.parent_rows/'result.json'),
        elapsed_seconds=time.time()-started, gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], events=len(events), seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
