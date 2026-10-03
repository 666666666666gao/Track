"""Recount stored M111 selections and weak-label confusion, without a forward."""
import argparse
import json
import math
from pathlib import Path
from accept_m110_outputs import sha, recount, compare


def main():
    p = argparse.ArgumentParser()
    for name in ['root', 'private-artifacts', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); d = Path(__file__).parent
    driver = json.loads((a.root / 'result.json').read_text())
    assert driver['status'] == 'complete_M111_explicit_category_pair'
    assert driver['source_sha256'] == sha(d / 'run_m111_category_pair.py')
    assert not driver['human_confirmed'] and driver['no_automatic_promotion']
    exits = ['controller', 'driver', 'prepare', 'sanity_weight0', 'sanity_weight1',
             'train_weight0', 'train_weight1']
    assert all((a.root / (name + '.exit')).read_text().strip() == '0' for name in exits)
    assert sha(a.private_artifacts / 'category_bank.pt') == driver['category_bank_sha256']
    arms = {}; row_hashes = {}; weak_categories = {}
    for arm, weight in [('weight0', 0), ('weight1', 1)]:
        folder = a.root / ('train_' + arm)
        r = json.loads((folder / 'result.json').read_text())
        assert r['status'] == 'complete_M111_explicit_category_arm'
        assert r['evidence_weight'] == weight and r['seed'] == 2027 and r['optimizer_steps'] == 480
        assert r['source_sha256'] == sha(d / 'train_m111_explicit_category_evidence.py')
        assert r['prototype_sha256'] == sha(d / 'instance_ab_prototype.py')
        assert r['final_sha256'] == sha(a.private_artifacts / ('train_' + arm) / 'final.pt')
        assert r['empty_all3039_scores_quality_exact'] and r['frozen_parameters_buffers_exact']
        assert r['final_state_roundtrip_exact'] and not r['human_confirmed']
        assert len(r['history']) == 12 and all(x['optimizer_steps'] == 40 for x in r['history'])
        assert sum(r['category_example_draw_counts'].values()) == 480 * 32
        for h in r['history']:
            loss, localization, preservation, ce = h['mean_losses']
            assert all(math.isfinite(v) for v in h['mean_losses'])
            assert abs(loss - localization - preservation - weight * ce) < 1e-6
        rows = [json.loads(x) for x in (folder / 'fit_events.jsonl').read_text().splitlines()]
        assert len(rows) == len({x['key'] for x in rows}) == 2544
        fit_keys = {x['key'].rsplit('@', 1)[0] for x in rows}
        compare(recount(rows), r['fit'])
        conditions = {}; condition_rows = {}
        for mode in ['empty', 'generic', 'weak_text']:
            path = folder / (mode + '_development.jsonl')
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            assert len(rows) == len({x['key'] for x in rows}) == 495
            assert not (fit_keys & {x['key'].rsplit('@', 1)[0] for x in rows})
            if mode == 'empty':
                assert all(x['selected'] == x['parent_selected'] and x['selected_iou'] == x['parent_iou'] for x in rows)
            conditions[mode] = recount(rows)
            compare(conditions[mode], r['content_conditions'][mode])
            condition_rows[mode] = rows; row_hashes[str(path.relative_to(a.root))] = sha(path)
        empty, weak = condition_rows['empty'], condition_rows['weak_text']
        assert [x['key'] for x in empty] == [x['key'] for x in weak]
        paired = dict(events=495,
            changed_selections=sum(x['selected'] != y['selected'] for x, y in zip(empty, weak)),
            iou_improved=sum(y['selected_iou'] > x['selected_iou'] for x, y in zip(empty, weak)),
            iou_worsened=sum(y['selected_iou'] < x['selected_iou'] for x, y in zip(empty, weak)),
            delta_mean_iou=sum(y['selected_iou'] - x['selected_iou'] for x, y in zip(empty, weak)) / 495,
            threshold_rescues=sum(x['selected_iou'] < .5 <= y['selected_iou'] for x, y in zip(empty, weak)),
            threshold_breaks=sum(y['selected_iou'] < .5 <= x['selected_iou'] for x, y in zip(empty, weak)))
        assert paired == r['weak_vs_own_empty']
        dev = conditions['weak_text']
        checks = dict(correct_exceeds_native=dev['all']['selected_iou50'] > dev['all']['native_iou50'],
                      mean_iou_exceeds_native=dev['all']['selected_mean_iou'] > dev['all']['native_mean_iou'],
                      healthy_breaks_zero=dev['healthy']['breaks'] == 0,
                      transition_correct_at_least_native=dev['transition']['selected_iou50'] >= dev['transition']['native_iou50'])
        assert checks == r['fixed_state_checks']
        for stage in ['initial_category_readout', 'final_category_readout']:
            for split, counts in [('fit', [80, 17, 1]), ('development', [15, 3, 1])]:
                c = r[stage][split]; matrix = c['confusion_target_by_prediction']
                assert [sum(row) for row in matrix] == counts
                assert sum(matrix[i][i] for i in range(3)) == c['correct']
                assert c['constant_majority_correct'] == max(counts)
                assert all(abs(matrix[i][i] / counts[i] - c['recalls'][i]) < 1e-7 for i in range(3))
                assert math.isfinite(c['cross_entropy']) and c['cross_entropy'] >= 0
        arms[arm] = dict(conditions=conditions, own_empty=paired, fixed_state_checks=checks,
                         final_sha256=r['final_sha256'], category_draw_counts=r['category_example_draw_counts'])
        weak_categories[arm] = r['final_category_readout']
    assert arms['weight0']['category_draw_counts'] == arms['weight1']['category_draw_counts']
    m110 = json.loads((d / 'm110_completed/train_weak_text/result.json').read_text())
    assert arms['weight0']['final_sha256'] == m110['final_sha256']
    receipt = dict(status='M111_CPU_recount_pass', source_sha256=sha(__file__),
                   seven_exit_codes_zero=True, category_encoding_complete=True,
                   arms=arms, category_confusion_count_checks=weak_categories,
                   development_row_hashes=row_hashes, weight0_checkpoint_matches_M110_weak_final=True,
                   category_CE_recomputed_from_logits=False, checkpoint_forward_replayed=False,
                   scope='Stored selection rows independently recounted; confusion arithmetic checked; no new GPU forward or official metrics.',
                   no_semantic_selection_gain=True, promotion_allowed=False)
    a.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(status=receipt['status'], control_reproduces_M110=True,
                         pair={k: v['own_empty'] for k, v in arms.items()})))


if __name__ == '__main__':
    main()
