"""Post-comparison fitting-only inspection; not a trained policy or metric promotion."""
from pathlib import Path
from collections import Counter
import hashlib
import json
from accept_m106_selected_geometry import overlap
HERE=Path(__file__).parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    sources=[HERE/'m103_completed/events.jsonl',HERE/'m106_completed/verification/geometry_boxes.jsonl',HERE/'m105_inputs/training_labels.json']
    probes={r['key']:r for r in (json.loads(s) for s in sources[0].read_text().splitlines())}
    labels=json.loads(sources[2].read_text())
    groups=Counter();events=[]
    for row in (json.loads(s) for s in sources[1].read_text().splitlines()):
        if row['split']!='fit' or not probes[row['key']]['refinement_eligible']:
            continue
        boxes=row['boxes']['parent']; target=labels[row['key']]['current']; selected=row['selected']
        values=[overlap(b,target) for b in boxes]; best=max(range(10),key=lambda i:values[i])
        b=boxes[selected]; center=[target[i]+.5*target[i+2] for i in (0,1)]
        def inside(factor):
            return all(abs(center[i]-(b[i]+.5*b[i+2]))<=.5*factor*b[i+2] for i in (0,1))
        event=dict(key=row['key'],selected=selected,GT_best_index=best,selected_iou=values[selected],GT_best_iou=values[best],
                   target_center_inside_selected_box=inside(1),target_center_inside_selected_factor2_context=inside(2),
                   same_selected_and_GT_best=selected==best,
                   selected_quality='correct' if values[selected]>=.5 else 'partial' if values[selected]>=.1 else 'low')
        events.append(event)
        groups['events']+=1;groups[event['selected_quality']]+=1
        for name in ['same_selected_and_GT_best','target_center_inside_selected_box','target_center_inside_selected_factor2_context']:
            groups[name]+=int(event[name])
    assert len(events)==2033
    result=dict(status='complete_fitting_only_postcomparison_selection_diagnostic',source_sha256=sha(Path(__file__)),
                inputs_sha256={str(p.relative_to(HERE)):sha(p) for p in sources},
                counts=dict(groups),scope='Only previously fixed eligible fitting states; no development/public cases used to define a new task. GT-best is training supervision only.',
                limitation='Spatial inclusion does not prove useful observed pixels/features or physical instance identity. This does not establish a causal explanation of M106 harm.',
                events=events)
    (HERE/'M106_FIT_SELECTION_DIAGNOSTIC.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(result['counts']))


if __name__=='__main__':
    main()
