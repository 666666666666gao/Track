"""CPU recount of stored M108 cached-state results; no model forward or tuning."""
import argparse,json,hashlib
from collections import defaultdict
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recount(rows):
    groups=defaultdict(list)
    for row in rows:
        groups['all'].append(row)
        for tag in row['strata']:groups[tag].append(row)
    result={}
    for tag,items in groups.items():
        result[tag]=dict(valid_gt=len(items),native_iou50=0,selected_iou50=0,oracle_iou50=0,
                         rescues=0,breaks=0,native_mean_iou=0.,selected_mean_iou=0.)
        r=result[tag]
        for x in items:
            n=x['native_iou']>=.5;s=x['selected_iou']>=.5
            r['native_iou50']+=int(n);r['selected_iou50']+=int(s)
            r['oracle_iou50']+=int(x['oracle_iou']>=.5)
            r['rescues']+=int(s and not n);r['breaks']+=int(n and not s)
            r['native_mean_iou']+=x['native_iou']/len(items)
            r['selected_mean_iou']+=x['selected_iou']/len(items)
    return result


def compare(actual,saved):
    assert actual.keys()==saved.keys()
    for group in actual:
        assert actual[group].keys()==saved[group].keys()
        for key,value in actual[group].items():
            assert abs(value-saved[group][key])<1e-12,(group,key,value,saved[group][key])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--parent-result',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root;d=Path(__file__).parent
    driver=json.loads((root/'result.json').read_text())
    assert driver['status']=='complete_M108_frozen_semantic_pair' and not driver['human_confirmed']
    assert driver['source_sha256']==sha(d/'run_m108_frozen_pair.py')
    for name in ['controller','driver','prepare','sanity_generic','sanity_weak_text','train_generic','train_weak_text']:
        assert (root/(name+'.exit')).read_text().strip()=='0',name
    parent=json.loads(a.parent_result.read_text());old={r['key']:r for r in parent['development_rows']}
    accepted={};content={};files={}
    for arm in ['generic','weak_text']:
        folder=root/('train_'+arm);result=json.loads((folder/'result.json').read_text())
        assert result['arm']==arm and result['optimizer_steps']==480 and result['seed']==2027
        assert result['source_sha256']==sha(d/'train_m108_frozen_semantics.py')
        assert result['prototype_sha256']==sha(d/'instance_ab_prototype.py')
        assert result['input_helper_sha256']==sha(d/'train_m107_weak_semantics.py')
        assert result['final_sha256']==sha(folder/'final.pt')
        assert result['empty_all3039_scores_quality_exact'] and result['frozen_parameters_buffers_exact'] and result['final_state_roundtrip_exact']
        assert result['bank_sha256']==driver['bank_sha256'] and result['labels_sha256']==driver['labels_sha256']
        rows=[json.loads(line) for line in (folder/'fit_events.jsonl').read_text().splitlines()]
        assert len(rows)==len({r['key'] for r in rows})==2544
        compare(recount(rows),result['fit']);conditions={};summary={}
        for mode in ['empty','generic','weak_text']:
            path=folder/(mode+'_development.jsonl')
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows)==len({r['key'] for r in rows})==495
            assert {r['key'] for r in rows}==old.keys()
            for r in rows:
                assert r['parent_selected']==old[r['key']]['selected'] and r['parent_iou']==old[r['key']]['selected_iou']
                if mode=='empty':assert r['selected']==r['parent_selected'] and r['selected_iou']==r['parent_iou']
            summary[mode]=recount(rows);compare(summary[mode],result['content_conditions'][mode])
            conditions[mode]={r['key']:r for r in rows};files[str(path.relative_to(root))]=sha(path)
        compare(summary[arm],result['development']);dev=summary[arm]
        gates=dict(correct_exceeds_native=dev['all']['selected_iou50']>dev['all']['native_iou50'],
                   mean_iou_exceeds_native=dev['all']['selected_mean_iou']>dev['all']['native_mean_iou'],
                   healthy_breaks_zero=dev['healthy']['breaks']==0,
                   transition_correct_at_least_native=dev['transition']['selected_iou50']>=dev['transition']['native_iou50'])
        assert gates==result['fixed_state_checks'];accepted[arm]=dict(summary=summary,gates=gates)
        x=conditions['weak_text'];y=conditions['generic'];z=conditions['empty']
        content[arm]={}
        for label,reference in [('reviewed_vs_generic',y),('reviewed_vs_empty',z)]:
            content[arm][label]=dict(changed_selection=sum(x[k]['selected']!=reference[k]['selected'] for k in x),
                improved_iou=sum(x[k]['selected_iou']>reference[k]['selected_iou'] for k in x),
                harmed_iou=sum(x[k]['selected_iou']<reference[k]['selected_iou'] for k in x),
                mean_iou_change=sum(x[k]['selected_iou']-reference[k]['selected_iou'] for k in x)/495)
    receipt=dict(status='M108_CPU_recount_pass',source_sha256=sha(__file__),arms=accepted,
                 same_weight_content=content,development_row_hashes=files,
                 scope='cached Train states only; no checkpoint forward replay or official metric')
    a.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':receipt['status'],'content':content}),flush=True)


if __name__=='__main__':main()
