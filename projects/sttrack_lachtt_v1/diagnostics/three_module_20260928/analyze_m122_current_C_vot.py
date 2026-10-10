"""Locked official VOT analysis after every multi-start trajectory is sealed."""
import argparse,importlib.util,json,math
from pathlib import Path
from bind_m122_official_initializations import sha
from prepare_m122_evaluation_suite import read,write


def main(root):
    selection=read(root/'selection.json')
    assert selection['status']=='M122_CurrentC_composite_final_fixed_before_metrics' and len(selection['models'])==1
    for row in selection['models']:
        target=root;bundle=read(target/'bundle.json');vot=target/'vot';run=vot/'run'
        assert sha(target/'bundle.json')==row['bundle_sha256'] and sha(bundle['final_path'])==row['final_sha256']
        for path,digest in bundle['source_sha256'].items():assert sha(path)==digest,path
        execution=read(vot/'execution.json')
        for path,digest in execution['source_sha256'].items():assert sha(path)==digest,path
        assert (target/'vot.exit').read_text().strip()=='0'
        merge=read(run/'merge_result.json')
        assert merge['status']=='complete_M122_full127_merge' and merge['anchor_count']==1765 and merge['result_file_count']==5295
        assert merge['source_manifest_sha256']==sha(run/'shard_manifest.json')
        for path,digest in merge['result_sha256'].items():assert sha(run/'master'/path)==digest,path
        analysis_path=run/'master/analysis'/(row['name']+'_full127.json')
        values=read(analysis_path)['results']['baseline']['results']
        metrics=dict(EAO=float(values[0][0][0])*100,ACC=float(values[2][0][0])*100,ROB=float(values[2][0][1])*100)
        assert all(math.isfinite(v) and 0<=v<=100 for v in metrics.values())
        assert sha(bundle['failure_source_path'])=='e96a375a3792a80ab12e5671db468f559d514558a05d1181fe7afec48bcb514a'
        assert sha(bundle['failure_common_path'])=='cbbe55132a4e64157011c0cbfa9162bd583f09588e3e497dcf741a8d5174a6bb'
        spec=importlib.util.spec_from_file_location('M122_locked_failure_audit',bundle['failure_source_path'])
        audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
        outcomes,failures,per,settings=audit.collect_confirmed_failure_outcomes(run/'master',merge['tracker'],expected_anchors=1765)
        assert len(outcomes)==1765 and len(per)==127
        write(vot/'result.json',dict(status='complete_full127',model=row['name'],final_sha256=row['final_sha256'],bundle_sha256=row['bundle_sha256'],
            bank_sha256=selection['bank_sha256'],binding_sha256=selection['binding_sha256'],metrics_percent=metrics,anchors=1765,sequences=127,
            confirmed_failures=failures,per_sequence_failures=per,failure_outcomes=outcomes,failure_settings=settings,
            merge_sha256=sha(run/'merge_result.json'),analysis_sha256=sha(analysis_path),external_optimizer_steps=0))
        print(json.dumps(dict(model=row['name'],metrics=metrics)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
