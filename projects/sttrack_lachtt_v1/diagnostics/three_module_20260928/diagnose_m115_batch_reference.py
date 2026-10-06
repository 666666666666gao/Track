"""Zero-update witness of cross-batch versus same-batch Parent score equality."""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from semantic_choice_prototype import SemanticChoicePrototype
from train_ab_visual_control import load_inputs
from train_m110_no_weak_rank import inputs,snapshot


def compare(x,y):
    x=x.detach().cpu();y=y.detach().cpu()
    return dict(exact=torch.equal(x,y),different_values=int((x!=y).sum()),
                max_absolute_delta=float((x-y).abs().max()),
                different_argmax=int((x.argmax(-1)!=y.argmax(-1)).sum()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    torch.set_num_threads(1);random.seed(2027);np.random.seed(2027)
    torch.manual_seed(2027);torch.cuda.manual_seed_all(2027);device=torch.device('cuda')
    panel=load_inputs(Path('/root/autodl-tmp/sttrack_m90_train_states_20260928'),
        Path('/root/autodl-tmp/sttrack_m98_train_contexts_20260928'),
        Path('/root/autodl-tmp/sttrack_m95_initial_origins_20260928'),False)['fit']
    bank=torch.load('/root/autodl-tmp/sttrack_m113_human_initialization_20261006/human_text.pt',map_location='cpu')
    assert bank['human_confirmed']
    model=SemanticChoicePrototype().to(device)
    model.initialize_parent(torch.load('/root/autodl-tmp/sttrack_m101_ab_native_pair_20260928/train_weight1/final.pt',map_location='cpu'))
    model.eval();original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    stored=snapshot(model,panel,bank,device)[0];result={}
    groups=dict(contiguous=torch.arange(64),random=torch.randperm(len(panel['key']),generator=torch.Generator().manual_seed(2027))[:64])
    for name,ids in groups.items():
        with torch.no_grad():
            full_no_grad=model(inputs(panel,ids,bank,'human_text',device))
            empty_no_grad=model(inputs(panel,ids,bank,'empty',device))
        with torch.enable_grad():
            full=model(inputs(panel,ids,bank,'human_text',device))
            empty=model(inputs(panel,ids,bank,'empty',device))
        result[name]=dict(cross_batch=compare(full['visual_selection_logits'],stored[ids]),
            same_batch_full_empty=compare(full['visual_selection_logits'],empty['visual_selection_logits']),
            same_batch_grad_nograd=compare(full['visual_selection_logits'],full_no_grad['visual_selection_logits']),
            same_batch_empty_grad_nograd=compare(empty['visual_selection_logits'],empty_no_grad['visual_selection_logits']),
            zero_update_selection_visual=compare(full['selection_logits'],full['visual_selection_logits']))
        assert all(result[name][k]['exact'] for k in ['same_batch_full_empty','same_batch_grad_nograd','same_batch_empty_grad_nograd','zero_update_selection_visual'])
    assert all(torch.equal(v.detach().cpu(),original[k]) for k,v in model.state_dict().items())
    receipt=dict(status='complete_M115_zero_update_batch_reference',torch_version=torch.__version__,seed=2027,
        optimizer_steps=0,parameters_buffers_exact=True,eval_mode=True,rows_per_group=64,groups=result)
    a.output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
