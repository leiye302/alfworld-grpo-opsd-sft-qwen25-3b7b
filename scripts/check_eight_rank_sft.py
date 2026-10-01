"""CPU eight-rank regression of the actual auxiliary code against a global CE oracle.

This tests the hardware guard, padding, two optimizer updates, global token mean,
and dummy collective participation. It is not an A100 FSDP/kernel test.
"""
import json, os
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn.parallel import DistributedDataParallel
from omegaconf import OmegaConf
from verl.trainer.ppo.expert_action_sft.actor import prepare_auxiliary,backward_auxiliary

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.3,dtype=torch.float64))
    def forward(self,batch):
        target=batch['responses'].to(torch.float64)
        return -((self.weight-target*.01)**2)

def main():
    torch.distributed.init_process_group('gloo')
    rank,world=torch.distributed.get_rank(),torch.distributed.get_world_size()
    assert world==8
    records=[]
    for i in range(11):
        targets=[i+2]*((i%3)+1)
        records.append({'sample_id':f'x:{i}','prompt_ids':[1]*(i%4+1),'response_ids':targets,
                        'action_mask':[1]*len(targets),'sft_mask':[1]*len(targets),'pad_token_id':0})
    tokens=sum(len(r['response_ids']) for r in records)
    config=OmegaConf.create({'expert_action_sft':{'enabled':True,'coef':.1,'mode':'complete_trajectory_reasoning_action',
             'decay_end_iteration':50,'decay_schedule':'cosine','trajectories_per_round':1,'loss_reduction':'token-mean'},
             'use_dynamic_bsz':False,'ppo_epochs':1,'use_sdar_loss':True})
    ddp=DistributedDataParallel(Model())
    actor=SimpleNamespace(config=config,use_ulysses_sp=False,actor_module=ddp,device_name='cpu')
    actor._forward_micro_batch=lambda batch,temperature,calculate_entropy:(None,ddp(batch))
    optimizer=torch.optim.SGD(ddp.parameters(),lr=.02)
    oracle=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
    cases=[]
    for iteration in (1,2):
        from verl.trainer.ppo.expert_action_sft.schedule import coefficient
        coef=coefficient(config.expert_action_sft,iteration)
        payload={'schema_version':3,'mode':'complete_trajectory_reasoning_action','rollout_iteration':iteration,'coef':coef,
          'global_samples':len(records),'records':records,'loss_reduction':'token-mean','trajectory_count':1,
          'trajectory_draws':[{'draw':0,'start':0,'decisions':11,'episode_id':'x','supervised_tokens':tokens}],
          'global_supervised_tokens':tokens}
        data=SimpleNamespace(meta_info={'expert_action_sft':payload},non_tensor_batch={},batch=SimpleNamespace(batch_size=[32]))
        plan=prepare_auxiliary(actor,data);assert plan['rank']==rank
        optimizer.zero_grad();ddp.module.weight.grad=torch.tensor(.07,dtype=torch.float64)
        state=torch.get_rng_state().clone()
        metrics=backward_auxiliary(actor,plan)
        assert torch.equal(state,torch.get_rng_state()),'Auxiliary computation consumed Student RNG'
        expected=coef*sum(((oracle-y*.01)**2) for r in records for y in r['response_ids'])/tokens
        grad=torch.autograd.grad(expected,oracle)[0]+.07
        assert torch.allclose(ddp.module.weight.grad,grad,atol=1e-12,rtol=1e-12),(rank,ddp.module.weight.grad,grad)
        optimizer.step();oracle=(oracle-.02*grad).detach().requires_grad_(True)
        assert torch.allclose(ddp.module.weight,oracle,atol=1e-12,rtol=1e-12)
        cases.append({'iteration':iteration,'coefficient':coef,'weight':float(oracle.detach()),'global_tokens':tokens})
    if rank==0:
        output=Path(os.environ['SDAR_RUN_ROOT'])/'reports/eight_rank_sft_cpu.json';output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps({'passed':True,'ranks':8,'optimizer_updates':2,'cases':cases,
             'actual_auxiliary_guard_and_backward_exercised':True,'GPU_FSDP_tested':False},indent=2)+'\n')
        print(output)
    torch.distributed.destroy_process_group()

if __name__=='__main__':main()
