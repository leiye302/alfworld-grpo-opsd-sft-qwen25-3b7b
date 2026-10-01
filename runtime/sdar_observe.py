"""Observe native optimizer calls without changing any tensors or RNG streams."""
import json,os,time
from pathlib import Path
import torch
from verl.workers.actor.dp_actor import DataParallelPPOActor

root=Path(os.environ['SDAR_RUN_ROOT'])
if torch.distributed.is_initialized():
    # Ray may assign rank after Python startup; apply the same private caches
    # before the native actor constructs compiled entropy and the vLLM engine.
    cache_rank=str(torch.distributed.get_rank())
    os.environ['TRITON_CACHE_DIR']=str(root/'cache'/('triton_rank'+cache_rank))
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(root/'cache'/('inductor_rank'+cache_rank))
native_step=DataParallelPPOActor._optimizer_step
native_update=DataParallelPPOActor.update_policy

def write(kind,record):
    record.update(kind=kind,time=time.time(),rank=torch.distributed.get_rank(),pid=os.getpid())
    with (root/'logs'/f'actor_audit_rank{record["rank"]}.jsonl').open('a') as f:
        f.write(json.dumps(record)+'\n')

def optimizer_step(self):
    params=[p for g in self.actor_optimizer.param_groups for p in g['params']]
    dtypes=sorted({str(p.dtype) for p in params})
    grad_dtypes=sorted({str(p.grad.dtype) for p in params if p.grad is not None})
    assert dtypes==['torch.float32'],dtypes
    self._audit_optimizer_steps=getattr(self,'_audit_optimizer_steps',0)+1
    snapshot=None
    if self._audit_optimizer_steps<=2:
        for p in params:
            if p.grad is not None and p.numel()>0:
                stride=max(1,p.numel()//4096)
                center=int(p.grad.detach().view(-1)[::stride].abs().argmax())*stride
                start=max(0,center-8);stop=min(p.numel(),center+9)
                snapshot=(p,start,stop,p.detach().view(-1)[start:stop].clone())
                break
    result=native_step(self)
    state_types=sorted({str(v.dtype) for state in self.actor_optimizer.state.values()
                        for k,v in state.items() if k in ('exp_avg','exp_avg_sq')})
    assert not state_types or state_types==['torch.float32'],state_types
    record={'optimizer_step':self._audit_optimizer_steps,'parameter_dtypes':dtypes,
            'gradient_dtypes':grad_dtypes,'adam_dtypes':state_types,
            'grad_norm':float(result),'lr':self.actor_optimizer.param_groups[0]['lr']}
    if snapshot:
        p,start,stop,before=snapshot
        difference=(p.detach().view(-1)[start:stop]-before).abs()
        record.update(sample_parameter_delta_max=float(difference.max()),sample_changed_elements=int(torch.count_nonzero(difference)))
    write('optimizer_step',record)
    return result

def update_policy(self,data):
    if os.environ.get('SDAR_RESPONSE_ONLY_LOGITS')=='1':
        from managed_gradient import prime_managed_pool
        prime_managed_pool()
    self._audit_rollout=getattr(self,'_audit_rollout',0)+1
    begin=getattr(self,'_audit_optimizer_steps',0)
    teacher=data.batch['teacher_log_probs']
    old=data.batch['old_log_probs']
    mask=data.batch['attention_mask'][:,-data.batch['responses'].shape[-1]:]
    gap=teacher-old
    write('update_start',{'rollout_iteration_in_process':self._audit_rollout,
          'local_decision_responses':int(old.shape[0]),'local_valid_tokens':int(mask.sum()),
          'normalized_mini_batch_size':int(self.config.ppo_mini_batch_size),
          'micro_batch_size':int(self.config.ppo_micro_batch_size_per_gpu),
          'response_head_memory_adapter':os.environ.get('SDAR_RESPONSE_ONLY_LOGITS')=='1',
          'gpu_storage_profile':os.environ.get('SDAR_GPU_STORAGE_PROFILE'),
          'teacher_gap_abs_mean':float((gap.abs()*mask).sum()/mask.sum().clamp_min(1)),
          'teacher_requires_grad':teacher.requires_grad,'old_requires_grad':old.requires_grad})
    result=native_update(self,data)
    write('update_end',{'rollout_iteration_in_process':self._audit_rollout,
          'optimizer_steps_this_iteration':getattr(self,'_audit_optimizer_steps',0)-begin,
          'gpu_allocated_bytes':torch.cuda.memory_allocated(),
          'gpu_reserved_bytes':torch.cuda.memory_reserved(),
          'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
          'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved()})
    return result

DataParallelPPOActor._optimizer_step=optimizer_step
DataParallelPPOActor.update_policy=update_policy

if os.environ.get('SDAR_RESPONSE_ONLY_LOGITS')=='1':
    from response_head_memory import install
    install(DataParallelPPOActor)
