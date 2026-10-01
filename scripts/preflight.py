#!/usr/bin/env python3
"""Actual CPU corpus/template/state checks, then optional recipient-GPU checks."""
from pathlib import Path
import argparse, hashlib, json, os, sqlite3, subprocess, sys, time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from handoff import DATA_SHA,MANIFEST_SHA,VAL_SHA,digest,write_json,arguments

def main():
    p=argparse.ArgumentParser(); p.add_argument('--work',type=Path,required=True);p.add_argument('--gpu',action='store_true')
    p.add_argument('--metadata-only',action='store_true',help='Maintainer CPU audit with only model config/tokenizer files.')
    o=p.parse_args(); work=o.work.resolve()
    import torch, numpy, yaml
    import importlib.metadata as metadata
    from transformers import AutoTokenizer
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from verl.trainer.ppo.expert_action_sft.mixed_dataset import MixedExpertTrajectoryDataset
    from verl.trainer.ppo.expert_action_sft.actor import validate_payload,build_microbatch
    from verl.trainer.ppo.expert_action_sft.schedule import coefficient
    versions={k:metadata.version(k) for k in ('torch','transformers','vllm','ray','alfworld','textworld','flash-attn','tensordict','numpy')}
    expected={'torch':'2.8.0','transformers':'4.57.3','vllm':'0.11.0','ray':'2.50.0','alfworld':'0.4.2','textworld':'1.6.2','flash-attn':'2.7.4.post1','tensordict':'0.8.3','numpy':'2.2.6'}
    for key,value in expected.items(): assert versions[key].split('+')[0]==value,(key,versions[key])
    assert digest(ROOT/'data/expert/episodes.jsonl')==DATA_SHA
    assert digest(ROOT/'data/expert/manifest.json')==MANIFEST_SHA
    manifest=json.loads((ROOT/'data/fixed_validation/manifest.json').read_text())
    assert manifest['sha256']==VAL_SHA
    canonical=dict(manifest);canonical.pop('sha256')
    assert hashlib.sha256(json.dumps(canonical,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()==VAL_SHA
    assert len(manifest['tasks'])==128 and len(set(t['task_id'] for t in manifest['tasks']))==128
    corpus={}
    for size in ('3b','7b'):
        model=work/'models'/('Qwen2.5-'+size.upper()+'-Instruct')
        tokenizer=AutoTokenizer.from_pretrained(model,use_fast=True,trust_remote_code=False,local_files_only=True)
        with initialize_config_dir(config_dir=str(ROOT/'framework/verl/trainer/config'),version_base=None):
            cfg=compose(config_name='ppo_trainer',overrides=arguments(work,work/'runs'/(size+'_sft'),size,'sft'))
            baseline=compose(config_name='ppo_trainer',overrides=arguments(work,work/'runs'/(size+'_baseline'),size,'baseline'))
        assert cfg.trainer.n_gpus_per_node==8 and cfg.env.rollout.n==8 and cfg.data.train_batch_size==16
        assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size==256
        assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu==32
        assert cfg.actor_rollout_ref.rollout.temperature==1 and cfg.actor_rollout_ref.actor.ppo_epochs==1
        assert cfg.trainer.total_epochs==150 and cfg.trainer.save_freq==25 and cfg.trainer.test_freq==10
        assert not baseline.actor_rollout_ref.actor.expert_action_sft.enabled
        assert cfg.algorithm.sdar.gate_beta==0 and cfg.algorithm.sdar.sdar_coef==.01
        options=cfg.actor_rollout_ref.actor.expert_action_sft
        dataset=MixedExpertTrajectoryDataset(options,tokenizer,{})
        assert len(dataset.episodes)==409
        total_tokens=0
        for record in dataset.encoded:
            assert record['sft_mask']==[1]*len(record['response_ids'])
            assert len(record['prompt_ids'])<=2048 and len(record['response_ids'])<=512
            assert record['response_ids'][-1]==tokenizer.eos_token_id
            total_tokens+=len(record['response_ids'])
        cases=[]
        for iteration in (1,25,49,50,100,150):
            coef=coefficient(options,iteration)
            if iteration>=50: assert coef==0;continue
            records,derived,draws=dataset.sample_trajectories(options.seed,iteration,16)
            payload={'schema_version':4,'mode':options.mode,'dataset_format':options.dataset_format,
                     'rollout_iteration':iteration,'coef':coef,'global_samples':len(records),'records':records,
                     'loss_reduction':'token-mean','trajectory_count':16,'trajectory_draws':draws,
                     'global_supervised_tokens':sum(len(r['response_ids']) for r in records)}
            validate_payload(payload,options)
            count=0
            for start in range(0,len(records),8):
                group=records[start:start+8]
                for rank,r in enumerate(group):
                    batch,mask=build_microbatch(group,rank,'cpu')
                    assert batch['input_ids'][batch['attention_mask'].bool()].tolist()==r['prompt_ids']+r['response_ids']
                    assert batch['responses'][mask.bool()].tolist()==r['response_ids']
                    count+=int(mask.sum())
            assert count==payload['global_supervised_tokens']
            cases.append({'iteration':iteration,'coefficient':coef,'trajectory_draws':16,'decisions':len(records),'tokens':count})
        if not o.metadata_only:
            index=json.loads((model/'model.safetensors.index.json').read_text())
            for shard in set(index['weight_map'].values()): assert (model/shard).is_file(),shard
        corpus[size]={'trajectories':409,'supervised_decisions':len(dataset.encoded),'supervised_tokens':total_tokens,
                      'chat_template_sha256':hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),'cases':cases}
    assert corpus['3b']['supervised_tokens']==corpus['7b']['supervised_tokens']
    # Use the actual ALFWorld wrapper, actual SQLite virtual filesystem and all
    # 128 seeds; no copied receipt can stand in for these reset assertions.
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment
    config=yaml.safe_load((ROOT/'framework/agent_system/environments/env_package/alfworld/configs/config_tw.yaml').read_text())
    base=get_environment(config['env']['type'])(config,train_eval='eval_in_distribution')
    resets=0
    for task in manifest['tasks']:
        with open(task['gamefile'],'rb') as f: assert hashlib.sha256(f.read()).hexdigest()==task['game_sha256']
        base.game_files=[task['gamefile']];base.num_games=1
        env=base.init_env(batch_size=1)
        try:
            env.seed(task['environment_seed']);obs,info=env.reset()
            state={'observation':obs[0],'admissible_commands':info['admissible_commands'][0]}
            state_sha=hashlib.sha256(json.dumps(state,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
            assert state_sha==task['initial_state_sha256'],task['task_id']
            resets+=1
        finally: env.close()
    import pyarrow.parquet as pq
    train_rows=pq.read_table(work/'assets_bundle/data/text/train.parquet').num_rows
    assert train_rows==16,('Native epoch/outer-iteration budget changed',train_rows)
    if o.gpu:
        assert torch.cuda.device_count()==8,'Eight allocated GPUs are required; do not reuse other users\' devices.'
        names=[torch.cuda.get_device_properties(i).name for i in range(8)]
        for i in range(8):
            props=torch.cuda.get_device_properties(i)
            assert 'A100' in props.name and props.total_memory>=75*1024**3,(i,props.name,props.total_memory)
        process=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        foreign=[int(x.strip()) for x in process.splitlines() if x.strip().isdigit() and int(x.strip())!=os.getpid()]
        assert not foreign,'GPU compute processes are already present. Do not stop another user\'s job.'
        import psutil
        available=psutil.virtual_memory().available
        # Container limits can be much smaller than the physical host reported
        # by psutil. Include the current cgroup budget before reserving host RAM.
        cgroup=Path('/sys/fs/cgroup')
        if (cgroup/'memory.max').is_file() and (cgroup/'memory.current').is_file():
            limit=(cgroup/'memory.max').read_text().strip()
            if limit!='max': available=min(available,int(limit)-int((cgroup/'memory.current').read_text()))
        assert available>=450*1024**3,'Eight ranks reserve384GiB pinned host memory; prepare at least450GiB actually available RAM (prefer1TiB installed).'
        import flash_attn
        from flash_attn import flash_attn_func
        q=torch.randn((1,16,4,64),device='cuda',dtype=torch.bfloat16,requires_grad=True)
        flash_attn_func(q,q,q,causal=True).float().sum().backward()
        assert torch.isfinite(q.grad).all()
        import ctypes
        ctypes.CDLL(str(ROOT/'runtime/libmanaged_gradient.so'))
    else:
        assert not torch.cuda.is_initialized(),'CPU corpus audit unexpectedly used a GPU'
    report={'passed':True,'metadata_only':o.metadata_only,'gpu_kernel_checked':o.gpu,'versions':versions,
            'corpus':corpus,'fixed_validation_sha256':VAL_SHA,'initial_states_verified':resets,
            'train_parquet_rows':train_rows,'training_iterations':0,'optimizer_updates':0,'time':time.time()}
    write_json(work/'preflight/acceptance.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
