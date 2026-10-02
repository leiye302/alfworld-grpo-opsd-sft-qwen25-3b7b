#!/usr/bin/env python3
"""Prepare and launch a standalone ALFWorld expert-SFT experiment."""
from pathlib import Path
import argparse, contextlib, hashlib, json, os, platform, shutil, subprocess, sys, sysconfig, tarfile, time, urllib.error, urllib.parse, urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from checkpoint_retention import POLICY as CHECKPOINT_POLICY, verify_checkpoint
from hardware_profile import PROFILE as HARDWARE_PROFILE
OWNER, NAME = 'leiye302', 'alfworld-grpo-opsd-sft-qwen25-3b7b'
DEFAULT_RUNS = (('3b', 'sft'),)
VALIDATION_ID = 'alfworld-valid-seen-128-v1'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    temp.replace(path)

def download_asset(asset, path, verify=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == asset['bytes']:
        if not verify or digest(path) == asset['sha256']: return
    # Public Release downloads do not require a GitHub account, gh, or a token.
    # No Authorization header is sent to GitHub or its asset redirect hosts.
    url = asset['browser_download_url']
    assert url.startswith('https://github.com/'+OWNER+'/'+NAME+'/releases/download/')
    req = urllib.request.Request(url, headers={'Accept':'application/octet-stream',
        'User-Agent':'alfworld-repro-handoff'})
    partial = path.with_suffix(path.suffix+'.partial')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as response, partial.open('wb') as f:
                shutil.copyfileobj(response,f,8*1024*1024)
            if partial.stat().st_size != asset['bytes']:
                raise RuntimeError('Asset download is incomplete; rerun prepare.')
            break
        except (OSError, urllib.error.URLError, RuntimeError):
            if attempt == 2: raise
            time.sleep(2 ** attempt)
    if verify and digest(partial) != asset['sha256']:
        raise RuntimeError('Optional download checksum failed; incomplete file retained.')
    partial.replace(path)

def asset_member(dest, name):
    """Reject archive/receipt paths that escape the selected data directory."""
    from pathlib import PurePosixPath
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or '..' in relative.parts or
            '\\' in name or ':' in name):
        raise RuntimeError('Unsafe asset path')
    target = dest.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(dest.resolve()):
        raise RuntimeError('Asset path points outside its directory')
    return target

def extract_verified(path, dest, verify=False):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    receipt = dest/'ASSET_MEMBERS.json'
    if receipt.exists():
        manifest = json.loads(receipt.read_text(encoding='utf-8'))
        files = [(asset_member(dest, x['path']), x) for x in manifest['files']]
        if files and all(member.is_file() and member.stat().st_size==x['bytes'] and
                        (not verify or digest(member)==x['sha256']) for member,x in files): return
    with tarfile.open(path,'r:gz') as t:
        members = t.getmembers()
        for m in members:
            if not m.isfile(): raise RuntimeError('Assets must contain regular files only')
            asset_member(dest, m.name)
        t.extractall(dest,filter='data')
    manifest = json.loads(receipt.read_text(encoding='utf-8'))
    for x in manifest['files']:
        member = asset_member(dest, x['path'])
        if not member.is_file() or member.stat().st_size != x['bytes']:
            raise RuntimeError('Incomplete extracted asset: '+x['path'])
        if verify and digest(member) != x['sha256']:
            raise RuntimeError('Optional asset checksum failed: '+x['path'])

def environment(work, run):
    work, run = Path(work).resolve(), Path(run).resolve()
    env = os.environ.copy()
    env.update(SDAR_RUN_ROOT=str(run), SDAR_BASE_ROOT=str(work/'assets_bundle'),SDAR_QUOTA_BASE=str(run.parent),
        SDAR_RESPONSE_ONLY_LOGITS='1', SDAR_CHECKPOINT_RETENTION_LAST_ONLY='1',
        SDAR_GPU_STORAGE_PROFILE=HARDWARE_PROFILE['name'],
        SDAR_ALFWORLD_SQLITE=str(work/'assets_bundle/assets/alfworld.sqlite'),
        ALFWORLD_DATA=str(work/'assets_bundle/assets/alfworld'), FULLTRAJ_FIXED_VALIDATION='1',
        SDAR_ALLOWED_GPU_COUNT='8',
        PYTHONPATH=os.pathsep.join(map(str,[run,ROOT/'framework',ROOT/'runtime',work/'assets_bundle/vendor/deps',work/'assets_bundle/vendor/restored_deps'])),
        PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', TOKENIZERS_PARALLELISM='false',
        HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        RAY_USAGE_STATS_ENABLED='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MAX_JOBS='1',
        CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', NCCL_DEBUG='WARN',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:False', WANDB_MODE='offline')
    # The shipped tiny allocator links libcudart; locate the recipient's pip
    # CUDA runtime instead of relying on the source host's library runpath.
    cuda_lib=Path(sysconfig.get_paths()['purelib'])/'nvidia/cuda_runtime/lib'
    if cuda_lib.is_dir(): env['LD_LIBRARY_PATH']=str(cuda_lib)+os.pathsep+env.get('LD_LIBRARY_PATH','')
    directories = {
        'TMPDIR':'tmp','TMP':'tmp','TEMP':'tmp','RAY_TMPDIR':'ray',
        'HF_HOME':'cache/hf','HF_HUB_CACHE':'cache/hf/hub','HF_DATASETS_CACHE':'cache/hf/datasets',
        'XDG_CACHE_HOME':'cache/xdg','XDG_CONFIG_HOME':'cache/config','XDG_DATA_HOME':'cache/data','XDG_STATE_HOME':'cache/state',
        'TORCH_HOME':'cache/torch','TORCH_EXTENSIONS_DIR':'cache/extensions','VLLM_CACHE_ROOT':'cache/vllm','VLLM_CONFIG_ROOT':'cache/vllm_config',
        'TRITON_CACHE_DIR':'cache/triton','TORCHINDUCTOR_CACHE_DIR':'cache/inductor','CUDA_CACHE_PATH':'cache/cuda',
        'WANDB_DIR':'logs/wandb','WANDB_CACHE_DIR':'cache/wandb','WANDB_CONFIG_DIR':'cache/wandb_config',
        'TENSORBOARD_DIR':'logs/tensorboard','PIP_CACHE_DIR':'cache/pip','MPLCONFIGDIR':'cache/matplotlib',
        'NUMBA_CACHE_DIR':'cache/numba','PYTHONPYCACHEPREFIX':'cache/pycache','MODELSCOPE_CACHE':'cache/modelscope',
    }
    for key, relative in directories.items():
        p=run/relative; p.mkdir(parents=True,exist_ok=True); env[key]=str(p)
    (run/'logs').mkdir(exist_ok=True)
    (run/'fixed_validation').mkdir(exist_ok=True)
    shutil.copyfile(ROOT/'data/fixed_validation/manifest.json',run/'fixed_validation/manifest.json')
    return env

def arguments(work, run, size, method, resume=False):
    work, run = Path(work), Path(run)
    original=json.loads((ROOT/'configs/train_defaults.json').read_text(encoding='utf-8'))
    swaps={
      'data.train_files':str(work/'assets_bundle/data/text/train.parquet'),
      'data.val_files':str(work/'assets_bundle/data/text/test.parquet'),
      'actor_rollout_ref.model.path':str(work/'models'/('Qwen2.5-'+size.upper()+'-Instruct')),
      'trainer.experiment_name':size+'_'+method+'_150_fixed128',
      'trainer.n_gpus_per_node':'8','ray_init.num_gpus':'8','trainer.total_epochs':'150',
      'trainer.save_freq':'50',
      'trainer.default_local_dir':str(run/'checkpoints'),'trainer.rollout_data_dir':str(run/'rollouts'),
      'trainer.validation_data_dir':str(run/'validation'),'ray_init._temp_dir':str(run/'ray'),
      'ray_init._plasma_directory':str(run/'tmp'),'trainer.resume_mode':'auto' if resume else 'disable',
      'actor_rollout_ref.actor.expert_action_sft.enabled':'true' if method=='sft' else 'false',
      'actor_rollout_ref.actor.expert_action_sft.dataset_path':str(ROOT/'data/expert/episodes.jsonl'),
      'actor_rollout_ref.actor.expert_action_sft.trajectory_manifest_path':str(ROOT/'data/expert/manifest.json')}
    values=[]
    for arg in original:
        key,value=arg.split('=',1); bare=key.lstrip('+')
        values.append(key+'='+swaps.get(bare,value))
    return values

def prepare(work, size='3b', verify=False):
    work.mkdir(parents=True,exist_ok=True)
    os.environ.update(HF_HOME=str(work/'cache/hf'),HF_HUB_CACHE=str(work/'cache/hf/hub'),
                      HF_HUB_OFFLINE='0',HF_HUB_DISABLE_TELEMETRY='1')
    lock=json.loads((ROOT/'configs/assets.lock.json').read_text(encoding='utf-8'))
    archive=work/lock['name']
    download_asset(lock,archive,verify); extract_verified(archive,work/'assets_bundle',verify)
    from huggingface_hub import snapshot_download
    models=json.loads((ROOT/'configs/models.lock.json').read_text(encoding='utf-8'))
    m=models[size]
    target=work/'models'/('Qwen2.5-'+size.upper()+'-Instruct')
    print('Download only '+m['model_id']+' at '+m['revision'],flush=True)
    snapshot_download(repo_id=m['model_id'],revision=m['revision'],local_dir=target,
        allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])
    if verify:
        for name,meta in m['metadata_files'].items():
            assert digest(target/name)==meta['sha256'],name
    print('Preparation complete. Next: check --size '+size+' --gpu, then run --size '+size+' --method sft.',flush=True)

def check(work, gpu=False, size='3b'):
    if sys.version_info[:2] != (3,12) or platform.system()!='Linux' or platform.machine()!='x86_64':
        raise RuntimeError('Use Linux x86_64 CPython 3.12; the released environment binaries use this ABI.')
    env=environment(work,work/'preflight')
    subprocess.run([sys.executable,str(ROOT/'scripts/preflight.py'),'--work',str(work),'--size',size]+(['--gpu'] if gpu else []),env=env,cwd=ROOT/'framework',check=True)

def run_one(work, size, method, resume):
    run=work/'runs'/(size+'_'+method)
    if (run/'COMPLETE.json').exists():
        print('Already complete: '+run.name,flush=True); return
    if (run/'reports/launch_args.json').exists() and not resume:
        raise RuntimeError('Existing run: use --resume to preserve the native full checkpoint; fresh runs never overwrite it.')
    if resume:
        marker = run/'checkpoints/latest_checkpointed_iteration.txt'
        if not marker.is_file():
            raise RuntimeError('No complete checkpoint to resume; refusing to restart from the base model.')
        verify_checkpoint(run, int(marker.read_text(encoding='utf-8').strip()))
    env=environment(work,run)
    if len(str(run/'ray'))>75: raise RuntimeError('Choose a shorter --work path, e.g. /data/alfwork, for Ray Unix sockets.')
    for name in ('repo','runtime'):
        dest=run/name
        if not dest.exists(): dest.symlink_to(ROOT/('framework' if name=='repo' else 'runtime'),target_is_directory=True)
    for name in ('quota_io.py','fulltraj_fixed_val.py'):
        dest=run/name
        if not dest.exists(): dest.symlink_to(ROOT/'runtime'/name)
    args=arguments(work,run,size,method,resume)
    write_json(run/'reports/launch_args.json',args)
    try: commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError): commit='source-archive'
    expert_manifest=json.loads((ROOT/'data/expert/manifest.json').read_text(encoding='utf-8'))
    identity={'repository_commit':commit,'model':json.loads((ROOT/'configs/models.lock.json').read_text(encoding='utf-8'))[size],
        'method':method,'outer_iterations':150,'gpus':8,'expert_dataset_id':expert_manifest['dataset_id'],
        'expert_trajectories':len(expert_manifest['episodes']),
        'fixed_validation_id':VALIDATION_ID,'resume':resume,'checkpoint_policy':CHECKPOINT_POLICY,
        'hardware_profile':HARDWARE_PROFILE}
    write_json(run/'reports/identity.json',identity)
    entry=run/'train_entry.py'
    entry.write_text("import json,os,runpy,sys\nfrom pathlib import Path\nimport ray\nfrom storage_guard import for_run\nr=Path(os.environ['SDAR_RUN_ROOT'])\nfor_run().initialize()\nsys.argv=[str(r/'train_entry.py')]+json.loads((r/'reports/launch_args.json').read_text())\ntry: runpy.run_module('verl.trainer.main_sdar',run_name='__main__')\nfinally:\n if ray.is_initialized(): ray.shutdown()\n")
    # Native resume restores actor + Adam + LR + dataloader + RNG; no HF-only resume.
    with (run/'logs/driver.log').open('ab',buffering=0) as log:
        result=subprocess.run([sys.executable,'-u',str(entry)],env=env,cwd=ROOT/'framework',stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError('Training failed; inspect '+str(run/'logs/driver.log')+'; no recipe settings have been changed.')
    evaluations=[run/'fixed_validation/results'/f'iteration_{k:06d}.json' for k in range(0,151,10)]
    for path in evaluations:
        if not path.exists(): raise RuntimeError('Missing fixed evaluation '+str(path)+'; do not mark complete.')
        result=json.loads(path.read_text(encoding='utf-8'))
        assert result['validation_id']==VALIDATION_ID and result['tasks']==128
    metrics=run/'logs/metrics.jsonl'
    if not metrics.exists(): raise RuntimeError('Missing local metrics after training; do not mark complete.')
    rows=[json.loads(x) for x in metrics.read_text(encoding='utf-8').splitlines() if x.strip()]
    steps=[int(x.get('step',x.get('global_steps',-1))) for x in rows]
    if max(steps,default=-1)<150: raise RuntimeError('Process exited before iteration150; native checkpoint is retained.')
    verify_checkpoint(run,150)
    for retired in (50,100):
        if (run/'checkpoints'/f'global_step_{retired}').exists():
            raise RuntimeError('Superseded checkpoint still exists at '+str(retired)+'; inspect checkpoint_retention.jsonl.')
    write_json(run/'COMPLETE.json',dict(identity,completed_at=time.time(),latest_logged_iteration=max(steps),validation_files=len(evaluations)))
    print('Completed '+run.name,flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','check','plan','run','run-all'])
    parser.add_argument('--work',type=Path,default=ROOT/'work')
    parser.add_argument('--size',choices=['3b','7b'],default='3b')
    parser.add_argument('--method',choices=['baseline','sft'],default='sft')
    parser.add_argument('--verify-downloads',action='store_true',help='Optional automatic asset/model checksums; not needed for ordinary use.')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--gpu',action='store_true')
    opts=parser.parse_args(); work=opts.work.expanduser().resolve()
    if opts.command=='plan':
        for size,method in ((opts.size,opts.method),):
            print(json.dumps({'group':size+'_'+method,'args':arguments(work,work/'runs'/(size+'_'+method),size,method,opts.resume)}))
        return
    if opts.command=='prepare': prepare(work,opts.size,opts.verify_downloads); return
    if opts.command=='check': check(work,opts.gpu,opts.size); return
    import fcntl
    work.mkdir(parents=True,exist_ok=True)
    with (work/'training_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        groups=[(opts.size,opts.method)] if opts.command=='run' else DEFAULT_RUNS
        for size,method in groups:
            check(work,True,size)
            run_one(work,size,method,opts.resume)

if __name__=='__main__': main()
