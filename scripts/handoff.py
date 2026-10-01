#!/usr/bin/env python3
"""Prepare and launch the two expert-SFT runs without changing the frozen trainer."""
from pathlib import Path
import argparse, contextlib, hashlib, json, os, platform, shutil, subprocess, sys, sysconfig, tarfile, time, urllib.error, urllib.parse, urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from checkpoint_retention import POLICY as CHECKPOINT_POLICY, verify_checkpoint
OWNER, NAME = 'leiye302', 'alfworld-grpo-opsd-sft-qwen25-3b7b'
DEFAULT_RUNS = (('3b', 'sft'), ('7b', 'sft'))
DATA_SHA = '3aa3796f8d07a3d8fc091c969281b9e22ae2249935038af575545b0d87a623b7'
MANIFEST_SHA = 'e080ddeac481c19c1e76a000d3a87ee1509a719737869274ecbe9d221d01100b'
VAL_SHA = 'd4bd4aa2d5e553e5eb9b61f60450315ad7d2f26032428e13c8f2e2d806aa4cc8'
# Logical path only: the bundled read-only SQLite adapter serves every byte.
# Keeping it identical preserves the signed fixed-validation manifest verbatim.
LOGICAL_ALFWORLD = '/mnt/zixuan/test/VLA_test/VLA_SDAR_GRPO_OPSD15_20260915/assets/alfworld'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)

def credentials():
    for name in ('GH_TOKEN','GITHUB_TOKEN'):
        if os.environ.get(name): return os.environ[name]
    try:
        return subprocess.check_output(['gh','auth','token'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        raise RuntimeError('Private repository assets require GH_TOKEN or gh auth login; use your own authorized GitHub account.') from None

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl): return None

def download_asset(asset, path):
    if path.exists() and digest(path) == asset['sha256']: return
    req = urllib.request.Request(asset['api_url'], headers={'Authorization':'Bearer '+credentials(),
        'Accept':'application/octet-stream','User-Agent':'alfworld-repro-handoff'})
    # Bearer credentials are sent only to api.github.com. The signed redirect is
    # fetched in a separate request with no Authorization header.
    assert asset['api_url'].startswith('https://api.github.com/repos/'+OWNER+'/'+NAME+'/releases/assets/')
    try:
        response = urllib.request.build_opener(NoRedirect()).open(req, timeout=90)
    except urllib.error.HTTPError as e:
        if e.code not in (301,302,303,307,308): raise
        target = e.headers['Location']; host = urllib.parse.urlparse(target).hostname or ''
        assert target.startswith('https://') and (host.endswith('.githubusercontent.com') or host.endswith('.github.com'))
        response = urllib.request.urlopen(urllib.request.Request(target,headers={'User-Agent':'alfworld-repro-handoff'}),timeout=90)
    partial = path.with_suffix(path.suffix+'.partial')
    with response, partial.open('wb') as f:
        shutil.copyfileobj(response,f,8*1024*1024)
    if partial.stat().st_size != asset['bytes'] or digest(partial) != asset['sha256']:
        raise RuntimeError('Asset size/SHA mismatch; incomplete file retained, training will not start.')
    partial.replace(path)

def extract_verified(path, dest):
    receipt = dest/'ASSET_MEMBERS.json'
    if receipt.exists():
        manifest = json.loads(receipt.read_text())
        if all((dest/x['path']).is_file() and (dest/x['path']).stat().st_size==x['bytes'] and digest(dest/x['path'])==x['sha256'] for x in manifest['files']): return
    with tarfile.open(path,'r:gz') as t:
        members = t.getmembers()
        for m in members:
            assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
        t.extractall(dest,filter='data')
    manifest = json.loads(receipt.read_text())
    for x in manifest['files']:
        if digest(dest/x['path']) != x['sha256']: raise RuntimeError('Asset member changed: '+x['path'])

def environment(work, run):
    work, run = Path(work).resolve(), Path(run).resolve()
    env = os.environ.copy()
    env.update(SDAR_RUN_ROOT=str(run), SDAR_BASE_ROOT=str(work/'assets_bundle'),SDAR_QUOTA_BASE=str(run.parent),
        SDAR_RESPONSE_ONLY_LOGITS='1', SDAR_CHECKPOINT_RETENTION_LAST_ONLY='1',
        SDAR_ALFWORLD_SQLITE=str(work/'assets_bundle/assets/alfworld.sqlite'),
        ALFWORLD_DATA=LOGICAL_ALFWORLD, FULLTRAJ_FIXED_VALIDATION='1',
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
    original=json.loads((ROOT/'configs/recorded_1p5b_launch_args.json').read_text())
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

def prepare(work):
    work.mkdir(parents=True,exist_ok=True)
    os.environ.update(HF_HOME=str(work/'cache/hf'),HF_HUB_CACHE=str(work/'cache/hf/hub'),
                      HF_HUB_OFFLINE='0',HF_HUB_DISABLE_TELEMETRY='1')
    lock=json.loads((ROOT/'configs/assets.lock.json').read_text())
    archive=work/lock['name']
    download_asset(lock,archive); extract_verified(archive,work/'assets_bundle')
    from huggingface_hub import snapshot_download
    models=json.loads((ROOT/'configs/models.lock.json').read_text())
    for size in ('3b','7b'):
        m=models[size]
        target=work/'models'/('Qwen2.5-'+size.upper()+'-Instruct')
        print('Download only '+m['model_id']+' at '+m['revision'],flush=True)
        snapshot_download(repo_id=m['model_id'],revision=m['revision'],local_dir=target,
            allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])
        # Config/tokenizer identities are locked separately from giant weights.
        for name,meta in m['metadata_files'].items():
            assert digest(target/name)==meta['sha256'],name
    print('Preparation complete. Next: check, then run-all.',flush=True)

def check(work, gpu=False):
    if sys.version_info[:2] != (3,12) or platform.system()!='Linux' or platform.machine()!='x86_64':
        raise RuntimeError('Use Linux x86_64 CPython 3.12; the released environment binaries use this ABI.')
    env=environment(work,work/'preflight')
    subprocess.run([sys.executable,str(ROOT/'scripts/preflight.py'),'--work',str(work)]+(['--gpu'] if gpu else []),env=env,cwd=ROOT/'framework',check=True)

def run_one(work, size, method, resume):
    run=work/'runs'/(size+'_'+method)
    if (run/'COMPLETE.json').exists():
        print('Already complete: '+run.name,flush=True); return
    if (run/'reports/launch_args.json').exists() and not resume:
        raise RuntimeError('Existing run: use --resume to preserve the native full checkpoint; fresh runs never overwrite it.')
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
    identity={'repository_commit':commit,'model':json.loads((ROOT/'configs/models.lock.json').read_text())[size],
        'method':method,'outer_iterations':150,'gpus':8,'expert_sha256':DATA_SHA,'expert_manifest_sha256':MANIFEST_SHA,
        'fixed_validation_sha256':VAL_SHA,'resume':resume,'checkpoint_policy':CHECKPOINT_POLICY}
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
        result=json.loads(path.read_text())
        assert result['manifest_sha256']==VAL_SHA and result['tasks']==128
    metrics=run/'logs/metrics.jsonl'
    if not metrics.exists(): raise RuntimeError('Missing local metrics after training; do not mark complete.')
    rows=[json.loads(x) for x in metrics.read_text().splitlines() if x.strip()]
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
    parser.add_argument('--size',choices=['3b','7b'])
    parser.add_argument('--method',choices=['baseline','sft'])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--gpu',action='store_true')
    opts=parser.parse_args(); work=opts.work.expanduser().resolve()
    if opts.command=='plan':
        for size,method in DEFAULT_RUNS:
            print(json.dumps({'group':size+'_'+method,'args':arguments(work,work/'runs'/(size+'_'+method),size,method,opts.resume)}))
        return
    if opts.command=='prepare': prepare(work); return
    if opts.command=='check': check(work,opts.gpu); return
    import fcntl
    work.mkdir(parents=True,exist_ok=True)
    with (work/'training_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check(work,True)
        groups=[(opts.size,opts.method)] if opts.command=='run' else DEFAULT_RUNS
        if any(s is None or m is None for s,m in groups): parser.error('run requires --size and --method')
        for size,method in groups: run_one(work,size,method,opts.resume)

if __name__=='__main__': main()
