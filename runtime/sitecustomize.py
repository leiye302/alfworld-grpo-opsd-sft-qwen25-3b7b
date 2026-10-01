import os
# GPU ranks compile concurrently on the shared filesystem. Keep compiler cache
# metadata private to each rank, including subprocess compiler workers.
if os.environ.get('SDAR_RUN_ROOT') and ('RANK' in os.environ or 'LOCAL_RANK' in os.environ):
    rank=os.environ.get('RANK',os.environ.get('LOCAL_RANK'))
    base=os.environ['SDAR_RUN_ROOT']+'/cache'
    os.environ['TRITON_CACHE_DIR']=base+'/triton_rank'+rank
    os.environ['TORCHINDUCTOR_CACHE_DIR']=base+'/inductor_rank'+rank
if os.environ.get('SDAR_ALFWORLD_SQLITE'):
    from asset_fs import install
    install()

# Install the JSON metric sink when the native tracker is imported naturally,
# including inside the Ray trainer process. No eager torch/CUDA import here.
if os.environ.get('SDAR_RUN_ROOT'):
    import importlib.abc, importlib.machinery, sys
    class _TrackerFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in ('verl.utils.tracking','vllm.platforms.interface','verl.trainer.ppo.ray_trainer'):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None:
                return None
            loader = spec.loader
            class Loader(importlib.abc.Loader):
                def create_module(self, spec):
                    return loader.create_module(spec)
                def exec_module(self, module):
                    loader.exec_module(module)
                    if fullname == 'vllm.platforms.interface':
                        from gpu_uuid_compat import install
                        install(module)
                        return
                    if fullname == 'verl.trainer.ppo.ray_trainer':
                        from storage_guard import install_trainer_hooks
                        install_trainer_hooks(module)
                        return
                    import json, time
                    original = module.Tracking.log
                    def log(self, data, step, backend=None):
                        result = original(self, data, step, backend)
                        with open(os.environ['SDAR_RUN_ROOT']+'/logs/metrics.jsonl', 'a') as f:
                            f.write(json.dumps({'step':step,'time':time.time(),'metrics':data},default=str)+'\n')
                        return result
                    module.Tracking.log = log
            spec.loader = Loader()
            return spec
    sys.meta_path.insert(0, _TrackerFinder())

# Task-local quota recovery; no model or sampling changes.
if os.environ.get("SDAR_RUN_ROOT"):
    import quota_io
    quota_io.install()
