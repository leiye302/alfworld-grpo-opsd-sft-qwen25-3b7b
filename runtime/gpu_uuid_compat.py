"""vLLM 0.11 NVML queries: accept UUID visibility without changing device order."""
import functools,json,os,time

@functools.lru_cache(None)
def physical_index(uuid):
    import pynvml
    pynvml.nvmlInit()
    try:
        result=pynvml.nvmlDeviceGetIndex(pynvml.nvmlDeviceGetHandleByUUID(uuid))
    finally:
        pynvml.nvmlShutdown()
    assert result in (0,1,2,3),('GPU outside authorized set',uuid,result)
    with open(os.environ['SDAR_RUN_ROOT']+'/logs/gpu_identity.jsonl','a') as f:
        f.write(json.dumps({'time':time.time(),'pid':os.getpid(),'uuid':uuid,'physical_index':result,'visible':os.environ.get('CUDA_VISIBLE_DEVICES')})+'\n')
    return result

def install(module):
    native=module.Platform.device_id_to_physical_device_id.__func__
    @classmethod
    def device_id_to_physical_device_id(cls,device_id):
        visible=os.environ.get(cls.device_control_env_var,'')
        if visible:
            name=visible.split(',')[device_id]
            if name.startswith('GPU-'):return physical_index(name)
        return native(cls,device_id)
    module.Platform.device_id_to_physical_device_id=device_id_to_physical_device_id
