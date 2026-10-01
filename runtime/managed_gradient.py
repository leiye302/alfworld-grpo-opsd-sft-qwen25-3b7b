"""Same GPU operations, with mapped pinned host backing for large temporaries.

The historical module name is retained; this does not use CUDA managed-memory
page migration. Host backing is fixed before backward collectives start.
"""
import json,os,time
from contextlib import contextmanager
from pathlib import Path
import torch

_allocator=None
_pool=None
_primed=False

def record_storage_event(kind,**details):
    root=os.environ.get('SDAR_RUN_ROOT')
    if not root:return
    rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    with (Path(root)/'logs'/f'temporary_storage_rank{rank}.jsonl').open('a') as f:
        f.write(json.dumps({'kind':kind,'time':time.time(),'pid':os.getpid(),'rank':rank,**details})+'\n')

def new_managed_pool():
    global _allocator,_pool
    if _allocator is None:
        _allocator=torch.cuda.memory.CUDAPluggableAllocator(
            str(Path(__file__).with_name('libmanaged_gradient.so')),'managed_gradient_alloc','managed_gradient_free')
    if _pool is None:_pool=torch.cuda.MemPool(_allocator.allocator())
    return _pool

def prime_managed_pool():
    """Reserve mapped host backing before backward collectives are in flight."""
    global _primed
    if _primed:return
    torch.cuda.synchronize()
    with torch.cuda.use_mem_pool(new_managed_pool()):
        reserve=torch.empty(48*1024**3,dtype=torch.uint8,device='cuda')
    reserve[:1024].zero_()
    assert reserve[:1024].sum().item()==0
    del reserve
    _primed=True
    record_storage_event('host_buffer_ready',reserved_host_gib=48,backing='mapped_pinned_host')

def page_head_backward_workspace(outputs,input_tensor):
    """Change allocation backing inside native head backward nodes only."""
    if not torch.is_grad_enabled():return
    pool=new_managed_pool()
    record_storage_event('head_workspace_host_backing')
    boundary=input_tensor.grad_fn
    seen=set()
    def visit(node):
        if node is None or node is boundary or node in seen:return
        seen.add(node)
        contexts=[]
        def before(grad_outputs):
            context=torch.cuda.use_mem_pool(pool)
            context.__enter__()
            contexts.append(context)
        def after(grad_inputs,grad_outputs):
            contexts.pop().__exit__(None,None,None)
        node.register_prehook(before)
        node.register_hook(after)
        for parent,_ in node.next_functions:visit(parent)
    for output in outputs:
        if output is not None:visit(output.grad_fn)

@contextmanager
def dense_gradient_buffer(like,shape):
    global _allocator
    required=shape[0]*shape[1]*like.element_size()
    # Host reservations are not physical GPU capacity.
    available=torch.cuda.mem_get_info()[0]
    force=os.environ.get('SDAR_TEST_MANAGED_GRADIENT')=='1'
    if not force and required+2*1024**3<available:
        try:
            result=like.new_zeros(shape)
        except torch.cuda.OutOfMemoryError:
            result=None
        if result is not None:
            yield result
            return
    pool=new_managed_pool()
    record_storage_event('dense_gradient_host_backing',required_gib=required/1024**3,physical_free_gpu_gib=available/1024**3)
    with torch.cuda.use_mem_pool(pool):
        result=like.new_zeros(shape)
    yield result
