"""Storage-only adaptation for eight A100 40GB GPUs.

Keep the native logical microbatch: its token-mean loss is reduced separately
inside each microbatch, so changing that size can change token weights.
"""

GIB = 1024**3
PROFILE = {
    'name': '8xa100_40gb',
    'gpus': 8,
    'minimum_gpu_memory_bytes': 35 * GIB,
    'actor_microbatch_per_gpu': 32,
    'logprob_microbatch_per_gpu': 32,
    'rollout_tensor_parallel_size': 2,
    'rollout_gpu_memory_utilization': 0.6,
    'model_saved_tensor_token_threshold': 16384,
    'head_saved_tensor_byte_threshold': 512 * 1024**2,
    'gpu_free_storage_margin_bytes': 8 * GIB,
    'mapped_host_reserve_gib_per_rank': 48,
    'minimum_available_host_memory_bytes': 450 * GIB,
}


def storage_flags(packed_tokens, logits_bytes, grad_enabled, free_gpu_bytes):
    """Move saved tensors to CPU sooner; never change tensor values or shapes."""
    if not grad_enabled:
        return False, False
    pressure = free_gpu_bytes < PROFILE['gpu_free_storage_margin_bytes']
    return (
        pressure or packed_tokens >= PROFILE['model_saved_tensor_token_threshold'],
        pressure or logits_bytes >= PROFILE['head_saved_tensor_byte_threshold'],
    )
