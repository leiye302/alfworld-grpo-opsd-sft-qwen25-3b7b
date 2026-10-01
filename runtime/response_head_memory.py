"""Keep native attention/batches; omit output-head rows discarded by the loss.

The original packed forward projects every prompt token into the full vocabulary,
then discards prompt scores. Native HF logits_to_keep selects only the original
response predecessor rows before that projection. No sequence is truncated and
the full-vocabulary normalization, entropy, reduction and backward remain native.
"""
import os,torch
from contextlib import contextmanager,nullcontext
from flash_attn.bert_padding import unpad_input,index_first_axis,pad_input
from einops import rearrange
from verl.utils.torch_functional import logprobs_from_logits
from managed_gradient import dense_gradient_buffer,page_head_backward_workspace

class _OriginalShapeLinearBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx,hidden,weight,selected,projected):
        ctx.save_for_backward(hidden,weight,selected)
        return projected
    @staticmethod
    def backward(ctx,gradient):
        hidden,weight,selected=ctx.saved_tensors
        # Preserve the native GEMM shapes and BF16 reduction axes. Only the
        # storage backing this temporary zero-filled matrix may page to CPU.
        with dense_gradient_buffer(gradient,(hidden.numel()//hidden.shape[-1],weight.shape[0])) as full_gradient:
            full_gradient.index_copy_(0,selected,gradient.reshape(-1,weight.shape[0]))
            grad_hidden=full_gradient.matmul(weight).reshape_as(hidden)
            grad_weight=full_gradient.t().matmul(hidden.reshape(-1,hidden.shape[-1]))
        return grad_hidden,grad_weight,None,None

@contextmanager
def original_linear_backward(actor_module,selected):
    if not torch.is_grad_enabled():
        yield
        return
    model=actor_module._fsdp_wrapped_module
    assert model.config.model_type=='qwen2' and model.lm_head.bias is None
    captured={}
    def capture(module,args,output):captured['hidden']=output[0]
    def head(module,args,output):
        return _OriginalShapeLinearBackward.apply(captured['hidden'],module.weight,selected,output.detach())
    first=model.model.register_forward_hook(capture)
    second=model.lm_head.register_forward_hook(head)
    try:yield
    finally:first.remove();second.remove()

def response_forward(self,micro_batch,temperature,calculate_entropy=False):
    native=self._native_complete_output_forward
    if (not self.use_remove_padding or self.use_fused_kernels or self.use_ulysses_sp
        or 'multi_modal_inputs' in micro_batch or micro_batch['position_ids'].dim()!=2):
        return native(micro_batch,temperature,calculate_entropy)
    response_length=micro_batch['responses'].size(-1)
    with torch.autocast(device_type=self.device_name,dtype=torch.bfloat16):
        input_ids=micro_batch['input_ids']
        batch_size,seqlen=input_ids.shape
        input_ids_rmpad,indices,*_=unpad_input(input_ids.unsqueeze(-1),micro_batch['attention_mask'])
        input_ids_rmpad=input_ids_rmpad.transpose(0,1)
        position_ids_rmpad=index_first_axis(
            rearrange(micro_batch['position_ids'].unsqueeze(-1),'b s ... -> (b s) ...'),indices
        ).transpose(0,1)
        # Original pad-back slicing is [:, -response_length-1:-1]. Preserve
        # those predictor rows, including the original packed rollover labels.
        columns=indices.remainder(seqlen)
        selected=torch.nonzero((columns>=seqlen-response_length-1)&(columns<seqlen-1),as_tuple=False).flatten()
        if selected.numel()==0:
            return native(micro_batch,temperature,calculate_entropy)
        labels=torch.roll(input_ids_rmpad,shifts=-1,dims=1).squeeze(0).index_select(0,selected)
        stage_model_saves=torch.is_grad_enabled() and (input_ids_rmpad.numel()>=32768
            or os.environ.get('SDAR_TEST_CPU_MODEL_SAVES')=='1')
        model_storage=torch.autograd.graph.save_on_cpu(pin_memory=True) if stage_model_saves else nullcontext()
        with model_storage,original_linear_backward(self.actor_module,selected):
            output=self.actor_module(input_ids=input_ids_rmpad,attention_mask=None,
                position_ids=position_ids_rmpad,use_cache=False,logits_to_keep=selected)
        logits=output.logits.squeeze(0)
        assert logits.shape[0]==selected.numel(),(logits.shape,selected.shape)
        if torch.is_grad_enabled() and temperature!=1.0:
            logits=logits/temperature
        elif not torch.is_grad_enabled():
            logits.div_(temperature)
        del output
        stage_head_saves=torch.is_grad_enabled() and (logits.numel()*logits.element_size()>=1024**3
            or os.environ.get('SDAR_TEST_CPU_HEAD_SAVES')=='1')
        storage=torch.autograd.graph.save_on_cpu(pin_memory=True) if stage_head_saves else nullcontext()
        with storage:
            log_probs_selected=logprobs_from_logits(logits=logits,labels=labels,
                inplace_backward=not calculate_entropy)
            log_probs=torch.zeros(input_ids_rmpad.numel(),device=logits.device,dtype=log_probs_selected.dtype)
            log_probs=log_probs.index_copy(0,selected,log_probs_selected)
            full_log_probs=pad_input(hidden_states=log_probs.unsqueeze(-1),indices=indices,batch=batch_size,seqlen=seqlen)
            log_probs=full_log_probs.squeeze(-1)[:,-response_length-1:-1]
            entropy=None
            if calculate_entropy:
                entropy_selected=self.compute_entropy_from_logits(logits)
                entropy_rmpad=torch.zeros(input_ids_rmpad.numel(),device=logits.device,dtype=entropy_selected.dtype)
                entropy_rmpad=entropy_rmpad.index_copy(0,selected,entropy_selected)
                full_entropy=pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),indices=indices,batch=batch_size,seqlen=seqlen)
                entropy=full_entropy.squeeze(-1)[:,-response_length-1:-1]
        if stage_head_saves:
            page_head_backward_workspace((log_probs,entropy),logits)
        return entropy,log_probs

def install(actor_class):
    if hasattr(actor_class,'_native_complete_output_forward'):return
    actor_class._native_complete_output_forward=actor_class._forward_micro_batch
    actor_class._forward_micro_batch=response_forward
