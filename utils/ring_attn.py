from typing import Optional
import os
import torch
import torch.distributed as dist
from transformers.modeling_flash_attention_utils import (
    _flash_supports_window_size,
    is_flash_attn_greater_or_equal
)
import transformers
from ring_flash_attn.zigzag_ring_flash_attn_varlen import zigzag_ring_flash_attn_varlen_func
from ring_flash_attn.adapters.hf_adapter import flash_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    
DATA_PARAMS = {}

def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float=0.0,
    position_ids: Optional[torch.Tensor]=None,
    softmax_scale: Optional[float]=None,
    sliding_window: Optional[int]=None,
    use_top_left_mask: bool=False,
    softcap: Optional[float]=None,
    deterministic: Optional[bool]=None,
    cu_seq_lens_q: Optional[torch.Tensor]=None,
    cu_seq_lens_k: Optional[torch.Tensor]=None,
    max_length_q: Optional[int]=None,
    max_length_k: Optional[int]=None,
    target_dtype: Optional[torch.dtype]=None,
    **kwargs
):
    
    use_sliding_windows = (
        _flash_supports_window_size
        and sliding_window is not None
        and key_states.shape[1] > sliding_window
    )
    flash_kwargs = (
        {"window_size": (sliding_window, sliding_window)}
        if use_sliding_windows
        else {}
    )
    flash_kwargs = {}

    if is_flash_attn_greater_or_equal("2.4.1"):
        if deterministic is None:
            deterministic = (
                os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
            )

    flash_kwargs["deterministic"] = deterministic
    flash_kwargs["group"] = DATA_PARAMS["group"]

    return zigzag_ring_flash_attn_varlen_func(
        query_states.squeeze(0), 
        key_states.squeeze(0),
        value_states.squeeze(0),
        cu_seqlens=DATA_PARAMS["cu_seqlens"],
        max_seqlen=DATA_PARAMS["max_seqlen"],
        dropout_p=dropout,
        softmax_scale=softmax_scale,
        causal=True,
        **flash_kwargs
    )

transformers.modeling_flash_attention_utils._flash_attention_forward = _flash_attention_forward
ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward

def update_params_of_ring_attn(
    cu_seqlens,
    device_mesh
):

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

    DATA_PARAMS.update({
        "group": device_mesh.get_group(),
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen
    })


class RingAttnAllGather(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        tensor,
        cu_seqlens,
        device_mesh
    ):

        ctx.cu_seqlens = cu_seqlens
        ctx.device_mesh = device_mesh
        
        tensors = [
            torch.zeros_like(tensor)
            for _ in range(device_mesh.size())
        ]
        dist.all_gather(tensors, tensor, group=device_mesh.get_group())

        output_tensors = []
        for start_idx, end_idx in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            middle_idx = (start_idx + end_idx) // 2

            inorder_tensors = []
            reversed_tensors = []
            for tensor in tensors:
                inorder_tensors.append(tensor[:, start_idx:middle_idx])
                reversed_tensors.append(tensor[:, middle_idx:end_idx])
            
            output_tensors.append(
                torch.cat(inorder_tensors + reversed_tensors[::-1], -1)
            )

        return torch.cat(output_tensors, -1)

    @staticmethod
    def backward(ctx, grad_outputs):

        cu_seqlens = ctx.device_mesh.size() * ctx.cu_seqlens
        rank = ctx.device_mesh.get_local_rank()
        multiple_of = 2 * ctx.device_mesh.size()
        grad_inputs = []
        for start_idx, end_idx in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            grad_seq = grad_outputs[:, start_idx:end_idx]
            half_seqlen = grad_seq.shape[-1] // multiple_of
            grad_input = torch.cat((
                grad_seq[:, rank * half_seqlen:(rank + 1) * half_seqlen],
                grad_seq[:, (multiple_of - rank - 1) * half_seqlen:(multiple_of - rank) * half_seqlen]
            ), -1)
            grad_inputs.append(grad_input)

        return torch.cat(grad_inputs, -1), None, None
    
def ring_attn_all_gather(minibatch, device_mesh):
    return {
        k: RingAttnAllGather.apply(
            v, minibatch["cu_seqlens"], device_mesh
        )
        for k, v in minibatch.items()
        if k != "cu_seqlens"
    }