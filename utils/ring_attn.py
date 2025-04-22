"""
Ring Attention implementation for efficient sequence parallelism.

This module implements zigzag ring attention, a technique for distributing
attention computation across multiple GPUs in a way that balances computation
and minimizes communication overhead.

The key idea is to partition sequences in a zigzag pattern across devices:
- First half of blocks are distributed in order (0,1,2,...,N-1)
- Second half of blocks are distributed in reverse (N-1,...,2,1,0)

This creates a balanced workload and enables efficient all-gather operations.
"""

from typing import Optional, Dict, Any
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

# Global parameters for ring attention
DATA_PARAMS: Dict[str, Any] = {}

def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    cu_seq_lens_q: Optional[torch.Tensor] = None,
    cu_seq_lens_k: Optional[torch.Tensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
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

# Override the default flash attention implementation
transformers.modeling_flash_attention_utils._flash_attention_forward = _flash_attention_forward
ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward

def update_params_of_ring_attn(
    cu_seqlens: torch.Tensor,
    device_mesh: dist.DeviceMesh
) -> None:
    """
    Updates global parameters for ring attention.
    
    This function should be called before performing attention computation
    to set up the necessary parameters for zigzag ring attention.
    
    Args:
        cu_seqlens: Cumulative sequence lengths tensor
        device_mesh: Device mesh for distributed computation
    """
    # Calculate maximum sequence length from cumulative sequence lengths
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    
    # Update global parameters
    DATA_PARAMS.update({
        "group": device_mesh.get_group(),
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen
    })