import torch
from torch.distributed.tensor.placement_types import Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module
)
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from transformers import Qwen2ForCausalLM

def prepare_lora_model(model, task_type: str, config):

    from peft import TaskType, LoraConfig, get_peft_model
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        task_type=getattr(TaskType, task_type),
        r=config.rank,
        lora_alpha=config.alpha,
        target_modules=config.target_modules,
        lora_dropout=config.dropout,
        bias="none"
    )
    return get_peft_model(model, lora_config)

def prepare_qwen2_tp_model(model, device_mesh):
    # TODO: support classification model
    for layer in model.model.layers:

        parallelize_plan = {
            "input_layernorm": SequenceParallel(),
            "self_attn.q_proj": ColwiseParallel(),
            "self_attn.k_proj": ColwiseParallel(),
            "self_attn.v_proj": ColwiseParallel(),
            "self_attn.o_proj": RowwiseParallel(
                output_layouts=Shard(1)
            ),
            "post_attention_layernorm": SequenceParallel(),
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.up_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(
                output_layouts=Shard(1)
            )
        }
        parallelize_module(
            module=layer,
            device_mesh=device_mesh,
            parallelize_plan=parallelize_plan
        )

    parallelize_plan = {
        "model.embed_tokens": ColwiseParallel(
            output_layouts=Shard(1)
        ),
        "model.norm": SequenceParallel(),
        "lm_head": ColwiseParallel()
    }
    parallelize_module(
        module=model,
        device_mesh=device_mesh,
        parallelize_plan=parallelize_plan
    )

def prepare_tp_model(model, device_mesh):

    assert model.config.num_key_value_heads % device_mesh.size() == 0, \
        f"Key and value heads {model.config.num_key_value_heads} must be divisible by tensor parallelism size {device_mesh.size()}."

    if isinstance(model, Qwen2ForCausalLM):
        prepare_qwen2_tp_model(model, device_mesh)
    else:
        raise NotImplementedError(
            f"Tensor parallelism is not supported for {model.__class__.__name__}."
        )
    
def prepare_dp_model(model, mixed_precision: bool, device_mesh):

    kwargs = {"device_mesh": device_mesh}
    if mixed_precision:
        kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16
        )
    for module in model.modules():
        if module.__class__.__name__ in model._no_split_modules or (
            isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings
        ):
            fully_shard(module, **kwargs)
    fully_shard(model, **kwargs) 