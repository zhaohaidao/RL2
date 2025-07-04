from torch.distributed.tensor.placement_types import Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module
)

# TODO: this is only for Qwen2.5
def prepare_tp_model(model, device_mesh):

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