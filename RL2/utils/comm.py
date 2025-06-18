import os
from datetime import timedelta
import torch
import torch.distributed as dist

def initialize_global_process_group(timeout_second=36000):
    
    dist.init_process_group("nccl", timeout=timedelta(seconds=timeout_second))

    local_rank = int(os.environ["LOCAL_RANK"])
    if dist.is_initialized():
        torch.cuda.set_device(local_rank)

def sum_across_processes(value):

    value = torch.Tensor([value]).to(torch.cuda.current_device())
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value.to("cpu").item()

def gather_and_concat_list(lst, device_mesh=None):

    if device_mesh is None:
        world_size = dist.get_world_size()
        is_dst = dist.get_rank() == 0
        dst = 0
        group = None
        group_dst = None
    else:
        world_size = device_mesh.size()
        is_dst = device_mesh.get_local_rank() == 0
        dst = None
        group = device_mesh.get_group()
        group_dst= 0
    
    lists = [None] * world_size if is_dst else None
    dist.gather_object(
        lst,
        lists,
        dst=dst,
        group=group,
        group_dst=group_dst
    )

    return sum(lists, []) if is_dst else None