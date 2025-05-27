from typing import List, Dict
import os
import math
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict
)
import transformers
from peft import LoraConfig, TaskType, get_peft_model
import wandb
from RL2.utils.seqlen_balance import get_seqlen_balanced_partitions
from RL2.utils.comm import gather_and_concat_list
        

class Worker:

    def __init__(self, config, device_mesh, train: bool):

        self.config = config
        self.device_mesh = device_mesh
        self.train = train

        self.sp_device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_dim_names=("dp", "sp"),
            mesh_shape=(
                device_mesh.size() // config.sp_size,
                config.sp_size
            )
        )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.model_name
        )

    def prepare_model_optimizer(self):

        if hasattr(self.config, "lora") and self.config.lora.rank > 0:
            # only support SFT and DPO for now
            self.model.enable_input_require_grads()

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora.rank,
                lora_alpha=self.config.lora.alpha,
                target_modules=self.config.lora.target_modules,
                lora_dropout=self.config.lora.dropout,
                bias="none"
            )
            self.model = get_peft_model(self.model, lora_config)

        if self.train and self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        kwargs = {
            "mesh": self.device_mesh
        }
        if self.train:
            kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                output_dtype=torch.bfloat16
            )
        for module in self.model.modules():
            if module.__class__.__name__ in self.model._no_split_modules or (isinstance(module, torch.nn.Embedding) and not self.model.config.tie_word_embeddings):
                fully_shard(module, **kwargs)
        fully_shard(self.model, **kwargs) 

        if self.train:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )

        self.offload_model_to_cpu()

    @torch.no_grad()
    def offload_model_to_cpu(self):
        if self.config.offload_model:
            for param in self.model.parameters():
                param.data = param.data.to("cpu", non_blocking=True)
    
    @torch.no_grad()
    def load_model_to_gpu(self):
        if self.config.offload_model:
            for param in self.model.parameters():
                param.data = param.data.to(
                    torch.cuda.current_device(), non_blocking=True
                )

    def scatter_and_pack_data_list(self, data_list, train: bool):

        if train:
            # If train, we partition data into multiple batches, where each
            # batch is used for an update and contains multiple minibatches.
            if self.device_mesh.get_rank() == 0:
                n_trajectories_per_update = len(data_list) // self.config.update_per_rollout
                return [
                    self.scatter_and_pack_data_list(
                        data_list[update * n_trajectories_per_update:(update + 1) * n_trajectories_per_update],
                        False
                    )
                    for update in range(self.config.update_per_rollout)
                ]
            else:
                return [
                    self.scatter_and_pack_data_list(None, False)
                    for _ in range(self.config.update_per_rollout)
                ]

        if self.device_mesh.get_rank() == 0:

            # We use ZigZag Ring Attention to partition sequences, where 
            # the length of each sequence needs to be multiple of 2 * 
            # sp size and each rank sequentially get the head and tail.
            # See https://zhuanlan.zhihu.com/p/683714620.
            multiple_of = 2 * self.sp_device_mesh["sp"].size()
            for ex in data_list:
                if ex["states"].shape[-1] % multiple_of == 0:
                    continue
                pad_tokens = multiple_of - ex["states"].shape[-1] % multiple_of
                for k, v in ex.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    ex[k] = torch.cat(
                        (v, torch.zeros((1, pad_tokens), dtype=v.dtype)),
                    -1)
                
            # We pack trajectories into minibatches for higher throughput.
            # To accommodate all trajectories, at least n_minibatches 
            # minibatches are needed.
            seq_len_list = [ex["states"].shape[-1] for ex in data_list]
            n_minibatches = math.ceil(
                sum(seq_len_list) / (
                    self.config.max_length_per_device * self.sp_device_mesh["sp"].size()
                )
            )
            
            # Every dp should has identical number of minibatches, thus the 
            # total number of minibatches must be a multiple of dp size.
            multiple_of = self.sp_device_mesh["dp"].size()
            if n_minibatches % multiple_of != 0:
                n_minibatches += (multiple_of - n_minibatches % multiple_of)
            n_minibatches_per_dp = n_minibatches // self.sp_device_mesh["dp"].size()

            # Partition data into n_minibatches balanced minibatches.
            partitions: List[List[int]] = get_seqlen_balanced_partitions(
                seq_len_list, k_partitions=n_minibatches, equal_size=False
            )
            # The n-th list contains data for rank n.
            data_lists = [
                [
                    [data_list[p] for p in partition]
                    for partition in partitions[rank * n_minibatches_per_dp:(rank + 1) * n_minibatches_per_dp]
                ]
                for rank in range(self.sp_device_mesh["dp"].size())
                for _ in range(self.sp_device_mesh["sp"].size())
            ]
        else:
            data_lists = [None for _ in range(self.device_mesh.size())]
        
        # Scatter data to all processes from rank 0.
        data_list = [None]
        dist.scatter_object_list(data_list, data_lists, src=0)
        data_list = data_list[0]
        
        rank = self.sp_device_mesh["sp"].get_local_rank()
        multiple_of = 2 * self.sp_device_mesh["sp"].size()
        minibatches = []
        for data in data_list:
            minibatch = {"uid": [ex.pop("uid") for ex in data]}
            for k in data[0].keys():
                tensors = []
                for ex in data:
                    tensor = ex[k]
                    # Every trajectory is evenly partitioned into 2 * sp 
                    # size segments and each rank sequentially get the head 
                    # and tail.
                    half_seqlen = tensor.shape[-1] // multiple_of
                    tensor = torch.cat((
                        tensor[:, rank * half_seqlen:(rank + 1) * half_seqlen],
                        tensor[:, (multiple_of - rank - 1) * half_seqlen: (multiple_of - rank) * half_seqlen]
                    ), -1)
                    tensors.append(tensor)
                minibatch[k] = torch.cat(tensors, -1).to(torch.cuda.current_device())
            # `update_params_of_ring_attn` requires `cu_seqlens` to mask 
            # the attention across trajectories within a minibatch. 
            seqlens = torch.IntTensor([tensor.shape[-1] for tensor in tensors])
            minibatch["cu_seqlens"] = torch.cumsum(
                torch.cat((torch.IntTensor([0]), seqlens)),
                0, dtype=torch.int32
            ).to(torch.cuda.current_device())
            minibatches.append(minibatch)
        
        return minibatches

    def resume_and_gather_data_list(self, minibatches):

        data_list = []
        for minibatch in minibatches:
            cu_seqlens = minibatch.pop("cu_seqlens")
            uid = minibatch.pop("uid")
            for idx, (start_idx, end_idx) in enumerate(
                zip(cu_seqlens[:-1], cu_seqlens[1:])
            ):
                ex = {}
                for k, v in minibatch.items():
                    tensor = v[:, start_idx:end_idx]
                    tensors = [
                        torch.zeros_like(tensor)
                        for _ in range(self.sp_device_mesh["sp"].size())
                    ]
                    dist.gather(
                        tensor,
                        tensors if self.sp_device_mesh["sp"].get_local_rank() == 0 else None,
                        group=self.sp_device_mesh["sp"].get_group(),
                        group_dst=0
                    )
                    # Devices with non-zero sp rank process zero tensors.
                    mid_idx = tensor.shape[-1] // 2
                    inorder_tensors, reversed_tensors = [], []
                    for tensor in tensors:
                        inorder_tensors.append(tensor[:, :mid_idx])
                        reversed_tensors.append(tensor[:, mid_idx:])
                    ex[k] = torch.cat((
                        inorder_tensors + reversed_tensors[::-1]
                    ), -1).to("cpu")

                if self.sp_device_mesh["sp"].get_local_rank() == 0:
                    length = torch.where(ex["eos_mask"])[1][0].item()
                    ex = {
                        k: v[:, :length + 1] for k, v in ex.items()
                    }
                    ex["uid"] = uid[idx]
                data_list.append(ex)
        
        if self.sp_device_mesh["sp"].get_local_rank() == 0:
            return gather_and_concat_list(
                data_list, self.sp_device_mesh["dp"]
            )
        else:
            return None
    
    @torch.no_grad()
    def optimizer_step(self):

        if self.config.offload_optimizer and self.optimizer.state:
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    state = self.optimizer.state[param]
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            state[key] = value.to(
                                torch.cuda.current_device(), non_blocking=True
                            )

        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.config.offload_optimizer:
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    state = self.optimizer.state[param]
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            state[key] = value.to("cpu", non_blocking=True)

    def log(self, metrics: Dict[str, List], step: int, device_mesh=None):

        metrics = {
            k: gather_and_concat_list(
                v, device_mesh if device_mesh is not None else self.device_mesh
            )
            for k, v in metrics.items()
        }
        
        if self.device_mesh.get_rank() == 0:
            wandb.log({
                k: torch.Tensor(v).mean().item()
                for k, v in metrics.items()
            }, step=step)

    def save(self, step):

        path = f"{self.config.save_dir}/step{step}"
        os.makedirs(path, exist_ok=True)
        options = StateDictOptions(
            full_state_dict=True, cpu_offload=True
        )
        state_dict = get_model_state_dict(
            self.model, options=options
        )
        if self.device_mesh.get_rank() == 0:

            self.tokenizer.save_pretrained(path)

            state_dict = {
                k: v.to(torch.bfloat16) for k, v in state_dict.items()
            }
            model_cls = getattr(
                transformers,
                self.model.__class__.__name__.removeprefix("FSDP")
            )
            with torch.device("meta"):
                model_to_save = model_cls._from_config(self.model.config)
            model_to_save.save_pretrained(
                path, state_dict=state_dict
            )

        if self.config.save_optimizer:
            torch.save(
                self.optimizer.state_dict(),
                f"{path}/optimizer_rank{self.device_mesh.get_rank()}.pt"
            )