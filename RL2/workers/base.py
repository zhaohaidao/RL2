from typing import List, Dict, Optional, Union
import os
import math
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict
)
from peft import LoraConfig, TaskType, get_peft_model
import wandb
from RL2.utils.fsdp import (
    shard_model,
    offload_fsdp_model_to_cpu,
    load_fsdp_model_to_gpu,
    offload_fsdp_optimizer,
    load_fsdp_optimizer
)
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

    def prepare_model_optimizer(self):

        if hasattr(self.config, "lora") and self.config.lora.rank > 0:
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

        shard_model(self.model, self.device_mesh)

        if self.train:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )

        self.offload_model_to_cpu()

    def offload_model_to_cpu(self):
        if self.config.offload_model:
            offload_fsdp_model_to_cpu(self.model)
    
    def load_model_to_gpu(self):
        if self.config.offload_model:
            load_fsdp_model_to_gpu(self.model)

    def offload_optimizer_to_cpu(self):
        if self.config.offload_optimizer:
            offload_fsdp_optimizer(self.optimizer)

    def load_optimizer_to_gpu(self):
        if self.config.offload_optimizer:
            load_fsdp_optimizer(self.optimizer, torch.cuda.current_device())

    def scatter_and_pack_data_list(
        self,
        data_list: Optional[List[Dict[str, torch.Tensor]]],
        train: bool
    ) -> Union[List[List[Dict[str, torch.Tensor]]], List[Dict[str, torch.Tensor]]]:
        """
        Distributes data across devices in a distributed training/inference setup with data parallelism and sequence parallelism.
        
        The distribution process works as follows:
        1. On rank 0: Partition the full dataset into balanced chunks based on sequence lengths
        2. Scatter these chunks to all processes across the device mesh (data parallelism dimension)
        3. Each process applies sequence parallelism by:
           - Padding sequences to required dimensions
           - Zigzag partitioning sequences across SP dimension for load balancing
           - Ensuring each SP rank processes different parts of the same sequences
        4. For training, further group minibatches into update batches
        
        Args:
            data_list: List of trajectory dictionaries (only populated on rank 0)
            train: Whether we're in training mode (affects batching)
            
        Returns:
            In training mode: List of lists of minibatches grouped by update step
            In inference mode: List of minibatches
        """
        
        # 1. Partition data on rank 0 and scatter to all processes
        # - Rank 0 divides data into balanced chunks based on sequence length
        # - Each process receives its assigned chunk via scatter operation
        partitioned_data = self._partition_and_scatter_data(data_list, train)
        
        # 2. Pack trajectories into minibatches with proper padding
        # - Each process pads its trajectories to required dimensions
        # - Applies zigzag partitioning for load balancing across SP dimension
        # - Concatenates trajectories into minibatches with sequence length tracking
        minibatches = self._pack_trajectories(partitioned_data)
        
        # 3. Group minibatches into batches if in training mode
        # - For training: organize minibatches into update groups
        # - For inference: return minibatches directly
        if train:
            return self._group_minibatches_for_training(minibatches)
        else:
            return minibatches

    def _partition_and_scatter_data(
        self, 
        data_list: Optional[List[Dict[str, torch.Tensor]]], 
        train: bool
    ) -> List[List[Dict[str, torch.Tensor]]]:
        """
        Partition data on rank 0 and scatter to all processes.

        Args:
            data_list: List of trajectory dictionaries (only populated on rank 0)
            train: Whether we're in training mode (affects batching)
            
        Returns:
            List of lists of minibatches grouped by update step
        """
        if self.device_mesh.get_rank() == 0:
            # Calculate sequence lengths and required minibatches
            seq_len_list = [ex["states"].shape[-1] for ex in data_list]
            n_minibatches = math.ceil(
                sum(seq_len_list) / (
                    self.config.max_length_per_device * self.sp_device_mesh["sp"].size()
                )
            )
            
            # At least n_minibatches minibatches are needed.
            # Every dp should has identical number of 
            # minibatches, thus the total number of minibatches 
            # must be a multiple of world size. Additinally, at 
            # training, the number of minibatches on each dp must 
            # be a multiple of updates so that they can be evenly 
            # devided into multiple batches, with each being used 
            # for an update.
            multiple_of = self.sp_device_mesh["dp"].size() * (self.config.update_per_rollout if train else 1)
            if n_minibatches % multiple_of != 0:
                n_minibatches += (multiple_of - n_minibatches % multiple_of)
            
            # Create balanced partitions
            partitions = get_seqlen_balanced_partitions(
                seq_len_list, k_partitions=n_minibatches, equal_size=False
            )

            data_lists: List[List[Dict[str, torch.Tensor]]] = [
                [data_list[p] for p in partition]
                for partition in partitions
            ] # Trajectories within an inner list will be packed into a minibatch.
            
            # Distribute minibatches across processes
            n_minibatches_per_process = n_minibatches // self.sp_device_mesh["dp"].size()
            data_lists = [
                data_lists[rank * n_minibatches_per_process:(rank + 1) * n_minibatches_per_process]
                for rank in range(self.sp_device_mesh["dp"].size())
                for _ in range(self.sp_device_mesh["sp"].size())
                # The n-th list contains data lists for rank n.
            ]
        else:
            data_lists = [None for _ in range(self.device_mesh.size())]
        
        # Scatter data to all processes
        data_list = [None]
        dist.scatter_object_list(data_list, data_lists, src=0)
        return data_list[0]

    def _pack_trajectories(self, data_list: List[List[Dict[str, torch.Tensor]]]) -> List[Dict[str, torch.Tensor]]:
        """
        Packs trajectories into minibatches with proper padding. This packing is mainly for the purpose for
        sequence parallelism.

        Args:
            data_list: List of lists of trajectory dictionaries
            
        Returns:
            List of minibatches
        """
        multiple_of = 2 * self.sp_device_mesh["sp"].size()
        rank = self.sp_device_mesh["sp"].get_local_rank()
        minibatches = []
        
        for data in data_list:
            minibatch = {"uid": [ex.pop("uid") for ex in data]}
            for k in data[0].keys():
                tensors = []
                for ex in data:
                    tensor = ex[k]
                    # Zigzag ring attention is used to balance 
                    # the load across devices, where the sequence 
                    # length needs to be multiple of 2 * 
                    # world_size and each rank sequentially get 
                    # the head and tail.
                    # See https://zhuanlan.zhihu.com/p/683714620.
                    tensor = self._pad_and_partition_tensor(tensor, multiple_of, rank)
                    tensors.append(tensor)
                minibatch[k] = torch.cat(tensors, -1).to(torch.cuda.current_device())
            
            # Create cumulative sequence lengths
            seqlens = torch.IntTensor([tensor.shape[-1] for tensor in tensors])
            minibatch["cu_seqlens"] = torch.cumsum(
                torch.cat((torch.IntTensor([0]), seqlens)),
                0, dtype=torch.int32
            ).to(torch.cuda.current_device())
            # Required by `update_params_of_ring_attn`.
            minibatches.append(minibatch)
        
        return minibatches

    def _pad_and_partition_tensor(self, tensor, multiple_of, rank):
        """
        Pads and partitions a tensor for sequence parallelism.
        
        Zigzag ring attention requires a specific data layout for efficient load balancing:
        
        +---------------------------------------------------------------+
        | GPU 0    | GPU 1    | GPU 2    | GPU 3    | ... | GPU N-1    |
        +---------------------------------------------------------------+
        | Block 0  | Block 1  | Block 2  | Block 3  | ... | Block N-1  |
        | Block 2N-1| Block 2N-2| Block 2N-3| Block 2N-4| ... | Block N  |
        +---------------------------------------------------------------+
        
        For example, with 4 GPUs (N=4), the block distribution would be:
        +-------------------------------------------+
        | GPU 0  | GPU 1  | GPU 2  | GPU 3  |
        +-------------------------------------------+
        | Block 0| Block 1| Block 2| Block 3|
        | Block 7| Block 6| Block 5| Block 4|
        +-------------------------------------------+
        
        Each rank processes two chunks:
        - One from the first half (at position equal to rank)
        - One from the second half (at position determined by (multiple_of - rank - 1))
        
        Args:
            tensor: Input tensor to pad and partition
            multiple_of: The multiple of the sequence length
            rank: The rank of the current process
            
        Returns:
            Padded and partitioned tensor
        """
        # Add padding if needed
        if tensor.shape[-1] % multiple_of != 0:
            padding_tokens = multiple_of - tensor.shape[-1] % multiple_of
            tensor = torch.cat(
                (tensor, torch.zeros((1, padding_tokens), dtype=tensor.dtype)),
            -1)
        
        # Zigzag partitioning for load balancing
        half_seqlen = tensor.shape[-1] // multiple_of
        return torch.cat((
            tensor[:, rank * half_seqlen:(rank + 1) * half_seqlen],
            tensor[:, (multiple_of - rank - 1) * half_seqlen: (multiple_of - rank) * half_seqlen]
        ), -1)

    def _group_minibatches_for_training(self, minibatches):
        """
        Groups minibatches into batches for training.

        Args:
            minibatches: List of minibatches
        
        Returns:
            List of batches
        """
        # Group minibatches into batches for training
        n_minibatches_per_update = len(minibatches) // self.config.update_per_rollout
        return [
            minibatches[update * n_minibatches_per_update:(update + 1) * n_minibatches_per_update]
            for update in range(self.config.update_per_rollout)
        ]

    def resume_and_gather_data_list(
        self,
        minibatches: List[Dict[str, torch.Tensor]]
    ) -> Optional[List[Dict[str, torch.Tensor]]]:
        """
        After model forward pass, reconstructs individual examples from minibatches and gathers them on rank 0.
        Sequence parallelism reorders the minibatches, so we need to gather and reorder them back.

        Args:
            minibatches: List of minibatches
        
        Returns:
            List of individual examples
        """
        # Step 1: Reconstruct individual examples from minibatches
        data_list = self._reconstruct_examples(minibatches)
        
        # Step 2: Gather and reorder data on rank 0
        return self._gather_and_reorder_data(data_list)

    def _reconstruct_examples(self, minibatches):
        """
        Reconstructs individual examples from minibatches and gathers them on rank 0.

        Args:
            minibatches: List of minibatches
        
        Returns:
            List of individual examples
        """
        data_list = []
        for minibatch in minibatches:
            cu_seqlens = minibatch.pop("cu_seqlens")
            uid = minibatch.pop("uid")
            for idx, (start_idx, end_idx) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
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
                        k: v[:, :length + 1]
                        for k, v in ex.items()
                    }
                    ex["uid"] = uid[idx]
                data_list.append(ex)
        
        return data_list

    def _gather_and_reorder_data(self, data_list):
        # Gather data across DP dimension if on SP rank 0
        if self.sp_device_mesh["sp"].get_local_rank() == 0:
            return gather_and_concat_list(data_list, self.sp_device_mesh["dp"])
        else:
            return None

    def log(self, metrics: Dict[str, List], step: int, device_mesh=None):

        metrics = {
            k: gather_and_concat_list(
                v,
                device_mesh if device_mesh is not None else self.device_mesh
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
            self.model.save_pretrained(
                path, state_dict=state_dict
            )

        torch.save(
            self.optimizer.state_dict(),
            f"{path}/optimizer_rank{self.device_mesh.get_rank()}.pt"
        )