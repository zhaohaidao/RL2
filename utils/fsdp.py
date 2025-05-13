# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp import MixedPrecisionPolicy

def shard_model(model, device_mesh):

    kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16
        ),
        "mesh": device_mesh
    }

    for name, module in model.named_modules():
        if module.__class__.__name__ in model._no_split_modules or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings):
            fully_shard(module, **kwargs)
    fully_shard(model, **kwargs)

@torch.no_grad()
def offload_fsdp_model_to_cpu(model, empty_cache: bool = True):
    for param in model.parameters():
        param.data = param.data.to("cpu", non_blocking=True)
    if empty_cache:
        torch.cuda.empty_cache()

@torch.no_grad()
def load_fsdp_model_to_gpu(model):
    for param in model.parameters():
        param.data = param.data.to(torch.cuda.current_device(), non_blocking=True)

@torch.no_grad()
def offload_fsdp_optimizer(optimizer):
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)

@torch.no_grad()
def load_fsdp_optimizer(optimizer, device_id):
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)

