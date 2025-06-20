from omegaconf import OmegaConf
import os
import math
import json
import asyncio
import importlib
from collections import defaultdict
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from tqdm.asyncio import tqdm
import wandb
from RL2.workers import Worker
from RL2.dataset import tokenize_messages
from RL2.algs import compute_baseline
from RL2.utils.comm import gather_and_concat_list
from RL2.utils.timing import time_logger


class Rollout(Worker):

    def __init__(self, config):

        self.config = config
        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cpu",
            mesh_dim_names=("dp", "tp"),
            mesh_shape=(
                dist.get_world_size() // self.config.tp_size,
                self.config.tp_size
            )
        )

        # Multi-node inference is not supported yet. We consider that 
        # FSDP suffers inferior efficiency for 100B+ models. If your 
        # model is smaller than 100B, then you probably should not use
        # multi-node inference for the maximal throughput.
        self.prepare_environment_variables()
        if self.device_mesh["tp"].get_local_rank() == 0:
            
            self.prepare_environment()
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name
            )

            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self.llm = Engine(
                model_path=config.model_name,
                dtype="bfloat16",
                tp_size=self.device_mesh["tp"].size(),
                mem_fraction_static=config.gpu_memory_utilization,
                enable_memory_saver=True,
                port=30000 + dist.get_rank()
            )
        
            self.train_sampling_params = OmegaConf.to_container(
                config.train_sampling_params
            )
            self.test_sampling_params = OmegaConf.to_container(
                config.test_sampling_params
            )

        dist.barrier()

    def prepare_environment_variables(self):

        if "TORCHELASTIC_USE_AGENT_STORE" in os.environ.keys():
            del os.environ["TORCHELASTIC_USE_AGENT_STORE"]
        monkey_patch_torch_reductions()
        cuda_visible_devices = self.device_mesh["tp"].size() * [None]
        dist.all_gather_object(
            cuda_visible_devices,
            os.environ["LOCAL_RANK"],
            self.device_mesh["tp"].get_group()
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible_devices)

    def prepare_environment(self):

        spec = importlib.util.spec_from_file_location(
            "custom_module", self.config.env_path
        )
        self.env = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.env)
        
    async def rollout(self, ex, train):

        messages, answer = ex["messages"], ex["answer"]
        metric = defaultdict(list)
        for turn in range(self.config.max_turns):

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tool=getattr(self.env, "TOOL", None),
                add_generation_prompt=True,
                tokenize=False
            )
            response = await self.llm.async_generate(
                prompt,
                sampling_params=self.train_sampling_params
                if train else self.test_sampling_params
            )
            messages.append(
                {"role": "assistant", "content": response["text"]}
            )

            meta_info = response["meta_info"]
            metric["response_length"].append(meta_info["completion_tokens"])
            metric["length_clip_ratio"].append(
                meta_info["finish_reason"]["type"] == "length"
            )

            # Do not invoke tools in the last turn.
            if turn + 1 == self.config.max_turns:
                break

            env_messages = self.env.interact(messages)
            # Terminate if no tool is invoked.
            if len(env_messages) == 0:
                break

            messages.extend(env_messages)

        reward = self.env.reward_fn(messages, answer)

        metric["n_turns"].append(turn + 1)
        metric["rewards"].append(reward)

        ex = tokenize_messages(self.tokenizer, messages)
        ex.update({
            "rewards": torch.FloatTensor((ex["states"].shape[-1] - 1) * [0] + [reward])
        })  

        return ex, messages, metric

    @time_logger("rollout")
    def __call__(self, data_list, train: bool, step: int):

        # Before each worker operation, the data is distributed from rank 0
        # and gathered before the next operation, which facilitates to do
        # model-agnostic operations, e.g., computing advantages, globally 
        # and guarantees the load balancing across all model computations.
        if self.device_mesh["tp"].get_local_rank() == 0:

            if dist.get_rank() == 0:
                data_per_dp = math.ceil(
                    len(data_list) / self.device_mesh["dp"].size()
                )
                data_lists = [
                    data_list[rank * data_per_dp:(rank + 1) * data_per_dp]
                    for rank in range(self.device_mesh["dp"].size())
                ]
            else:
                data_lists = [None for _ in range(self.device_mesh["dp"].size())]
            data_list = [None]
            dist.scatter_object_list(
                data_list,
                data_lists,
                group=self.device_mesh["dp"].get_group(),
                group_src=0    
            )
            data_list = data_list[0]

            loop = asyncio.get_event_loop()
            outputs = loop.run_until_complete(
                tqdm.gather(
                    *(self.rollout(ex, train) for ex in data_list),
                    desc="Rollout", position=1, leave=False,
                    disable=(dist.get_rank() != 0)
                )
            )
            if train:
                # If test, llm will soon be called again. See `Trainer.train`.
                self.llm.release_memory_occupation()

        dist.barrier()

        if self.device_mesh["tp"].get_local_rank() == 0:

            data_list, all_messages, metrics = map(list, zip(*outputs))

            metrics = {
                k: sum([metric[k] for metric in metrics], [])
                for k in metrics[0].keys()
            }
            suffix = "train" if train else "test"
            self.log(
                {f"{k}/{suffix}": v for k, v in metrics.items()},
                step=step,
                device_mesh=self.device_mesh["dp"]
            )

            data_list = gather_and_concat_list(
                data_list, self.device_mesh["dp"]
            )

            if dist.get_rank() == 0:
                tqdm.write(json.dumps(all_messages[0], indent=4))

                # Filter out groups with too low or too high average rewards, 
                # e.g., all trajectories within the group succeed or fail.
                _, baseline = compute_baseline(
                    data_list, self.config.responses_per_prompt
                )
                is_group_filtered = [
                    b <= self.config.group_filtering.lower or 
                    b >= self.config.group_filtering.upper
                    for b in baseline
                ]
                wandb.log({
                    f"group_filtering_ratio/{suffix}": sum(is_group_filtered) / len(is_group_filtered)
                }, step=step)
                return sum([
                    data_list[idx * self.config.responses_per_prompt:(idx + 1) * self.config.responses_per_prompt]
                    for idx, is_filtered in enumerate(is_group_filtered)
                    if not is_filtered
                ], [])
        
    def update(self, actor):

        torch.cuda.empty_cache()
        # or llm.resume_memory_occupation() may OOM
        if self.device_mesh["tp"].get_local_rank() == 0:
            self.llm.resume_memory_occupation()

        named_tensors = [(k, v) for k, v in actor.model.state_dict().items()]
        for idx, (name, tensor) in enumerate(named_tensors):
            serialized_tensor = MultiprocessingSerializer.serialize(
                tensor.full_tensor()
            )
            serialized_tensors = [
                None for _ in range(self.device_mesh["tp"].size())
            ] if self.device_mesh["tp"].get_local_rank() == 0 else None
            dist.gather_object(
                serialized_tensor,
                serialized_tensors,
                group_dst=0,
                group=self.device_mesh["tp"].get_group(),
            )
            if self.device_mesh["tp"].get_local_rank() == 0:
                self.llm.update_weights_from_tensor(
                    named_tensors=[(
                        name, LocalSerializedTensor(values=serialized_tensors)
                    )],
                    flush_cache=(idx == len(named_tensors) - 1)
                )
        dist.barrier()
        actor.offload_model_to_cpu()
        # Offload params here, or the params cannot be loaded.