from typing import Dict, List, Tuple
import os
import asyncio
import requests
import importlib
import concurrent.futures
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModelForCausalLM
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from tqdm import tqdm
from RL2.workers.base import Worker
from RL2.algs import compute_kl_term
from RL2.utils.ring_attn import update_params_of_ring_attn
from RL2.utils.comm import gather_and_concat_list, sum_across_processes


class Actor(Worker):

    def __init__(self, config, device_mesh, train: bool):
        super().__init__(config, device_mesh, train)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            attn_implementation="flash_attention_2"
        )
        
        self.prepare_model_optimizer()
        if hasattr(config, "rollout") and train:
            self.prepare_reward_fn()
            self.prepare_inference_engine()
    
    def prepare_reward_fn(self):

        spec = importlib.util.spec_from_file_location(
            "custom_module", self.config.rollout.reward_fn_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.reward_fn = module.reward_fn

    def prepare_inference_engine(self):

        self.rollout_device_mesh = dist.device_mesh.init_device_mesh(
            "cpu",
            mesh_dim_names=("dp", "tp"),
            mesh_shape=(
                self.device_mesh.size() // self.config.rollout.tp_size,
                self.config.rollout.tp_size
            )
        )

        if "TORCHELASTIC_USE_AGENT_STORE" in os.environ.keys():
            del os.environ["TORCHELASTIC_USE_AGENT_STORE"]
        monkey_patch_torch_reductions()
        cuda_visible_devices = self.rollout_device_mesh["tp"].size() * [None]
        dist.all_gather_object(
            cuda_visible_devices,
            os.environ["RANK"],
            self.rollout_device_mesh["tp"].get_group()
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible_devices)

        if self.rollout_device_mesh["tp"].get_local_rank() == 0:
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name
            )

            self.llm = Engine(
                model_path=self.config.model_name,
                dtype="bfloat16",
                tp_size=self.rollout_device_mesh["tp"].size(),
                mem_fraction_static=self.config.rollout.gpu_memory_utilization,
                enable_memory_saver=True
            )
        
            self.train_sampling_params = {
                "temperature": self.config.rollout.train_temperature,
                "max_new_tokens": self.config.rollout.max_response_length
            }

            self.test_sampling_params = {
                "temperature": self.config.rollout.test_temperature,
                "max_new_tokens": self.config.rollout.max_response_length
            }

        dist.barrier()

    async def single_rollout(self, messages, train):

        stat = defaultdict(list)
        for turn in range(self.config.rollout.n_turns):

            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            response = await self.llm.async_generate(
                prompt, sampling_params=self.train_sampling_params if train else self.test_sampling_params
            )
            messages.append(
                {"role": "assistant", "content": response["text"]}
            )

            meta_info = response["meta_info"]
            stat["response_length"].append(meta_info["completion_tokens"])
            stat["rollout_length_clip_ratio"].append(meta_info["finish_reason"]["type"] == "length")

            # Do not invoke tools in the last turn.
            if turn + 1 == self.config.rollout.n_turns:
                break

            # The environment should be launched as a service which 
            # receives previous messages return a *list of messages*, as 
            # the model may call multiple tools simultaneously. An empty 
            # list should be returned if no function is called.
            env_messages = requests.post(
                self.config.env_url, json=messages
            ).json()

            # Terminate if no tool is invoked.
            if len(env_messages) == 0:
                break

            messages.extend(env_messages)

        stat["n_turns"].append(turn + 1)
        return stat

    def tokenize_messages(self, ex, reward):

        messages = ex["messages"]
        states, actions, action_mask = [], [], []
        for idx, message in enumerate(messages):

            state = self.tokenizer.apply_chat_template(
                messages[:idx + 1],
                add_generation_prompt=idx + 1 < len(messages) and messages[idx + 1]["role"] == "assistant"
            )[len(states):]

            states.extend(state)
            actions.extend(
                state if message["role"] == "assistant"
                else len(state) * [0]
            )
            action_mask.extend(len(state) * [
                1 if message["role"] == "assistant" else 0
            ])

        states = states[:-1]
        actions = actions[1:]
        action_mask = action_mask[1:]

        rewards = (len(states) - 1) * [0] + [reward]
        position_ids = list(range(len(states)))
        eos_mask = (len(states) - 1) * [0] + [1]

        return {
            "uid": ex["uid"],
            "states": torch.LongTensor(states).unsqueeze(0),
            "actions": torch.LongTensor(actions).unsqueeze(0),
            "rewards": torch.FloatTensor(rewards).unsqueeze(0),
            "position_ids": torch.LongTensor(position_ids).unsqueeze(0),
            "action_mask": torch.LongTensor(action_mask).unsqueeze(0),
            "eos_mask": torch.LongTensor(eos_mask).unsqueeze(0)
        }

    def rollout(self, data_list, train: bool, step: int):

        if self.rollout_device_mesh["tp"].get_local_rank() == 0:

            loop = asyncio.get_event_loop()
            stats: Tuple[Dict[List]] = loop.run_until_complete(
                asyncio.gather(*(
                    self.single_rollout(ex["messages"], train)
                    for ex in data_list
                ))
            )
            stats: Dict[List] = {
                k: sum([stat[k] for stat in stats], [])
                for k in stats[0].keys()
            }

            if train:
                # If test, llm will soon be called again. See `Trainer.train`.
                self.llm.release_memory_occupation()
        
        dist.barrier()

        if self.rollout_device_mesh["tp"].get_local_rank() == 0:

            if self.config.rollout.multi_thread_scoring:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    rewards = list(executor.map(
                        lambda ex: self.reward_fn(ex["messages"], ex["answer"]),
                        data_list
                    ))
            else:
                # Math-Verify is thread-unsafe. See https://github.com/huggingface/Math-Verify/issues/42.
                rewards = [
                    self.reward_fn(ex["messages"], ex["answer"])
                    for ex in data_list
                ]

            with concurrent.futures.ThreadPoolExecutor() as executor:
                data_list = list(executor.map(
                    lambda ex, reward: self.tokenize_messages(ex, reward),
                    data_list, rewards
                ))
            
            suffix = "train" if train else "test"
            self.log(
                {
                    f"reward/{suffix}": rewards,
                    **{
                        f"{prefix}/{suffix}": v
                        for prefix, v in stats.items()
                    }
                },
                step=step,
                device_mesh=self.rollout_device_mesh["dp"]
            )

        dist.barrier()

        if self.rollout_device_mesh["tp"].get_local_rank() == 0:
            return gather_and_concat_list(
                data_list,
                self.rollout_device_mesh["dp"]
            )

    def forward(self, minibatch: Dict[str, torch.Tensor]) -> torch.Tensor:
        update_params_of_ring_attn(
            minibatch["cu_seqlens"], self.sp_device_mesh["sp"]
        )

        logits = self.model(
            input_ids=minibatch["states"],
            position_ids=minibatch["position_ids"],
            use_cache=False
        ).logits / (
            self.config.rollout.train_temperature
            if hasattr(self.config, "rollout") else 1.0
        )

        return torch.gather(
            logits.log_softmax(-1),
            dim=-1,
            index=minibatch["actions"].unsqueeze(-1)
        ).squeeze(-1) * minibatch["action_mask"]

    @torch.no_grad()
    def compute_logps(
        self,
        data_list: List[Dict[str, torch.Tensor]],
        step: int
    ) -> List[Dict[str, torch.Tensor]]:
        self.load_model_to_gpu()
        minibatches = self.scatter_and_pack_data_list(data_list, False)

        prefix = "old" if self.train else "ref"

        total_actions = sum_across_processes(
            sum([minibatch["action_mask"].sum() for minibatch in minibatches])
        )
        
        self.model.eval()
        metrics = defaultdict(list)
        for minibatch in (
            tqdm(minibatches, desc=f"Step {step + 1}, compute {prefix} logps")
            if self.device_mesh.get_rank() == 0 else minibatches
        ):
            minibatch[f"{prefix}_logps"] = self.forward(minibatch)

            if not self.train and self.config.kl.type == "reward":
                kl_term = compute_kl_term(
                    minibatch["old_logps"],
                    minibatch["ref_logps"],
                    self.config.kl.estimator
                )
                minibatch["rewards"] -= self.config.kl.coef * kl_term
                kl = kl_term.sum() / total_actions
                metrics["kl"].append(self.device_mesh.size() * len(minibatches) * kl.item())

        self.log(metrics, step)

        self.offload_model_to_cpu()
        return self.resume_and_gather_data_list(minibatches) 

    def update(self, data_list: List[Dict[str, torch.Tensor]], step: int):
        self.load_model_to_gpu()
        self.load_optimizer_to_gpu()
        batches = self.scatter_and_pack_data_list(data_list, True)

        self.model.train()
        metrics = defaultdict(list)
        if self.device_mesh.get_rank() == 0:
            tbar = tqdm(total=len(batches) * len(batches[0]), desc=f"Step {step + 1}, update actor")
        for batch in batches:
            
            total_actions = sum_across_processes(
                sum([minibatch["action_mask"].sum() for minibatch in batch])
            )
            
            for minibatch in batch:

                logps = self.forward(minibatch)
                ratio = (logps - minibatch["old_logps"]).exp()
                clipped_ratio = torch.clamp(
                    ratio,
                    1 - self.config.clip,
                    1 + self.config.clip
                )
                objective = minibatch["advantages"] * ratio
                clipped_objective = minibatch["advantages"] * clipped_ratio
                loss = - torch.min(objective, clipped_objective).sum() / total_actions
                clip_ratio = (objective > clipped_objective).sum() / total_actions

                metrics["actor/loss"].append(self.device_mesh.size() * len(batch) * loss.item())
                # The losses on different data processes (resp. 
                # of minibatches within a batch) are accumulated 
                # but the value will be averaged in `Worker.log`. 
                # Therefore we multiply the world size (resp. bsz) 
                # here to get the correct value.
                metrics["actor/clip_ratio"].append(self.device_mesh.size() * len(batch) * clip_ratio.item())

                if self.config.kl.coef > 0 and self.config.kl.type == "loss":
                    kl = compute_kl_term(
                        logps, minibatch["ref_logps"], self.config.kl.estimator
                    ).sum() / total_actions
                    loss = loss + self.config.kl.coef * kl
                    metrics["kl"].append(self.device_mesh.size() * len(batch) * kl.item())

                loss.backward() 
                if self.device_mesh.get_rank() == 0:
                    tbar.update()
            grad_norm = clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.max_grad_norm
            )
            metrics["actor/grad_norm"].append(grad_norm.full_tensor().item())
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.log(metrics, step)
        if (step + 1) % self.config.save_freq == 0:
            self.save(step)

        self.offload_optimizer_to_cpu()
        torch.cuda.empty_cache()
        # or llm.resume_memory_occupation() may OOM
        if self.rollout_device_mesh["tp"].get_local_rank() == 0:
            self.llm.resume_memory_occupation()

        named_tensors = [(k, v) for k, v in self.model.state_dict().items()]
        for idx, (name, tensor) in enumerate(named_tensors):
            serialized_tensor = MultiprocessingSerializer.serialize(
                tensor.full_tensor()
            )
            serialized_tensors = [
                None for _ in range(self.rollout_device_mesh["tp"].size())
            ] if self.rollout_device_mesh["tp"].get_local_rank() == 0 else None
            dist.gather_object(
                serialized_tensor,
                serialized_tensors,
                group_dst=0,
                group=self.rollout_device_mesh["tp"].get_group(),
            )
            if self.rollout_device_mesh["tp"].get_local_rank() == 0:
                self.llm.update_weights_from_tensor(
                    named_tensors=[(
                        name, LocalSerializedTensor(values=serialized_tensors)
                    )],
                    flush_cache=(idx == len(named_tensors) - 1)
                )
        dist.barrier()
        self.offload_model_to_cpu()
        # Offload params here, or the params cannot be loaded.