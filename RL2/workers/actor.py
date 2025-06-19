from collections import defaultdict
import torch
import torch.distributed as dist
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from RL2.workers import Worker
from RL2.algs import compute_kl_term
from RL2.utils.ring_attn import update_params_of_ring_attn
from RL2.utils.cut_cross_entropy import (
    substitute_lm_head_forward,
    update_params_of_linear_cross_entropy
)
from RL2.utils.comm import sum_across_processes
from RL2.utils.timing import time_logger


class Actor(Worker):

    def __init__(self, config, train: bool):
        super().__init__(config, train)
        
        self.model = AutoLigerKernelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32 if train else torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        substitute_lm_head_forward(self.model)
        self.prepare_model_optimizer()

    def forward(self, minibatch):
        update_params_of_ring_attn(
            minibatch["cu_seqlens"], self.sp_device_mesh["sp"]
        )
        update_params_of_linear_cross_entropy(
            minibatch["actions"],
            getattr(self.config, "temperature", 1.0)
        )
        logps, entropy = self.model(
            input_ids=minibatch["states"],
            position_ids=minibatch["position_ids"],
            use_cache=False
        ).logits
        logps = logps.unsqueeze(0) * minibatch["action_mask"]
        entropy = entropy.unsqueeze(0) * minibatch["action_mask"]
        return logps, entropy

    @time_logger("compute_logps")
    @torch.no_grad()
    def compute_logps(self, data_list, step):
        self.load_model_to_gpu()
        minibatches = self.scatter_and_pack_data_list(data_list)

        prefix = "old" if self.train else "ref"

        self.model.eval()
        for minibatch in self.tqdm(
            minibatches, desc=f"Compute {prefix} logps"
        ):
            minibatch[f"{prefix}_logps"], _ = self.forward(minibatch)

        self.offload_model_to_cpu()
        return self.resume_and_gather_data_list(minibatches) 
    
    @time_logger("update_actor")
    def update(self, data_list, step: int):
        self.load_model_to_gpu()
        if step < self.config.freeze_steps:
            return
        batches = self.scatter_and_pack_data_list(data_list, True)

        self.model.train()
        tbar = self.tqdm(
            total=sum([len(batch) for batch in batches]),
            desc="Update actor"
        )
        metrics = defaultdict(list)
        for batch in batches:
            
            total_actions = sum_across_processes(
                sum([minibatch["action_mask"].sum() for minibatch in batch])
            )
            
            for minibatch in batch:

                logps, entropy = self.forward(minibatch)
                ratio = (logps - minibatch["old_logps"]).exp()
                clipped_ratio = torch.clamp(
                    ratio,
                    1 - self.config.clip,
                    1 + self.config.clip
                )
                objective = minibatch["advantages"] * ratio
                clipped_objective = minibatch["advantages"] * clipped_ratio
                entropy = entropy.sum() / total_actions
                policy_loss = - torch.min(objective, clipped_objective).sum() / total_actions
                loss = policy_loss - self.config.entropy.coef * entropy
                clip_ratio = (objective > clipped_objective).sum() / total_actions

                if self.config.kl.coef > 0 and self.config.kl.type == "loss":
                    kl_loss = compute_kl_term(
                        logps,
                        minibatch["ref_logps"],
                        self.config.kl.loss_estimator
                    ).sum() / total_actions
                    loss = loss + self.config.kl.coef * kl_loss

                (loss * dist.get_world_size()).backward() 

                tbar.update()
                # The losses on each device (resp. of minibatches within a 
                # batch) are accumulated but the value will be averaged in 
                # `Worker.log`. Therefore we multiply the world size (resp. 
                # bsz) here to get the correct value.
                metrics["actor/entropy"].append(dist.get_world_size() * len(batch) * entropy.item())
                metrics["actor/loss"].append(dist.get_world_size() * len(batch) * loss.item())
                metrics["actor/clip_ratio"].append(dist.get_world_size() * len(batch) * clip_ratio.item())

            grad_norm = self.optimizer_step()
            metrics["actor/grad_norm"].append(grad_norm)

        self.log(metrics, step)
        if self.config.save_freq is not None and (step + 1) % self.config.save_freq == 0:
            self.save(step)