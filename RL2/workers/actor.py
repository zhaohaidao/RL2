from collections import defaultdict
import torch
import torch.distributed as dist
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from RL2.workers import Worker
from RL2.algs import compute_logsumexp_by_chunk, compute_kl_term
from RL2.utils.comm import gather_and_concat_list
from RL2.utils.ring_attn import update_params_of_ring_attn
from RL2.utils.timing import time_logger


class Actor(Worker):

    def __init__(self, config, train: bool):
        super().__init__(config, train)
        
        self.model = AutoLigerKernelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32 if train else torch.bfloat16,
            attn_implementation="flash_attention_2"
        )

        self.prepare_model_optimizer()

    def forward(self, minibatch, compute_entropy=False):
        update_params_of_ring_attn(
            minibatch["cu_seqlens"], self.sp_device_mesh["sp"]
        )

        logits = self.model(
            input_ids=minibatch["states"],
            position_ids=minibatch["position_ids"],
            use_cache=False
        ).logits / getattr(
            self.config, "temperature", 1.0
        )
        
        action_logits = torch.gather(
            logits, dim=-1, index=minibatch["actions"].unsqueeze(-1)
        ).squeeze(-1)
        logsumexp = compute_logsumexp_by_chunk(logits)
        logps = (action_logits - logsumexp) * minibatch["action_mask"]
        
        if compute_entropy:
            probs = logits.softmax(-1)
            entropy = (
                logsumexp - (probs * logits).sum(-1)
            ) * minibatch["action_mask"]
            return logps, entropy
        else:
            return logps

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
            minibatch[f"{prefix}_logps"] = self.forward(minibatch)
        
        if not self.train:
            self.offload_model_to_cpu()
        return self.unpack_and_gather_data_list(minibatches) 
    
    @time_logger("update_actor")
    def update(self, data_list, step: int):
        if step < self.config.freeze_steps:
            self.offload_model_to_cpu()
            return
        if self.config.kl.coef == 0 and self.config.update_per_rollout == 1:
            self.load_model_to_gpu()
        batches = self.scatter_and_pack_data_list(data_list, True)

        self.model.train()
        tbar = self.tqdm(
            total=sum([len(batch) for batch in batches]),
            desc="Update actor"
        )
        metrics = defaultdict(list)
        for update, batch in enumerate(batches):
            
            total_actions = self.count_total_actions(batch)
            metric = defaultdict(list)
            for minibatch in batch:

                logps, entropy = self.forward(minibatch, True)
                if update == 0:
                    loss = - (minibatch["advantages"] * logps).sum() / total_actions
                    clip_ratio = torch.zeros_like(loss)
                else:
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

                entropy_loss = - entropy.sum() / total_actions
                loss = loss + self.config.entropy.coef * entropy_loss

                if self.config.kl.coef > 0 and self.config.kl.type == "loss":
                    kl_loss = compute_kl_term(
                        logps,
                        minibatch["ref_logps"],
                        self.config.kl.loss_estimator
                    ).sum() / total_actions
                    loss = loss + self.config.kl.coef * kl_loss

                (loss * dist.get_world_size()).backward() 

                tbar.update()
                metric["actor/entropy_loss"].append(entropy_loss.item())
                metric["actor/loss"].append(loss.item())
                metric["actor/clip_ratio"].append(clip_ratio.item())

            grad_norm = self.optimizer_step()

            for k, v in metric.items():
                v = gather_and_concat_list(v)
                if dist.get_rank() == 0:
                    metrics[k].append(sum(v))
            metrics["actor/grad_norm"].append(grad_norm)

        self.rank0_log(metrics, step)
        if self.config.save_freq is not None and (step + 1) % self.config.save_freq == 0:
            self.save(step)

        if self.config.adv_estimator == "gae":
            self.offload_model_to_cpu()