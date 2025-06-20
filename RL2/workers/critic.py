from collections import defaultdict
import torch
import torch.distributed as dist
from transformers import AutoModelForTokenClassification
from RL2.workers import Worker
from RL2.utils.ring_attn import update_params_of_ring_attn
from RL2.utils.timing import time_logger


class Critic(Worker):
    task_type = "TOKEN_CLS"

    def __init__(self, config):
        super().__init__(config, True)

        self.model = AutoModelForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=1,
            attn_implementation="flash_attention_2"
        )

        self.prepare_model_optimizer()

    def forward(self, minibatch) -> torch.Tensor:
        update_params_of_ring_attn(
            minibatch["cu_seqlens"], self.sp_device_mesh["sp"]
        )

        return self.model(
            input_ids=minibatch["states"],
            position_ids=minibatch["position_ids"],
            use_cache=False
        ).logits.squeeze(-1) * minibatch["action_mask"]

    @time_logger("compute_values")
    @torch.no_grad()
    def compute_values(self, data_list, step):
        self.load_model_to_gpu()
        minibatches = self.scatter_and_pack_data_list(data_list)

        self.model.eval()
        for minibatch in self.tqdm(minibatches, desc="Compute values"):
            minibatch["values"] = self.forward(minibatch)
        
        # No need to offload model because it will be updated soon. See `Trainer.train`.
        return self.resume_and_gather_data_list(minibatches)

    @time_logger("update_critic")
    def update(self, data_list, step: int):
        # Model has been loaded in `compute_values`. See `Trainer.train`.
        batches = self.scatter_and_pack_data_list(data_list, True)

        self.model.train()
        tbar = self.tqdm(
            total=sum([len(batch) for batch in batches]),
            desc="Update critic"
        )
        metrics = defaultdict(list)
        grad_norms = []
        for batch in batches:

            total_actions = self.count_total_actions(batch)
            for minibatch in batch:

                values = self.forward(minibatch)
                clipped_values = torch.clamp(
                    values,
                    minibatch["values"] - self.config.clip,
                    minibatch["values"] + self.config.clip
                )
                mse = (values - minibatch["returns"]).pow(2)
                clipped_mse = (clipped_values - minibatch["returns"]).pow(2)
                loss = torch.max(mse, clipped_mse).sum() / total_actions
                clip_ratio = (mse < clipped_mse).sum() / total_actions
                
                (loss * dist.get_world_size()).backward()

                tbar.update()
                metrics["critic/loss"].append(loss.item())
                metrics["critic/clip_ratio"].append(clip_ratio.item())

            grad_norm = self.optimizer_step()
            grad_norms.append(grad_norm)

        self.log(metrics, step, op="sum")
        self.log({"critic/grad_norm": grad_norms}, step)
        if self.config.save_freq is not None and (step + 1) % self.config.save_freq == 0:
            self.save(step)

        self.offload_model_to_cpu()