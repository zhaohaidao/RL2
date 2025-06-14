from collections import defaultdict
import torch
from transformers import AutoModelForTokenClassification
from RL2.workers import Worker
from RL2.utils.ring_attn import update_params_of_ring_attn
from RL2.utils.comm import sum_across_processes
from RL2.utils.timing import time_logger


class Critic(Worker):

    def __init__(self, config, device_mesh):
        super().__init__(config, device_mesh, True)

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
        for batch in batches:

            total_actions = sum_across_processes(
                sum([minibatch["action_mask"].sum() for minibatch in batch])
            )

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
                
                (loss * self.device_mesh.size()).backward()

                tbar.update()
                metrics["critic/loss"].append(self.device_mesh.size() * len(batch) * loss.item())
                metrics["critic/clip_ratio"].append(self.device_mesh.size() * len(batch) * clip_ratio.item())

            grad_norm = self.optimizer_step()
            metrics["critic/grad_norm"].append(grad_norm)

        self.log(metrics, step)
        if self.config.save_freq is not None and (step + 1) % self.config.save_freq == 0:
            self.save(step)

        self.offload_model_to_cpu()