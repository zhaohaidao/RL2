from collections import defaultdict
import torch
import torch.distributed as dist
from transformers import AutoModelForTokenClassification
from RL2.workers import Worker
from RL2.utils.models import prepare_lora_model
from RL2.utils.comm import gather_and_concat_list
from RL2.utils.ring_attn import update_params_of_ring_attn
from RL2.utils.timing import time_logger


class Critic(Worker):

    def __init__(self, config):
        super().__init__(config, True)

        self.model = AutoModelForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=1,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )

        if hasattr(self.config, "lora") and self.config.lora.rank > 0:
            self.model = prepare_lora_model(
                self.model, "TOKEN_CLS", config.lora
            )

        self.prepare_model_optimizer()

    def forward(self, minibatch) -> torch.Tensor:
        update_params_of_ring_attn(
            minibatch["cu_seqlens"], self.data_device_mesh["sp"]
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
        
        self.offload_model_to_cpu()
        return self.unpack_and_gather_data_list(minibatches)

    @time_logger("update_critic")
    def update(self, data_list, step: int):
        self.load_model_to_gpu()
        batches = self.scatter_and_pack_data_list(data_list, True)

        self.model.train()
        tbar = self.tqdm(
            total=sum([len(batch) for batch in batches]),
            desc="Update critic"
        )
        metrics = defaultdict(list)
        for batch in batches:

            total_actions = self.count_total_actions(batch)
            metric = defaultdict(list)
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
                
                self.backward(loss)

                tbar.update()
                metric["critic/loss"].append(loss.item())
                metric["critic/clip_ratio"].append(clip_ratio.item())

            grad_norm = self.optimizer_step()
            
            for k, v in metric.items():
                v = gather_and_concat_list(v, self.data_device_mesh["sp"])
                v = gather_and_concat_list(v, self.data_device_mesh["dp"])
                if dist.get_rank() == 0:
                    metrics[k].append(sum(v))
            metrics["actor/grad_norm"].append(grad_norm)

        self.rank0_log(metrics, step)
        if self.config.save_freq is not None and (step + 1) % self.config.save_freq == 0:
            self.save(step)

        self.offload_model_to_cpu()