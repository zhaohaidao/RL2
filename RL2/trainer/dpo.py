import hydra
from collections import defaultdict
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb
from RL2.trainer import Trainer
from RL2.dataset import DPODataset
from RL2.workers import Actor
from RL2.utils.comm import initialize_global_process_group


class DPOTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

        tokenizer = AutoTokenizer.from_pretrained(config.actor.model_name)
        dataset = DPODataset(
            config.data.path, tokenizer, config.data.max_length
        )
        _, self.dataloader = self.prepare_sampler_dataloader(
            dataset, self.config.data.batch_size, True
        )

        self.actor = Actor(config.actor, self.device_mesh, True)
        self.ref_actor = Actor(config.actor, self.device_mesh, False)

    def train(self):

        step = 0
        for _ in range(self.config.trainer.n_epochs):
            for data_list in (
                tqdm(self.dataloader) if self.device_mesh.get_rank() == 0 else self.dataloader
            ):
                
                data_list = self.ref_actor.compute_logps(data_list, step)
                data_list = self.actor.compute_logps(data_list, step)

                metrics = defaultdict(list)
                if self.device_mesh.get_rank() == 0:
                    uid2margin = defaultdict(float)
                    for ex in data_list:
                        chosen_mask = ex["chosen_mask"][0, 0].item()
                        reward = self.config.trainer.beta * (ex["old_logps"] - ex["ref_logps"]).sum().item()
                        uid2margin[ex["uid"]] += chosen_mask * reward
                        metrics["rewards/" + ("chosen" if chosen_mask == 1 else "rejected")].append(reward)
                    
                    for ex in data_list:
                        margin = uid2margin[ex["uid"]]
                        ex["grad"] = - self.config.trainer.beta * F.sigmoid(
                            - torch.tensor(margin)
                        ).item() * ex["chosen_mask"]
                        metrics["rewards/margin"].append(margin)
                        metrics["loss"].append(- F.logsigmoid(
                            torch.tensor(margin)
                        ).item())
                        metrics["accuracy"].append(margin > 0)

                minibatches = self.actor.scatter_and_pack_data_list(data_list)
                for minibatch in minibatches:
                    logps = self.actor.forward(minibatch)
                    loss = (logps * minibatch["grad"]).sum() / self.config.data.batch_size
                    (loss * self.actor.device_mesh.size()).backward()

                grad_norm = clip_grad_norm_(
                    self.actor.model.parameters(),
                    max_norm=self.actor.config.max_grad_norm
                )
                self.actor.optimizer_step()
                metrics["grad_norm"].append(grad_norm.full_tensor().item())
                if self.device_mesh.get_rank() == 0:
                    wandb.log({
                        k: torch.Tensor(v).mean().item()
                        for k, v in metrics.items()
                    }, step=step)
                step += 1

                if self.actor.config.save_freq is not None and step % self.actor.config.save_freq == 0:
                    self.actor.save(step)

        self.actor.save(step)


@hydra.main(config_path="config", config_name="dpo", version_base=None)
def main(config):

    initialize_global_process_group()

    trainer = DPOTrainer(config)
    trainer.train()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()