import hydra
from collections import defaultdict
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from RL2.trainer.base import Trainer
from RL2.dataset.dpo import DPODataset
from RL2.workers.actor import Actor
from RL2.utils.comm import initialize_global_process_group

def forward(actor, batch):

    logps = actor.forward(batch)

    cu_seqlens = batch["cu_seqlens"]
    partial_logps = torch.stack([
        logps[:, start_idx:end_idx].sum()
        for start_idx, end_idx
        in zip(cu_seqlens[:-1], cu_seqlens[1:])
    ])
    logps = partial_logps.detach()
    dist.all_reduce(
        logps,
        op=dist.ReduceOp.SUM,
        group=actor.sp_device_mesh["sp"].get_group()
    )
    logps = logps + partial_logps - partial_logps.detach()
    chosen_logps, rejected_logps = logps.view(2, -1)

    chosen_actions = torch.stack([
        batch["action_mask"][:, start_idx:end_idx].sum()
        for start_idx, end_idx
        in zip(cu_seqlens[:len(chosen_logps)], cu_seqlens[1:len(chosen_logps) + 1])
    ])
    dist.all_reduce(
        chosen_actions,
        op=dist.ReduceOp.SUM,
        group=actor.sp_device_mesh["sp"].get_group()
    )
    chosen_avg_logps = chosen_logps / (chosen_actions + torch.finfo(chosen_logps.dtype).eps)

    return chosen_logps, rejected_logps, chosen_avg_logps


class DPOTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

        self.actor = Actor(config.actor, self.device_mesh, True)
        self.ref_actor = Actor(config.actor, self.device_mesh, False)

        tokenizer = AutoTokenizer.from_pretrained(config.actor.model_name)
        dataset = DPODataset(
            config.data.path,
            tokenizer,
            config.data.max_length,
            self.actor.sp_device_mesh["sp"]
        )
        self.sampler = DistributedSampler(
            dataset,
            num_replicas=self.actor.sp_device_mesh["dp"].size(),
            rank=self.actor.sp_device_mesh["dp"].get_local_rank(),
            shuffle=True,
            drop_last=True
        )
        self.dataloader = DataLoader(
            dataset,
            self.config.data.batch_size_per_device,
            sampler=self.sampler,
            collate_fn=dataset.collate_fn
        )

    def train(self):
        gradient_accumulation_steps = self.config.data.batch_size // (self.actor.sp_device_mesh["dp"].size() * self.config.data.batch_size_per_device)

        step = 0
        metrics = defaultdict(list)
        for epoch in range(self.config.trainer.n_epochs):
            self.sampler.set_epoch(epoch)
            for batch in (
                tqdm(self.dataloader) if self.device_mesh.get_rank() == 0 else self.dataloader
            ):

                with torch.no_grad():
                    ref_chosen_logps, ref_rejected_logps, _ = forward(
                        self.ref_actor, batch
                    )
                chosen_logps, rejected_logps, chosen_avg_logps = forward(
                    self.actor, batch
                )
                chosen_rewards = self.config.trainer.beta * (chosen_logps - ref_chosen_logps)
                rejected_rewards = self.config.trainer.beta * (rejected_logps - ref_rejected_logps)
                reward_margins = chosen_rewards - rejected_rewards

                dpo_loss = - F.logsigmoid(reward_margins).mean()
                sft_loss = - chosen_avg_logps.mean()
                loss = dpo_loss + self.config.trainer.alpha * sft_loss
                loss.backward()

                accuracies = (chosen_rewards > rejected_rewards).float()

                metrics["loss/total"].append(loss.item())
                metrics["loss/dpo"].append(dpo_loss.item())
                metrics["loss/sft"].append(sft_loss.item())
                metrics["accuracy"].append(accuracies.mean().item())
                metrics["rewards/chosen"].append(chosen_rewards.mean().item())
                metrics["rewards/rejected"].append(rejected_rewards.mean().item())
                metrics["rewards/margins"].append(reward_margins.mean().item())

                step += 1
                if step % gradient_accumulation_steps == 0:
                    self.actor.optimizer.step()
                    self.actor.optimizer.zero_grad()

                    self.actor.log(metrics, step, self.actor.sp_device_mesh["dp"])
                    metrics = defaultdict(list)

            self.actor.save(step)


@hydra.main(config_path="config", config_name="dpo", version_base=None)
def main(config):

    initialize_global_process_group()

    trainer = DPOTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()