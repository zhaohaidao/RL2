import hydra
from collections import defaultdict
import torch
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from transformers import AutoTokenizer
from tqdm import tqdm
from RL2.trainer import Trainer
from RL2.dataset import SFTDataset
from RL2.workers import Actor
from RL2.utils.comm import initialize_global_process_group

def compute_avg_logps(batch, logps, device_mesh):

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
        group=device_mesh.get_group()
    )
    logps = logps + partial_logps - partial_logps.detach()

    actions = torch.stack([
        batch["action_mask"][:, start_idx:end_idx].sum()
        for start_idx, end_idx
        in zip(cu_seqlens[:-1], cu_seqlens[1:])
    ])
    dist.all_reduce(
        actions,
        op=dist.ReduceOp.SUM,
        group=device_mesh.get_group()
    )
    return logps / (actions + torch.finfo(logps.dtype).eps)


class SFTTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

        tokenizer = AutoTokenizer.from_pretrained(config.actor.model_name)
        dataset = SFTDataset(
            config.data.path, tokenizer, config.data.max_length
        )
        _, self.dataloader = self.prepare_sampler_dataloader(
            dataset, self.config.data.batch_size, True
        )

        self.actor = Actor(config.actor, self.device_mesh, True)

    def train(self):

        step = 0
        for _ in range(self.config.trainer.n_epochs):
            for data_list in (
                tqdm(self.dataloader) if self.device_mesh.get_rank() == 0 else self.dataloader
            ):
                minibatches = self.actor.scatter_and_pack_data_list(data_list)

                metrics = defaultdict(list)
                for minibatch in minibatches:
                    logps = self.actor.forward(minibatch)
                    logps = compute_avg_logps(
                        minibatch, logps, self.actor.sp_device_mesh["sp"]
                    )
                    loss = - logps.sum() / self.config.data.batch_size
                    (loss * self.actor.device_mesh.size()).backward()
                    metrics["loss"].append(
                        self.actor.sp_device_mesh["dp"].size() * len(minibatches) * loss.item()
                    )

                grad_norm = clip_grad_norm_(
                    self.actor.model.parameters(),
                    max_norm=self.actor.config.max_grad_norm
                )
                self.actor.optimizer_step()
                metrics["grad_norm"].append(grad_norm.full_tensor().item())
                self.actor.log(
                    metrics, step, self.actor.sp_device_mesh["dp"]
                )
                step += 1

                if self.actor.config.save_freq is not None and step % self.actor.config.save_freq == 0:
                    self.actor.save(step)

        self.actor.save(step)


@hydra.main(config_path="config", config_name="sft", version_base=None)
def main(config):

    initialize_global_process_group()

    trainer = SFTTrainer(config)
    trainer.train()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()