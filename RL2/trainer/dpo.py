import hydra
from collections import defaultdict
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from RL2.trainer import Trainer
from RL2.dataset import DPODataset
from RL2.workers import Actor
from RL2.algs import sequence_all_reduce
from RL2.utils.comm import initialize_global_process_group
from RL2.utils.timing import time_logger


class DPOTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

        tokenizer = AutoTokenizer.from_pretrained(config.actor.model_name)
        dataset = DPODataset(
            config.data.path, tokenizer, config.data.max_length
        )
        self.dataloader = DataLoader(
            dataset,
            self.config.data.batch_size,
            collate_fn=dataset.collate_fn
        )

        self.actor = Actor(config.actor, True)
        self.ref_actor = Actor(config.ref_actor, False)
        
        num_training_steps = self.config.trainer.n_epochs * len(self.dataloader)
        num_warmup_steps = int(self.config.actor.warmup_ratio * num_training_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.actor.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    @time_logger("update_actor")
    def update_actor(self, data_list, step):

        minibatches = self.actor.scatter_and_pack_data_list(data_list, pair=True)

        metrics = defaultdict(list)
        losses = []
        for minibatch in self.actor.tqdm(minibatches):
            logps, _ = self.actor.forward(minibatch)
            chosen_rewards, rejected_rewards = sequence_all_reduce(
                minibatch,
                self.config.actor.beta * (logps - minibatch["ref_logps"]),
                self.actor.sp_device_mesh["sp"]
            ).view(-1, 2).T
            reward_margins = chosen_rewards - rejected_rewards
            loss = - F.logsigmoid(reward_margins).sum() / self.config.data.batch_size
            (loss * dist.get_world_size()).backward()

            metrics["rewards/chosen"].extend(chosen_rewards.tolist())
            metrics["rewards/rejected"].extend(rejected_rewards.tolist())
            metrics["rewards/margin"].extend(reward_margins.tolist())
            metrics["accuray"].extend((reward_margins > 0).tolist())
            losses.append(loss.item())

        grad_norm = self.actor.optimizer_step()
        self.scheduler.step()
        metrics["grad_norm"].append(grad_norm)
        self.actor.log(
            metrics, step, device_mesh=self.actor.sp_device_mesh["dp"]
        )
        self.actor.log(
            {"loss": losses}, step, False, self.actor.sp_device_mesh["dp"]
        )

    def train(self):

        step = 0
        for epoch in range(self.config.trainer.n_epochs):
            for data_list in tqdm(
                self.dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=(dist.get_rank() != 0)
            ):
                data_list = self.ref_actor.compute_logps(data_list, step)
                self.update_actor(data_list, step)
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