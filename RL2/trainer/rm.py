import hydra
from collections import defaultdict
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from RL2.trainer import Trainer
from RL2.dataset import RMDataset
from RL2.workers import Critic
from RL2.algs import sequence_all_reduce
from RL2.utils.comm import initialize_global_process_group
from RL2.utils.timing import time_logger


class RMTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

        tokenizer = AutoTokenizer.from_pretrained(config.critic.model_name)
        dataset = RMDataset(
            config.data.path, tokenizer, config.data.max_length
        )
        self.dataloader = DataLoader(
            dataset,
            self.config.data.batch_size,
            collate_fn=dataset.collate_fn
        )

        self.critic = Critic(config.critic)

        num_training_steps = self.config.trainer.n_epochs * len(self.dataloader)
        num_warmup_steps = int(self.config.critic.warmup_ratio * num_training_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.critic.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    @time_logger("update_critic")
    def update_critic(self, data_list, step):

        minibatches = self.critic.scatter_and_pack_data_list(data_list, pair=True)

        metrics = defaultdict(list)
        losses = []
        for minibatch in self.critic.tqdm(minibatches):
            rewards = self.critic.forward(minibatch)
            chosen_rewards, rejected_rewards = sequence_all_reduce(
                minibatch, rewards, self.critic.sp_device_mesh["sp"]
            ).view(-1, 2).T
            reward_margins = chosen_rewards - rejected_rewards
            loss = - F.logsigmoid(reward_margins).sum() / self.config.data.batch_size
            (loss * dist.get_world_size()).backward()

            metrics["accuray"].extend((reward_margins > 0).tolist())
            losses.append(loss.item())

        grad_norm = self.critic.optimizer_step()
        self.scheduler.step()
        metrics["grad_norm"].append(grad_norm)
        self.critic.log(metrics, step)
        self.critic.log(
            {"loss": losses}, step, op="sum", device_mesh=self.critic.sp_device_mesh["dp"]
        )

    def train(self):

        step = 0
        for epoch in range(self.config.trainer.n_epochs):
            for data_list in tqdm(
                self.dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=(dist.get_rank() != 0)
            ):
                self.update_critic(data_list, step)
                step += 1

                if self.critic.config.save_freq is not None and step % self.critic.config.save_freq == 0:
                    self.critic.save(step, rm=True)

        self.critic.save(rm=True)


@hydra.main(config_path="config", config_name="rm", version_base=None)
def main(config):

    initialize_global_process_group()

    trainer = RMTrainer(config)
    trainer.train()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()