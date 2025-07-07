import hydra
from collections import defaultdict
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer
from tqdm import tqdm
from RL2.trainer import Trainer
from RL2.dataset import RMDataset
from RL2.workers import Critic
from RL2.utils.comm import initialize_global_process_group
from RL2.algs import sequence_all_reduce
from RL2.utils.timing import time_logger


class RMTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

        tokenizer = AutoTokenizer.from_pretrained(config.critic.model_name)
        dataset = RMDataset(
            config.data.path, tokenizer, config.data.max_length
        )
        self.dataloader = self.prepare_dataloader(dataset)
        self.critic = Critic(config.critic)
        self.scheduler = self.prepare_scheduler(self.critic)

    @time_logger("update_critic")
    def update_critic(self, data_list, step):

        minibatches = self.critic.scatter_and_pack_data_list(data_list, pair=True)
        metrics = defaultdict(list)
        for minibatch in self.critic.tqdm(
            minibatches, desc="Update critic"
        ):
            rewards = self.critic.forward(minibatch)
            chosen_rewards, rejected_rewards = sequence_all_reduce(
                rewards,
                minibatch["cu_seqlens"],
                self.critic.device_mesh["sp"]
            ).view(-1, 2).T
            reward_margins = chosen_rewards - rejected_rewards
            loss = - F.logsigmoid(reward_margins).sum() / self.config.data.batch_size
            self.critic.backward(loss)
            metrics["loss"].append(loss.item())
            metrics["accuray"].extend((reward_margins > 0).tolist())

        grad_norm = self.critic.optimizer_step()
        self.scheduler.step()
        metrics["grad_norm"].append(grad_norm)
        self.critic.gather_and_log(metrics, step)

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