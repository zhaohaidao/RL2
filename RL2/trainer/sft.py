import hydra
from torch.utils.data import DataLoader
import torch.distributed as dist
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from RL2.trainer import Trainer
from RL2.dataset import SFTDataset
from RL2.workers import Actor
from RL2.utils.comm import initialize_global_process_group
from RL2.utils.timing import time_logger


class SFTTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

        tokenizer = AutoTokenizer.from_pretrained(config.actor.model_name)
        dataset = SFTDataset(
            config.data.path, tokenizer, config.data.max_length
        )
        self.dataloader = DataLoader(
            dataset,
            self.config.data.batch_size,
            collate_fn=dataset.collate_fn
        )

        self.actor = Actor(config.actor, True)

        num_training_steps = self.config.trainer.n_epochs * len(self.dataloader)
        num_warmup_steps = int(self.config.actor.warmup_ratio * num_training_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.actor.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    @time_logger("update_actor")
    def update_actor(self, data_list, step):

        minibatches = self.actor.scatter_and_pack_data_list(data_list)

        total_actions = self.actor.count_total_actions(minibatches)
        losses = []
        for minibatch in self.actor.tqdm(minibatches):
            logps = self.actor.forward(minibatch)
            loss = - logps.sum() / total_actions
            (loss * dist.get_world_size()).backward()
            losses.append(loss.item())

        grad_norm = self.actor.optimizer_step()
        self.scheduler.step()
        self.actor.log({"loss": losses}, step, op="sum")
        self.actor.log({"grad_norm": [grad_norm]}, step)

    def train(self):

        step = 0
        for epoch in range(self.config.trainer.n_epochs):
            for data_list in tqdm(
                self.dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=(dist.get_rank() != 0)
            ):
                self.update_actor(data_list, step)
                step += 1

                if self.actor.config.save_freq is not None and step % self.actor.config.save_freq == 0:
                    self.actor.save(step)

        self.actor.save()


@hydra.main(config_path="config", config_name="sft", version_base=None)
def main(config):

    initialize_global_process_group()

    trainer = SFTTrainer(config)
    trainer.train()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()