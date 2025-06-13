import hydra
from collections import defaultdict
import torch.distributed as dist
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from RL2.trainer import Trainer
from RL2.dataset import SFTDataset
from RL2.workers import Actor
from RL2.algs import sequence_all_reduce
from RL2.utils.comm import initialize_global_process_group


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

        num_training_steps = self.config.trainer.n_epochs * len(self.dataloader)
        num_warmup_steps = int(self.config.actor.warmup_ratio * num_training_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.actor.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def train(self):

        step = 0
        for epoch in range(self.config.trainer.n_epochs):
            for data_list in tqdm(
                self.dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=(self.device_mesh.get_rank() != 0)
            ):
                minibatches = self.actor.scatter_and_pack_data_list(data_list)

                metrics = defaultdict(list)
                for minibatch in self.actor.tqdm(minibatches):
                    logps = self.actor.forward(minibatch)
                    logps = sequence_all_reduce(
                        minibatch,
                        logps,
                        self.actor.sp_device_mesh["sp"],
                        "mean"
                    )
                    loss = - logps.sum() / self.config.data.batch_size
                    (loss * self.actor.device_mesh.size()).backward()
                    metrics["loss"].append(
                        self.actor.sp_device_mesh["dp"].size() * len(minibatches) * loss.item()
                    )

                grad_norm = self.actor.optimizer_step()
                self.scheduler.step()
                metrics["grad_norm"].append(grad_norm)
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