import hydra
from collections import defaultdict
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer
from tqdm import tqdm
from RL2.trainer.base import Trainer
from RL2.dataset.sft import SFTDataset
from RL2.workers.actor import Actor
from RL2.algs import compute_seq_and_avg_logps
from RL2.utils.comm import initialize_global_process_group

class SFTTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

        self.actor = Actor(config.actor, self.device_mesh, True)

        tokenizer = AutoTokenizer.from_pretrained(config.actor.model_name)
        dataset = SFTDataset(
            config.data.path,
            tokenizer,
            config.data.max_length,
            self.actor.sp_device_mesh["sp"]
        )
        self.sampler, self.dataloader = self.prepare_sampler_dataloader(
            dataset,
            self.config.data.batch_size_per_device,
            True,
            self.actor.sp_device_mesh["dp"]
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

                logps = self.actor.forward(batch)
                _, avg_logps = compute_seq_and_avg_logps(
                    batch, logps, self.actor.sp_device_mesh["sp"]
                )

                loss = - avg_logps.mean()
                (loss * self.actor.sp_device_mesh["sp"].size() / gradient_accumulation_steps).backward()

                metrics["loss"].append(loss.item())

                step += 1
                if step % gradient_accumulation_steps == 0:
                    grad_norm = clip_grad_norm_(
                        self.actor.model.parameters(),
                        max_norm=self.actor.config.max_grad_norm
                    )
                    metrics["grad_norm"].append(grad_norm.full_tensor().item())
                    self.actor.optimizer.step()
                    self.actor.optimizer.zero_grad()

                    self.actor.log(metrics, step, self.actor.sp_device_mesh["dp"])
                    metrics = defaultdict(list)

            self.actor.save(step)


@hydra.main(config_path="config", config_name="sft", version_base=None)
def main(config):

    initialize_global_process_group()

    trainer = SFTTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()