from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import torch.distributed as dist
from transformers import get_cosine_schedule_with_warmup
import wandb

class Trainer:
    
    def __init__(self, config):
        
        OmegaConf.resolve(config)
        self.config = config

        if dist.get_rank() == 0:
            print(OmegaConf.to_yaml(config))
            if not config.trainer.disable_wandb:
                wandb.init(
                    project=config.trainer.project,
                    name=config.trainer.experiment_name,
                    config=OmegaConf.to_container(config)
                )
            else:
                wandb.log = lambda *args, **kwargs: None

    # TODO (P1): resume training
    def prepare_dataloader(self, dataset, batch_size, shuffle):
        return DataLoader(
            dataset,
            batch_size,
            shuffle=shuffle,
            drop_last=True,
            collate_fn=dataset.collate_fn
        )
    
    def prepare_scheduler(self, worker):

        num_training_steps = self.config.trainer.n_epochs * len(self.dataloader)
        num_warmup_steps = int(worker.config.warmup_ratio * num_training_steps)

        return get_cosine_schedule_with_warmup(
            worker.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )