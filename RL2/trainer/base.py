from omegaconf import OmegaConf
import os
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader
import torch.distributed as dist
import wandb

class Trainer:
    
    def __init__(self, config):
        
        OmegaConf.resolve(config)
        self.config = config
        world_size = int(os.environ["WORLD_SIZE"])
        # TODO: support 2D device mesh for HSDP
        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_shape=(world_size,)
        )

        if self.device_mesh.get_rank() == 0:
            if not config.trainer.disable_wandb:
                wandb.init(
                    project=config.trainer.project,
                    name=config.trainer.experiment_name,
                    config=OmegaConf.to_container(config)
                )
            else:
                wandb.log = lambda *args, **kwargs: None

    def prepare_sampler_dataloader(
        self,
        dataset,
        batch_size_per_device,
        train,
        device_mesh=None
    ):

        sampler = DistributedSampler(
            dataset,
            num_replicas=device_mesh.size(),
            rank=device_mesh.get_local_rank(),
            shuffle=train,
            drop_last=True
        ) if device_mesh is not None else RandomSampler(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size_per_device,
            sampler=sampler,
            collate_fn=dataset.collate_fn
        )

        return sampler, dataloader