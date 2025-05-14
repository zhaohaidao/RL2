from omegaconf import OmegaConf
import os
import torch.distributed as dist
import wandb

class Trainer:
    
    def __init__(self, config):
        
        OmegaConf.resolve(config)
        self.config = config
        world_size = int(os.environ["WORLD_SIZE"])
        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_shape=(world_size,)
        )

        if not config.trainer.disable_wandb:
            if self.device_mesh.get_rank() == 0:
                wandb.init(
                    project=config.trainer.project,
                    name=config.trainer.experiment_name,
                    config=OmegaConf.to_container(config)
                )
        else:
            wandb.log = lambda *args, **kwargs: None