import time
import functools
import torch.distributed as dist
import wandb

def time_logger(name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args):
            start = time.time()
            output = func(*args)
            if dist.get_rank() == 0:
                wandb.log({
                    f"timing/{name}": time.time() - start
                }, step=args[-1])
            return output
        return wrapper
    return decorator