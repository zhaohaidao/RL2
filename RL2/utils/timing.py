import time
import functools
import wandb

def time_logger(name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start = time.time()
            output = func(self, *args, **kwargs)
            if self.device_mesh.get_rank() == 0:
                wandb.log({
                    f"timing/{name}": time.time() - start
                }, step=args[-1])
            return output
        return wrapper
    return decorator