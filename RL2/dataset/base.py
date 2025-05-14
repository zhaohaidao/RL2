import json
import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    
    def __init__(
        self,
        data_path,
        tokenizer,
        max_length,
        device_mesh
    ):

        with open(data_path) as f:
            self.dataset = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device_mesh = device_mesh
    
    def truncate_and_scatter(self, ex):
        
        ex = {
            k: v[:self.max_length] for k, v in ex.items()
        }

        multiple_of = 2 * self.device_mesh.size()
        if len(ex["states"]) % multiple_of != 0:
            pad_tokens = multiple_of - len(ex["states"]) % multiple_of
            for v in ex.values():
                v.extend(pad_tokens * [0])

        rank = self.device_mesh.get_local_rank()
        half_seqlen = len(ex["states"]) // multiple_of
        return {
            k: v[rank * half_seqlen:(rank + 1) * half_seqlen] + v[(multiple_of - rank - 1) * half_seqlen:(multiple_of - rank) * half_seqlen]
            for k, v in ex.items()
        }

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        
        seqlens = torch.IntTensor(
            [len(ex["states"]) for ex in batch]
        )
        cu_seqlens = torch.cumsum(
            torch.cat((torch.IntTensor([0]), seqlens)),
            0, dtype=torch.int32
        ).to(torch.cuda.current_device())
        batch = {
            k: torch.LongTensor(
                sum([ex[k] for ex in batch], [])
            ).unsqueeze(0).to(torch.cuda.current_device())
            for k in batch[0].keys()
        }
        batch["cu_seqlens"] = cu_seqlens

        return batch