import os
import datasets
import torch
from torch.utils.data import Dataset

def load_dataset(data_path):
    # TODO: support concatenating multiple datasets

    if "@" in data_path:
        split, data_path = data_path.split("@")
    else:
        split = "train"
    
    ext = os.path.splitext(data_path)[-1].strip(".")
    if ext in ["json", "jsonl", "csv", "parquet", "arrow"]:
        if ext == "jsonl":
            ext = "json"
        return datasets.load_dataset(ext, data_files=data_path, split=split)
    else:
        return datasets.load_dataset(data_path, split=split)

def tokenize_messages(tokenizer, messages):

    states, actions, action_mask = [], [], []
    for idx, message in enumerate(messages):

        state = tokenizer.apply_chat_template(
            messages[:idx + 1],
            add_generation_prompt=idx + 1 < len(messages) and messages[idx + 1]["role"] == "assistant"
        )[len(states):]

        states.extend(state)
        actions.extend(
            state if message["role"] == "assistant"
            else len(state) * [0]
        )
        action_mask.extend(len(state) * [
            1 if message["role"] == "assistant" else 0
        ])

    return {
        "states": states[:-1],
        "actions": actions[1:],
        "action_mask": action_mask[1:],
        "position_ids": list(range(len(states) - 1))
    }


class BaseDataset(Dataset):
    
    def __init__(
        self,
        data_path,
        tokenizer,
        max_length,
        device_mesh
    ):

        self.dataset = load_dataset(data_path)
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