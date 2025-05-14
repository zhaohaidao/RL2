import json
import torch
from torch.utils.data import Dataset


class DPODataset(Dataset):

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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        ex = self.dataset[idx]
        messages = ex["messages"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        chosen = self.tokenize_messages_completion(messages, chosen)
        rejected = self.tokenize_messages_completion(messages, rejected)

        ex = {
            k: torch.LongTensor(chosen[k] + rejected[k]).unsqueeze(0)
            for k in chosen.keys()
        }
        ex["seqlens"] = torch.IntTensor([
            len(chosen["states"]), len(rejected["states"])
        ])

        return ex

    def tokenize_messages_completion(self, messages, completion):

        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        completion = self.tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": completion}]
        )[len(prompt):]

        ex = {
            "states": prompt + completion[:-1],
            "actions": (len(prompt) - 1) * [0] + completion,
            "position_ids": list(range(len(prompt) + len(completion) - 1)),
            "action_mask": (len(prompt) - 1) * [0] + len(completion) * [1]
        }

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

    def collate_fn(self, batch):
        
        seqlens = torch.cat([ex.pop("seqlens") for ex in batch]) 
        cu_seqlens = torch.cumsum(
            torch.cat((torch.IntTensor([0]), seqlens)),
            0, dtype=torch.int32
        ).to(torch.cuda.current_device())
        batch = {
            k: torch.cat(
                [ex[k] for ex in batch], -1
            ).to(torch.cuda.current_device())
            for k in batch[0].keys()
        }
        batch["cu_seqlens"] = cu_seqlens

        return batch