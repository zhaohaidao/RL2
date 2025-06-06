import torch
from RL2.dataset import BaseDataset, tokenize_messages


class DPODataset(BaseDataset):

    def __getitem__(self, idx):

        ex = self.dataset[idx]
        messages = ex["messages"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        chosen = self.tokenize_messages_completion(messages, chosen)
        rejected = self.tokenize_messages_completion(messages, rejected)

        return chosen, rejected
    
    def tokenize_messages_completion(self, messages, completion):

        ex = tokenize_messages(
            self.tokenizer,
            messages + [{"role": "assistant", "content": completion}]
        )
        ex.update({
            "eos_mask": torch.LongTensor((ex["states"].shape[-1] - 1) * [0] + [1]).unsqueeze(0)
        })
        return {k: v[:, :self.max_length] for k, v in ex.items()}

    def collate_fn(self, batch):
        return sum([list(ex) for ex in batch], [])