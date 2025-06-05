import uuid
import torch
from RL2.dataset import BaseDataset, tokenize_messages


class DPODataset(BaseDataset):

    def __getitem__(self, idx):

        ex = self.dataset[idx]
        uid = str(uuid.uuid4())
        messages = ex["messages"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        chosen = self.tokenize_messages_completion(
            uid, messages, chosen, True
        )
        rejected = self.tokenize_messages_completion(
            uid, messages, rejected, False
        )

        return chosen, rejected
    
    def tokenize_messages_completion(
        self, uid, messages, completion, chosen: bool
    ):

        ex = tokenize_messages(
            self.tokenizer,
            messages + [{"role": "assistant", "content": completion}]
        )
        ex.update({
            "uid": uid,
            "eos_mask": torch.LongTensor((ex["states"].shape[-1] - 1) * [0] + [1]).unsqueeze(0),
            "chosen_mask": torch.LongTensor(ex["states"].shape[-1] * [1 if chosen else -1]).unsqueeze(0)
        })
        return ex

    def collate_fn(self, batch):
        return super().collate_fn(
            sum([list(ex) for ex in batch], [])
        )