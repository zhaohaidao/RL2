import torch
from RL2.dataset import DPODataset


class RMDataset(DPODataset):

    def tokenize_messages_completion(self, messages, completion):

        states = self.tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": completion}]
        )[:self.max_length]
        action_mask = (len(states) - 1) * [0] + [1]

        return {
            "states": torch.LongTensor(states),
            "action_mask": torch.LongTensor(action_mask),
            "position_ids": torch.arange(len(states))
        }