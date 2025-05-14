from RL2.dataset.base import BaseDataset


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
        return self.truncate_and_scatter(ex)

    def collate_fn(self, batch):

        return super().collate_fn(
            [ex[0] for ex in batch] + [ex[1] for ex in batch]
        )