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
        "states": torch.LongTensor(states[:-1]),
        "actions": torch.LongTensor(actions[1:]),
        "action_mask": torch.LongTensor(action_mask[1:]),
        "position_ids": torch.arange(len(states) - 1)
    }


class BaseDataset(Dataset):
    
    def __init__(
        self,
        data_path,
        tokenizer,
        max_length
    ):

        self.dataset = load_dataset(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)