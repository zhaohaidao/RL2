from typing import Tuple, Dict, List
import json
import copy
from torch.utils.data import Dataset

class RLDataset(Dataset):
    
    def __init__(self, data_path, responses_per_prompt):

        with open(data_path, "r") as f:
            self.dataset = json.load(f)
        self.responses_per_prompt = responses_per_prompt
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        ex = self.dataset[idx]
        messages = ex["messages"]
        answer = ex["answer"]
        # Check if the last message is from the assistant (indicating enforce thinking for base model)
        # add_generation_prompt = message[-1]["role"] != "assistant"
        
        # prompt = self.tokenizer.apply_chat_template(
        #     message,
        #     add_generation_prompt=add_generation_prompt,
        #     tokenize=False
        # )
        # prompt_id = self.tokenizer.encode(
        #     prompt,
        #     add_special_tokens=False,
        #     max_length=self.max_length,
        #     truncation=True
        # )

        return {
            "messages": messages,
            "answer": answer
        }

    def collate_fn(self, batch: Tuple[Dict]) -> List[Dict]:
        return [
            copy.deepcopy(ex)
            for ex in batch
            for _ in range(self.responses_per_prompt)
        ]