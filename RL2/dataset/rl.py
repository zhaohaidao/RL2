from typing import Tuple, Dict, List
import json
import copy
import uuid
from torch.utils.data import Dataset

class RLDataset(Dataset):
    
    def __init__(self, data_path, responses_per_prompt):

        with open(data_path) as f:
            self.dataset = json.load(f)
        self.responses_per_prompt = responses_per_prompt
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        ex = self.dataset[idx]
        uid = str(uuid.uuid4())
        messages = ex["messages"]
        answer = ex["answer"]

        return {
            "uid": uid,
            "messages": messages,
            "answer": answer
        }

    def collate_fn(self, batch: Tuple[Dict]) -> List[Dict]:
        return [
            copy.deepcopy(ex)
            for ex in batch
            for _ in range(self.responses_per_prompt)
        ]