from RL2.dataset.base import BaseDataset
from RL2.algs import tokenize_messages

class SFTDataset(BaseDataset):
    
    def __getitem__(self, idx):
        
        messages = self.dataset[idx]
        ex = tokenize_messages(self.tokenizer, messages)
        return self.truncate_and_scatter(ex)