from RL2.dataset import BaseDataset, tokenize_messages

class SFTDataset(BaseDataset):
    
    def __getitem__(self, idx):
        
        messages = self.dataset[idx]["messages"]
        ex = tokenize_messages(self.tokenizer, messages)
        return self.truncate_and_scatter(ex)