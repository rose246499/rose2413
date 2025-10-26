import torch
from torch.utils.data import Dataset,DataLoader
import os
from os import path
import pandas as pd
class CoraDataset(Dataset):

    def __init__(self,cora_cites,cora_content):
        self.cora_cites = cora_cites
        self.cora_content = cora_content

        self.content_df = pd.read_csv(cora_content,sep='\t')
        self.cites_df = pd.read_csv(cora_cites,sep='\t')

        
    def __getitem__(self,idx):
        row = self.content_df.iloc[idx]
        cora = torch.tensor(row[1:-1].values,dtype=torch.float)
        label = row[-1]
        return cora, label
    
    def __len__(self):
        return len(self.content_df)

cora_content = "/home/rose/cora.content"
cora_cites = "/home/rose/cora.cites"
cora_dataset = CoraDataset(cora_cites,cora_content)
print(f"Dataset size: {len(cora_dataset)}")