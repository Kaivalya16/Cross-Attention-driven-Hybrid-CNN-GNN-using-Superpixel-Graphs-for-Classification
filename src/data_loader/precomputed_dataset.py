import os
import torch
from torch.utils.data import Dataset

class PrecomputedGraphDataset(Dataset):
    def __init__(self, processed_dir):
        """Loads precomputed image/graph/label dictionaries from disk."""
        self.processed_dir = processed_dir
        # Get a list of all .pt files in the directory
        self.files = [f for f in os.listdir(processed_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.processed_dir, self.files[idx])
        
        # Load the dictionary saved during preprocessing
        data = torch.load(file_path, weights_only=False)
        
        return data['image'], data['graph'], data['label']