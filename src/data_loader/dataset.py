import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from src.data_loader.graph_builder import build_radius_graph

class SuperpixelDataset(Dataset):
    def __init__(self, base_dataset, n_superpixels: int, radius: float):
        """
        Wrapper dataset that generates SLIC superpixel graphs on the fly.
        
        Args:
            base_dataset (Dataset): A PyTorch dataset returning (image, label) tuples.
                                    The image should be a PyTorch Tensor of shape (C, H, W)
                                    with values scaled between [0, 1].
            n_superpixels (int): Target number of superpixels for SLIC.
            radius (float): Radius threshold for connecting superpixel centroids.
        """
        self.base_dataset = base_dataset
        self.n_superpixels = n_superpixels
        self.radius = radius

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # 1. Get standard image tensor and label from the base dataset
        image_tensor, label = self.base_dataset[idx]
        
        # 2. Convert the image tensor (C, H, W) back to numpy (H, W, C) for SLIC
        # SLIC and skimage functions expect standard numpy image formats
        image_np = image_tensor.permute(1, 2, 0).numpy()
        
        # If it's grayscale (1, H, W), permuting makes it (H, W, 1). 
        # We squeeze it to (H, W) so our graph_builder handles it properly.
        if image_np.shape[-1] == 1:
            image_np = image_np.squeeze(-1)
            
        # 3. Generate the PyTorch Geometric graph data
        graph_data = build_radius_graph(
            image=image_np, 
            n_superpixels=self.n_superpixels, 
            radius=self.radius
        )
        
        # 4. Return the CNN input, GNN input, and the ground truth label
        return image_tensor, graph_data, label