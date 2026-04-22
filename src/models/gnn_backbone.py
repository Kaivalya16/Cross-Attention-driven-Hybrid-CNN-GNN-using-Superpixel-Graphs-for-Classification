import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GNNBackbone(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        
        # Layer 1: in_channels -> 32
        self.conv1 = GATConv(in_channels, 32, heads=1, concat=True)
        
        # Layer 2: 32 -> 64
        self.conv2 = GATConv(32, 64, heads=1, concat=True)
        
        # Layer 3: 64 -> out_channels (64)
        self.conv3 = GATConv(64, out_channels, heads=1, concat=False)
        
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        # x shape: (Total_Nodes_in_Batch, Features)
        # edge_index shape: (2, Total_Edges_in_Batch)
        
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        
        x = self.conv3(x, edge_index)
        
        # Output shape: (Total_Nodes_in_Batch, out_channels)
        # Again, we don't apply Global Pooling here because our Mid-Fusion handles it.
        return x