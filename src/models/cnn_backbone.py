import torch
import torch.nn as nn

class CNNBackbone(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        
        # Block 1: in_channels -> 32
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: 32 -> 64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: 64 -> out_channels (64)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        # Input shape: (Batch, Channels, Height, Width)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Output shape: (Batch, out_channels, H', W')
        # We don't flatten here because our Cross-Attention module handles the sequence reshaping!
        return x