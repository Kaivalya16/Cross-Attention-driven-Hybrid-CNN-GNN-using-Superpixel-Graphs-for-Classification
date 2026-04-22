import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader # Standard PyTorch DataLoader this time

from src.models.cnn_backbone import CNNBackbone

# We need a quick wrapper to add a classifier head to the CNN backbone
class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, embed_dim=64):
        super().__init__()
        self.cnn = CNNBackbone(in_channels=in_channels, out_channels=embed_dim)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        features = self.cnn(x)
        return self.classifier(features)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load standard dataset (NO superpixels this time)
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data/raw', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data/raw', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = BaselineCNN(num_classes=10, in_channels=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    print("Training Baseline CNN...")
    for epoch in range(10): # Match the hybrid epochs
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Quick evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Epoch [{epoch+1}/10] | Test Acc: {correct/total:.4f}")

if __name__ == "__main__":
    main()