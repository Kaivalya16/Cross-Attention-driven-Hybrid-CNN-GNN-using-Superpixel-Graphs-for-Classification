import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from src.models.cnn_backbone import CNNBackbone

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=4, in_channels=1, embed_dim=64):
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
    print(f"Training COVID-19 Baseline on {device}...")

    # We must use the exact same transforms we used during preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])

    print("Loading raw X-ray dataset...")
    full_dataset = datasets.ImageFolder(root='./data/raw/COVID-19', transform=transform)
    
    # 80/20 split to match the hybrid run
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Batch size 64 to avoid VRAM overflow with 299x299 images
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = BaselineCNN(num_classes=4, in_channels=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    num_epochs = 10
    print("Starting Training...")
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluation
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
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Test Acc: {correct/total:.4f}")

if __name__ == "__main__":
    main()