import sys
import os

# Ensure the 'src' directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch_geometric.loader import DataLoader 

from src.data_loader.dataset import SuperpixelDataset
from src.models.hybrid_model import HybridCrossAttentionModel
from src.core.train import train_one_epoch, evaluate

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters updated for CIFAR-10
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 1e-3
    num_epochs = 15  # CIFAR-10 usually takes a bit longer to converge
    n_superpixels = 100  # Specified for CIFAR in the paper
    radius = 0.15       

    print("Loading CIFAR-10 dataset...")
    transform = transforms.ToTensor() 

    base_train = datasets.CIFAR10(root='./data/raw', train=True, download=True, transform=transform)
    base_test = datasets.CIFAR10(root='./data/raw', train=False, download=True, transform=transform)

    # Wrap with our on-the-fly superpixel generator
    train_dataset = SuperpixelDataset(base_train, n_superpixels=n_superpixels, radius=radius)
    test_dataset = SuperpixelDataset(base_test, n_superpixels=n_superpixels, radius=radius)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Initializing model for RGB images...")
    # CIFAR-10 adjustments: 3 image channels (RGB), 5 graph features (R, G, B, norm_y, norm_x)
    model = HybridCrossAttentionModel(
        num_classes=10, 
        in_channels_img=3, 
        in_channels_graph=5, 
        embed_dim=64
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print("Starting CIFAR-10 training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.4f}")

    print("\nTraining Complete!")
    os.makedirs('./models_saved', exist_ok=True)
    torch.save(model.state_dict(), './models_saved/hybrid_model_cifar10.pth')
    print("Model saved to ./models_saved/hybrid_model_cifar10.pth")

if __name__ == "__main__":
    main()