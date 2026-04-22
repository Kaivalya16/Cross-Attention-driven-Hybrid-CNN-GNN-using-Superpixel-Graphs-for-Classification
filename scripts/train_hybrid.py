import sys
import os

# Ensure the 'src' directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch_geometric.loader import DataLoader  # PyG DataLoader is required

from src.data_loader.dataset import SuperpixelDataset
from src.models.hybrid_model import HybridCrossAttentionModel
from src.core.train import train_one_epoch, evaluate

def main():
    # 1. Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Hyperparameters (Based on paper's grid search)
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 1e-3
    num_epochs = 10
    n_superpixels = 75  # Specified for MNIST in the paper
    radius = 0.15       # Tune this based on normalized coordinate scale

    # 3. Load Datasets (Using MNIST as the baseline test)
    print("Loading datasets...")
    transform = transforms.ToTensor() # Converts to [0, 1] range

    base_train = datasets.MNIST(root='./data/raw', train=True, download=True, transform=transform)
    base_test = datasets.MNIST(root='./data/raw', train=False, download=True, transform=transform)

    # Wrap with our on-the-fly superpixel generator
    train_dataset = SuperpixelDataset(base_train, n_superpixels=n_superpixels, radius=radius)
    test_dataset = SuperpixelDataset(base_test, n_superpixels=n_superpixels, radius=radius)

    # Note: For testing the pipeline, you might want to subset the data first to ensure it runs quickly.
    # train_dataset = torch.utils.data.Subset(train_dataset, range(1000))

    # 4. Create DataLoaders
    # MUST use torch_geometric.loader.DataLoader to handle batches of variable-size graphs
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 5. Initialize Model
    print("Initializing model...")
    # MNIST has 1 channel, our graph builder outputs 3 features for grayscale (mean_intensity, norm_y, norm_x)
    model = HybridCrossAttentionModel(
        num_classes=10, 
        in_channels_img=1, 
        in_channels_graph=3, 
        embed_dim=64
    ).to(device)

    # 6. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 7. Training Loop
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.4f}")

    print("\nTraining Complete!")
    
    # Save the model
    os.makedirs('./models_saved', exist_ok=True)
    torch.save(model.state_dict(), './models_saved/hybrid_model_mnist.pth')
    print("Model saved to ./models_saved/hybrid_model_mnist.pth")

if __name__ == "__main__":
    main()