import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from src.data_loader.precomputed_dataset import PrecomputedGraphDataset
from src.models.hybrid_model import HybridCrossAttentionModel
from src.core.train import train_one_epoch, evaluate

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Precomputed Dataset
    print("Loading precomputed COVID-19 dataset...")
    full_dataset = PrecomputedGraphDataset(processed_dir='./data/processed/COVID-19')
    
    if len(full_dataset) == 0:
        print("Error: No .pt files found! Make sure preprocess_covid.py has finished running.")
        return

    # Split into 80% train, 20% test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # We use a smaller batch size (64) because 299x299 images take up much more VRAM
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 2. Initialize Model
    print("Initializing Hybrid Model for COVID-19...")
    # X-rays are grayscale (1 channel). Graph features: intensity, norm_y, norm_x (3 features). 4 Classes.
    model = HybridCrossAttentionModel(
        num_classes=4, 
        in_channels_img=1, 
        in_channels_graph=3, 
        embed_dim=64
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    # 3. Train
    num_epochs = 10
    print("Starting Training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.4f}")

    print("\nTraining Complete!")
    os.makedirs('./models_saved', exist_ok=True)
    torch.save(model.state_dict(), './models_saved/hybrid_model_covid.pth')
    print("Model saved to ./models_saved/hybrid_model_covid.pth")

if __name__ == "__main__":
    main()