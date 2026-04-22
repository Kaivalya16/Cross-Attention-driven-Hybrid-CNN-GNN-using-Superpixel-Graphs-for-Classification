import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one single epoch.
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Wrap dataloader in tqdm for a nice progress bar
    progress_bar = tqdm(dataloader, desc="Training")
    
    for images, graphs, labels in progress_bar:
        # Move data to the appropriate device (CPU/GPU)
        images = images.to(device)
        graphs = graphs.to(device)
        labels = labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: Our hybrid model takes both the image and the graph
        logits = model(images, graphs)

        # Calculate loss (Standard CrossEntropy since we use Mid-Fusion)
        loss = criterion(logits, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * labels.size(0)
        
        # Get predictions (highest logit)
        _, predicted = torch.max(logits.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / total_samples, 
            'acc': correct_predictions / total_samples
        })

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on a validation or test set.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad(): # Disable gradient calculation for faster inference
        for images, graphs, labels in dataloader:
            images = images.to(device)
            graphs = graphs.to(device)
            labels = labels.to(device)

            logits = model(images, graphs)
            loss = criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            
            _, predicted = torch.max(logits.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    
    return epoch_loss, epoch_acc