import os
import sys
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader.graph_builder import build_radius_graph

def preprocess_and_save(data_dir, output_dir, n_superpixels=200, radius=0.27):
    """Pre-computes and saves (image, graph, label) tuples."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Standardize image size to 299x299 as specified in the paper and convert to Grayscale
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    print(f"Found {len(dataset)} images belonging to {len(dataset.classes)} classes.")
    
    for idx in tqdm(range(len(dataset)), desc="Processing X-Rays"):
        img_tensor, label = dataset[idx]
        
        # Convert to numpy for SLIC (H, W)
        img_np = img_tensor.squeeze(0).numpy()
        
        # Build the graph (Paper uses n=200 for COVID)
        graph_data = build_radius_graph(img_np, n_superpixels=n_superpixels, radius=radius)
        
        # Save the processed package
        save_path = os.path.join(output_dir, f"sample_{idx}.pt")
        torch.save({'image': img_tensor, 'graph': graph_data, 'label': label}, save_path)
        
    print(f"Preprocessing complete! Files saved to {output_dir}")

if __name__ == "__main__":
    raw_dir = './data/raw/COVID-19'
    processed_dir = './data/processed/COVID-19'
    
    if not os.path.exists(raw_dir):
        print(f"Error: Could not find raw data at {raw_dir}")
    else:
        preprocess_covid = preprocess_and_save(raw_dir, processed_dir)