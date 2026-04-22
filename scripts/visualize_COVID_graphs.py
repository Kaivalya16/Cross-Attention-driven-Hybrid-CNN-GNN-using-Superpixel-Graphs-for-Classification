import os
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torchvision import datasets, transforms
from skimage.segmentation import slic
from skimage.measure import regionprops
from skimage.color import label2rgb
from scipy.spatial import distance_matrix

def visualize_geometric_graph(
    ax,
    graph_x,
    graph_edge_index,
    graph_pos,
    img=None
):
    """
    Draws the Superpixel GNN using NetworkX over the image,
    now using high-contrast colors (Red nodes, Blue edges).
    """
    src = graph_edge_index[0]
    dst = graph_edge_index[1]
    edgelist = list(zip(src, dst))

    if img is not None:
        ax.imshow(img, alpha=1.0)

    g = nx.Graph()
    g.add_nodes_from(list(range(0, graph_x.shape[0])))
    g.add_edges_from(edgelist)

    # Convert coordinates to a dictionary for NetworkX
    pos_dic = dict(zip(list(range(0, graph_x.shape[0])), graph_pos))
    
    # Draw using NetworkX with custom vibrant colors
    nx.draw_networkx(
        g,
        pos=pos_dic,
        ax=ax,
        node_color="red",          # Red nodes
        edge_color="blue",         # Blue edges
        node_size=150,
        font_size=7,
        font_color="white",        # White text inside the red nodes
        edgecolors="black"         # Black border around the red nodes for sharpness
    )

def process_and_draw_covid(ax, img_tensor):
    img_np = img_tensor.squeeze(0).numpy()
    
    # Generate organic superpixels
    segments = slic(
        img_np, 
        n_segments=200, 
        compactness=0.1, 
        sigma=1.0, 
        slic_zero=True, 
        start_label=0, 
        channel_axis=None
    )

    # 1. Node Positions (Centroids)
    props = regionprops(segments)
    centroids = np.array([prop.centroid for prop in props])
    cy, cx = centroids[:, 0], centroids[:, 1]
    graph_pos = np.column_stack((cx, cy))

    # 2. Edge Index (Radius Graph)
    H, W = img_np.shape
    norm_centroids = np.column_stack((cy / H, cx / W))
    dist_matrix = distance_matrix(norm_centroids, norm_centroids)
    
    radius = 0.12 
    src, dst = np.where((dist_matrix <= radius) & (dist_matrix > 0))
    graph_edge_index = np.stack([src, dst], axis=0)

    # 3. Background Image
    img_rgb = np.stack((img_np,)*3, axis=-1)
    bg_img = label2rgb(segments, img_rgb, kind='avg')
    graph_x = bg_img[cy.astype(int), cx.astype(int)]

    # Plot
    visualize_geometric_graph(ax, graph_x, graph_edge_index, graph_pos, img=bg_img)

def main():
    print("Loading COVID-19 dataset...")
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])
    
    dataset_path = './data/raw/COVID-19'
    if not os.path.exists(dataset_path):
        print(f"Error: Could not find dataset at {dataset_path}")
        return
        
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    
    idx = random.randint(0, len(dataset) - 1)
    img_tensor, label_idx = dataset[idx]
    class_name = dataset.classes[label_idx]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    print(f"Generating High-Contrast NetworkX Graph for class '{class_name}'...")
    process_and_draw_covid(ax, img_tensor)
    
    ax.set_axis_off()
    caption = f"Figure X: NetworkX Superpixel Representation of a COVID-19 Dataset X-Ray ({class_name})"
    fig.text(0.5, 0.05, caption, ha='center', fontsize=14, fontweight='bold', fontfamily='serif')
    
    plt.subplots_adjust(bottom=0.15)
    
    plt.savefig("COVID_NetworkX_Replication.png", bbox_inches="tight")
    print("Saved exact replication to COVID_NetworkX_Replication.png")
    plt.show()

if __name__ == "__main__":
    main()