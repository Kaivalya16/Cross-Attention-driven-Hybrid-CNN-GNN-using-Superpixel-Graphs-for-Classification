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
    img=None,
    is_cifar=False
):
    """
    Directly adapted from the reference paper's visualization code.
    Draws the Superpixel GNN using NetworkX.
    """
    colors = graph_x
    src = graph_edge_index[0]
    dst = graph_edge_index[1]
    edgelist = list(zip(src, dst))

    if img is not None:
        # The paper uses alpha=0.7 for the base images
        alpha_val = 0.7 if not is_cifar else 1.0
        ax.imshow(img, alpha=alpha_val, cmap='viridis' if not is_cifar else None)

    g = nx.Graph()
    g.add_nodes_from(list(range(0, colors.shape[0])))
    g.add_edges_from(edgelist)

    # Convert coordinates to a dictionary for NetworkX
    pos_dic = dict(zip(list(range(0, colors.shape[0])), graph_pos))
    
    # Draw using NetworkX as implemented in the reference paper
    nx.draw_networkx(
        g,
        pos=pos_dic,
        ax=ax,
        node_color=colors,
        node_size=150,
        font_size=7,
        edgecolors="black",
        cmap='viridis' if not is_cifar else None
    )

def process_and_draw(ax, img_tensor, is_cifar):
    # Convert tensor to correct numpy shape
    if not is_cifar:
        img_np = img_tensor.squeeze(0).numpy()
        # N=81, high compactness for rigid grid (MNIST)
        segments = slic(img_np, n_segments=81, compactness=100, start_label=0, channel_axis=None)
    else:
        img_np = img_tensor.permute(1, 2, 0).numpy()
        # N=100, low compactness for organic borders (CIFAR)
        segments = slic(img_np, n_segments=100, compactness=10, start_label=0, channel_axis=-1)

    # 1. Node Positions (Centroids)
    props = regionprops(segments)
    centroids = np.array([prop.centroid for prop in props])
    cy, cx = centroids[:, 0], centroids[:, 1]
    
    # NetworkX expects (x, y) coordinates
    graph_pos = np.column_stack((cx, cy))

    # 2. Edge Index (Radius Graph)
    H, W = img_np.shape[:2]
    norm_centroids = np.column_stack((cy / H, cx / W))
    dist_matrix = distance_matrix(norm_centroids, norm_centroids)
    
    radius = 0.17 if not is_cifar else 0.16
    src, dst = np.where((dist_matrix <= radius) & (dist_matrix > 0))
    graph_edge_index = np.stack([src, dst], axis=0)

    # 3. Node Colors and Background Image
    if not is_cifar:
        # Intensity values for MNIST node colors
        graph_x = img_np[cy.astype(int), cx.astype(int)]
        bg_img = img_np
    else:
        # Color-averaged background for CIFAR
        bg_img = label2rgb(segments, img_np, kind='avg')
        # RGB values for CIFAR node colors
        graph_x = bg_img[cy.astype(int), cx.astype(int)]

    # Plot using the paper's NetworkX function
    visualize_geometric_graph(ax, graph_x, graph_edge_index, graph_pos, img=bg_img, is_cifar=is_cifar)

def main():
    print("Loading datasets for NetworkX plotting...")
    transform = transforms.ToTensor()
    
    mnist_data = datasets.MNIST(root='./data/raw', train=False, download=True, transform=transform)
    cifar_data = datasets.CIFAR10(root='./data/raw', train=False, download=True, transform=transform)
    
    # Grab RANDOM images from both datasets
    mnist_idx = random.randint(0, len(mnist_data) - 1)
    cifar_idx = random.randint(0, len(cifar_data) - 1)
    
    mnist_img, mnist_label = mnist_data[mnist_idx]
    cifar_img, cifar_label = cifar_data[cifar_idx]
    cifar_class_name = cifar_data.classes[cifar_label]
    
    # Setup Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    print(f"Generating MNIST NetworkX Graph for digit '{mnist_label}'...")
    process_and_draw(axes[0], mnist_img, is_cifar=False)
    
    print(f"Generating CIFAR-10 NetworkX Graph for class '{cifar_class_name}'...")
    process_and_draw(axes[1], cifar_img, is_cifar=True)
    
    # Format layout
    for ax in axes:
        ax.set_axis_off()
        
    caption = f"Figure 1: Superpixels for the digit '{mnist_label}' image (left), a {cifar_class_name} (right)"
    fig.text(0.5, 0.05, caption, ha='center', fontsize=14, fontweight='bold', fontfamily='serif')
    
    plt.subplots_adjust(bottom=0.15)
    
    # Save the figure
    plt.savefig("Graph_Replication_Random.png", bbox_inches="tight")
    print("Saved exact replication to Graph_Replication_Random.png")
    plt.show()

if __name__ == "__main__":
    main()