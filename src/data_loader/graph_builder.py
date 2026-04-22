import torch
import numpy as np
from skimage.segmentation import slic
from skimage.measure import regionprops
from torch_geometric.data import Data
from scipy.spatial import distance_matrix

def build_radius_graph(image: np.ndarray, n_superpixels: int, radius: float) -> Data:
    """
    Constructs a PyTorch Geometric graph from an image using SLIC superpixels.
    """
    # Handle grayscale vs RGB
    is_grayscale = len(image.shape) == 2
    if is_grayscale:
        # Add channel dimension for consistency
        image = np.expand_dims(image, axis=-1)

    H, W, C = image.shape

    # 1. Generate superpixels using SLIC
    # compactness can be tuned; 10 is a standard default for images
    segments = slic(image, n_segments=n_superpixels, compactness=10, start_label=0)

    # 2. Extract node features
    # The paper uses normalized pixel value and centroid location.
    props = regionprops(segments, intensity_image=image)
    
    num_nodes = len(props)
    features = np.zeros((num_nodes, C + 2)) 
    centroids = np.zeros((num_nodes, 2))

    for i, prop in enumerate(props):
        # Extract and normalize centroid coordinates to [0, 1] relative to image dimensions
        cy, cx = prop.centroid
        centroids[i] = [cy, cx]
        norm_cy, norm_cx = cy / H, cx / W
        
        # Calculate mean intensity (color) for the superpixel
        mean_color = prop.image_intensity.mean(axis=(0, 1)) 
        
        # Concatenate color and location for the node feature vector
        features[i] = np.concatenate([mean_color, [norm_cy, norm_cx]])

    # 3. Construct edges based on the Radius Graph logic
    # Calculate pairwise Euclidean distances between all centroids
    dist_matrix = distance_matrix(centroids, centroids)
    
    # Find pairs where the distance is less than or equal to the radius
    # We exclude 0 to avoid self-loops
    edges = np.argwhere((dist_matrix <= radius) & (dist_matrix > 0))
    
    # Convert to PyTorch tensors
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)

    # 4. Package into a PyTorch Geometric Data object
    graph_data = Data(x=x, edge_index=edge_index)
    
    return graph_data