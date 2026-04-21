```markdown
# Cross-Attention Driven Hybrid CNN-GNN using Superpixel Graphs

This repository contains the official PyTorch implementation of the **Cross-Attention Driven Hybrid CNN-GNN**. This architecture utilizes SLIC superpixel segmentation and Mid-Fusion Cross-Attention to provide structural regularization for image classification, effectively preventing model collapse on high-resolution, noisy medical radiography.

---

## 🛑 The Problem
Convolutional Neural Networks (CNNs) are the standard for computer vision, but they process images as rigid grids of pixels. This localized, pixel-level view creates two major blind spots:
1. **Lack of Global Topology:** Standard CNNs struggle to map long-range structural relationships (e.g., contextualizing a fuzzy opacity relative to a ribcage).
2. **Vulnerability to Noise:** On high-resolution, complex images like COVID-19 X-rays, shallow CNNs become overwhelmed by high-frequency clinical static, leading to catastrophic overfitting and "model collapse."

## 💡 Our Solution (Methodology)
To solve this, we abstract the raw pixel grid into a perceptually meaningful graph and fuse it with the visual data. Our architecture is built on three pillars:

* **Graph Generation (Relational Pathway):** We use the SLIC algorithm to group pixels into organic "superpixels" based on color and proximity. We extract the centroids of these superpixels to form a spatial Radius Graph, effectively creating a macro-level anatomical blueprint.
* **Dual-Stream Feature Extraction:** 
  * A 3-block CNN scans the raw image grid for local visual textures.
  * A 3-layer Graph Attention Network (GAT) processes the Radius Graph to map global topology.
* **Mid-Fusion Cross-Attention:** Instead of simple late fusion, the CNN spatial features act as a Query (Q) to dynamically attend to the GNN's structural node embeddings (K, V). This forces the visual filters to actively contextualize anomalies within the global anatomy.

---

## 📊 Datasets
The model was evaluated across three datasets representing a progressive gradient of spatial complexity and noise:

| Dataset | Complexity Level | Classes | Train Set | Test Set |
| :--- | :--- | :--- | :--- | :--- |
| **MNIST** | Structural Control (No Noise) | 10 | 60K | 10K |
| **CIFAR-10** | Color & Background Clutter | 10 | 50K | 10K |
| **COVID-19 X-Ray** | High-Res Clinical Static | 4 | 19K | 2K |

**Results Summary:** On the COVID-19 dataset, the baseline CNN collapsed to 57.04% accuracy by Epoch 10. Our Hybrid Model utilized the superpixel graph as an anchor to filter the clinical noise, holding stable at **70.14%**.

---

## ⚙️ Execution and Setup

This project is optimized for a Linux (Ubuntu) environment using Python and CUDA for accelerated graph and tensor operations. 

### Prerequisites
Ensure you have the following installed:
* Python 3.8+
* PyTorch (with CUDA support)
* PyTorch Geometric (PyG)
* scikit-image (for SLIC segmentation)
* NetworkX & Matplotlib (for visualization)

### Installation
1. Clone this repository:
```bash
git clone [https://github.com/Kaivalya16/Cross-Attention-driven-Hybrid-CNN-GNN-ussing-Superpixel-Graphs-for-Classification.git](https://github.com/Kaivalya16/Cross-Attention-driven-Hybrid-CNN-GNN-ussing-Superpixel-Graphs-for-Classification.git)
cd Cross-Attention-driven-Hybrid-CNN-GNN-ussing-Superpixel-Graphs-for-Classification
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Code

**1. Model Training**
To train the hybrid model on a specific dataset, execute the main training script. *(Note: Adjust arguments based on your specific script setup)*
```bash
python train.py --dataset covid19 --epochs 20 --batch_size 32
```

**2. Generating Superpixel Visualizations**
We provide specific scripts to replicate the NetworkX graph topography renderings for both standard and medical datasets:
```bash
# Generate visualizations for MNIST and CIFAR-10
python visualize_small_datasets.py 

# Generate anatomical graph visualizations for COVID-19 X-Rays
python replicate_covid_nx.py
```

---

## 👥 Authors

This project was developed by:
* **Kaivalya Vanmali** (M25DS003)
* **Vanshika Srivastava** (M25DS014)
* **Varun Shirbhayye** (M25DS015)
* **Akhileshkumar Yadav** (M25DS018)

*MTech in Data Science & Artificial Intelligence - ITT Bhilai*

---
```