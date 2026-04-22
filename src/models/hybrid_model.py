import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from .cnn_backbone import CNNBackbone
from .gnn_backbone import GNNBackbone

class VisionGraphCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # PyTorch's native MultiheadAttention requires (Seq_Len, Batch, Embed_Dim) if batch_first=False
        # We will use batch_first=True for easier handling: (Batch, Seq_Len, Embed_Dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, cnn_features, gnn_features, gnn_padding_mask):
        """
        cnn_features: (Batch, Num_Patches, Embed_Dim) -> Query
        gnn_features: (Batch, Num_Superpixels, Embed_Dim) -> Key, Value
        gnn_padding_mask: (Batch, Num_Superpixels) -> Boolean mask for padding
        """
        # Cross Attention: CNN queries the GNN
        attn_output, _ = self.cross_attn(
            query=cnn_features, 
            key=gnn_features, 
            value=gnn_features, 
            key_padding_mask=gnn_padding_mask
        )
        
        # Add & Norm (Residual Connection)
        x = self.layer_norm(cnn_features + attn_output)
        
        # Feed Forward Network
        ffn_output = self.ffn(x)
        
        # Add & Norm
        output = self.layer_norm2(x + ffn_output)
        return output

class HybridCrossAttentionModel(nn.Module):
    def __init__(self, num_classes: int, in_channels_img: int, in_channels_graph: int, embed_dim: int = 64):
        super().__init__()
        
        # 1. Backbones
        self.cnn = CNNBackbone(in_channels=in_channels_img, out_channels=embed_dim)
        self.gnn = GNNBackbone(in_channels=in_channels_graph, out_channels=embed_dim)
        
        # 2. Cross-Attention Fusion
        self.fusion = VisionGraphCrossAttention(embed_dim=embed_dim)
        
        # 3. Final Classification Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), # Pool sequence dimension
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, image, graph_data):
        batch_size = image.size(0)
        
        # --- CNN Pipeline ---
        # Returns shape: (Batch, embed_dim, H', W')
        cnn_feat = self.cnn(image) 
        
        # Flatten spatial dimensions: (Batch, Embed_Dim, H'*W') -> (Batch, H'*W', Embed_Dim)
        cnn_feat = cnn_feat.flatten(2).transpose(1, 2) 
        
        # --- GNN Pipeline ---
        # Returns shape: (Total_Nodes_in_Batch, embed_dim)
        gnn_feat_flat = self.gnn(graph_data.x, graph_data.edge_index, graph_data.batch)
        
        # Convert flat GNN batch into dense batches: (Batch, Max_Nodes, Embed_Dim)
        # mask shape: (Batch, Max_Nodes) - True where nodes are real, False where padded
        gnn_feat_dense, mask = to_dense_batch(gnn_feat_flat, graph_data.batch)
        
        # MultiheadAttention expects padding mask to be True for elements to IGNORE
        padding_mask = ~mask 

        # --- Mid-Fusion (Cross-Attention) ---
        # CNN queries the GNN superpixels
        fused_feat = self.fusion(cnn_features=cnn_feat, gnn_features=gnn_feat_dense, gnn_padding_mask=padding_mask)
        
        # --- Classification ---
        # Change shape for AdaptiveAvgPool1d: (Batch, Embed_Dim, Seq_Len)
        fused_feat = fused_feat.transpose(1, 2) 
        logits = self.classifier(fused_feat)
        
        return logits