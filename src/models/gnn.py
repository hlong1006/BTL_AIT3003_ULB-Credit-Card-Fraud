"""
GNN Architectures for Elliptic Bitcoin Illicit Transaction Detection.
Includes:
- GCN (Graph Convolutional Network) with optional Skip/Residual connections
- GraphSAGE (Sample and Aggregate)
- GAT (Graph Attention Network)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (N, C), targets: (N,)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) with optional skip-connections.
    Supports variable depth (1, 2, 3, 4+ layers) to study over-smoothing.
    """
    def __init__(self, in_channels=165, hidden_channels=128, out_channels=2,
                 num_layers=2, dropout=0.3, use_skip=True):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_skip = use_skip
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
                
            self.convs.append(GCNConv(hidden_channels, out_channels))
            
        if self.use_skip and num_layers > 1:
            self.skip = nn.Linear(in_channels, out_channels)
        else:
            self.skip = None

    def forward(self, x, edge_index, return_embeddings=False):
        if self.num_layers == 1:
            logits = self.convs[0](x, edge_index)
            if return_embeddings:
                return logits, x
            return logits

        h = x
        for i in range(self.num_layers - 1):
            h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
        embedding = h
        logits = self.convs[-1](h, edge_index)
        
        if self.skip is not None:
            logits = logits + self.skip(x)
            
        if return_embeddings:
            return logits, embedding
        return logits


class GraphSAGE(nn.Module):
    """
    GraphSAGE Network.
    Uses inductive neighborhood aggregation with SAGEConv.
    """
    def __init__(self, in_channels=165, hidden_channels=128, out_channels=2,
                 num_layers=2, dropout=0.3, aggr='mean'):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels, aggr=aggr))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
                
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))

    def forward(self, x, edge_index, return_embeddings=False):
        if self.num_layers == 1:
            logits = self.convs[0](x, edge_index)
            if return_embeddings:
                return logits, x
            return logits

        h = x
        for i in range(self.num_layers - 1):
            h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
        embedding = h
        logits = self.convs[-1](h, edge_index)
        
        if return_embeddings:
            return logits, embedding
        return logits


class GAT(nn.Module):
    """
    Graph Attention Network (GAT).
    Learns dynamic edge attention weights between connected transactions.
    """
    def __init__(self, in_channels=165, hidden_channels=64, out_channels=2,
                 num_layers=2, heads=2, dropout=0.3):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        if num_layers == 1:
            self.convs.append(GATConv(in_channels, out_channels, heads=1, concat=False, dropout=dropout))
        else:
            self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
            
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=dropout))
                self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
                
            self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index, return_embeddings=False):
        if self.num_layers == 1:
            logits = self.convs[0](x, edge_index)
            if return_embeddings:
                return logits, x
            return logits

        h = x
        for i in range(self.num_layers - 1):
            h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
        embedding = h
        logits = self.convs[-1](h, edge_index)
        
        if return_embeddings:
            return logits, embedding
        return logits


def build_gnn_model(model_name, in_channels=165, hidden_channels=128, out_channels=2,
                     num_layers=2, dropout=0.3, **kwargs):
    """Factory helper to build GNN models."""
    name = model_name.lower().replace("-", "").replace("_", "")
    if name == "gcn":
        use_skip = kwargs.get("use_skip", True)
        return GCN(in_channels, hidden_channels, out_channels, num_layers, dropout, use_skip=use_skip)
    elif name == "skipgcn":
        return GCN(in_channels, hidden_channels, out_channels, num_layers, dropout, use_skip=True)
    elif name == "vanillagcn":
        return GCN(in_channels, hidden_channels, out_channels, num_layers, dropout, use_skip=False)
    elif name in ["graphsage", "sage"]:
        aggr = kwargs.get("aggr", "mean")
        return GraphSAGE(in_channels, hidden_channels, out_channels, num_layers, dropout, aggr=aggr)
    elif name == "gat":
        heads = kwargs.get("heads", 2)
        # For GAT, if hidden_channels=128, with heads=2 we can use hidden_channels=64
        h_dim = kwargs.get("gat_hidden", hidden_channels // heads if hidden_channels >= 64 else 32)
        return GAT(in_channels, h_dim, out_channels, num_layers, heads, dropout)
    else:
        raise ValueError(f"Unknown GNN model: {model_name}. Supported: GCN, GraphSAGE, GAT")
