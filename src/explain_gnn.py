"""
GNNExplainer for Elliptic Bitcoin Illicit Transaction Detection.
Explains predictions for true positive illicit transactions:
- Identifies critical payment edges (transaction flows)
- Identifies top contributing features (local vs aggregated)
- Visualizes ego-subgraphs with edge importance weights
Outputs:
- outputs/figures/gnn_explainer_case_*.png
- outputs/tables/gnn_explainer_cases.csv
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.explain import Explainer, GNNExplainer

from src.models.gnn import build_gnn_model


class ModelWrapper(nn.Module):
    """Wrapper to output probabilities for Explainer."""
    def __init__(self, base_model):
        super(ModelWrapper, self).__init__()
        self.base_model = base_model

    def forward(self, x, edge_index):
        logits = self.base_model(x, edge_index)
        return F.softmax(logits, dim=-1)


def run_gnn_explainer(num_cases=3, epochs=100):
    print("=" * 70)
    print("STARTING GNNEXPLAINER CASE STUDIES")
    print("=" * 70)
    
    checkpoint_path = 'outputs/models/best_gnn_model.pt'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Run training first.")
        
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_name = ckpt['model_name']
    state_dict = ckpt['state_dict']
    print(f"Loaded best checkpoint: {model_name} (Seed {ckpt['seed']})")
    
    npz = np.load('data/processed/processed_data.npz')
    X = torch.tensor(npz['X_full'], dtype=torch.float32)
    edge_index = torch.tensor(npz['edge_index'], dtype=torch.long)
    y = npz['y']
    tx_ids = npz['tx_ids']
    timestep = npz['timestep']
    test_mask = npz['test_mask']
    
    # Instantiate and load model
    base_model = build_gnn_model(model_name, in_channels=165, hidden_channels=128, out_channels=2, num_layers=2)
    base_model.load_state_dict(state_dict)
    base_model.eval()
    
    model = ModelWrapper(base_model)
    
    # Get test predictions
    best_probs = np.load('outputs/predictions/best_gnn_probs.npy')
    
    # Select candidate True Positive nodes in Test set with highest confidence and interesting degrees
    test_indices = np.where(test_mask)[0]
    tp_indices = [idx for idx in test_indices if y[idx] == 1 and best_probs[idx] >= 0.70]
    
    # Compute in/out degree for each TP
    src_nodes, dst_nodes = edge_index[0].numpy(), edge_index[1].numpy()
    total_degrees = np.bincount(src_nodes, minlength=len(X)) + np.bincount(dst_nodes, minlength=len(X))
    
    # Sort TP candidates by degree to find informative subgraphs
    candidate_records = []
    for idx in tp_indices:
        deg = total_degrees[idx]
        if deg >= 2:  # at least connected to some transactions
            candidate_records.append((int(idx), deg, best_probs[idx]))
            
    candidate_records.sort(key=lambda x: (x[1], x[2]), reverse=True)
    selected_nodes = [int(rec[0]) for rec in candidate_records[:num_cases]]
    
    if len(selected_nodes) == 0:
        # Fallback to top confidence TP
        selected_nodes = [int(i) for i in sorted(tp_indices, key=lambda i: best_probs[i], reverse=True)[:num_cases]]
        
    print(f"Selected True Positive illicit nodes for explanation: {selected_nodes}")
    
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='probs',
        ),
    )
    
    case_results = []
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/tables', exist_ok=True)
    
    for case_num, target_idx in enumerate(selected_nodes, start=1):
        target_tx = int(tx_ids[target_idx])
        target_t = int(timestep[target_idx])
        target_prob = float(best_probs[target_idx])
        
        print(f"\n---> Analyzing Case {case_num}: Node Index {target_idx} (txId: {target_tx}, Timestep: {target_t}, Illicit Prob: {target_prob:.4f})")
        
        # Extract 2-hop ego subgraph
        target_idx = int(target_idx)
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            target_idx, num_hops=2, edge_index=edge_index, relabel_nodes=True
        )
        sub_x = X[subset]
        target_local_idx = mapping.item()
        
        # Run explainer on the ego subgraph
        explanation = explainer(sub_x, sub_edge_index, index=target_local_idx)
        
        edge_mask = explanation.edge_mask.detach().cpu().numpy()
        node_mask = explanation.node_mask.detach().cpu().numpy()
        
        # Feature importance for target node
        target_feat_imp = node_mask[target_local_idx]
        top_feat_indices = np.argsort(target_feat_imp)[::-1][:10]
        top_feat_scores = target_feat_imp[top_feat_indices]
        
        # Distinguish local (0..92) vs aggregate (93..164) features
        top_features_desc = []
        for fi, score in zip(top_feat_indices, top_feat_scores):
            f_type = "Local" if fi < 93 else "Aggregated (1-hop)"
            top_features_desc.append(f"Feat_{fi} ({f_type}): {score:.4f}")
            
        case_results.append({
            'case_num': case_num,
            'node_idx': target_idx,
            'tx_id': target_tx,
            'timestep': target_t,
            'illicit_prob': target_prob,
            'num_ego_nodes': len(subset),
            'num_ego_edges': sub_edge_index.size(1),
            'top_feature_indices': list(top_feat_indices),
            'top_feature_scores': [float(s) for s in top_feat_scores],
            'top_features_desc': "; ".join(top_features_desc)
        })
        
        # Visualize Ego-network and Feature Importance
        fig = plt.figure(figsize=(15, 6))
        
        # 1. Subgraph visualization
        ax1 = fig.add_subplot(1, 2, 1)
        G = nx.DiGraph()
        
        # Add nodes
        subset_np = subset.cpu().numpy()
        for local_i, global_i in enumerate(subset_np):
            label_val = y[global_i]
            G.add_node(local_i, global_id=global_i, label=label_val, is_target=(local_i == target_local_idx))
            
        # Add edges with weights
        sub_edge_np = sub_edge_index.cpu().numpy()
        for e_i in range(sub_edge_np.shape[1]):
            u = int(sub_edge_np[0, e_i])
            v = int(sub_edge_np[1, e_i])
            w = float(edge_mask[e_i]) if e_i < len(edge_mask) else 0.1
            G.add_edge(u, v, weight=w)
            
        # Layout
        pos = nx.spring_layout(G, seed=42)
        
        # Node colors
        node_colors = []
        node_sizes = []
        for n in G.nodes():
            if n == target_local_idx:
                node_colors.append('#e74c3c')  # red target illicit
                node_sizes.append(900)
            else:
                lbl = G.nodes[n]['label']
                if lbl == 1:
                    node_colors.append('#e67e22')  # orange illicit neighbor
                elif lbl == 0:
                    node_colors.append('#2ecc71')  # green licit neighbor
                else:
                    node_colors.append('#95a5a6')  # gray unknown
                node_sizes.append(450)
                
        # Edge weights & styling
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_w = max(edge_weights) if len(edge_weights) > 0 and max(edge_weights) > 0 else 1.0
        edge_widths = [1.0 + 4.0 * (w / max_w) for w in edge_weights]
        edge_colors = [plt.cm.Blues(0.3 + 0.7 * (w / max_w)) for w in edge_weights]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax1, edgecolors='black', linewidths=1.2)
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, arrowsize=14, arrowstyle='-|>', ax=ax1, connectionstyle='arc3,rad=0.1')
        
        # Node labels (txId or local index)
        labels = {n: f"Target\n#{subset_np[n]}" if n == target_local_idx else f"#{subset_np[n]}" for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold', ax=ax1)
        
        ax1.set_title(
            f"Case {case_num}: Ego Graph for Target Node #{target_idx}\n"
            f"(P(illicit) = {target_prob:.2%}, Timestep = {target_t})\n"
            f"Thicker edges = higher GNNExplainer attribution",
            fontsize=12, fontweight='bold'
        )
        ax1.axis('off')
        
        # Legend for nodes
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Target Illicit (Predicted)', markerfacecolor='#e74c3c', markersize=12, markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', label='Illicit Neighbor', markerfacecolor='#e67e22', markersize=10, markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', label='Licit Neighbor', markerfacecolor='#2ecc71', markersize=10, markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', label='Unknown Neighbor', markerfacecolor='#95a5a6', markersize=10, markeredgecolor='black'),
        ]
        ax1.legend(handles=legend_elements, loc='lower left', fontsize=9)
        
        # 2. Feature importance bar chart
        ax2 = fig.add_subplot(1, 2, 2)
        y_pos = np.arange(len(top_feat_indices))
        feat_labels = [f"Feat {fi} ({'Local' if fi < 93 else 'Agg 1-hop'})" for fi in top_feat_indices]
        bar_colors = ['#3498db' if fi < 93 else '#9b59b6' for fi in top_feat_indices]
        
        ax2.barh(y_pos, top_feat_scores[::-1], color=bar_colors[::-1], edgecolor='black', alpha=0.85)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(feat_labels[::-1], fontsize=10)
        ax2.set_xlabel('GNNExplainer Attribution Score', fontsize=11)
        ax2.set_title(f'Top 10 Most Influential Features for Node #{target_idx}', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        feat_legends = [
            Line2D([0], [0], color='#3498db', lw=6, label='Local Feature (0-92)'),
            Line2D([0], [0], color='#9b59b6', lw=6, label='Aggregated Feature (93-164)')
        ]
        ax2.legend(handles=feat_legends, loc='lower right', fontsize=9)
        
        plt.tight_layout()
        fig_path = f'outputs/figures/gnn_explainer_case_{case_num}_node_{target_idx}.png'
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Saved explanation plot: {fig_path}")
        
    df_cases = pd.DataFrame(case_results)
    df_cases.to_csv('outputs/tables/gnn_explainer_cases.csv', index=False)
    print("\nSaved GNNExplainer cases table to outputs/tables/gnn_explainer_cases.csv")
    return df_cases


if __name__ == '__main__':
    run_gnn_explainer(num_cases=3, epochs=100)
