"""
Over-smoothing Experiment for GNNs on Elliptic Bitcoin Dataset.
Tests depths L in [1, 2, 3, 4] for:
- VanillaGCN (standard GCN without skip connections)
- Skip-GCN (GCN with linear skip connection from input)
- GraphSAGE
- GAT

Measures whether increasing depth degrades performance (over-smoothing)
or whether residual/skip connections mitigate over-smoothing.
Outputs:
- outputs/tables/gnn_oversmoothing_depth.csv
- outputs/figures/gnn_depth_oversmoothing.png
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from src.models.gnn import build_gnn_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_depth_experiment(depths=[1, 2, 3, 4], seed=42, epochs=60):
    print("=" * 70)
    print("RUNNING GNN OVER-SMOOTHING DEPTH EXPERIMENT")
    print(f"Depths: {depths}")
    print("=" * 70)
    
    set_seed(seed)
    npz = np.load('data/processed/processed_data.npz')
    
    X = torch.tensor(npz['X_full'], dtype=torch.float32)
    edge_index = torch.tensor(npz['edge_index'], dtype=torch.long)
    y = torch.tensor(npz['y'], dtype=torch.long)
    train_mask = torch.tensor(npz['train_mask'], dtype=torch.bool)
    val_mask = torch.tensor(npz['val_mask'], dtype=torch.bool)
    test_mask = torch.tensor(npz['test_mask'], dtype=torch.bool)
    
    pos_weight = 2.33
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], dtype=torch.float32))
    
    y_val_np = y[val_mask].cpu().numpy()
    y_test_np = y[test_mask].cpu().numpy()
    
    model_configs = [
        ('VanillaGCN', 'VanillaGCN'),
        ('Skip-GCN', 'SkipGCN'),
        ('GraphSAGE', 'GraphSAGE'),
        ('GAT', 'GAT')
    ]
    
    results = []
    
    for display_name, m_name in model_configs:
        for L in depths:
            print(f"---> Testing {display_name} with {L} layer(s) ...")
            set_seed(seed)
            model = build_gnn_model(
                m_name,
                in_channels=X.size(1),
                hidden_channels=128,
                out_channels=2,
                num_layers=L,
                dropout=0.3
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
            
            best_val_pr = -1.0
            best_state = None
            
            for epoch in range(1, epochs + 1):
                model.train()
                optimizer.zero_grad()
                logits = model(X, edge_index)
                loss = criterion(logits[train_mask], y[train_mask])
                loss.backward()
                optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    val_logits = model(X, edge_index)
                    val_probs = F.softmax(val_logits, dim=-1)[:, 1][val_mask].cpu().numpy()
                    val_pr = average_precision_score(y_val_np, val_probs)
                    if val_pr > best_val_pr:
                        best_val_pr = val_pr
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        
            # Evaluate best model
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                test_logits = model(X, edge_index)
                test_probs = F.softmax(test_logits, dim=-1)[:, 1][test_mask].cpu().numpy()
                test_pred = (test_probs >= 0.5).astype(int)
                
                test_pr_auc = average_precision_score(y_test_np, test_probs)
                test_f1 = f1_score(y_test_np, test_pred, zero_division=0)
                test_p = precision_score(y_test_np, test_pred, zero_division=0)
                test_r = recall_score(y_test_np, test_pred, zero_division=0)
                
            print(f"     L={L}: Val PR-AUC={best_val_pr:.4f} | Test PR-AUC={test_pr_auc:.4f} | Test F1={test_f1:.4f} | Prec={test_p:.4f} | Rec={test_r:.4f}")
            results.append({
                'model': display_name,
                'num_layers': L,
                'val_pr_auc': float(best_val_pr),
                'test_pr_auc': float(test_pr_auc),
                'test_f1': float(test_f1),
                'test_precision': float(test_p),
                'test_recall': float(test_r)
            })
            
    df_depth = pd.DataFrame(results)
    os.makedirs('outputs/tables', exist_ok=True)
    df_depth.to_csv('outputs/tables/gnn_oversmoothing_depth.csv', index=False)
    print("\nSaved depth experiment results to outputs/tables/gnn_oversmoothing_depth.csv")
    
    # Plotting Depth vs Metrics (Over-smoothing check)
    os.makedirs('outputs/figures', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    styles = {
        'VanillaGCN': {'color': '#e74c3c', 'marker': 'o', 'linestyle': '--'},
        'Skip-GCN': {'color': '#2980b9', 'marker': 's', 'linestyle': '-'},
        'GraphSAGE': {'color': '#27ae60', 'marker': '^', 'linestyle': '-'},
        'GAT': {'color': '#8e44ad', 'marker': 'D', 'linestyle': '-.'}
    }
    
    for display_name, _ in model_configs:
        sub = df_depth[df_depth['model'] == display_name]
        st = styles.get(display_name, {'marker': 'o'})
        axes[0].plot(sub['num_layers'], sub['test_f1'], label=display_name, linewidth=2.2, markersize=8, **st)
        axes[1].plot(sub['num_layers'], sub['test_pr_auc'], label=display_name, linewidth=2.2, markersize=8, **st)
        
    axes[0].set_title('Test F1 Score (Illicit) vs Layer Depth', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Number of GNN Layers (Depth L)', fontsize=11)
    axes[0].set_ylabel('Test F1 Score (Illicit)', fontsize=11)
    axes[0].set_xticks(depths)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    
    axes[1].set_title('Test PR-AUC vs Layer Depth (Over-smoothing Analysis)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Number of GNN Layers (Depth L)', fontsize=11)
    axes[1].set_ylabel('Test PR-AUC (Average Precision)', fontsize=11)
    axes[1].set_xticks(depths)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('outputs/figures/gnn_depth_oversmoothing.png', dpi=300)
    plt.close()
    print("Saved figure: outputs/figures/gnn_depth_oversmoothing.png")
    return df_depth


if __name__ == '__main__':
    run_depth_experiment(depths=[1, 2, 3, 4], seed=42, epochs=60)
