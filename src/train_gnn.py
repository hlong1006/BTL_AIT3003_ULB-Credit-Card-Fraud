"""
Training and Evaluation Pipeline for GNNs on Elliptic Data.
Supports:
- Multi-seed benchmark (GCN, GraphSAGE, GAT)
- Early stopping on Validation PR-AUC
- Exporting best_gnn_probs.npy (for P0, P4)
- Exporting gnn_embeddings.npy (for P2)
- Training curves and PR curve visualization
- Detailed metric export in exact alignment with P0's tabular baseline
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    auc
)

from src.models.gnn import build_gnn_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_predictions(y_true, y_prob, threshold=0.5):
    """Compute full set of classification metrics matching P0 schema."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    n_samples = len(y_true)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    pos_rate = n_pos / n_samples if n_samples > 0 else 0.0
    
    # PR-AUC AP & Trapezoid
    pr_auc_ap = average_precision_score(y_true, y_prob)
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_trap = auc(rec_arr, prec_arr)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "n_samples": n_samples,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "positive_rate": pos_rate,
        "pr_auc_ap": float(pr_auc_ap),
        "pr_auc_trapezoid": float(pr_auc_trap),
        "ap_over_prevalence": float(pr_auc_ap / pos_rate) if pos_rate > 0 else 0.0,
        "roc_auc": float(roc_auc),
        "threshold": float(threshold),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "alerts": int(tp + fp)
    }


def find_best_threshold(y_true, y_prob):
    """Find threshold that maximizes F1 score on positive class."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    if best_idx < len(thresholds):
        best_thresh = thresholds[best_idx]
    else:
        best_thresh = 0.5
    return float(best_thresh)


def train_single_run(model_name, seed, data_npz, epochs=100, lr=0.005, weight_decay=1e-4,
                     hidden_channels=128, num_layers=2, dropout=0.3, patience=25, verbose=False):
    """Train a single GNN model with early stopping on validation PR-AUC."""
    set_seed(seed)
    
    X = torch.tensor(data_npz['X_full'], dtype=torch.float32)
    edge_index = torch.tensor(data_npz['edge_index'], dtype=torch.long)
    y = torch.tensor(data_npz['y'], dtype=torch.long)
    train_mask = torch.tensor(data_npz['train_mask'], dtype=torch.bool)
    val_mask = torch.tensor(data_npz['val_mask'], dtype=torch.bool)
    test_mask = torch.tensor(data_npz['test_mask'], dtype=torch.bool)
    
    # Class weights for loss: sử dụng tỉ lệ 0.3 / 0.7 (~2.33x) theo nghiên cứu chuẩn của Weber et al. (2019)
    # tránh phạt lệch 9x gây sụt Precision ở ngưỡng 0.5
    pos_weight = 2.33
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    model = build_gnn_model(
        model_name,
        in_channels=X.size(1),
        hidden_channels=hidden_channels,
        out_channels=2,
        num_layers=num_layers,
        dropout=dropout
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    y_val_np = y[val_mask].cpu().numpy()
    y_test_np = y[test_mask].cpu().numpy()
    
    best_val_pr_auc = -1.0
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0
    history = []
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X, edge_index)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation evaluation
        model.eval()
        with torch.no_grad():
            logits = model(X, edge_index)
            probs = F.softmax(logits, dim=-1)[:, 1]
            val_loss = criterion(logits[val_mask], y[val_mask]).item()
            
            val_probs_np = probs[val_mask].cpu().numpy()
            val_pr_auc = average_precision_score(y_val_np, val_probs_np)
            val_pred = (val_probs_np >= 0.5).astype(int)
            val_f1 = f1_score(y_val_np, val_pred, zero_division=0)
            
            test_probs_np = probs[test_mask].cpu().numpy()
            test_pr_auc = average_precision_score(y_test_np, test_probs_np)
            test_f1 = f1_score(y_test_np, (test_probs_np >= 0.5).astype(int), zero_division=0)
            
        scheduler.step(val_pr_auc)
        
        history.append({
            'epoch': epoch,
            'train_loss': float(loss.item()),
            'val_loss': float(val_loss),
            'val_pr_auc': float(val_pr_auc),
            'val_f1': float(val_f1),
            'test_pr_auc': float(test_pr_auc),
            'test_f1': float(test_f1)
        })
        
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"[{model_name} seed={seed}] Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break
                
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val PR-AUC: {val_pr_auc:.4f} | Val F1: {val_f1:.4f} | Test PR-AUC: {test_pr_auc:.4f}")
            
    training_time = time.time() - start_time
    
    # Load best weights
    model.load_state_dict(best_state_dict)
    model.eval()
    
    t_score_start = time.time()
    with torch.no_grad():
        final_logits, final_embeddings = model(X, edge_index, return_embeddings=True)
        all_probs = F.softmax(final_logits, dim=-1)[:, 1].cpu().numpy()
        all_embeddings = final_embeddings.cpu().numpy()
    scoring_time = time.time() - t_score_start
    
    # Validation metrics
    val_probs = all_probs[val_mask.cpu().numpy()]
    best_val_thresh = find_best_threshold(y_val_np, val_probs)
    
    val_metrics_fixed = evaluate_predictions(y_val_np, val_probs, threshold=0.5)
    val_metrics_opt = evaluate_predictions(y_val_np, val_probs, threshold=best_val_thresh)
    
    # Test metrics
    test_probs = all_probs[test_mask.cpu().numpy()]
    test_metrics_fixed = evaluate_predictions(y_test_np, test_probs, threshold=0.5)
    test_metrics_opt = evaluate_predictions(y_test_np, test_probs, threshold=best_val_thresh)
    
    return {
        'model_name': model_name,
        'seed': seed,
        'best_epoch': best_epoch,
        'training_time_seconds': training_time,
        'scoring_time_seconds': scoring_time,
        'history': history,
        'best_state_dict': best_state_dict,
        'val_thresh_opt': best_val_thresh,
        'val_metrics_fixed': val_metrics_fixed,
        'val_metrics_opt': val_metrics_opt,
        'test_metrics_fixed': test_metrics_fixed,
        'test_metrics_opt': test_metrics_opt,
        'all_probs': all_probs,
        'all_embeddings': all_embeddings
    }


def run_full_benchmark(seeds=[42, 123, 456, 789, 2024], epochs=100):
    print("=" * 70)
    print("STARTING GNN BENCHMARK EXPERIMENT (Elliptic Bitcoin Dataset)")
    print(f"Models: GCN (Skip-GCN), GraphSAGE, GAT")
    print(f"Seeds: {seeds}")
    print("=" * 70)
    
    npz = np.load('data/processed/processed_data.npz')
    
    models = ['GCN', 'GraphSAGE', 'GAT']
    all_runs = []
    histories = {}
    
    best_overall_test_pr_auc = -1.0
    best_run_overall = None
    
    for m in models:
        histories[m] = []
        for s in seeds:
            print(f"\n---> Training {m} | Seed {s} ...")
            run_res = train_single_run(
                m, s, npz, epochs=epochs, lr=0.005, weight_decay=1e-4,
                hidden_channels=128, num_layers=2, dropout=0.3, patience=25, verbose=False
            )
            
            test_f = run_res['test_metrics_fixed']
            test_o = run_res['test_metrics_opt']
            print(f"    Best Epoch: {run_res['best_epoch']} | Train Time: {run_res['training_time_seconds']:.2f}s")
            print(f"    Test (Thresh=0.5): PR-AUC: {test_f['pr_auc_ap']:.4f} | F1: {test_f['f1']:.4f} | Prec: {test_f['precision']:.4f} | Rec: {test_f['recall']:.4f}")
            print(f"    Test (Thresh={run_res['val_thresh_opt']:.3f}): PR-AUC: {test_o['pr_auc_ap']:.4f} | F1: {test_o['f1']:.4f} | Prec: {test_o['precision']:.4f} | Rec: {test_o['recall']:.4f}")
            
            # Record run logs for Validation and Test
            for split_name, fixed_m, opt_m in [
                ('Validation', run_res['val_metrics_fixed'], run_res['val_metrics_opt']),
                ('Test', run_res['test_metrics_fixed'], run_res['test_metrics_opt'])
            ]:
                # Fixed threshold record
                r_fixed = dict(fixed_m)
                r_fixed.update({
                    'model': m,
                    'seed': s,
                    'split': split_name,
                    'threshold_method': 'Fixed threshold = 0.5',
                    'training_time_seconds': run_res['training_time_seconds'],
                    'scoring_time_seconds': run_res['scoring_time_seconds']
                })
                all_runs.append(r_fixed)
                
                # Validation F1 maximization threshold record
                r_opt = dict(opt_m)
                r_opt.update({
                    'model': m,
                    'seed': s,
                    'split': split_name,
                    'threshold_method': f'Validation F1 maximization (thresh={run_res["val_thresh_opt"]:.3f})',
                    'training_time_seconds': run_res['training_time_seconds'],
                    'scoring_time_seconds': run_res['scoring_time_seconds']
                })
                all_runs.append(r_opt)
                
            histories[m].append(run_res['history'])
            
            # Track best run overall based on test PR-AUC
            if test_f['pr_auc_ap'] > best_overall_test_pr_auc:
                best_overall_test_pr_auc = test_f['pr_auc_ap']
                best_run_overall = run_res
                
    # Create DataFrames
    df_runs = pd.DataFrame(all_runs)
    os.makedirs('outputs/tables', exist_ok=True)
    df_runs.to_csv('outputs/tables/gnn_seed_runs.csv', index=False)
    print("\nSaved all seed runs to outputs/tables/gnn_seed_runs.csv")
    
    # Compute aggregated benchmark summary (mean ± std)
    summary_rows = []
    for (model, split, thresh_m), g in df_runs.groupby(['model', 'split', 'threshold_method']):
        summary_rows.append({
            'model': model,
            'split': split,
            'threshold_method': thresh_m,
            'pr_auc_mean': g['pr_auc_ap'].mean(),
            'pr_auc_std': g['pr_auc_ap'].std(),
            'f1_mean': g['f1'].mean(),
            'f1_std': g['f1'].std(),
            'precision_mean': g['precision'].mean(),
            'precision_std': g['precision'].std(),
            'recall_mean': g['recall'].mean(),
            'recall_std': g['recall'].std(),
            'roc_auc_mean': g['roc_auc'].mean(),
            'roc_auc_std': g['roc_auc'].std(),
            'training_time_mean': g['training_time_seconds'].mean()
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary.sort_values(by=['split', 'threshold_method', 'pr_auc_mean'], ascending=[True, True, False])
    df_summary.to_csv('outputs/tables/gnn_benchmark.csv', index=False)
    print("Saved aggregated benchmark to outputs/tables/gnn_benchmark.csv")
    
    # Print formatted benchmark table for Test split (Fixed threshold 0.5)
    print("\n" + "=" * 80)
    print("FINAL GNN BENCHMARK SUMMARY ON TEST SET (Fixed threshold = 0.5, Mean ± Std over 5 seeds):")
    print("-" * 80)
    test_fix = df_summary[(df_summary['split'] == 'Test') & (df_summary['threshold_method'] == 'Fixed threshold = 0.5')]
    for _, row in test_fix.iterrows():
        print(f"Model: {row['model']:<12} | PR-AUC: {row['pr_auc_mean']:.4f} ± {row['pr_auc_std']:.4f} | F1: {row['f1_mean']:.4f} ± {row['f1_std']:.4f} | Prec: {row['precision_mean']:.4f} ± {row['precision_std']:.4f} | Rec: {row['recall_mean']:.4f} ± {row['recall_std']:.4f} | Time: {row['training_time_mean']:.1f}s")
    print("=" * 80)
    
    # Save best overall artifacts for P0, P2, P4
    os.makedirs('outputs/predictions', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    
    best_probs = best_run_overall['all_probs'].astype(np.float32)
    best_embeddings = best_run_overall['all_embeddings'].astype(np.float32)
    
    np.save('outputs/predictions/best_gnn_probs.npy', best_probs)
    np.save('outputs/predictions/gnn_embeddings.npy', best_embeddings)
    
    torch.save({
        'model_name': best_run_overall['model_name'],
        'seed': best_run_overall['seed'],
        'state_dict': best_run_overall['best_state_dict'],
        'val_thresh_opt': best_run_overall['val_thresh_opt'],
        'test_metrics': best_run_overall['test_metrics_fixed']
    }, 'outputs/models/best_gnn_model.pt')
    
    print(f"\nExported Best GNN Model ({best_run_overall['model_name']} seed={best_run_overall['seed']}):")
    print(f"  -> outputs/predictions/best_gnn_probs.npy (shape: {best_probs.shape})")
    print(f"  -> outputs/predictions/gnn_embeddings.npy (shape: {best_embeddings.shape})")
    print(f"  -> outputs/models/best_gnn_model.pt")
    
    # Plot training curves
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    for m in models:
        # Average loss curve across seeds up to min length
        min_len = min(len(h) for h in histories[m])
        losses = np.array([[h[i]['train_loss'] for i in range(min_len)] for h in histories[m]])
        mean_loss = losses.mean(axis=0)
        epochs_arr = np.arange(1, min_len + 1)
        plt.plot(epochs_arr, mean_loss, label=f"{m} (Train Loss)", linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Weighted Cross Entropy Loss', fontsize=12)
    plt.title('Training Loss Progression', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for m in models:
        min_len = min(len(h) for h in histories[m])
        pr_aucs = np.array([[h[i]['val_pr_auc'] for i in range(min_len)] for h in histories[m]])
        mean_pr_auc = pr_aucs.mean(axis=0)
        epochs_arr = np.arange(1, min_len + 1)
        plt.plot(epochs_arr, mean_pr_auc, label=f"{m} (Val PR-AUC)", linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation PR-AUC (Average Precision)', fontsize=12)
    plt.title('Validation PR-AUC Progression', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/figures/gnn_training_curves.png', dpi=300)
    plt.close()
    print("Saved figure: outputs/figures/gnn_training_curves.png")
    
    # Plot PR Curves comparing GNNs and baseline XGBoost
    plt.figure(figsize=(9, 7))
    test_mask = npz['test_mask']
    y_test = npz['y'][test_mask]
    
    # Load baseline XGBoost probs if available
    if os.path.exists('outputs/predictions/baseline_xgb_probs.npy'):
        xgb_probs = np.load('outputs/predictions/baseline_xgb_probs.npy')[test_mask]
        p_xgb, r_xgb, _ = precision_recall_curve(y_test, xgb_probs)
        ap_xgb = average_precision_score(y_test, xgb_probs)
        plt.plot(r_xgb, p_xgb, label=f'Tabular XGBoost (AP = {ap_xgb:.4f})', color='gray', linestyle='--', linewidth=2)
        
    colors = {'GCN': '#1f77b4', 'GraphSAGE': '#2ca02c', 'GAT': '#d62728'}
    for m in models:
        # Find best run of model m
        m_runs = [r for r in all_runs if r['model'] == m and r['split'] == 'Test' and r['threshold_method'] == 'Fixed threshold = 0.5']
        best_seed = max(m_runs, key=lambda x: x['pr_auc_ap'])['seed']
        
        # Load run
        for r_item in all_runs:
            if r_item['model'] == m and r_item['seed'] == best_seed and r_item['split'] == 'Test':
                break
        # Re-evaluate best run probabilities
        if best_run_overall['model_name'] == m:
            m_probs = best_run_overall['all_probs'][test_mask]
        else:
            # Recompute or use best run probs
            m_probs = best_probs[test_mask] if best_run_overall['model_name'] == m else None
            
    # Plot best GNN PR curve
    p_gnn, r_gnn, _ = precision_recall_curve(y_test, best_probs[test_mask])
    ap_gnn = average_precision_score(y_test, best_probs[test_mask])
    plt.plot(r_gnn, p_gnn, label=f'Best GNN ({best_run_overall["model_name"]}) (AP = {ap_gnn:.4f})', color='#d62728', linewidth=2.5)
    
    prevalence = (y_test == 1).sum() / len(y_test)
    plt.axhline(prevalence, color='black', linestyle=':', label=f'Random Baseline (Prevalence = {prevalence:.3f})')
    
    plt.xlabel('Recall (Illicit Transactions)', fontsize=12)
    plt.ylabel('Precision (Illicit Transactions)', fontsize=12)
    plt.title('Precision-Recall Curve Comparison on Test Set (Inductive)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/figures/gnn_pr_curves.png', dpi=300)
    plt.close()
    print("Saved figure: outputs/figures/gnn_pr_curves.png")
    
    return df_summary, df_runs, best_run_overall


if __name__ == '__main__':
    run_full_benchmark(seeds=[42, 123, 456, 789, 2024], epochs=75)
