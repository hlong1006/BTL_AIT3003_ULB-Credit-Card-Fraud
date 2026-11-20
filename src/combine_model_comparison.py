"""
Merges P0 baseline models (XGBoost, Isolation Forest, Autoencoder)
with P3 GNN models (GCN, GraphSAGE, GAT) into a unified benchmark table.
Outputs:
- outputs/tables/all_models_comparison.csv
"""

import os
import pandas as pd


def combine_benchmarks():
    p0_path = 'outputs/tables/p0_model_comparison.csv'
    gnn_runs_path = 'outputs/tables/gnn_seed_runs.csv'
    
    if not os.path.exists(p0_path):
        print(f"Warning: {p0_path} not found.")
        return
    if not os.path.exists(gnn_runs_path):
        print(f"Warning: {gnn_runs_path} not found.")
        return
        
    df_p0 = pd.read_csv(p0_path)
    df_gnn_runs = pd.read_csv(gnn_runs_path)
    
    # Calculate average across seeds for each GNN model, split, and threshold_method
    numeric_cols = [
        'n_samples', 'n_positive', 'n_negative', 'positive_rate',
        'pr_auc_ap', 'pr_auc_trapezoid', 'ap_over_prevalence', 'roc_auc',
        'threshold', 'precision', 'recall', 'f1', 'tp', 'fp', 'tn', 'fn',
        'alerts', 'training_time_seconds', 'scoring_time_seconds'
    ]
    
    gnn_summary = []
    for (model, split, thresh_m), group in df_gnn_runs.groupby(['model', 'split', 'threshold_method']):
        row = {
            'model': f"{model} (5-seed avg)",
            'split': split,
            'threshold_method': thresh_m,
            'score_direction': 'Higher probability = more illicit',
        }
        for col in numeric_cols:
            if col in group.columns:
                row[col] = group[col].mean()
        gnn_summary.append(row)
        
    df_gnn_avg = pd.DataFrame(gnn_summary)
    
    # Also include the best single run for each GNN model on Test
    best_single_runs = []
    for model, group in df_gnn_runs[df_gnn_runs['split'] == 'Test'].groupby('model'):
        # Pick best seed by pr_auc_ap under fixed threshold
        sub = group[group['threshold_method'] == 'Fixed threshold = 0.5']
        if len(sub) > 0:
            best_r = sub.loc[sub['pr_auc_ap'].idxmax()].to_dict()
            best_r['model'] = f"{model} (Best Seed {best_r['seed']})"
            best_single_runs.append(best_r)
            
    df_best_single = pd.DataFrame(best_single_runs)
    
    # Combine everything
    cols_order = df_p0.columns.tolist()
    # Filter only matching columns
    df_gnn_avg_sub = df_gnn_avg[[c for c in cols_order if c in df_gnn_avg.columns]]
    df_best_single_sub = df_best_single[[c for c in cols_order if c in df_best_single.columns]]
    
    df_combined = pd.concat([df_p0, df_gnn_avg_sub, df_best_single_sub], ignore_index=True)
    df_combined.to_csv('outputs/tables/all_models_comparison.csv', index=False)
    print("Saved combined comparison table to outputs/tables/all_models_comparison.csv")
    
    # Print formatted comparison for Test set
    print("\n" + "=" * 95)
    print("ALL MODELS COMPARISON ON TEST SET (Timesteps 35-49, Inductive Setting):")
    print("-" * 95)
    test_rows = df_combined[df_combined['split'] == 'Test']
    for _, r in test_rows.iterrows():
        print(f"{r['model']:<25} | Thresh: {str(r['threshold_method'])[:20]:<20} | PR-AUC: {r['pr_auc_ap']:.4f} | F1: {r['f1']:.4f} | Prec: {r['precision']:.4f} | Rec: {r['recall']:.4f}")
    print("=" * 95)


if __name__ == '__main__':
    combine_benchmarks()
