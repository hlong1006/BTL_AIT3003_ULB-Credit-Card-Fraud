# -*- coding: utf-8 -*-
"""
P2 (Quyen) - Nhom C: Hybrid (GNN embeddings + original features -> XGBoost).
Embeddings: gnn_embeddings.npy (203769 x 128) from P3's best GraphSAGE (seed 456).

Variants:
  emb128          : GNN embedding only
  local93+emb128  : 93 local features + embedding
  full165+emb128  : 165 full features + embedding

Leakage note (checked, not assumed):
  The GNN was trained with loss on train_mask (ts 1-28) only; validation was used
  for early stopping. Test nodes (ts 35-49) are therefore OUT-OF-SAMPLE for the
  GNN, so hybrid test metrics are clean. Train-node embeddings are in-sample for
  the GNN -> we never report train metrics, and we quantify the val/test gap.

Outputs: hybrid_benchmark.csv, hybrid_benchmark_summary.csv,
         hybrid_leakage_check.json, best_hybrid_probs.npy (for P0 & P4),
         best_hybrid_per_timestep.csv, best hybrid model + info.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from p2_common import (
    SEEDS, TABLES, PREDICTIONS, MODELS, GNN_EMBEDDINGS_NPY, GNN_PROBS_NPY,
    load_data, build_feature_configs, evaluate, per_timestep_metrics, save_json,
)
from p2_tabular import run_xgb


def main():
    print("[P2-hybrid] Loading shared data + GNN embeddings ...", flush=True)
    data = load_data()
    base_cfgs = build_feature_configs(data)
    emb = np.load(GNN_EMBEDDINGS_NPY).astype(np.float32)
    assert emb.shape == (203769, 128)
    emb_names = [f"emb_{i}" for i in range(128)]
    emb_groups = ["gnn_embedding"] * 128

    variants = {
        "emb128": {"X": emb, "names": emb_names, "groups": emb_groups},
        "local93+emb128": {
            "X": np.hstack([base_cfgs["local93"]["X"], emb]),
            "names": base_cfgs["local93"]["names"] + emb_names,
            "groups": base_cfgs["local93"]["groups"] + emb_groups},
        "full165+emb128": {
            "X": np.hstack([base_cfgs["full165"]["X"], emb]),
            "names": base_cfgs["full165"]["names"] + emb_names,
            "groups": base_cfgs["full165"]["groups"] + emb_groups},
    }

    tr, va, te = data["train_mask"], data["val_mask"], data["test_mask"]
    y = data["y"]
    all_rows, all_tuning, fitted = [], [], {}
    for vname, cfg in variants.items():
        print(f"[P2-hybrid] XGB x {vname} (dim={cfg['X'].shape[1]}) ...", flush=True)
        rows, tuning, model = run_xgb(vname, cfg, data, tr, va, te)
        all_rows += rows
        all_tuning += tuning
        fitted[vname] = (model, rows, cfg)
        print(f"    val PR-AUC={rows[0]['val_pr_auc']:.4f} "
              f"test PR-AUC={rows[0]['test_pr_auc']:.4f} F1={rows[0]['test_f1']:.4f}",
              flush=True)

    bench = pd.DataFrame(all_rows)
    bench.to_csv(TABLES / "hybrid_benchmark.csv", index=False)
    pd.DataFrame(all_tuning).to_csv(TABLES / "hybrid_tuning.csv", index=False)
    summary = (bench.groupby(["model", "config"])
               .agg(n_seeds=("seed", "count"),
                    val_pr_auc_mean=("val_pr_auc", "mean"),
                    test_pr_auc_mean=("test_pr_auc", "mean"),
                    test_pr_auc_std=("test_pr_auc", "std"),
                    test_f1_mean=("test_f1", "mean"),
                    test_f1_std=("test_f1", "std"),
                    test_precision_mean=("test_precision", "mean"),
                    test_recall_mean=("test_recall", "mean"),
                    fit_time_s_mean=("fit_time_s", "mean"))
               .reset_index().sort_values("test_pr_auc_mean", ascending=False))
    summary.to_csv(TABLES / "hybrid_benchmark_summary.csv", index=False)
    print("\n[P2-hybrid] Summary (test, mean over seeds):")
    print(summary.to_string(index=False), flush=True)

    # ---- leakage / overfit check ----
    gnn_probs = np.load(GNN_PROBS_NPY)
    gnn_test = evaluate(y[te], gnn_probs[te])
    gnn_val = evaluate(y[va], gnn_probs[va])
    check = {
        "note": ("GNN trained on ts1-28 (train_mask), early stopping on val (ts29-34). "
                 "Test ts35-49 is out-of-sample for the GNN -> hybrid test metrics clean. "
                 "Train-node embeddings are in-sample for the GNN; train metrics never reported."),
        "gnn_probs_val_pr_auc": gnn_val["pr_auc"],
        "gnn_probs_test_pr_auc": gnn_test["pr_auc"],
        "variants": {},
    }
    for vname, (model, rows, cfg) in fitted.items():
        r = rows[0]
        check["variants"][vname] = {
            "val_pr_auc": r["val_pr_auc"], "test_pr_auc": r["test_pr_auc"],
            "val_minus_test_pr_auc": r["val_pr_auc"] - r["test_pr_auc"],
        }
    save_json(check, TABLES / "hybrid_leakage_check.json")

    # ---- best hybrid by validation PR-AUC ----
    seed42 = bench[bench["seed"] == SEEDS[0]]
    best = seed42.loc[seed42["val_pr_auc"].idxmax()]
    vname = best["config"]
    model, _, cfg = fitted[vname]
    probs_all = model["clf"].predict_proba(cfg["X"])[:, 1].astype(np.float32)
    np.save(PREDICTIONS / "best_hybrid_probs.npy", probs_all)
    model_file = "best_hybrid_xgb.json"
    model["clf"].save_model(MODELS / model_file)

    info = {
        "branch": "P2 (Quyen)", "model": "XGB", "config": vname, "seed": SEEDS[0],
        "params": model["params"], "model_file": model_file,
        "feature_names": cfg["names"], "feature_groups": cfg["groups"],
        "embedding_source": "P3 best GraphSAGE (seed 456), gnn_embeddings.npy",
        "validation": evaluate(y[va], probs_all[va]),
        "test": evaluate(y[te], probs_all[te]),
        "probs_file": "best_hybrid_probs.npy",
    }
    save_json(info, TABLES / "best_hybrid_info.json")
    pt = pd.concat([
        per_timestep_metrics(data, probs_all, "val_mask").assign(split="validation"),
        per_timestep_metrics(data, probs_all, "test_mask").assign(split="test"),
    ], ignore_index=True)
    pt.to_csv(TABLES / "best_hybrid_per_timestep.csv", index=False)
    print(f"\n[P2-hybrid] BEST by val PR-AUC: XGB x {vname} "
          f"(test PR-AUC={best['test_pr_auc']:.4f}, F1={best['test_f1']:.4f}). Saved.",
          flush=True)


if __name__ == "__main__":
    main()
