"""P2 / Bai 2b-hybrid - GNN hidden embedding (P3) + original features -> XGBoost.

Research question: does hybrid beat tabular-only and GNN-only?

LEAKAGE CAVEAT (stated in plan, step 8): P3's embeddings come from a GNN trained with the
labels of train nodes (ts 1-28), so train-node embeddings are 'too clean' and XGBoost can
over-trust them. We therefore report two protocols:

  A. standard   : fit XGB on train (ts 1-28) with embeddings as delivered.   [optimistic risk]
  B. clean-fit  : fit XGB ONLY on nodes the GNN never saw labels for:
                  ts 29-32 for fitting, ts 33-34 for early stopping, test ts 35-49.
                  Tabular-only is re-run under the SAME protocol for a fair comparison.

Run:  python -m src.p2_hybrid
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from src import p2_common as C

HYBRID_CFGS = ["local93+gnn", "full165+gnn", "local93+graph+gnn"]
TAB_CFGS = ["local93", "full165", "local93+graph"]


def fit_xgb(X, y, tr, es, seed, max_depth):
    spw = float((y[tr] == 0).sum() / (y[tr] == 1).sum())
    m = xgb.XGBClassifier(
        n_estimators=600, max_depth=max_depth, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=spw, tree_method="hist", eval_metric="aucpr",
        early_stopping_rounds=40, random_state=seed, n_jobs=-1)
    m.fit(X[tr], y[tr], eval_set=[(X[es], y[es])], verbose=False)
    return m.predict_proba(X)[:, 1].astype(np.float32), m


def run_protocol(tag, sets, cfgs, y, tr, es, eval_va, te):
    out, runs, probs_store = [], [], {}
    for cfg in cfgs:
        X = sets[cfg][0]
        depth = max((4, 6), key=lambda d: C.eval_split(y, fit_xgb(X, y, tr, es, 42, d)[0], es)["pr_auc"])
        ps = []
        for s in C.SEEDS:
            pr, _ = fit_xgb(X, y, tr, es, s, depth)
            ps.append(pr)
            for split, idx in (("train", tr), ("val_or_es", es), ("test", te)):
                r = C.eval_split(y, pr, idx); r.update(protocol=tag, config=cfg, seed=s, split=split, depth=depth)
                runs.append(r)
        probs_store[cfg] = np.mean(ps, 0).astype(np.float32)
        print(f"[{tag}] {cfg} depth={depth} done", flush=True)
    return runs, probs_store


def main():
    C.ensure_dirs()
    base = C.load_base()
    y = base["y"].astype(int)
    ts = base["timestep"]
    tr, va, te = C.split_idx(base)
    sets = C.feature_sets(base)

    # ---- Protocol A: standard (train 1-28, early-stop on 29-34)
    runsA, probsA = run_protocol("A_standard", sets, TAB_CFGS + HYBRID_CFGS, y, tr, va, va, te)
    # ---- Protocol B: clean-fit (labels the GNN never trained on)
    lab = base["labeled_mask"]
    fitB = np.where(lab & (ts >= 29) & (ts <= 32))[0]
    esB = np.where(lab & (ts >= 33) & (ts <= 34))[0]
    runsB, probsB = run_protocol("B_clean_fit", sets, TAB_CFGS + HYBRID_CFGS, y, fitB, esB, esB, te)

    runs = pd.DataFrame(runsA + runsB)
    runs.to_csv(C.TABLE_DIR / "hybrid_seed_runs.csv", index=False)
    m = ["pr_auc", "roc_auc", "precision", "recall", "f1"]
    agg = runs.groupby(["protocol", "config", "split"])[m].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(C.TABLE_DIR / "hybrid_benchmark.csv", index=False)

    # GNN-only reference (P3 best_gnn_probs), same test nodes
    gp = np.load(C.PRED_DIR / "best_gnn_probs.npy")
    ref = {"gnn_only_test": C.eval_split(y, gp, te), "gnn_only_val": C.eval_split(y, gp, va)}
    json.dump(ref, open(C.TABLE_DIR / "gnn_reference_metrics.json", "w"), indent=2)

    # pick best hybrid by VALIDATION (protocol A, early-stop split) PR-AUC
    a = agg[(agg.protocol == "A_standard") & (agg.split == "val_or_es") & agg.config.isin(HYBRID_CFGS)]
    best = a.sort_values("pr_auc_mean", ascending=False).iloc[0]["config"]
    np.save(C.PRED_DIR / "best_hybrid_probs.npy", probsA[best])
    for cfg, pr in probsA.items():
        np.save(C.PRED_DIR / f"hybrid_A_{cfg}_probs.npy", pr)
    json.dump({"best_hybrid_config_by_val": best}, open(C.TABLE_DIR / "best_hybrid_info.json", "w"))
    print("best hybrid by val:", best)


if __name__ == "__main__":
    main()
