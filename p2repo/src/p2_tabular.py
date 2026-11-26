"""P2 / Bai 2a - Tabular benchmark: LogReg, RandomForest, XGBoost x feature configs.

Configs: local93 | full165 | local93+graph (P1)  [+ full165+graph as an extra ablation]
Protocol: hyperparameters picked on validation PR-AUC (seed 42) -> best config re-run
over SEEDS -> mean +- std. Models are fit on train timesteps only; test is never used
for selection. Outputs probs for ALL nodes (for P0 cost + P4 drift).

Run:  python -m src.p2_tabular
"""
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import p2_common as C

CONFIGS = ["local93", "full165", "local93+graph", "full165+graph"]
GRIDS = {
    "LogReg": [{"C": c} for c in (0.01, 0.1, 1.0)],
    "RandomForest": [{"max_depth": d, "min_samples_leaf": 2} for d in (12, 24)],
    "XGBoost": [{"max_depth": d, "learning_rate": 0.1} for d in (4, 6)],
}


def fit_predict(name, params, X, y, tr, va, seed):
    """Fit on train; return probs for all nodes + fitted model."""
    ytr = y[tr]
    if name == "LogReg":
        m = make_pipeline(StandardScaler(), LogisticRegression(
            C=params["C"], class_weight="balanced", max_iter=3000, random_state=seed))
        m.fit(X[tr], ytr)
    elif name == "RandomForest":
        m = RandomForestClassifier(
            n_estimators=200, max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"], class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed)
        m.fit(X[tr], ytr)
    else:
        spw = float((ytr == 0).sum() / (ytr == 1).sum())
        m = xgb.XGBClassifier(
            n_estimators=600, max_depth=params["max_depth"], learning_rate=params["learning_rate"],
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw, tree_method="hist",
            eval_metric="aucpr", early_stopping_rounds=40, random_state=seed, n_jobs=-1)
        m.fit(X[tr], ytr, eval_set=[(X[va], y[va])], verbose=False)
    return m.predict_proba(X)[:, 1].astype(np.float32), m


def val_opt_threshold(y, probs, va):
    p, r, t = precision_recall_curve(y[va], probs[va])
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.argmax(f1))])


def main():
    C.ensure_dirs()
    base = C.load_base()
    y = base["y"].astype(int)
    tr, va, te = C.split_idx(base)
    sets = C.feature_sets(base, with_emb=False)
    rows, runs, store = [], [], {}

    for cfg in CONFIGS:
        X, cols = sets[cfg]
        for name, grid in GRIDS.items():
            t0 = time.time()
            best = max(grid, key=lambda g: C.eval_split(y, fit_predict(name, g, X, y, tr, va, 42)[0], va)["pr_auc"])
            probs_seeds, models = [], []
            for s in (C.SEEDS if name != "LogReg" else C.SEEDS[:1]):
                pr, m = fit_predict(name, best, X, y, tr, va, s)
                thr = val_opt_threshold(y, pr, va)
                for split, idx in (("val", va), ("test", te)):
                    r = C.eval_split(y, pr, idx, 0.5)
                    r.update(model=name, config=cfg, seed=s, split=split, threshold=0.5)
                    runs.append(r)
                r = C.eval_split(y, pr, te, thr)
                r.update(model=name, config=cfg, seed=s, split="test", threshold=thr)
                runs.append(r)
                probs_seeds.append(pr); models.append(m)
            store[(name, cfg)] = (np.mean(probs_seeds, 0).astype(np.float32), models[0], best)
            print(f"[{cfg:14s}] {name:12s} best={best} ({time.time()-t0:.0f}s)", flush=True)

    runs = pd.DataFrame(runs)
    runs.to_csv(C.TABLE_DIR / "tabular_seed_runs.csv", index=False)
    runs["thr_kind"] = np.where(runs["threshold"] == 0.5, "fixed_0.5", "val_F1_opt")
    metrics = ["pr_auc", "roc_auc", "precision", "recall", "f1"]
    agg = runs.groupby(["config", "model", "split", "thr_kind"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(C.TABLE_DIR / "tabular_benchmark.csv", index=False)

    # best model chosen on VALIDATION PR-AUC (never on test)
    v = agg[(agg.split == "val") & (agg.thr_kind == "fixed_0.5")].sort_values("pr_auc_mean", ascending=False)
    top = v.iloc[0]
    key = (top["model"], top["config"])
    probs, model, params = store[key]
    np.save(C.PRED_DIR / "best_tabular_probs.npy", probs)
    # keep every XGB config's probs/model for hybrid + SHAP + downstream
    for (name, cfg), (pr, m, _) in store.items():
        if name == "XGBoost":
            np.save(C.PRED_DIR / f"tabular_xgb_{cfg}_probs.npy", pr)
            m.save_model(str(C.MODEL_DIR / f"tabular_xgb_{cfg}.json"))
    info = {"selected_by": "validation PR-AUC (fixed threshold 0.5)", "model": key[0], "config": key[1],
            "params": params, "seeds_averaged": C.SEEDS if key[0] != "LogReg" else C.SEEDS[:1],
            "val_pr_auc": float(top["pr_auc_mean"])}
    json.dump(info, open(C.TABLE_DIR / "best_tabular_info.json", "w"), indent=2)
    print("BEST:", info)


if __name__ == "__main__":
    main()
