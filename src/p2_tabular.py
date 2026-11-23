# -*- coding: utf-8 -*-
"""
P2 (Quyen) - Nhom A: Tabular baselines.
Logistic Regression / Random Forest / XGBoost x 3 feature configurations:
  local93         (93 local features, timestep excluded)
  full165         (93 local + 72 aggregated)
  local93+graph11 (93 local + 11 structural features from P1)

- Scaler fit on TRAIN only (LogReg).
- Imbalance: class_weight='balanced' (LogReg/RF), scale_pos_weight (XGB).
- Small hyperparameter grid selected on VALIDATION (PR-AUC).
- Multi-seed (42, 43, 44) for RF/XGB -> mean +/- std.
Outputs: tabular_benchmark.csv, tabular_benchmark_summary.csv,
         best_tabular_probs.npy (all 203,769 nodes, for P0 & P4),
         best_tabular_per_timestep.csv, best model file + info json.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from p2_common import (
    SEEDS, THRESHOLD, TABLES, PREDICTIONS, MODELS,
    load_data, build_feature_configs, set_seed, evaluate,
    per_timestep_metrics, save_json, Timer,
)


def _row(model, config, seed, params, val_m, test_m, fit_s, selected=True):
    return {
        "model": model, "config": config, "seed": seed,
        "params": json.dumps(params, sort_keys=True), "selected_params": selected,
        "val_pr_auc": val_m["pr_auc"], "val_f1": val_m["f1"],
        "test_pr_auc": test_m["pr_auc"], "test_roc_auc": test_m["roc_auc"],
        "test_f1": test_m["f1"], "test_precision": test_m["precision"],
        "test_recall": test_m["recall"],
        "test_tp": test_m["tp"], "test_fp": test_m["fp"], "test_fn": test_m["fn"],
        "fit_time_s": round(fit_s, 3),
    }


def run_logreg(name, cfg, data, tr, va, te):
    X, y = cfg["X"], data["y"]
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xva, Xte = scaler.transform(X[tr]), scaler.transform(X[va]), scaler.transform(X[te])
    tuning, best = [], None
    for C in (0.01, 0.1, 1.0):
        set_seed(SEEDS[0])
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
        with Timer() as t:
            clf.fit(Xtr, y[tr])
        pva = clf.predict_proba(Xva)[:, 1]
        vm = evaluate(y[va], pva)
        tuning.append({"model": "LogReg", "config": name, "C": C,
                       "val_pr_auc": vm["pr_auc"], "fit_time_s": round(t.seconds, 3)})
        if best is None or vm["pr_auc"] > best[0]:
            best = (vm["pr_auc"], C, clf, t.seconds)
    _, C, clf, fit_s = best
    pte = clf.predict_proba(Xte)[:, 1]
    row = _row("LogReg", name, SEEDS[0], {"C": C, "class_weight": "balanced"},
               evaluate(y[va], clf.predict_proba(Xva)[:, 1]), evaluate(y[te], pte), fit_s)
    return [row], tuning, {"type": "logreg", "scaler": scaler, "clf": clf, "params": {"C": C}}


def run_rf(name, cfg, data, tr, va, te):
    X, y = cfg["X"], data["y"]
    base = dict(n_estimators=300, min_samples_leaf=2,
                class_weight="balanced_subsample", n_jobs=-1)
    tuning, fits = [], {}
    for depth in (16, 28):
        set_seed(SEEDS[0])
        clf = RandomForestClassifier(max_depth=depth, random_state=SEEDS[0], **base)
        with Timer() as t:
            clf.fit(X[tr], y[tr])
        pva = clf.predict_proba(X[va])[:, 1]
        vm = evaluate(y[va], pva)
        tuning.append({"model": "RF", "config": name, "max_depth": depth,
                       "val_pr_auc": vm["pr_auc"], "fit_time_s": round(t.seconds, 3)})
        fits[depth] = (vm["pr_auc"], clf, t.seconds)
    best_depth = max(fits, key=lambda d: fits[d][0])
    rows = []
    for seed in SEEDS:
        if seed == SEEDS[0]:
            _, clf, fit_s = fits[best_depth]
        else:
            set_seed(seed)
            clf = RandomForestClassifier(max_depth=best_depth, random_state=seed, **base)
            with Timer() as t:
                clf.fit(X[tr], y[tr])
            fit_s = t.seconds
        vm = evaluate(y[va], clf.predict_proba(X[va])[:, 1])
        tm = evaluate(y[te], clf.predict_proba(X[te])[:, 1])
        rows.append(_row("RF", name, seed, {"max_depth": best_depth, **base}, vm, tm, fit_s))
        if seed == SEEDS[0]:
            best_model = {"type": "rf", "clf": clf,
                          "params": {"max_depth": best_depth, **base}}
    return rows, tuning, best_model


def run_xgb(name, cfg, data, tr, va, te):
    X, y = cfg["X"], data["y"]
    spw = float((y[tr] == 0).sum() / (y[tr] == 1).sum())
    base = dict(learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=5, scale_pos_weight=spw, tree_method="hist",
                n_jobs=8, eval_metric="aucpr")
    tuning, fits = [], {}
    for depth in (4, 6, 8):
        for n_est in (300, 600):
            set_seed(SEEDS[0])
            clf = XGBClassifier(max_depth=depth, n_estimators=n_est,
                                random_state=SEEDS[0], **base)
            with Timer() as t:
                clf.fit(X[tr], y[tr])
            pva = clf.predict_proba(X[va])[:, 1]
            vm = evaluate(y[va], pva)
            tuning.append({"model": "XGB", "config": name, "max_depth": depth,
                           "n_estimators": n_est, "val_pr_auc": vm["pr_auc"],
                           "fit_time_s": round(t.seconds, 3)})
            fits[(depth, n_est)] = (vm["pr_auc"], clf, t.seconds)
    best_key = max(fits, key=lambda k: fits[k][0])
    rows = []
    for seed in SEEDS:
        if seed == SEEDS[0]:
            _, clf, fit_s = fits[best_key]
        else:
            set_seed(seed)
            clf = XGBClassifier(max_depth=best_key[0], n_estimators=best_key[1],
                                random_state=seed, **base)
            with Timer() as t:
                clf.fit(X[tr], y[tr])
            fit_s = t.seconds
        vm = evaluate(y[va], clf.predict_proba(X[va])[:, 1])
        tm = evaluate(y[te], clf.predict_proba(X[te])[:, 1])
        params = {"max_depth": best_key[0], "n_estimators": best_key[1], **base}
        rows.append(_row("XGB", name, seed, params, vm, tm, fit_s))
        if seed == SEEDS[0]:
            best_model = {"type": "xgb", "clf": clf, "params": params}
    return rows, tuning, best_model


def main():
    print("[P2-tabular] Loading shared data ...", flush=True)
    data = load_data()
    configs = build_feature_configs(data)
    tr, va, te = data["train_mask"], data["val_mask"], data["test_mask"]
    y = data["y"]
    print(f"  train={tr.sum()} val={va.sum()} test={te.sum()} "
          f"(illicit train={(y[tr]==1).sum()}, test={(y[te]==1).sum()})", flush=True)

    all_rows, all_tuning, fitted = [], [], {}
    runners = {"LogReg": run_logreg, "RF": run_rf, "XGB": run_xgb}
    for cname, cfg in configs.items():
        for mname, runner in runners.items():
            print(f"[P2-tabular] {mname} x {cname} (dim={cfg['X'].shape[1]}) ...", flush=True)
            rows, tuning, model = runner(cname, cfg, data, tr, va, te)
            all_rows += rows
            all_tuning += tuning
            fitted[(mname, cname)] = (model, rows)
            best_seed42 = rows[0]
            print(f"    val PR-AUC={best_seed42['val_pr_auc']:.4f} "
                  f"test PR-AUC={best_seed42['test_pr_auc']:.4f} "
                  f"F1={best_seed42['test_f1']:.4f}", flush=True)

    bench = pd.DataFrame(all_rows)
    bench.to_csv(TABLES / "tabular_benchmark.csv", index=False)
    pd.DataFrame(all_tuning).to_csv(TABLES / "tabular_tuning.csv", index=False)

    # mean +/- std across seeds (LogReg deterministic -> 1 seed)
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
    summary.to_csv(TABLES / "tabular_benchmark_summary.csv", index=False)
    print("\n[P2-tabular] Summary (test, mean over seeds):")
    print(summary.to_string(index=False), flush=True)

    # ---- best model by VALIDATION PR-AUC (seed 42 runs) ----
    seed42 = bench[bench["seed"] == SEEDS[0]]
    best = seed42.loc[seed42["val_pr_auc"].idxmax()]
    mname, cname = best["model"], best["config"]
    model, _ = fitted[(mname, cname)]
    cfg = configs[cname]
    print(f"\n[P2-tabular] BEST by val PR-AUC: {mname} x {cname} "
          f"(val={best['val_pr_auc']:.4f}, test={best['test_pr_auc']:.4f})", flush=True)

    # probabilities for ALL nodes (message-passing-free, pure tabular)
    Xall = cfg["X"]
    if model["type"] == "logreg":
        probs_all = model["clf"].predict_proba(model["scaler"].transform(Xall))[:, 1]
    else:
        probs_all = model["clf"].predict_proba(Xall)[:, 1]
    probs_all = probs_all.astype(np.float32)
    np.save(PREDICTIONS / "best_tabular_probs.npy", probs_all)

    # save model
    if model["type"] == "xgb":
        model_file = "best_tabular_xgb.json"
        model["clf"].save_model(MODELS / model_file)
    elif model["type"] == "rf":
        model_file = "best_tabular_rf.joblib"
        joblib.dump(model["clf"], MODELS / model_file)
    else:
        model_file = "best_tabular_logreg.joblib"
        joblib.dump({"scaler": model["scaler"], "clf": model["clf"]}, MODELS / model_file)

    info = {
        "branch": "P2 (Quyen)", "model": mname, "config": cname, "seed": SEEDS[0],
        "params": model["params"], "model_file": model_file,
        "feature_names": cfg["names"], "feature_groups": cfg["groups"],
        "selection": "max validation PR-AUC among seed-42 runs",
        "validation": evaluate(y[va], probs_all[va]),
        "test": evaluate(y[te], probs_all[te]),
        "probs_file": "best_tabular_probs.npy",
    }
    save_json(info, TABLES / "best_tabular_info.json")

    pt = pd.concat([
        per_timestep_metrics(data, probs_all, "val_mask").assign(split="validation"),
        per_timestep_metrics(data, probs_all, "test_mask").assign(split="test"),
    ], ignore_index=True)
    pt.to_csv(TABLES / "best_tabular_per_timestep.csv", index=False)
    print("[P2-tabular] Saved benchmark, best probs, per-timestep metrics, model.", flush=True)


if __name__ == "__main__":
    main()
