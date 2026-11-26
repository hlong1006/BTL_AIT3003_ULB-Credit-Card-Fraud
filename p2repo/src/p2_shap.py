"""P2 / SHAP + False-Negative analysis on the tabular XGBoost models.

Questions answered:
  Q1. Which features matter most; local (93) vs aggregated (72) vs graph (P1)?
  Q2. What distinguishes false negatives from true positives (degree, timestep, SHAP)?

SHAP values use XGBoost's exact TreeSHAP (pred_contribs) -> margin (log-odds) space.
Run:  python -m src.p2_shap
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import precision_recall_curve

from src import p2_common as C

CONFIGS = ["full165", "local93+graph"]
N_BG = 3000          # random labeled test nodes used for global SHAP (+ all FN/TP below)


def group_of(col):
    if col.startswith("local_"): return "local"
    if col.startswith("agg_"): return "aggregated"
    return "graph(P1)"


def contribs(model, X):
    b = model.get_booster()
    c = b.predict(xgb.DMatrix(X), pred_contribs=True)
    return c[:, :-1], c[:, -1]


def main():
    C.ensure_dirs()
    base = C.load_base()
    y = base["y"].astype(int); ts = base["timestep"]
    tr, va, te = C.split_idx(base)
    sets = C.feature_sets(base, with_emb=False)
    G, gcols = C.load_graph_features(base["tx_ids"])
    gdf = pd.DataFrame(G, columns=gcols)
    rng = np.random.default_rng(42)
    summary = {}

    for cfg in CONFIGS:
        X, cols = sets[cfg]
        m = xgb.XGBClassifier(); m.load_model(str(C.MODEL_DIR / f"tabular_xgb_{cfg}.json"))
        probs = np.load(C.PRED_DIR / f"tabular_xgb_{cfg}_probs.npy")
        p, r, t = precision_recall_curve(y[va], probs[va])
        thr = float(t[int(np.argmax(2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)))])
        pred = probs >= thr
        tp = te[(y[te] == 1) & pred[te]]; fn = te[(y[te] == 1) & ~pred[te]]
        bg = rng.choice(te, size=min(N_BG, len(te)), replace=False)
        idx = np.unique(np.concatenate([bg, tp, fn]))
        S, _ = contribs(m, X[idx])
        pos = {n: i for i, n in enumerate(idx)}

        # ---- Q1 global importance
        imp = pd.DataFrame({"feature": cols, "mean_abs_shap": np.abs(S[[pos[i] for i in bg]]).mean(0)})
        imp["group"] = imp.feature.map(group_of)
        imp = imp.sort_values("mean_abs_shap", ascending=False)
        imp.to_csv(C.TABLE_DIR / f"shap_importance_{cfg}.csv", index=False)
        grp = imp.groupby("group").mean_abs_shap.agg(["sum", "mean", "count"])
        grp["share_of_total"] = grp["sum"] / grp["sum"].sum()
        grp.to_csv(C.TABLE_DIR / f"shap_group_share_{cfg}.csv")

        Sb = S[[pos[i] for i in bg]]
        shap.summary_plot(Sb, X[bg], feature_names=cols, max_display=20, show=False)
        plt.title(f"SHAP summary - XGBoost [{cfg}] (random test sample, log-odds)")
        plt.tight_layout(); plt.savefig(C.FIG_DIR / f"shap_summary_{cfg}.png", dpi=150); plt.close()
        top = imp.head(15).iloc[::-1]
        colors = top.group.map({"local": "#1f77b4", "aggregated": "#ff7f0e", "graph(P1)": "#2ca02c"})
        plt.figure(figsize=(7, 5)); plt.barh(top.feature, top.mean_abs_shap, color=colors)
        plt.xlabel("mean |SHAP|"); plt.title(f"Top-15 features [{cfg}]  blue=local orange=aggregated green=graph")
        plt.tight_layout(); plt.savefig(C.FIG_DIR / f"shap_top15_{cfg}.png", dpi=150); plt.close()

        # ---- Q2 false negatives vs true positives
        def prof(nodes, label):
            d = pd.DataFrame({"node_idx": nodes, "tx_id": base["tx_ids"][nodes], "timestep": ts[nodes],
                              "prob": probs[nodes], "group": label,
                              "in_degree": np.expm1(gdf.in_degree.values[nodes]),
                              "out_degree": np.expm1(gdf.out_degree.values[nodes]),
                              "pagerank": gdf.pagerank.values[nodes],
                              "core_number": gdf.core_number.values[nodes],
                              "clustering": gdf.clustering.values[nodes]})
            return d
        P = pd.concat([prof(tp, "TP"), prof(fn, "FN")])
        P.to_csv(C.TABLE_DIR / f"fn_vs_tp_nodes_{cfg}.csv", index=False)
        cmp_ = P.groupby("group")[["prob", "in_degree", "out_degree", "pagerank", "core_number", "clustering", "timestep"]].median()
        cmp_["n"] = P.groupby("group").size()
        cmp_.to_csv(C.TABLE_DIR / f"fn_vs_tp_summary_{cfg}.csv")
        by_t = pd.DataFrame({"fn": pd.Series(ts[fn]).value_counts(), "tp": pd.Series(ts[tp]).value_counts()}).fillna(0).astype(int).sort_index()
        by_t["recall"] = by_t.tp / (by_t.tp + by_t.fn)
        by_t.to_csv(C.TABLE_DIR / f"fn_by_timestep_{cfg}.csv")

        # SHAP profile: which features push FN vs TP
        S_fn, S_tp = S[[pos[i] for i in fn]], S[[pos[i] for i in tp]]
        d = pd.DataFrame({"feature": cols, "mean_shap_FN": S_fn.mean(0), "mean_shap_TP": S_tp.mean(0)})
        d["gap_TP_minus_FN"] = d.mean_shap_TP - d.mean_shap_FN
        d.reindex(d.gap_TP_minus_FN.abs().sort_values(ascending=False).index).head(20).to_csv(
            C.TABLE_DIR / f"fn_vs_tp_shap_gap_{cfg}.csv", index=False)

        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        ax[0].bar(by_t.index, by_t.recall, color="#d62728"); ax[0].axvline(43.5, ls="--", c="k", lw=.8)
        ax[0].set_title("Recall of illicit by test timestep"); ax[0].set_xlabel("timestep"); ax[0].set_ylim(0, 1)
        ax[1].hist([np.log10(P[P.group == "TP"].prob + 1e-6), np.log10(P[P.group == "FN"].prob + 1e-6)],
                   bins=25, label=["TP", "FN"], color=["#2ca02c", "#d62728"])
        ax[1].set_xlabel("log10 predicted prob"); ax[1].legend(); ax[1].set_title("Score of TP vs FN")
        plt.tight_layout(); plt.savefig(C.FIG_DIR / f"fn_analysis_{cfg}.png", dpi=150); plt.close()

        summary[cfg] = {"val_F1_opt_threshold": thr, "n_TP": int(len(tp)), "n_FN": int(len(fn)),
                        "group_share": grp["share_of_total"].round(4).to_dict(),
                        "top5": imp.head(5)[["feature", "mean_abs_shap"]].round(4).to_dict("records")}
        print(cfg, json.dumps(summary[cfg], indent=1), flush=True)

    json.dump(summary, open(C.TABLE_DIR / "shap_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
