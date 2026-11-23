# -*- coding: utf-8 -*-
"""
P2 (Quyen) - SHAP explanation + false-negative error analysis + case studies.
- SHAP (TreeExplainer) for best tabular XGB and best hybrid XGB on TEST nodes.
- Local vs aggregated vs graph vs embedding feature-group contribution.
- FN vs TP illicit: structural comparison (Mann-Whitney U + rank-biserial effect),
  FN rate per timestep (drift evidence for P4).
- 2-3 case studies: correctly detected illicit nodes, 1-hop ego subgraphs.
"""
from __future__ import annotations

import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import shap
from scipy.stats import mannwhitneyu
from xgboost import XGBClassifier

from p2_common import (
    THRESHOLD, TABLES, FIGURES, PREDICTIONS, MODELS, GRAPH_FEATURE_COLS,
    GNN_EMBEDDINGS_NPY, load_data, build_feature_configs, set_seed,
)

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})


def load_tree_model(info):
    mf = info["model_file"]
    if mf.endswith(".json"):
        m = XGBClassifier()
        m.load_model(MODELS / mf)
        return m
    if "logreg" in mf:
        raise RuntimeError("Best model is not tree-based; SHAP script expects XGB/RF.")
    return joblib.load(MODELS / mf)


def shap_analysis(tag, info, data, X_all):
    model = load_tree_model(info)
    te = data["test_mask"]
    Xte = X_all[te]
    names, groups = info["feature_names"], info["feature_groups"]
    explainer = shap.TreeExplainer(model)
    sv_file = PREDICTIONS / f"shap_values_{tag}_test.npy"
    if sv_file.exists():
        sv = np.load(sv_file)  # reuse cached (RF explain took ~8 min)
    else:
        sv = np.asarray(explainer.shap_values(Xte), dtype=np.float32)
        np.save(sv_file, sv)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv, dtype=np.float32)
    # RandomForest returns (n_samples, n_features, 2); XGB returns (n, d).
    if sv.ndim == 3:
        sv = sv[..., 1] if sv.shape[-1] == 2 else sv[..., -1]
    assert sv.shape == Xte.shape, f"unexpected SHAP shape {sv.shape} vs {Xte.shape}"
    np.save(sv_file, sv)
    np.save(PREDICTIONS / f"shap_{tag}_test_node_indices.npy", np.where(te)[0])

    # beeswarm + bar
    plt.figure()
    shap.summary_plot(sv, Xte, feature_names=names, max_display=20, show=False)
    plt.title(f"SHAP summary - {tag} (test nodes)")
    plt.savefig(FIGURES / f"shap_{tag}_beeswarm.png")
    plt.close("all")
    plt.figure()
    shap.summary_plot(sv, Xte, feature_names=names, plot_type="bar",
                      max_display=20, show=False)
    plt.title(f"SHAP mean |value| - {tag}")
    plt.savefig(FIGURES / f"shap_{tag}_bar.png")
    plt.close("all")

    # group contribution
    imp = np.abs(sv).mean(axis=0)
    gdf = pd.DataFrame({"feature": names, "group": groups, "mean_abs_shap": imp})
    gdf.sort_values("mean_abs_shap", ascending=False).to_csv(
        TABLES / f"shap_{tag}_feature_importance.csv", index=False)
    share = gdf.groupby("group")["mean_abs_shap"].sum()
    share = (share / share.sum()).sort_values(ascending=False)
    share.rename("share").to_csv(TABLES / f"shap_{tag}_group_share.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    share.plot.bar(ax=ax, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(share)])
    ax.set_ylabel("Share of total mean |SHAP|")
    ax.set_title(f"Feature-group contribution - {tag}")
    for i, v in enumerate(share.values):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center")
    plt.savefig(FIGURES / f"shap_{tag}_group_share.png")
    plt.close("all")
    print(f"[P2-shap] {tag}: group share = {share.round(4).to_dict()}", flush=True)
    return sv, share


def fn_analysis(data, probs, info):
    te, y, ts = data["test_mask"], data["y"], data["timestep"]
    illicit_te = te & (y == 1)
    pred = probs >= THRESHOLD
    tp_m = illicit_te & pred
    fn_m = illicit_te & ~pred
    Xg = data["X_graph"]
    cols = GRAPH_FEATURE_COLS

    rows = []
    for c_i, c in enumerate(cols):
        a, b = Xg[tp_m, c_i].astype(float), Xg[fn_m, c_i].astype(float)
        if len(a) > 0 and len(b) > 0 and len(np.unique(np.concatenate([a, b]))) > 1:
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            rbc = 1.0 - 2.0 * u / (len(a) * len(b))  # rank-biserial effect size
        else:
            p, rbc = np.nan, np.nan
        rows.append({"feature": c, "TP_median": np.median(a), "FN_median": np.median(b),
                     "TP_mean": a.mean(), "FN_mean": b.mean(),
                     "mannwhitney_p": p, "rank_biserial_effect": rbc,
                     "n_TP": len(a), "n_FN": len(b)})
    stats = pd.DataFrame(rows)
    stats.to_csv(TABLES / "fn_vs_tp_structural_stats.csv", index=False)
    print("[P2-shap] FN vs TP structural stats:\n", stats.to_string(index=False), flush=True)

    # per-node table (illicit test nodes)
    idx = np.where(illicit_te)[0]
    per_node = pd.DataFrame({
        "node_idx": idx, "tx_id": data["tx_ids"][idx], "timestep": ts[idx],
        "outcome": np.where(pred[idx], "TP", "FN"), "prob": probs[idx],
    })
    for c_i, c in enumerate(cols):
        per_node[c] = Xg[idx, c_i]
    per_node.to_csv(TABLES / "fn_analysis_per_node.csv", index=False)

    # FN rate per timestep
    rows = []
    for t in np.unique(ts[te]):
        m = illicit_te & (ts == t)
        n = int(m.sum())
        nfn = int((m & ~pred).sum())
        rows.append({"timestep": int(t), "n_illicit": n, "n_fn": nfn,
                     "fn_rate": nfn / n if n else np.nan})
    fnt = pd.DataFrame(rows)
    fnt.to_csv(TABLES / "fn_rate_by_timestep.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    bp = ax.boxplot([Xg[tp_m, 0] + Xg[tp_m, 1], Xg[fn_m, 0] + Xg[fn_m, 1]],
                    tick_labels=[f"TP (n={tp_m.sum()})", f"FN (n={fn_m.sum()})"],
                    showfliers=False)
    ax.set_yscale("symlog")
    ax.set_ylabel("total degree (in+out), symlog")
    ax.set_title(f"Degree of detected vs missed illicit (test, thr={THRESHOLD})")
    ax = axes[1]
    ax.plot(fnt["timestep"], fnt["fn_rate"], "o-", color="#d62728", label="FN rate")
    ax.set_xlabel("timestep")
    ax.set_ylabel("FN rate", color="#d62728")
    ax.set_ylim(0, 1)
    ax2 = ax.twinx()
    ax2.bar(fnt["timestep"], fnt["n_illicit"], alpha=0.25, color="#1f77b4",
            label="n illicit")
    ax2.set_ylabel("n illicit (labeled)", color="#1f77b4")
    ax.set_title("Missed illicit by timestep (drift evidence)")
    fig.tight_layout()
    plt.savefig(FIGURES / "fn_analysis.png")
    plt.close("all")
    return tp_m, fn_m


def case_studies(data, probs, tp_m):
    Xg = data["X_graph"]
    g = {c: Xg[:, i] for i, c in enumerate(GRAPH_FEATURE_COLS)}
    tp_idx = np.where(tp_m)[0]
    chosen, reasons = [], []

    def pick(mask_fn, reason, exclude):
        cand = [i for i in tp_idx if mask_fn(i) and i not in exclude]
        if not cand:
            return None
        best = max(cand, key=lambda i: probs[i])
        chosen.append(best)
        reasons.append(reason)
        return best

    pick(lambda i: g["fan_out"][i] >= 3, "fan-out (tán tiền)", set())
    pick(lambda i: bool(g["one_in_one_out_chain"][i]), "peeling chain 1-vào-1-ra", set(chosen))
    pick(lambda i: True, "degree cao nhất trong tập TP (tương đối)", set(chosen))

    ei = data["edge_index"]
    ts = data["timestep"]
    y = data["y"]
    rows = []
    for k, (node, reason) in enumerate(zip(chosen, reasons), 1):
        t = int(ts[node])
        in_ts = np.where(ts == t)[0]
        m = np.isin(ei[0], in_ts) & np.isin(ei[1], in_ts)
        sub = nx.DiGraph()
        sub.add_nodes_from(in_ts.tolist())
        sub.add_edges_from(zip(ei[0][m].tolist(), ei[1][m].tolist()))
        ego = {node} | set(sub.predecessors(node)) | set(sub.successors(node))
        H = sub.subgraph(ego).copy()
        pos = nx.spring_layout(H, seed=42, k=1.2)
        colors = ["#d62728" if y[v] == 1 else "#2ca02c" if y[v] == 0 else "#bbbbbb"
                  for v in H.nodes]
        sizes = [900 if v == node else 380 for v in H.nodes]
        edges_w = [2.5 if (u == node or v == node) else 0.8 for u, v in H.edges]
        fig, ax = plt.subplots(figsize=(8, 6.5))
        nx.draw_networkx_edges(H, pos, ax=ax, width=edges_w, alpha=0.6,
                               edge_color="#555555", arrows=True, arrowsize=12)
        nx.draw_networkx_nodes(H, pos, ax=ax, node_color=colors, node_size=sizes,
                               edgecolors=["black" if v == node else "white" for v in H.nodes],
                               linewidths=[2.2 if v == node else 0.6 for v in H.nodes])
        nx.draw_networkx_labels(H, pos, ax=ax, font_size=7,
                                labels={v: str(data["tx_ids"][v])[:8] for v in H.nodes})
        ax.set_title(f"Case {k}: node {node} (tx {data['tx_ids'][node]}), ts={t}, "
                     f"p(illicit)={probs[node]:.3f}\n{reason} | "
                     f"đỏ=illicit, xanh=licit, xám=unknown, viền đen=node gốc")
        ax.axis("off")
        plt.savefig(FIGURES / f"case_study_{k}_node_{node}.png")
        plt.close("all")
        rows.append({"case": k, "node_idx": int(node), "tx_id": int(data["tx_ids"][node]),
                     "timestep": t, "prob_illicit": float(probs[node]), "reason": reason,
                     "n_ego_nodes": H.number_of_nodes(), "n_ego_edges": H.number_of_edges(),
                     "in_degree": int(g["in_degree"][node]),
                     "out_degree": int(g["out_degree"][node])})
    pd.DataFrame(rows).to_csv(TABLES / "case_studies.csv", index=False)
    print("[P2-shap] Case studies:\n", pd.DataFrame(rows).to_string(index=False), flush=True)


def main():
    set_seed(42)
    print("[P2-shap] Loading data + best models ...", flush=True)
    data = load_data()
    configs = build_feature_configs(data)

    tab_info = json.loads((TABLES / "best_tabular_info.json").read_text(encoding="utf-8"))
    X_tab = configs[tab_info["config"]]["X"]
    probs_tab = np.load(PREDICTIONS / "best_tabular_probs.npy")
    shap_analysis("tabular", tab_info, data, X_tab)

    hyb_info = json.loads((TABLES / "best_hybrid_info.json").read_text(encoding="utf-8"))
    emb = np.load(GNN_EMBEDDINGS_NPY).astype(np.float32)
    if hyb_info["config"] == "emb128":
        X_hyb = emb
    elif hyb_info["config"] == "local93+emb128":
        X_hyb = np.hstack([data["X_local"], emb])
    else:
        X_hyb = np.hstack([data["X_full"], emb])
    shap_analysis("hybrid", hyb_info, data, X_hyb)

    tp_m, fn_m = fn_analysis(data, probs_tab, tab_info)
    case_studies(data, probs_tab, tp_m)
    print("[P2-shap] Done.", flush=True)


if __name__ == "__main__":
    main()
