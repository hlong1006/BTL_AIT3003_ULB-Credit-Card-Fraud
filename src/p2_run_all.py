# -*- coding: utf-8 -*-
"""
P2 (Quyen) - orchestration: tabular -> hybrid -> SHAP -> final comparison.
Run:  python src/p2_run_all.py
Final deliverables land in outputs/quyen/ and probs are copied to
shared/outputs/predictions/ for P0 (cost optimization) and P4 (drift).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from p2_common import (
    ROOT, SHARED_OUT, TABLES, FIGURES, PREDICTIONS, OUT,
    load_data, per_timestep_metrics,
)

SRC = Path(__file__).resolve().parent


def run_stage(script):
    print(f"\n{'='*70}\n[run_all] {script}\n{'='*70}", flush=True)
    r = subprocess.run([sys.executable, str(SRC / script)], cwd=SRC)
    if r.returncode != 0:
        raise SystemExit(f"{script} failed with code {r.returncode}")


def final_comparison():
    data = load_data()
    rows = []

    # P0 baseline + unsupervised (their published test numbers)
    p0 = pd.read_csv(SHARED_OUT / "tables" / "p0_test_model_comparison.csv")
    for _, r in p0.iterrows():
        rows.append({
            "group": "Tabular (P0)" if r["model"] == "XGBoost" else "Unsupervised (P0)",
            "model": r["model"], "config": "full165" if r["model"] == "XGBoost" else "-",
            "test_pr_auc_mean": r["pr_auc_ap"], "test_pr_auc_std": np.nan,
            "test_f1_mean": r["f1"], "test_f1_std": np.nan,
            "test_precision_mean": r["precision"], "test_recall_mean": r["recall"],
            "threshold": r["threshold"], "source": "P0 p0_test_model_comparison.csv",
        })

    # GNN (P3), test @ fixed 0.5, mean over seeds
    gnn = pd.read_csv(SHARED_OUT / "tables" / "gnn_benchmark.csv")
    gnn = gnn[(gnn["split"] == "Test") & (gnn["threshold_method"] == "Fixed 0.5")]
    for _, r in gnn.iterrows():
        rows.append({
            "group": "GNN (P3)", "model": r["model"], "config": "full165 + graph",
            "test_pr_auc_mean": r["pr_auc_mean"], "test_pr_auc_std": r["pr_auc_std"],
            "test_f1_mean": r["f1_mean"], "test_f1_std": r["f1_std"],
            "test_precision_mean": r["precision_mean"], "test_recall_mean": r["recall_mean"],
            "threshold": 0.5, "source": "P3 gnn_benchmark.csv",
        })

    # P2 best tabular + best hybrid (mean over seeds of selected params)
    for tag, info_file, summary_file, group in [
        ("tabular", "best_tabular_info.json", "tabular_benchmark_summary.csv", "Tabular (P2)"),
        ("hybrid", "best_hybrid_info.json", "hybrid_benchmark_summary.csv", "Hybrid (P2)"),
    ]:
        info = json.loads((TABLES / info_file).read_text(encoding="utf-8"))
        s = pd.read_csv(TABLES / summary_file)
        sel = s[(s["model"] == info["model"]) & (s["config"] == info["config"])].iloc[0]
        rows.append({
            "group": group, "model": info["model"], "config": info["config"],
            "test_pr_auc_mean": sel["test_pr_auc_mean"], "test_pr_auc_std": sel["test_pr_auc_std"],
            "test_f1_mean": sel["test_f1_mean"], "test_f1_std": sel["test_f1_std"],
            "test_precision_mean": sel["test_precision_mean"],
            "test_recall_mean": sel["test_recall_mean"],
            "threshold": 0.5, "source": f"P2 {summary_file} (3 seeds)",
        })

    comp = pd.DataFrame(rows).sort_values("test_pr_auc_mean", ascending=False)
    comp.to_csv(TABLES / "p2_final_comparison.csv", index=False)
    print("\n[run_all] FINAL COMPARISON (test, illicit class):")
    print(comp.to_string(index=False), flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    labels = [f"{r['model']}\n{r['config']}" for _, r in comp.iterrows()]
    colors = {"Tabular (P0)": "#7f7f7f", "Tabular (P2)": "#1f77b4",
              "Hybrid (P2)": "#ff7f0e", "GNN (P3)": "#2ca02c",
              "Unsupervised (P0)": "#d62728"}
    cs = [colors[g] for g in comp["group"]]
    for ax, metric, title in [(axes[0], "test_pr_auc_mean", "PR-AUC (illicit)"),
                              (axes[1], "test_f1_mean", "F1 illicit @0.5")]:
        ax.bar(range(len(comp)), comp[metric], color=cs)
        std_col = metric.replace("_mean", "_std")
        ax.errorbar(range(len(comp)), comp[metric],
                    yerr=comp[std_col].fillna(0), fmt="none", ecolor="k", capsize=3)
        ax.set_xticks(range(len(comp)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(title)
        ax.set_ylim(0, 1)
    fig.suptitle("P2 final benchmark - test timestep 35-49 (mean +/- std over seeds)")
    fig.tight_layout()
    plt.savefig(FIGURES / "p2_final_comparison.png")
    plt.close("all")

    # per-timestep F1: P0 baseline vs P2 tabular vs P2 hybrid vs P3 GNN (for P4)
    probs_map = {
        "p0_xgb_baseline": np.load(SHARED_OUT / "predictions" / "baseline_xgb_probs.npy"),
        "p2_best_tabular": np.load(PREDICTIONS / "best_tabular_probs.npy"),
        "p2_best_hybrid": np.load(PREDICTIONS / "best_hybrid_probs.npy"),
        "p3_best_gnn": np.load(SHARED_OUT / "predictions" / "best_gnn_probs.npy"),
    }
    merged = None
    for name, p in probs_map.items():
        pt = per_timestep_metrics(data, p, "test_mask")
        pt = pt.rename(columns={c: f"{c}_{name}" for c in pt.columns
                                if c not in ("timestep", "n_labeled", "n_illicit")})
        merged = pt if merged is None else merged.merge(pt, on=["timestep", "n_labeled", "n_illicit"])
    merged.to_csv(TABLES / "p2_models_per_timestep_test.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, style in [("p0_xgb_baseline", "o--"), ("p2_best_tabular", "s-"),
                        ("p2_best_hybrid", "^-"), ("p3_best_gnn", "v-")]:
        ax.plot(merged["timestep"], merged[f"f1_{name}"], style, label=name, alpha=0.85)
    ax.set_xlabel("timestep")
    ax.set_ylabel("F1 illicit @0.5")
    ax.set_title("F1 illicit per timestep (test) - all branches")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.savefig(FIGURES / "p2_models_f1_by_timestep.png")
    plt.close("all")

    # hand over probs to P0 & P4 via shared folder
    for f in ("best_tabular_probs.npy", "best_hybrid_probs.npy"):
        shutil.copy2(PREDICTIONS / f, SHARED_OUT / "predictions" / f)
    print("[run_all] Copied best_tabular_probs.npy & best_hybrid_probs.npy "
          "to shared/outputs/predictions/ (for P0 & P4).", flush=True)
    return comp


def write_readme(comp):
    tab_info = json.loads((TABLES / "best_tabular_info.json").read_text(encoding="utf-8"))
    hyb_info = json.loads((TABLES / "best_hybrid_info.json").read_text(encoding="utf-8"))
    tab_sum = pd.read_csv(TABLES / "tabular_benchmark_summary.csv")
    hyb_sum = pd.read_csv(TABLES / "hybrid_benchmark_summary.csv")
    leakage = json.loads((TABLES / "hybrid_leakage_check.json").read_text(encoding="utf-8"))
    fn_stats = pd.read_csv(TABLES / "fn_vs_tp_structural_stats.csv")
    share_tab = pd.read_csv(TABLES / "shap_tabular_group_share.csv", index_col=0)["share"]

    def fmt_row(df, model, config):
        r = df[(df["model"] == model) & (df["config"] == config)].iloc[0]
        return (f"{r['test_pr_auc_mean']:.4f} ± {r['test_pr_auc_std']:.4f}",
                f"{r['test_f1_mean']:.4f} ± {r['test_f1_std']:.4f}")

    tab_pr, tab_f1 = fmt_row(tab_sum, tab_info["model"], tab_info["config"])
    hyb_pr, hyb_f1 = fmt_row(hyb_sum, hyb_info["model"], hyb_info["config"])

    lines = [
        "# P2 (Quyền) — Tabular + Hybrid + SHAP",
        "",
        "## Cấu hình thí nghiệm",
        "- Dữ liệu: `processed_data.npz` của P0 (train ts 1–28 / val 29–34 / test 35–49, chỉ node có nhãn).",
        "- Chống rò rỉ: bỏ timestep khỏi feature; scaler fit trên train; chọn hyperparameter trên validation.",
        "- Imbalance: `class_weight='balanced'` (LogReg/RF), `scale_pos_weight≈8.9` (XGB).",
        "- 3 seed (42, 43, 44) cho RF/XGB/hybrid; báo mean ± std. PR-AUC = average precision.",
        "",
        "## Kết quả chính (test ts 35–49, lớp illicit)",
        f"- **Best tabular**: {tab_info['model']} × `{tab_info['config']}` → PR-AUC **{tab_pr}**, F1@0.5 **{tab_f1}**.",
        f"- **Best hybrid**: XGB × `{hyb_info['config']}` → PR-AUC **{hyb_pr}**, F1@0.5 **{hyb_f1}**.",
        "- So sánh đầy đủ với P0 (XGB baseline, Isolation Forest, Autoencoder) và P3 (GCN/SAGE/GAT): "
        "`tables/p2_final_comparison.csv` + `figures/p2_final_comparison.png`.",
        "",
        "### Trả lời câu hỏi nghiên cứu",
        "1. **93 vs 165 feature**: xem `tabular_benchmark_summary.csv` — chênh lệch local93/full165 "
        "cho biết 72 feature tổng hợp đóng góp bao nhiêu (kiểm chứng giả thuyết '72 feature ≈ 1 lớp message passing').",
        "2. **Feature cấu trúc thay được 72 feature tổng hợp không?** So `local93+graph11` với `full165` và `local93`.",
        "3. **Hybrid có hơn tabular/GNN thuần không?** So best hybrid với best tabular và GNN trong bảng final.",
        "",
        "### Kiểm tra rò rỉ embedding (hybrid)",
        f"- GNN train trên ts 1–28, early stopping theo val → test (35–49) là out-of-sample của GNN.",
        f"- Gap val−test PR-AUC: emb128 = {leakage['variants']['emb128']['val_minus_test_pr_auc']:.4f}, "
        f"full165+emb128 = {leakage['variants']['full165+emb128']['val_minus_test_pr_auc']:.4f} "
        "(xem `tables/hybrid_leakage_check.json`).",
        "",
        "### SHAP",
        "- Tỷ trọng đóng góp theo nhóm feature (tabular): "
        + ", ".join(f"{k} = {v:.1%}" for k, v in share_tab.items()) + ".",
        "- Hình: `figures/shap_tabular_beeswarm.png`, `shap_tabular_bar.png`, `shap_tabular_group_share.png`, "
        "`shap_hybrid_*.png`.",
        "",
        "### Phân tích lỗi (False Negative)",
        f"- FN vs TP illicit: xem `tables/fn_vs_tp_structural_stats.csv` (Mann-Whitney U + rank-biserial).",
        f"- FN rate theo timestep: `tables/fn_rate_by_timestep.csv`, `figures/fn_analysis.png`.",
        f"- Case study (TP): `tables/case_studies.csv`, `figures/case_study_*.png`.",
        "",
        "## Deliverables gửi P0 & P4",
        "- `best_tabular_probs.npy`, `best_hybrid_probs.npy` (203.769 node, float32) — đã copy vào "
        "`shared/outputs/predictions/`.",
        "- `best_tabular_per_timestep.csv`, `best_hybrid_per_timestep.csv`, `p2_models_per_timestep_test.csv` cho P4.",
        "",
        "## Cách chạy lại",
        "```bash",
        "python src/p2_run_all.py   # tabular -> hybrid -> SHAP -> final comparison",
        "```",
        "",
        "## File chính",
        "| Loại | File |",
        "|---|---|",
        "| Benchmark | `tables/tabular_benchmark*.csv`, `tables/hybrid_benchmark*.csv`, `tables/p2_final_comparison.csv` |",
        "| Dự đoán | `predictions/best_tabular_probs.npy`, `predictions/best_hybrid_probs.npy` |",
        "| SHAP | `predictions/shap_values_*_test.npy`, `tables/shap_*_feature_importance.csv`, `figures/shap_*.png` |",
        "| Lỗi & case | `tables/fn_*.csv`, `tables/case_studies.csv`, `figures/fn_analysis.png`, `figures/case_study_*.png` |",
        "| Model | `models/best_tabular_*`, `models/best_hybrid_xgb.json` |",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("[run_all] Wrote outputs/quyen/README.md", flush=True)


def main():
    if "--finalize" not in sys.argv:
        run_stage("p2_tabular.py")
        run_stage("p2_hybrid.py")
        run_stage("p2_shap.py")
    comp = final_comparison()
    write_readme(comp)
    print("\n[run_all] P2 BRANCH COMPLETE.", flush=True)


if __name__ == "__main__":
    main()
