# P2 (Quyền) — Tabular + Hybrid + SHAP

## Cấu hình thí nghiệm
- Dữ liệu: `processed_data.npz` của P0 (train ts 1–28 / val 29–34 / test 35–49, chỉ node có nhãn).
- Chống rò rỉ: bỏ timestep khỏi feature; scaler fit trên train; chọn hyperparameter trên validation.
- Imbalance: `class_weight='balanced'` (LogReg/RF), `scale_pos_weight≈8.9` (XGB).
- 3 seed (42, 43, 44) cho RF/XGB/hybrid; báo mean ± std. PR-AUC = average precision.

## Kết quả chính (test ts 35–49, lớp illicit)
- **Best tabular**: RF × `full165` → PR-AUC **0.7833 ± 0.0017**, F1@0.5 **0.7978 ± 0.0014**.
- **Best hybrid**: XGB × `full165+emb128` → PR-AUC **0.7798 ± 0.0030**, F1@0.5 **0.7277 ± 0.0181**.
- So sánh đầy đủ với P0 (XGB baseline, Isolation Forest, Autoencoder) và P3 (GCN/SAGE/GAT): `tables/p2_final_comparison.csv` + `figures/p2_final_comparison.png`.

### Trả lời câu hỏi nghiên cứu
1. **93 vs 165 feature**: xem `tabular_benchmark_summary.csv` — chênh lệch local93/full165 cho biết 72 feature tổng hợp đóng góp bao nhiêu (kiểm chứng giả thuyết '72 feature ≈ 1 lớp message passing').
2. **Feature cấu trúc thay được 72 feature tổng hợp không?** So `local93+graph11` với `full165` và `local93`.
3. **Hybrid có hơn tabular/GNN thuần không?** So best hybrid với best tabular và GNN trong bảng final.

### Kiểm tra rò rỉ embedding (hybrid)
- GNN train trên ts 1–28, early stopping theo val → test (35–49) là out-of-sample của GNN.
- Gap val−test PR-AUC: emb128 = 0.2736, full165+emb128 = 0.1976 (xem `tables/hybrid_leakage_check.json`).

### SHAP
- Tỷ trọng đóng góp theo nhóm feature (tabular): local = 75.7%, aggregated = 24.3%.
- Hình: `figures/shap_tabular_beeswarm.png`, `shap_tabular_bar.png`, `shap_tabular_group_share.png`, `shap_hybrid_*.png`.

### Phân tích lỗi (False Negative)
- FN vs TP illicit: xem `tables/fn_vs_tp_structural_stats.csv` (Mann-Whitney U + rank-biserial).
- FN rate theo timestep: `tables/fn_rate_by_timestep.csv`, `figures/fn_analysis.png`.
- Case study (TP): `tables/case_studies.csv`, `figures/case_study_*.png`.

## Deliverables gửi P0 & P4
- `best_tabular_probs.npy`, `best_hybrid_probs.npy` (203.769 node, float32) — đã copy vào `shared/outputs/predictions/`.
- `best_tabular_per_timestep.csv`, `best_hybrid_per_timestep.csv`, `p2_models_per_timestep_test.csv` cho P4.

## Cách chạy lại
```bash
python src/p2_run_all.py   # tabular -> hybrid -> SHAP -> final comparison
```

## File chính
| Loại | File |
|---|---|
| Benchmark | `tables/tabular_benchmark*.csv`, `tables/hybrid_benchmark*.csv`, `tables/p2_final_comparison.csv` |
| Dự đoán | `predictions/best_tabular_probs.npy`, `predictions/best_hybrid_probs.npy` |
| SHAP | `predictions/shap_values_*_test.npy`, `tables/shap_*_feature_importance.csv`, `figures/shap_*.png` |
| Lỗi & case | `tables/fn_*.csv`, `tables/case_studies.csv`, `figures/fn_analysis.png`, `figures/case_study_*.png` |
| Model | `models/best_tabular_*`, `models/best_hybrid_xgb.json` |