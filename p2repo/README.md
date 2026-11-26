# P2 – Tabular + Hybrid + SHAP (Quyền)

Nhánh: `feature/tabular-hybrid-shap`
Bài toán: phát hiện giao dịch illicit trên đồ thị Bitcoin Elliptic (Bài 2a trong kế hoạch).

## 1. Cấu trúc

```
data/
  processed/processed_data.npz      # P0 (không push, lấy từ Drive)
  external/graph_features.parquet   # P1 (không push, lấy từ Drive)
outputs/                            # kết quả, không push, chia sẻ qua Drive
  predictions/  tables/  figures/  models/
src/
  p2_common.py     # nạp dữ liệu, ghép bộ feature, hàm đánh giá
  p2_tabular.py    # LogReg / RandomForest / XGBoost x 4 cấu hình feature
  p2_hybrid.py     # embedding GNN (P3) + feature gốc -> XGBoost
  p2_shap.py       # SHAP + phân tích False Negative
  p2_run_all.py    # chạy cả 3 bước theo thứ tự
requirements.txt
```

## 2. Dữ liệu đầu vào (lấy từ Drive nhóm)

Mọi file đều theo thứ tự node 0..203768 của `processed_data.npz`.

| File | Do | Đặt vào |
|---|---|---|
| `processed_data.npz` | P0 | `data/processed/` |
| `graph_features.parquet` | P1 | `data/external/` |
| `gnn_embeddings.npy` (128 chiều) | P3 | `outputs/predictions/` |
| `best_gnn_probs.npy` | P3 | `outputs/predictions/` (chỉ để so sánh GNN thuần) |

Lưu ý: bản `processed_data.npz` hợp lệ có kích thước khoảng 82 MB. File chỉ vài trăm byte là con trỏ Git LFS bị hỏng, không dùng được.
Code tự kiểm tra `tx_id` của P1 khớp với P0 và số dòng embedding khớp số node.

## 3. Cách chạy

```bash
pip install -r requirements.txt
python -m src.p2_run_all        # chạy cả 3 bước (khoảng 15-20 phút trên 1 CPU)
```

Hoặc chạy từng bước:

```bash
python -m src.p2_tabular   # benchmark tabular + best_tabular_probs.npy
python -m src.p2_hybrid    # hybrid (cần gnn_embeddings.npy của P3)
python -m src.p2_shap      # SHAP + phân tích False Negative
```

## 4. Quy ước thực nghiệm

- Chia theo thời gian: train ts 1-28, validation ts 29-34, test ts 35-49. Không random split.
- Chỉ dùng node có nhãn để huấn luyện và đánh giá. Node unknown không tính loss, không tính metric.
- Không dùng timestep làm feature.
- Hyperparameter và mô hình tốt nhất được chọn theo PR-AUC trên validation, không dùng test.
- Chạy 3 seed (42, 123, 456), báo trung bình ± độ lệch chuẩn.
- Xử lý mất cân bằng: `class_weight` (LogReg, RF) và `scale_pos_weight` (XGBoost).

## 5. Đầu ra (lưu trên Drive)

| File | Nội dung | Gửi cho |
|---|---|---|
| `predictions/best_tabular_probs.npy` | Xác suất illicit của toàn bộ node. Mô hình: RandomForest, `full165+graph`, trung bình 3 seed | P0, P4 |
| `predictions/best_hybrid_probs.npy` | Xác suất của hybrid tốt nhất theo validation | P0, P4 (nếu cần) |
| `tables/tabular_benchmark.csv` | Benchmark tabular (fixed 0.5 và ngưỡng chọn theo validation) | báo cáo |
| `tables/hybrid_benchmark.csv` | Benchmark hybrid, hai protocol A và B | báo cáo |
| `tables/shap_*.csv`, `fn_*.csv` | Độ quan trọng feature, phân tích FN vs TP | báo cáo |
| `figures/shap_*.png`, `fn_analysis_*.png` | Hình SHAP và phân tích lỗi | báo cáo |

## 6. Kết quả (test ts 35-49, PR-AUC lớp illicit, ngưỡng 0.5, trung bình ± std 3 seed)

| Model | 93 local | 165 đầy đủ | 93 + graph | 165 + graph |
|---|---|---|---|---|
| LogReg | 0.186 | 0.199 | 0.185 | 0.200 |
| RandomForest | 0.779 ± 0.000 | 0.784 ± 0.002 | 0.779 ± 0.003 | 0.787 ± 0.001 |
| XGBoost | 0.756 ± 0.006 | 0.785 ± 0.002 | 0.753 ± 0.004 | 0.787 ± 0.001 |

Hybrid (embedding GNN + feature gốc -> XGBoost), test PR-AUC:

| Cấu hình | Protocol A: train ts 1-28 | Protocol B: fit ts 29-32 |
|---|---|---|
| 165 feature (không embedding) | 0.785 | 0.784 |
| 165 feature + GNN | 0.767 | 0.771 |
| 93 feature (không embedding) | 0.756 | 0.780 |
| 93 feature + GNN | 0.747 | 0.772 |
| GNN thuần (P3) | 0.576 | |

Protocol B chỉ huấn luyện XGBoost trên các node mà GNN của P3 chưa từng thấy nhãn, để tránh rò rỉ nhãn qua embedding.

## 7. Trả lời câu hỏi nghiên cứu

1. **165 feature có hơn 93 feature không?** Có với XGBoost (0.756 -> 0.785), gần như không đổi với RF (0.779 -> 0.784). 72 feature tổng hợp chiếm khoảng 38% tổng |SHAP|.
2. **Feature cấu trúc của P1 có thay được 72 feature tổng hợp không?** Không. 93 + graph xấp xỉ 93 (chênh trong khoảng ±0.003); graph chỉ chiếm khoảng 3.6% tổng |SHAP|. Thêm graph vào 165 feature chỉ tăng 0.002, nằm trong nhiễu. Đây là tín hiệu yếu.
3. **Hybrid có hơn tabular và GNN thuần không?** Hơn GNN thuần rõ rệt (0.77 so với 0.58) nhưng không hơn tabular (0.767 < 0.785). Kết luận giữ nguyên ở cả hai protocol.
4. **Illicit bị bỏ sót có đặc điểm gì?** Recall giảm mạnh từ timestep 43: ts 43, 45, 47 đạt 0%, ts 44, 48, 49 dưới 10%, trong khi ts 35-42 đạt 60-97%. Điểm dự đoán của FN rất thấp (median khoảng 0.004), nghĩa là mô hình không nhận ra chứ không phải do chọn ngưỡng. Nguyên nhân (dark market đóng quanh ts 43) chỉ là giả thuyết từ paper Elliptic gốc, dataset không tự chứng minh.

## 8. Hạn chế

- Node là giao dịch chứ không phải ví, nên các chuỗi trong phân tích là chuỗi giao dịch.
- `best_gnn_probs.npy` và `gnn_embeddings.npy` của P3 được chọn theo PR-AUC trên test (`best_overall_test_pr_auc` trong `train_gnn.py`). Đây là rò rỉ chọn mô hình, số GNN có thể hơi lạc quan. Kết luận về hybrid không bị đảo vì bất lợi cho hybrid, nhưng cần ghi vào báo cáo.
- Embedding của node train đã "thấy" nhãn khi GNN huấn luyện; protocol B giảm nhưng không loại bỏ hoàn toàn ảnh hưởng này.
- LogReg dùng `class_weight=balanced` nên ngưỡng 0.5 gắn cờ quá nhiều (precision khoảng 0.13, ROC-AUC vẫn khoảng 0.85). Xem thêm cột `val_F1_opt` trong `tabular_benchmark.csv`.
- Grid hyperparameter nhỏ (LogReg 3, RF 2, XGBoost 2 giá trị), chưa dùng Optuna.
- SHAP tính trên XGBoost (TreeSHAP chính xác), mẫu ngẫu nhiên 3000 node test cộng toàn bộ TP và FN.