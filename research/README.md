# research/ — Phần nghiên cứu (khoa học)

Toàn bộ **mã nguồn, thực nghiệm và kết quả** của khóa luận. Đây là phần chứng minh
tính khoa học (ablation, chống data leakage, ST-GCN, ensemble) đứng sau ứng dụng web.

```
research/
├── code/        # Mã nguồn train + đánh giá (xem code/README.md)
│   ├── src/                 # thư viện dùng chung (models, geometric_features, loaders)
│   ├── scripts/             # pipeline dữ liệu → train → đánh giá (01..08)
│   ├── evaluate_all_models.py  # so sánh Hảo/V1.4/V2.0/V2.4/Ensemble
│   ├── data/                # CSV 56-feat + .npy
│   └── models/              # trọng số: hao.h5 / v1.4.keras / v2.0.pt / v2.4.pt
│
├── notebooks/   # Quá trình thực nghiệm (Colab) — bằng chứng ablation cho báo cáo
│   ├── V1_baseline_vs_split_by_clip.ipynb   # V1.0 vs V1.1 — phát hiện data leakage
│   ├── V1.2_56features_by_clip.ipynb        # 56 đặc trưng hình học
│   ├── V1.3_regularize_augment.ipynb        # BiLSTM + Focal + Augment
│   ├── V1.4_stratified_multiseed.ipynb      # model dùng cho demo web (V1.4)
│   ├── V2.0_stgcn_pytorch.ipynb             # ST-GCN — đóng góp chính
│   └── V2.3_TTA_ensemble.ipynb              # kết quả tốt nhất F1 0,8067
│
└── results/     # Số liệu xuất ra để đưa vào báo cáo / web
    ├── model_comparison_scores.csv          # điểm per-clip mọi model (evaluate_all_models)
    └── eval_v14_allclips.csv                # đánh giá V1.4 per-clip
```

## Chạy nhanh

```powershell
# So sánh toàn bộ model (độc lập, không cần KLTN gốc):
..\backend\.venv\Scripts\python.exe code\evaluate_all_models.py
```

## Vai trò trong báo cáo
- **code/ + notebooks/** → Chương Phương pháp & Thực nghiệm (ablation table, leakage fix).
- **results/** → bảng số liệu, biểu đồ so sánh model trong báo cáo.
- Model tốt nhất **V1.4** chính là model đang chạy trong web app (`backend/`).
