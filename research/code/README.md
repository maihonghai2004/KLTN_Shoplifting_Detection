# KLTN — Ứng dụng học sâu nhận diện hành vi bất thường trong siêu thị

**Khóa luận tốt nghiệp Kỹ thuật Dữ liệu**
Nhóm: Mai Hồng Hải (22133014) · Nguyễn Ngọc Hiếu Hảo (22133015)
GVHD: ThS. Phan Thị Thể

Hệ thống phát hiện hành vi **cất giấu hàng hóa trái phép** từ camera CCTV, dùng phương pháp **skeleton-based action recognition** (phân tích chuyển động khung xương người).

---

## Kết quả chính

| Metric | Giá trị | KPI đề cương |
|---|---:|---:|
| **F1-macro** (V2.3 Ensemble, 5-seed) | **0,8067 ± 0,0153** | ≥ 0,80 ✅ |
| Precision | 0,8124 | ≥ 0,78 ✅ |
| Recall | 0,8076 | ≥ 0,80 ✅ |

Mô hình tốt nhất: **V2.3 = Ensemble (ST-GCN + LSTM-CNN) + TTA**, đánh giá nghiêm ngặt split-by-clip.

---

## Cấu trúc thư mục

> Đây là thư mục `research/code/` — **mã nguồn nghiên cứu** (train + đánh giá).
> Notebook (`../notebooks/`) và kết quả (`../results/`) nằm ở cấp `research/`.
> Tài liệu/báo cáo nằm ở `report/` (cấp gốc dự án).

```
research/code/
├── README.md                  # File này
├── ARCHITECTURE.md            # Kiến trúc pipeline 4-stage
├── requirements.txt           # Dependencies (pin version) cho phần nghiên cứu
├── evaluate_all_models.py     # So sánh toàn bộ model (Hảo/V1.4/V2.0/V2.4/Ensemble)
├── configs/config.yaml        # Cấu hình tổng
│
├── src/                       # Thư viện dùng chung cho train/eval
│   ├── state_machine.py       # Rule-based FSM 4-state
│   ├── models/stgcn.py        # ST-GCN (V2.0) + MS-G3D (V2.4) + graph utilities
│   ├── data_preprocessing/    # geometric_features.py (56 đặc trưng)
│   ├── behavior/              # hao_baseline.py (LSTM-CNN) + keras_v14.py (loader V1.4)
│   └── utils/one_euro.py      # Bộ lọc làm mượt keypoint
│
├── tests/                     # Unit tests (pytest)
│   ├── test_geometric_features.py  # Test 56 đặc trưng hình học
│   ├── test_state_machine.py       # Test FSM transitions
│   └── test_models.py              # Test ST-GCN / MS-G3D forward pass
│
├── scripts/                   # Pipeline dữ liệu → train → đánh giá (theo số thứ tự)
│   ├── 01_extract_keypoints_from_video.py
│   ├── 02_clean_csv_filter.py
│   ├── 03_extract_56_geometric_features.py
│   ├── 04_prepare_sliding_window.py
│   ├── 05_train_baseline_hao.py
│   ├── 08_batch_eval_csv.py         # đánh giá batch V1.4 (split-by-clip)
│   └── README.md
│
├── data/
│   ├── clean_csv_56feat/      # CSV sạch 56 đặc trưng hình học (đầu vào đánh giá)
│   └── processed_npy/         # X_data, y_data (.npy) cho train
│
└── models/                    # Trọng số model: hao .h5 / v1.4 .keras / v2.0 .pt / v2.4 .pt
```

---

## Cài đặt nhanh

```powershell
# Cách 1: dùng lại venv của web app (đã đủ TF 2.15 + torch + sklearn)
..\..\backend\.venv\Scripts\python.exe evaluate_all_models.py

# Cách 2: tạo venv riêng cho phần nghiên cứu
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Yêu cầu: Python 3.10/3.11.

### Chạy tests

```powershell
python -m pytest tests/ -v
```

---

## Sử dụng nhanh

> **Lưu ý:** phần **demo/ứng dụng** (realtime + dashboard) đã chuyển sang **web app**
> ở thư mục gốc dự án (`backend/` + `src/`). Thư mục `research/` này chỉ giữ
> **mã nguồn nghiên cứu** (mô hình, train, đánh giá) phục vụ báo cáo.

```powershell
# So sánh TOÀN BỘ mô hình (Hảo / V1.4 / ST-GCN / MS-G3D / Ensemble) — độc lập
python evaluate_all_models.py

# Đánh giá batch V1.4 trên CSV (split-by-clip)
python scripts/08_batch_eval_csv.py --split test --seed 42

# Train lại baseline (split-by-clip) — cần dataset gốc
python scripts/05_train_baseline_hao.py --features 56 --split by_clip
```

---

## Kiến trúc pipeline 4-stage

```
Video → [1] YOLO Detection → [2] ByteTrack → [3] YOLO-Pose (17 kp)
      → [4] Behavior Analysis (ST-GCN + LSTM + Rule → Ensemble) → Alert
```

Chi tiết: `ARCHITECTURE.md` và Chương 2 trong báo cáo.

---

## Quá trình thực nghiệm (ablation)

| Phiên bản | F1-macro | Cải tiến |
|---|---:|---|
| V1.0 (random split) | 0,7658 | Baseline (có leakage) |
| V1.1 (split-by-clip) | 0,5526 | Phát hiện & fix data leakage |
| V1.2 (56 features) | 0,7182 | Đặc trưng hình học |
| V1.3 (regularize) | 0,7568 ± 0,034 | BiLSTM + Dropout + Focal + Augment |
| V2.0 (ST-GCN) | 0,7734 ± 0,036 | Kiến trúc đồ thị (đóng góp chính) |
| **V2.3 (Ensemble + TTA)** | **0,8067 ± 0,015** | **Kết quả tốt nhất** |

Chi tiết phân tích: Chương 3 báo cáo (`report/`) và các notebook `../notebooks/`.

---

## Đóng góp chính của khóa luận

1. **Phát hiện và sửa data leakage** trong phương pháp đánh giá baseline — chuyển từ random split sang split-by-clip (GroupShuffleSplit), đảm bảo kết quả F1 phản ánh đúng khả năng tổng quát hóa.

2. **Ứng dụng ST-GCN (Spatial-Temporal Graph Convolutional Network)** cho bài toán nhận diện hành vi cất giấu hàng hóa — mô hình hóa skeleton người dưới dạng đồ thị không gian-thời gian, khai thác cấu trúc topo xương thay vì chuỗi đặc trưng phẳng.

3. **Phương pháp Ensemble + TTA** kết hợp ST-GCN (graph-based) + LSTM-CNN (sequence-based) + Rule-based FSM, đạt F1-macro = 0,8067 ± 0,0153 trên đánh giá multi-seed 5-fold split-by-clip.

4. **Hệ thống end-to-end hoàn chỉnh**: từ trích xuất keypoint (YOLOv8-Pose), tiền xử lý (56 đặc trưng hình học), suy luận mô hình, đến **web app phân tích video** (FastAPI + React) hiển thị khung xương, phát hiện từng sự kiện bất thường, lưu ảnh bằng chứng và gửi email cảnh báo.

---

## Tài liệu (thư mục `report/` ở cấp gốc dự án)

- `report/KLTN-KTDL- Nhóm Hải + Hảo.docx` — Báo cáo chính
- `report/system_architecture.html` — Sơ đồ kiến trúc hệ thống
- `report/bieumau-kltn/` — Biểu mẫu khóa luận (nhiệm vụ, nhận xét GVHD…)
- `report/*.pdf` — Paper tham khảo (IEEE / abnormal behavior detection)
