# Scripts — Pipeline dữ liệu → train → đánh giá

Các script tổ chức theo thứ tự pipeline. (Phần **demo/ứng dụng** đã chuyển sang
**web app** ở thư mục gốc dự án: `backend/` + `src/` — không còn Streamlit/realtime ở đây.)

## Thứ tự sử dụng

| STT | File | Vai trò | Input | Output |
|-----|------|---------|-------|--------|
| 1 | `01_extract_keypoints_from_video.py` | YOLO-Pose trích 17 keypoint từ video | `data/raw_videos/{normal,shoplifting}/*.mp4` | `data/raw_keypoints/` (34 cột) |
| 2 | `02_clean_csv_filter.py` | Lọc file mất pose | `data/raw_keypoints/` | `data/clean_csv_34feat/` |
| 3 | `03_extract_56_geometric_features.py` | Trích 56 đặc trưng hình học | `data/raw_keypoints/` | `data/clean_csv_56feat/` |
| 4 | `04_prepare_sliding_window.py` | Sliding window → .npy | `data/clean_csv_56feat/` | `data/processed_npy/` |
| 5 | `05_train_baseline_hao.py` | Train LSTM-CNN baseline (Keras) | `data/processed_npy/` | `models/*.h5` |
| 8 | `08_batch_eval_csv.py` | Đánh giá batch V1.4 trên CSV (split-by-clip) | `data/clean_csv_56feat/` + model | bảng metrics + CSV |

> So sánh **toàn bộ mô hình** (Hảo / V1.4 / ST-GCN / MS-G3D / Ensemble):
> `python ../evaluate_all_models.py` (chạy độc lập từ dữ liệu trong `data/` + `models/`).

## Train lại từ đầu (split-by-clip — đúng phương pháp)

```powershell
# Bước 1-4 (chỉ chạy 1 lần khi mới có video gốc):
python scripts\01_extract_keypoints_from_video.py
python scripts\02_clean_csv_filter.py
python scripts\03_extract_56_geometric_features.py
python scripts\04_prepare_sliding_window.py --features 56

# Bước 5 — Train với split-by-clip (chống data leakage)
python scripts\05_train_baseline_hao.py --features 56 --split by_clip
```

## Nguồn gốc

Pipeline được tổ chức lại từ source ban đầu của Hảo (extract/filter/prepare),
cải tiến: đường dẫn tương đối (không hardcode), bổ sung `clip_ids` chống data
leakage, dùng chung `src/data_preprocessing/extract_geometric_features` cho cả
train lẫn inference.
