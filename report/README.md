# report/ — Báo cáo & tài liệu khóa luận

Thư mục **tài liệu báo cáo** — đặt ở cấp gốc dự án vì báo cáo chính mô tả **TOÀN BỘ
dự án**: cả phần nghiên cứu (`research/`) lẫn ứng dụng web demo (`backend/` + `src/`).

```
report/
├── KLTN-KTDL- Nhóm Hải + Hảo.docx     # BÁO CÁO CHÍNH (deliverable chấm điểm)
├── system_architecture.html           # Sơ đồ kiến trúc hệ thống (hình minh hoạ)
├── bieumau-kltn/                      # Biểu mẫu khoa: nhiệm vụ KLTN, nhận xét GVHD, ...
├── IJCEO_V02_01_01_06_02012024.pdf    # Paper tham khảo
└── Paper_5-An_Abnormal_Behavior_Detection_Method.pdf  # Paper tham khảo
```

## Báo cáo chính viết những gì
| Chương | Nội dung | Lấy từ |
|---|---|---|
| Mở đầu | Bài toán, mục tiêu, KPI | đề cương |
| Cơ sở lý thuyết | YOLO-Pose, ST-GCN, LSTM-CNN | paper trong `report/` |
| Phương pháp | 56 đặc trưng, split-by-clip, ensemble | `research/code` + `research/notebooks` |
| Thực nghiệm | Bảng ablation, F1 0,8067, so sánh model | `research/results` + notebooks |
| **Ứng dụng (Demo)** | **Web app real-time, UI, cảnh báo email** | `backend/` + `src/` |
| Kết luận | Đóng góp, hạn chế, hướng phát triển | tổng hợp |

> Báo cáo là **một tài liệu duy nhất** bao trùm cả nghiên cứu lẫn ứng dụng — nên nó
> nằm ở cấp gốc, ngang hàng với `research/` và web app, không nằm bên trong `research/`.
