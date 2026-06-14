# KLTN_Shoplifting_Detection

### Ứng dụng học sâu nhận diện hành vi bất thường (cất giấu hàng hóa) trong siêu thị

> **Khóa luận tốt nghiệp — Ngành Kỹ thuật Dữ liệu, Khoa CNTT**
> **Trường Đại học Sư phạm Kỹ thuật TP. Hồ Chí Minh (HCMUTE)**
>
> Sinh viên: **Mai Hồng Hải** (22133014) · **Nguyễn Ngọc Hiếu Hảo** (22133015)
> Giảng viên hướng dẫn: **ThS. Phan Thị Thể**

Hệ thống phát hiện hành vi **cất giấu hàng hóa trái phép** từ camera giám sát, dùng
phương pháp **nhận dạng hành động dựa trên khung xương** (skeleton-based action
recognition): trích khung xương người bằng **YOLOv8-Pose** → tính **56 đặc trưng hình
học** → phân loại bằng mô hình học sâu **BiLSTM-CNN (V1.4)**. Kèm **web app demo** phân
tích video, vẽ khung xương theo thời gian, ghi nhận từng sự kiện bất thường, lưu ảnh bằng
chứng và gửi **email cảnh báo**.

---

## 1. Tổng quan

| | |
|---|---|
| **Bài toán** | Phát hiện hành vi cất giấu hàng hóa (trộm cắp) trong siêu thị từ video CCTV |
| **Phương pháp** | Skeleton-based action recognition (YOLOv8-Pose + 56 đặc trưng hình học + Deep Learning) |
| **Mô hình demo** | V1.4 — BiLSTM-CNN (chính xác & ổn định nhất cho suy luận) |
| **Kết quả tốt nhất** | **F1-macro = 0,807** (Ensemble V2.3, đánh giá split-by-clip 5 seed) |
| **Đóng góp chính** | Phát hiện & sửa **data leakage** · ứng dụng **ST-GCN** · **Ensemble + TTA** · web app end-to-end |

### Kết quả thực nghiệm (held-out, split-by-clip)

| Mô hình | F1-macro |
|---|---:|
| **V1.4 BiLSTM-CNN** (dùng trong demo) | **0,92** |
| Ensemble (V1.4 + V2.4) | 0,86 |
| Hảo baseline (LSTM-CNN) | 0,85 |
| V2.4 MS-G3D | 0,79 |
| Ensemble V2.3 (V1.4 + V2.0) | 0,76 |
| V2.0 ST-GCN | 0,44 |

> Chi tiết ablation (V1.0 → V2.3) và việc xử lý data leakage xem trong `research/`.

---

## 2. Tính năng

- 📹 **Phân tích video**: tải video giám sát, hệ thống xử lý và trả kết quả.
- 🦴 **Hiển thị khung xương** (COCO-17) bám theo người + **điểm nghi ngờ theo thời gian**.
- 🎯 **Phát hiện theo sự kiện**: mỗi đoạn hành vi bất thường = 1 sự kiện riêng (ảnh + số liệu).
- 🖼️ **Lưu ảnh bằng chứng** tại thời điểm đỉnh của mỗi sự kiện.
- 📧 **Cảnh báo email**: 1 video → 1 email tổng hợp đầy đủ các sự kiện (không spam).
- 🚨 **Trang Cảnh báo**: bảng sự kiện kèm ảnh, thời gian thực, độ tin cậy.
- 📊 **Trang Thống kê giám sát**: KPI, phân tầng mức độ, phân bố độ tin cậy, nhận định tự động.

---

## 3. Kiến trúc & pipeline

```
Video CCTV
   │
   ├─ [1] YOLOv8-Pose  ──► 17 điểm khớp / khung hình
   ├─ [2] Trích 56 đặc trưng hình học (tương đối + vector xương + tay–hông)
   ├─ [3] Cửa sổ trượt 90 khung hình (step 30)
   ├─ [4] Mô hình V1.4 BiLSTM-CNN  ──► điểm nghi ngờ mỗi cửa sổ
   └─ [5] Gom sự kiện + ảnh bằng chứng + cảnh báo email
```

> **Vì sao xử lý offline (không realtime)?** Để giữ đúng YOLOv8-Pose như lúc huấn luyện
> (tránh lệch phân phối keypoint), bảo đảm độ chính xác. Chi tiết: `report/ghi_chu_demo_offline.md`.

---

## 4. Cấu trúc thư mục

```
KLTN_Shoplifting_Detection/
├── src/                 # FRONTEND — React + Vite + TailwindCSS
│   ├── pages/           #   Home · Monitoring(Phân tích) · Alerts · Analytics
│   ├── components/      #   ModelTester · ModelOverlayPlayer · Sidebar · Logo
│   └── context/         #   AnalysisContext (lưu kết quả vào localStorage)
│
├── backend/             # BACKEND — FastAPI
│   ├── main.py          #   API: YOLOv8-Pose → V1.4 → sự kiện + ảnh + email
│   ├── geometric_features.py · v14_loader.py
│   ├── models/          #   v1.4_best.keras (+ yolov8s-pose.pt tự tải)
│   └── requirements.txt · .env.example
│
├── research/            # NGHIÊN CỨU (khoa học sau ứng dụng)
│   ├── code/            #   train + đánh giá (evaluate_all_models.py, scripts/, src/, tests/)
│   ├── notebooks/       #   thực nghiệm Colab (ablation V1.0 → V2.3)
│   └── results/         #   số liệu so sánh model
│
├── report/              # BÁO CÁO & tài liệu (báo cáo chính .docx, biểu mẫu, paper)
│
├── index.html · package.json · tailwind.config.js · postcss.config.js
└── README.md
```

---

## 5. Công nghệ

| Thành phần | Công nghệ |
|---|---|
| Frontend | React 18, Vite 5, TailwindCSS 3, Chart.js, react-router-dom |
| Backend | FastAPI, Uvicorn, OpenCV, imageio-ffmpeg |
| Mô hình | YOLOv8-Pose (Ultralytics 8.2.103), TensorFlow 2.15 / Keras (V1.4), PyTorch 2.2 (ST-GCN) |

---

## 6. Yêu cầu hệ thống

- **Python 3.11** (khuyến nghị; 3.10 cũng được)
- **Node.js 18+** và npm
- Hệ điều hành: Windows / macOS / Linux
- *Không cần cài ffmpeg riêng* — đã đi kèm qua `imageio-ffmpeg`.

---

## 7. Cài đặt

```bash
git clone https://github.com/maihonghai2004/KLTN_Shoplifting_Detection.git
cd KLTN_Shoplifting_Detection
```

### 7.1. Backend (Python)

```powershell
# Windows (PowerShell)
python -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

```bash
# macOS / Linux
python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt
```

> Lần chạy đầu, Ultralytics sẽ **tự tải `yolov8s-pose.pt`** (~23MB) vào `backend/models/`.

### 7.2. Cấu hình email cảnh báo (tùy chọn)

```powershell
copy backend\.env.example backend\.env      # Windows
# cp backend/.env.example backend/.env       # macOS/Linux
```

Mở `backend/.env` và điền:

```
GMAIL_USER=email-gui@gmail.com
GMAIL_APP_PASSWORD=app-password-16-ky-tu
RECIPIENT_EMAIL=email-nhan@example.com
```

> `GMAIL_APP_PASSWORD` là **App Password** của Gmail (cần bật xác thực 2 bước), tạo tại
> https://myaccount.google.com/apppasswords — **không** phải mật khẩu đăng nhập thường.
> File `.env` đã được `.gitignore` loại trừ, **không bao giờ commit**. Nếu bỏ trống, app
> vẫn chạy bình thường, chỉ không gửi email.

### 7.3. Frontend (Node)

```bash
npm install
```

---

## 8. Chạy demo

Mở **2 cửa sổ terminal**:

**Terminal 1 — Backend** (cổng 8000):
```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
# macOS/Linux: backend/.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — Frontend** (cổng 5173):
```bash
npm run dev
```

Mở trình duyệt: **http://localhost:5173**

### Cách dùng
1. Vào trang **Phân tích video** → tải lên video giám sát (MP4/AVI/MOV, một người trong khung hình).
2. Đợi xử lý (video ngắn ~30–60s) → xem video phát lại + khung xương + điểm nghi ngờ.
3. Nếu phát hiện bất thường → ảnh bằng chứng + email được ghi nhận.
4. Xem **Cảnh báo** (danh sách sự kiện + ảnh) và **Thống kê** (báo cáo tổng hợp).

---

## 9. Kiểm thử & đánh giá mô hình (phần nghiên cứu)

Dùng chung môi trường `backend/.venv` (đã đủ TensorFlow + PyTorch + scikit-learn):

```powershell
# So sánh TOÀN BỘ mô hình trên dữ liệu 56-feat (Hảo / V1.4 / ST-GCN / MS-G3D / Ensemble)
backend\.venv\Scripts\python.exe research\code\evaluate_all_models.py

# Đánh giá V1.4 trên tập held-out (split-by-clip)
backend\.venv\Scripts\python.exe research\code\scripts\08_batch_eval_csv.py --split test --seed 42

# Unit tests
cd research\code
..\..\backend\.venv\Scripts\python.exe -m pytest tests\ -v
```

> Quá trình thực nghiệm chi tiết (ablation, fix data leakage) nằm trong `research/notebooks/`.

---

## 10. Build production (frontend)

```bash
npm run build      # tạo thư mục dist/
npm run preview    # xem thử bản build
```

---

## 11. Giới hạn & hướng phát triển

- **Bài toán nhị phân** (bình thường / cất giấu hàng hóa) — chưa phân biệt nhiều loại hành vi.
- **Xử lý offline** theo lô, không realtime (đánh đổi để giữ độ chính xác).
- Trên **video ngoài phân phối huấn luyện**, độ tin cậy ở vùng ~50–65% có thể chồng lấp giữa
  bình thường và bất thường → cần người giám sát kiểm tra lại.
- **Hướng phát triển**: triển khai YOLOv8-Pose trên GPU + streaming WebSocket để tiến gần
  realtime; mở rộng đa lớp hành vi; định vị sự kiện theo thời gian (temporal localization).

---

## 12. Nhóm thực hiện

| Vai trò | Họ tên | MSSV |
|---|---|---|
| Sinh viên | Mai Hồng Hải | 22133014 |
| Sinh viên | Nguyễn Ngọc Hiếu Hảo | 22133015 |
| GVHD | ThS. Phan Thị Thể | — |

Khoa Công nghệ Thông tin — Trường Đại học Sư phạm Kỹ thuật TP.HCM (HCMUTE).

---

*Dự án phục vụ mục đích học thuật (khóa luận tốt nghiệp). Bộ dữ liệu video gốc thuộc về
nguồn công khai trên Kaggle.*
