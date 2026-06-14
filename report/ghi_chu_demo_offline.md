# Ghi chú cho báo cáo — Vì sao demo xử lý offline (batch), không realtime

> Đoạn này dùng cho **Chương Ứng dụng / Demo**, mục lý giải lựa chọn kiến trúc xử lý.
> Có thể chép thẳng vào báo cáo (.docx).

## 1. Mô tả cơ chế xử lý của ứng dụng

Khi người dùng tải video lên, hệ thống thực hiện **phân tích theo lô (offline batch
inference)** thay vì suy luận trực tiếp (real-time). Quy trình gồm ba bước chạy tuần tự
ở phía máy chủ trước khi trả kết quả:

1. **Trích khung xương:** mô hình **YOLOv8-Pose** xử lý lần lượt từng khung hình của
   toàn bộ video để lấy 17 điểm khớp (keypoints) của người.
2. **Suy luận hành vi:** chuỗi keypoint được chuyển thành **56 đặc trưng hình học**,
   gom theo **cửa sổ trượt 90 khung hình** rồi đưa vào mô hình **V1.4 (BiLSTM-CNN)** để
   tính điểm nghi ngờ hành vi cất giấu hàng hóa.
3. **Chuẩn hóa video phát lại:** video được mã hóa lại sang chuẩn **H.264** để phát
   được trên trình duyệt, kèm lớp phủ khung xương đồng bộ theo từng khung hình.

Vì vậy người dùng phải chờ một khoảng (khoảng 30–60 giây với video ngắn). Đây là
**lựa chọn thiết kế có chủ đích**, không phải hạn chế kỹ thuật, với các lý do sau.

## 2. Vì sao chọn offline thay vì real-time

### a) Bảo đảm tính nhất quán với dữ liệu huấn luyện
Mô hình hành vi (V1.4) được huấn luyện trên đặc trưng trích từ **YOLOv8-Pose**. Nếu
chuyển sang suy luận real-time trong trình duyệt, ta buộc phải dùng các mô hình pose
nhẹ (MoveNet / TensorFlow.js) vì YOLOv8-Pose quá nặng để chạy trực tiếp trên máy
khách. Các mô hình nhẹ này sinh ra **keypoint có phân phối khác** với lúc huấn luyện,
gây hiện tượng **lệch phân phối dữ liệu (distribution mismatch / domain shift)**, khiến
mô hình dự đoán sai dù độ chính xác trên tập kiểm thử rất cao.

> Thực nghiệm trong quá trình phát triển xác nhận điều này: phiên bản demo real-time
> ban đầu cho kết quả tracking và dự đoán không ổn định, đúng do nguyên nhân lệch phân
> phối keypoint giữa pipeline huấn luyện và pipeline suy luận.

Giải pháp offline cho phép chạy **đúng YOLOv8-Pose như khi huấn luyện** ở phía máy chủ,
giữ nguyên độ chính xác đã đánh giá (F1-macro ≈ 0,92 trên held-out của V1.4).

### b) Bản chất bài toán cần cửa sổ thời gian
Hành vi cất giấu hàng hóa là **chuỗi động tác kéo dài**, không thể nhận diện từ một
khung hình đơn lẻ. Mô hình cần tích lũy đủ **90 khung hình liên tiếp** mới đưa ra một
dự đoán. Do đó việc "hiển thị kết quả ngay lập tức từ khung hình đầu" là bất khả thi về
mặt phương pháp.

### c) Phù hợp ngữ cảnh sử dụng
Ứng dụng hướng tới **phân tích lại video giám sát (post-event / forensic review)** — rà
soát đoạn ghi hình để phát hiện và lập bằng chứng hành vi bất thường. Trong ngữ cảnh
này, độ trễ vài chục giây là **chấp nhận được**, và **độ chính xác quan trọng hơn độ
tức thời**.

## 3. Đánh đổi (trade-off)

| Tiêu chí | Real-time (browser) | **Offline (đề tài chọn)** |
|---|---|---|
| Mô hình pose | Nhẹ (MoveNet) — lệch phân phối | **YOLOv8-Pose — khớp huấn luyện** |
| Độ chính xác | Giảm mạnh, không ổn định | **Giữ F1 ≈ 0,92** |
| Độ trễ | Tức thời | 30–60s/video |
| Phù hợp | Cảnh báo tức thời | **Rà soát giám sát, lập bằng chứng** |

→ Đề tài ưu tiên **độ chính xác và tính nhất quán phương pháp**, nên chọn offline.

## 4. Hướng phát triển

Có thể đạt suy luận gần real-time mà vẫn giữ độ chính xác bằng cách triển khai
YOLOv8-Pose trên **máy chủ có GPU**, truyền luồng kết quả về client theo từng đoạn qua
**WebSocket/SSE** (streaming). Đây là hướng mở rộng, không thuộc phạm vi đồ án hiện tại.
