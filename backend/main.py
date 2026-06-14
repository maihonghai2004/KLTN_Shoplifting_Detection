"""
Backend — Hệ thống Nhận diện hành vi bất thường trong siêu thị (HCMUTE).

Độc lập, không phụ thuộc dự án KLTN. Luồng trang "Phân tích video":
    upload -> YOLOv8-Pose (17 khớp) -> 56 đặc trưng hình học -> cửa sổ 90 frame
    -> mô hình V1.4 (BiLSTM-CNN) -> điểm nghi ngờ + verdict
    -> (nếu bất thường) chụp ảnh bằng chứng + GỬI EMAIL cảnh báo.

Chạy:
    .venv/Scripts/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import json
import os
import re
import shutil
import smtplib
import subprocess
import tempfile
import uuid
import warnings
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from geometric_features import extract_geometric_features
from v14_loader import load_v14

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# --------------------------------------------------------------------------- #
# Cấu hình
# --------------------------------------------------------------------------- #
load_dotenv()
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")

BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "models"
STATIC_DIR = BACKEND_DIR / "static"
RUNS_DIR = STATIC_DIR / "runs"          # mỗi lần phân tích = 1 thư mục con trong đây
STATIC_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

SEQ, STEP = 90, 30
THRESHOLD = 0.5          # ngưỡng xác suất 1 cửa sổ bị coi là "nghi ngờ"
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]
MODEL_LABEL = "V1.4 BiLSTM-CNN"

_yolo = None
_v14 = None

# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="HCMUTE Abnormal Behavior API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def get_yolo():
    global _yolo
    if _yolo is None:
        print("[Backend] Loading YOLOv8-Pose...")
        _yolo = YOLO(str(MODELS_DIR / "yolov8s-pose.pt"))
    return _yolo


def get_v14():
    global _v14
    if _v14 is None:
        print("[Backend] Loading V1.4 model...")
        _v14 = load_v14(str(MODELS_DIR / "v1.4_best.keras"), (SEQ, 56))
    return _v14


# --------------------------------------------------------------------------- #
# Email
# --------------------------------------------------------------------------- #
def send_email_alert(message: str) -> bool:
    if not all([GMAIL_USER, GMAIL_APP_PASSWORD, RECIPIENT_EMAIL]):
        print("⚠️  Email chưa cấu hình (.env: GMAIL_USER / GMAIL_APP_PASSWORD / RECIPIENT_EMAIL).")
        return False
    try:
        msg = MIMEMultipart()
        msg["From"] = GMAIL_USER
        msg["To"] = RECIPIENT_EMAIL
        msg["Subject"] = "🚨 Cảnh báo hành vi bất thường trong siêu thị"
        msg.attach(MIMEText(message, "plain"))
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_USER, RECIPIENT_EMAIL, msg.as_string())
        print(f"✅ Đã gửi email cảnh báo tới {RECIPIENT_EMAIL}")
        return True
    except Exception as exc:
        print(f"❌ Gửi email thất bại: {exc}")
        return False


# --------------------------------------------------------------------------- #
# Xử lý
# --------------------------------------------------------------------------- #
def extract_features(video_path: str, max_frames: int = 1500):
    yolo = get_yolo()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(400, "Không mở được video")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    feats, feat_frames, kp_by_frame = [], [], {}
    n_frames, n_person = 0, 0
    while cap.isOpened() and n_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo.predict(frame, verbose=False, conf=0.5)
        kp = None
        for r in results:
            if r.keypoints is not None and len(r.keypoints.data) > 0:
                k = r.keypoints.xyn[0].cpu().numpy()
                if k.shape == (17, 2):
                    kp = k
                break
        if kp is not None and not np.allclose(kp, 0):
            n_person += 1
            kp_by_frame[n_frames] = kp
            feats.append(extract_geometric_features(kp))
            feat_frames.append(n_frames)
        n_frames += 1
    cap.release()
    return np.asarray(feats, dtype=np.float32), feat_frames, kp_by_frame, n_frames, n_person, fps


def make_windows(arr):
    T = arr.shape[0]
    if T < SEQ:
        pad = np.repeat(arr[-1:], SEQ - T, axis=0)
        return np.concatenate([arr, pad], axis=0)[None]
    return np.asarray([arr[i:i + SEQ] for i in range(0, T - SEQ + 1, STEP)], dtype=np.float32)


def render_snapshot(video_path, frame_idx, kp):
    """Vẽ khung xương + nhãn lên frame và trả về JPEG bytes (None nếu lỗi)."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    color = (0, 0, 255)
    if kp is not None:
        pts = [(int(x * w), int(y * h)) for (x, y) in kp]
        for a, b in COCO_EDGES:
            if not (kp[a][0] == 0 and kp[a][1] == 0) and not (kp[b][0] == 0 and kp[b][1] == 0):
                cv2.line(frame, pts[a], pts[b], color, 2)
        valid = [(i, p) for i, p in enumerate(pts) if not (kp[i][0] == 0 and kp[i][1] == 0)]
        for _, p in valid:
            cv2.circle(frame, p, 3, (0, 255, 255), -1)
        if valid:
            xs = [p[0] for _, p in valid]; ys = [p[1] for _, p in valid]
            cv2.rectangle(frame, (min(xs) - 8, min(ys) - 8), (max(xs) + 8, max(ys) + 8), color, 2)
    cv2.putText(frame, "SHOPLIFTING", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    scale = 480 / w if w > 480 else 1.0
    if scale != 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else None


def new_run_dir(filename):
    """Tạo 1 thư mục cho lần phân tích này: runs/<ngày-giờ>_<tên video>_<id>/.
    Tên có timestamp để sắp xếp theo thời gian, có slug tên video để dễ nhận biết."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", Path(filename or "video").stem)[:30].strip("-") or "video"
    name = f"{ts}_{slug}_{uuid.uuid4().hex[:6]}"
    d = RUNS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d, name


def save_meta(run_dir, meta):
    """Ghi meta.json tóm tắt kết quả vào thư mục run (dễ tra cứu/maintain sau này)."""
    try:
        (run_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def transcode_h264(src, dst):
    subprocess.run(
        [FFMPEG, "-y", "-i", src, "-c:v", "libx264", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst],
        check=True, capture_output=True,
    )


def clear_static_files():
    """Xóa toàn bộ thư mục run (video + ảnh + meta) — gọi khi bấm 'Xóa tất cả'."""
    n = 0
    if RUNS_DIR.exists():
        for d in RUNS_DIR.iterdir():
            try:
                if d.is_dir():
                    shutil.rmtree(d)
                else:
                    d.unlink()
                n += 1
            except OSError:
                pass
    # Dọn cả file phẳng cũ (cấu trúc trước đây) nếu còn sót.
    for pattern in ("play_*.mp4", "snap_*.jpg"):
        for p in STATIC_DIR.glob(pattern):
            try:
                p.unlink()
            except OSError:
                pass
    return n


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/models")
def list_models():
    return {"models": [{"key": "v1.4", "label": MODEL_LABEL}]}


@app.post("/clear-alerts")
def clear_alerts():
    """Xóa mọi ảnh bằng chứng + video phát lại trên server (frontend gọi khi bấm 'Xóa tất cả')."""
    return {"deleted": clear_static_files()}


@app.post("/analyze-model")
async def analyze_model(file: UploadFile = File(...), model: str = Form("v1.4")):
    suffix = Path(file.filename).suffix or ".mp4"
    tmp = Path(tempfile.gettempdir()) / f"hcmute_upload{suffix}"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)

    run_dir, run_name = new_run_dir(file.filename)
    playback_url = None
    try:
        transcode_h264(str(tmp), str(run_dir / "playback.mp4"))
        playback_url = f"/static/runs/{run_name}/playback.mp4"
    except Exception as exc:
        print(f"[Backend] transcode failed: {exc}")

    try:
        feats, feat_frames, kp_by_frame, n_frames, n_person, fps = extract_features(str(tmp))
        if len(feats) == 0:
            save_meta(run_dir, {"file": file.filename, "label": "no_person",
                                "analyzed_at": datetime.now().isoformat(timespec="seconds")})
            return {"model": "v1.4", "label": "no_person", "verdict": "Không phát hiện người",
                    "score": 0.0, "n_frames": n_frames, "n_person_frames": 0, "n_windows": 0,
                    "fps": fps, "timeline": [], "playback_url": playback_url,
                    "snapshot_url": None, "email_sent": False}

        wins = make_windows(feats)
        p = get_v14().predict(wins, verbose=0)[:, 1]
        score = float(p.mean())          # điểm trung bình toàn video (tham khảo)
        score_max = float(p.max())       # đỉnh nghi ngờ

        # Frame "đại diện" cho mỗi cửa sổ (frame cuối của cửa sổ).
        if len(feats) >= SEQ:
            window_frames = [feat_frames[start + SEQ - 1]
                             for start in range(0, len(feats) - SEQ + 1, STEP)]
        else:
            window_frames = [feat_frames[-1]]

        score_by_frame = {window_frames[j]: float(p[j]) for j in range(len(p))}

        timeline, cur = [], None
        for fr in range(n_frames):
            if fr in score_by_frame:
                cur = score_by_frame[fr]
            kp = kp_by_frame.get(fr)
            timeline.append({
                "f": fr,
                "kp": [[round(float(x), 4), round(float(y), 4)] for x, y in kp] if kp is not None else None,
                "s": round(cur, 3) if cur is not None else None,
            })

        # --- Phát hiện TỪNG SỰ KIỆN bất thường ---------------------------------
        # Mỗi ĐOẠN cửa sổ liên tiếp có điểm >= ngưỡng = MỘT sự kiện. Bất kỳ sự kiện
        # nào cũng được ghi nhận (ảnh bằng chứng + số liệu). Không cần "kéo dài >=N".
        events = []
        i = 0
        while i < len(p):
            if p[i] >= THRESHOLD:
                j = i
                while j + 1 < len(p) and p[j + 1] >= THRESHOLD:
                    j += 1
                seg = p[i:j + 1]
                k = i + int(np.argmax(seg))      # cửa sổ đỉnh trong đoạn
                ev_frame = int(window_frames[k])
                idx = len(events) + 1
                buf = render_snapshot(str(tmp), ev_frame, kp_by_frame.get(ev_frame))
                snap_url = None
                if buf is not None:
                    (run_dir / f"snap_{idx}.jpg").write_bytes(buf)
                    snap_url = f"/static/runs/{run_name}/snap_{idx}.jpg"
                events.append({
                    "index": idx,
                    "score": round(float(seg.max()), 4),
                    "peak_frame": ev_frame,
                    "peak_time": round(ev_frame / fps, 2) if fps else 0.0,
                    "start_time": round(window_frames[i] / fps, 2) if fps else 0.0,
                    "end_time": round(window_frames[j] / fps, 2) if fps else 0.0,
                    "n_windows": int(j - i + 1),
                    "snapshot_url": snap_url,
                })
                i = j + 1
            else:
                i += 1

        is_shoplift = len(events) > 0

        # --- MỘT email duy nhất cho cả video, gộp đầy đủ mọi sự kiện -----------
        # (1 video / 1 người -> chỉ gửi 1 thư, không spam nhiều mail.)
        email_sent = False
        if is_shoplift:
            lines = [
                f"  {e['index']}. Thoi diem {e['peak_time']:.1f}s (frame {e['peak_frame']}) "
                f"- do nghi ngo {round(e['score'] * 100)}% "
                f"[doan {e['start_time']:.1f}s-{e['end_time']:.1f}s]"
                for e in events
            ]
            message = (
                f"CANH BAO: Phat hien {len(events)} hanh vi bat thuong trong video\n\n"
                f"Thoi gian phan tich: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Video: {file.filename}\n"
                f"Mo hinh: {MODEL_LABEL}\n"
                f"Ket luan: HANH VI CAT GIAU HANG HOA ({len(events)} su kien)\n\n"
                "Danh sach su kien:\n" + "\n".join(lines) + "\n\n"
                "Anh bang chung tung su kien xem tai trang Canh bao cua he thong.\n"
                "Day la canh bao tu dong tu He thong Nhan dien hanh vi bat thuong - HCMUTE."
            )
            email_sent = send_email_alert(message)

        # Sự kiện nổi bật nhất (điểm cao nhất) — ảnh đại diện + tương thích cũ.
        top = max(events, key=lambda e: e["score"]) if events else None

        save_meta(run_dir, {
            "file": file.filename,
            "analyzed_at": datetime.now().isoformat(timespec="seconds"),
            "model": MODEL_LABEL,
            "label": "shoplifting" if is_shoplift else "normal",
            "score_mean": round(score, 4),
            "score_max": round(score_max, 4),
            "n_events": len(events),
            "email_sent": email_sent,
            "recipient": RECIPIENT_EMAIL if email_sent else None,
            "events": events,
        })

        return {
            "model": "v1.4",
            "model_label": MODEL_LABEL,
            "label": "shoplifting" if is_shoplift else "normal",
            "verdict": (f"⚠ {len(events)} HÀNH VI GIẤU HÀNG" if is_shoplift else "✅ Bình thường"),
            "score": round(score, 4),
            "score_max": round(score_max, 4),
            "threshold": THRESHOLD,
            "n_events": len(events),
            "events": events,
            "peak_frame": top["peak_frame"] if top else None,
            "peak_time": top["peak_time"] if top else 0.0,
            "n_frames": n_frames,
            "n_person_frames": n_person,
            "n_windows": int(len(wins)),
            "fps": round(fps, 3),
            "timeline": timeline,
            "playback_url": playback_url,
            "snapshot_url": top["snapshot_url"] if top else None,
            "snapshot_frame": top["peak_frame"] if top else None,
            "email_sent": email_sent,
            "recipient_email": RECIPIENT_EMAIL if email_sent else None,
        }
    finally:
        if tmp.exists():
            tmp.unlink()


@app.get("/")
def root():
    return {"message": "HCMUTE Abnormal Behavior API", "model": MODEL_LABEL}
