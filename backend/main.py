"""
Backend — Hệ thống Nhận diện hành vi bất thường trong siêu thị (HCMUTE).

Đã nâng cấp: 
1. Hỗ trợ MULTI-PERSON TRACKING sử dụng ByteTrack.
2. Vẽ TRỰC TIẾP khung xương, ID và nhãn cảnh báo lên video playback sau khi phân tích.

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
RUNS_DIR = STATIC_DIR / "runs"          
STATIC_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

SEQ, STEP = 90, 30
THRESHOLD = 0.5          # Ngưỡng xác suất bất thường
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]
MODEL_LABEL = "V1.4 BiLSTM-CNN + ByteTrack"

_yolo = None
_v14 = None

# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="HCMUTE Abnormal Behavior API with Video Rendering")
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
        msg["Subject"] = "🚨 [Tracking] Cảnh báo hành vi bất thường nhiều đối tượng"
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
# Xử lý và tính toán điểm (Two-pass processing)
# --------------------------------------------------------------------------- #
def make_windows(arr):
    T = arr.shape[0]
    if T < SEQ:
        pad = np.repeat(arr[-1:], SEQ - T, axis=0)
        return np.concatenate([arr, pad], axis=0)[None]
    return np.asarray([arr[i:i + SEQ] for i in range(0, T - SEQ + 1, STEP)], dtype=np.float32)


def extract_features_and_predict(video_path: str, max_frames: int = 1500):
    """
    Bước 1: Trích xuất tọa độ xương của từng đối tượng qua ByteTrack.
    Bước 2: Dự đoán điểm bất thường cho từng ID bằng mô hình V1.4 để có điểm số theo thời gian thực.
    """
    yolo = get_yolo()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(400, "Không mở được video")
        
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    track_raw_data = {}
    n_frames = 0
    total_detected_persons = set()

    # Thu thập dữ liệu thô từ video
    while cap.isOpened() and n_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = yolo.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.4)
        
        for r in results:
            if r.keypoints is None or r.boxes is None or r.boxes.id is None:
                continue
                
            track_ids = r.boxes.id.int().cpu().tolist()
            kpts_normalized = r.keypoints.xyn.cpu().numpy()
            
            for idx, track_id in enumerate(track_ids):
                total_detected_persons.add(track_id)
                kp = kpts_normalized[idx]
                
                if kp.shape == (17, 2) and not np.allclose(kp, 0):
                    if track_id not in track_raw_data:
                        track_raw_data[track_id] = {"feats": [], "frames": [], "kpts": []}
                    
                    geom_feat = extract_geometric_features(kp)
                    track_raw_data[track_id]["feats"].append(geom_feat)
                    track_raw_data[track_id]["frames"].append(n_frames)
                    track_raw_data[track_id]["kpts"].append(kp)
                    
        n_frames += 1
    cap.release()

    # Tính toán điểm số bất thường theo từng frame cho từng ID
    scores_by_id_and_frame = {}
    track_clean_data = {}

    for track_id, data in track_raw_data.items():
        feats_arr = np.asarray(data["feats"], dtype=np.float32)
        feat_frames = data["frames"]
        
        wins = make_windows(feats_arr)
        p = get_v14().predict(wins, verbose=0)[:, 1]
        
        if len(feats_arr) >= SEQ:
            window_frames = [feat_frames[start + SEQ - 1] for start in range(0, len(feats_arr) - SEQ + 1, STEP)]
        else:
            window_frames = [feat_frames[-1]]

        score_by_frame = {}
        cur_score = 0.0
        # Điền điểm số lan tỏa theo dòng thời gian của frame
        for f_idx, fr in enumerate(feat_frames):
            if fr in window_frames:
                w_idx = window_frames.index(fr)
                if w_idx < len(p):
                    cur_score = float(p[w_idx])
            score_by_frame[fr] = cur_score
            
        scores_by_id_and_frame[track_id] = score_by_frame
        
        track_clean_data[track_id] = {
            "p": p,
            "window_frames": window_frames,
            "feat_frames": feat_frames,
            "kpts": data["kpts"],
            "score_by_frame": score_by_frame
        }

    return track_clean_data, scores_by_id_and_frame, n_frames, len(total_detected_persons), fps


# --------------------------------------------------------------------------- #
# Kết xuất Video (Render)
# --------------------------------------------------------------------------- #
def render_output_video(video_path: str, output_path: str, track_clean_data: dict, max_frames: int):
    """
    Đọc lại video gốc và ghi đè các thông tin nhận diện (Khung xương, ID, Trạng thái) 
    lên từng khung hình dựa trên kết quả phân tích.
    """
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    n_frames = 0
    while cap.isOpened() and n_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        any_shoplift_in_frame = False
        
        # Duyệt qua tất cả các ID xem có ai xuất hiện ở frame này không
        for track_id, data in track_clean_data.items():
            if n_frames in data["feat_frames"]:
                idx = data["feat_frames"].index(n_frames)
                kp = data["kpts"][idx]
                score = data["score_by_frame"].get(n_frames, 0.0)
                
                is_abnormal = score >= THRESHOLD
                # Đỏ nếu bất thường, Xanh lá nếu bình thường
                color = (0, 0, 255) if is_abnormal else (0, 255, 0)
                if is_abnormal:
                    any_shoplift_in_frame = True
                
                # Vẽ khung xương
                pts = [(int(x * w), int(y * h)) for (x, y) in kp]
                for a, b in COCO_EDGES:
                    if not (kp[a][0] == 0 and kp[a][1] == 0) and not (kp[b][0] == 0 and kp[b][1] == 0):
                        cv2.line(frame, pts[a], pts[b], color, 2)
                for i, p in enumerate(pts):
                    if not (kp[i][0] == 0 and kp[i][1] == 0):
                        cv2.circle(frame, p, 3, (0, 255, 255), -1)
                        
                # Vẽ bounding box giả lập quanh khung xương
                valid_pts = [p for i, p in enumerate(pts) if not (kp[i][0] == 0 and kp[i][1] == 0)]
                if valid_pts:
                    xs = [p[0] for p in valid_pts]
                    ys = [p[1] for p in valid_pts]
                    x_min, y_min = max(0, min(xs) - 10), max(0, min(ys) - 10)
                    x_max, y_max = min(w, max(xs) + 10), min(h, max(ys) + 10)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    
                    # Hiện ID và độ nghi ngờ (%)
                    label_str = f"ID: {track_id} ({round(score * 100)}%)"
                    cv2.putText(frame, label_str, (x_min, max(20, y_min - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
        # Nếu có bất kỳ đối tượng nào vi phạm trong frame, hiện chữ cảnh báo lớn góc màn hình
        if any_shoplift_in_frame:
            cv2.putText(frame, "🚨 SUSPICIOUS BEHAVIOR DETECTED", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                        
        out.write(frame)
        n_frames += 1
        
    cap.release()
    out.release()


def render_snapshot_with_id(video_path, frame_idx, kp, track_id):
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
            x_min, y_min = max(0, min(xs) - 8), max(0, min(ys) - 8)
            x_max, y_max = min(w, max(xs) + 8), min(h, max(ys) + 8)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x_min, max(20, y_min - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
    cv2.putText(frame, f"SUSPICIOUS BEHAVIOR", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    scale = 480 / w if w > 480 else 1.0
    if scale != 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else None


def new_run_dir(filename):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", Path(filename or "video").stem)[:30].strip("-") or "video"
    name = f"{ts}_{slug}_{uuid.uuid4().hex[:6]}"
    d = RUNS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d, name


def save_meta(run_dir, meta):
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
    return n


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/models")
def list_models():
    return {"models": [{"key": "v1.4", "label": MODEL_LABEL}]}


@app.post("/clear-alerts")
def clear_alerts():
    return {"deleted": clear_static_files()}


@app.post("/analyze-model")
async def analyze_model(file: UploadFile = File(...), model: str = Form("v1.4")):
    suffix = Path(file.filename).suffix or ".mp4"
    tmp_upload = Path(tempfile.gettempdir()) / f"hcmute_upload_{uuid.uuid4().hex}{suffix}"
    tmp_rendered = Path(tempfile.gettempdir()) / f"hcmute_rendered_{uuid.uuid4().hex}.mp4"
    
    with open(tmp_upload, "wb") as f:
        shutil.copyfileobj(file.file, f)

    run_dir, run_name = new_run_dir(file.filename)
    playback_url = None

    try:
        # 1. Chạy tracking và dự đoán hành vi đa đối tượng
        track_clean_data, scores_by_id_and_frame, n_frames, total_persons, fps = extract_features_and_predict(str(tmp_upload))
        
        if not track_clean_data:
            transcode_h264(str(tmp_upload), str(run_dir / "playback.mp4"))
            playback_url = f"/static/runs/{run_name}/playback.mp4"
            save_meta(run_dir, {"file": file.filename, "label": "no_person", "analyzed_at": datetime.now().isoformat(timespec="seconds")})
            return {"model": "v1.4", "label": "no_person", "verdict": "Không phát hiện người nào",
                    "score_max": 0.0, "n_frames": n_frames, "total_persons_tracked": 0, "n_windows": 0,
                    "fps": fps, "timeline": [], "playback_url": playback_url, "snapshot_url": None, "email_sent": False}

        # 2. Vẽ thông tin nhận diện & hành vi trực tiếp lên video mới
        render_output_video(str(tmp_upload), str(tmp_rendered), track_clean_data, n_frames)
        
        # 3. Transcode video đã vẽ sang chuẩn H.264 để chạy được trên Web Browser
        try:
            transcode_h264(str(tmp_rendered), str(run_dir / "playback.mp4"))
            playback_url = f"/static/runs/{run_name}/playback.mp4"
        except Exception as exc:
            print(f"[Backend] transcode failed: {exc}")
            # Fallback về video gốc nếu lỗi transcode video vẽ
            transcode_h264(str(tmp_upload), str(run_dir / "playback.mp4"))
            playback_url = f"/static/runs/{run_name}/playback.mp4"

        all_events = []
        global_timeline = {fr: {"f": fr, "targets": []} for fr in range(n_frames)}
        global_max_score = 0.0
        total_windows_processed = 0

        # 4. Phân tích chuỗi sự kiện và gom danh sách bất thường
        for track_id, data in track_clean_data.items():
            p = data["p"]
            window_frames = data["window_frames"]
            feat_frames = data["feat_frames"]
            kpts_list = data["kpts"]
            
            total_windows_processed += len(p)
            if len(p) > 0 and float(p.max()) > global_max_score:
                global_max_score = float(p.max())

            # Map ngược sang cấu trúc khung xương để lưu timeline JSON
            frame_to_kp_map = {feat_frames[idx]: kpts_list[idx] for idx in range(len(feat_frames))}

            for fr in range(n_frames):
                cur_score = data["score_by_frame"].get(fr, 0.0)
                if fr in frame_to_kp_map:
                    global_timeline[fr]["targets"].append({
                        "id": track_id,
                        "kp": [[round(float(x), 4), round(float(y), 4)] for x, y in frame_to_kp_map[fr]],
                        "s": round(cur_score, 3)
                    })

            # Bóc tách sự kiện liên tục vượt ngưỡng
            i = 0
            while i < len(p):
                if p[i] >= THRESHOLD:
                    j = i
                    while j + 1 < len(p) and p[j + 1] >= THRESHOLD:
                        j += 1
                    seg = p[i:j + 1]
                    k = i + int(np.argmax(seg))
                    ev_frame = int(window_frames[k])
                    
                    event_idx = len(all_events) + 1
                    
                    # Chụp hình snapshot bằng chứng
                    buf = render_snapshot_with_id(str(tmp_upload), ev_frame, frame_to_kp_map.get(ev_frame), track_id)
                    snap_url = None
                    if buf is not None:
                        (run_dir / f"snap_{event_idx}.jpg").write_bytes(buf)
                        snap_url = f"/static/runs/{run_name}/snap_{event_idx}.jpg"
                        
                    all_events.append({
                        "index": event_idx,
                        "track_id": track_id,
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

        is_shoplift = len(all_events) > 0

        # Gửi email nếu phát hiện dấu hiệu vi phạm
        email_sent = False
        if is_shoplift:
            lines = [
                f"  {e['index']}. [ID {e['track_id']}] Khung hinh {e['peak_frame']} ({e['peak_time']:.1f}s) - Ti le nghi ngo {round(e['score'] * 100)}%"
                for e in all_events
            ]
            message = (
                f"HE THONG CANH BAO QUET MULTI-PERSON (HCMUTE)\n\n"
                f"Tep tin phan tich: {file.filename}\n"
                f"Trạng thái: PHAT HIEN {len(all_events)} HANH VI BAT THUONG THEO THEO DOI ID\n\n"
                "Chi tiet danh sach:\n" + "\n".join(lines) + "\n\n"
                "Video ghi hinh kem tag ID va tracking da duoc dong bo hoa thanh cong tai o dia static."
            )
            email_sent = send_email_alert(message)

        top = max(all_events, key=lambda e: e["score"]) if all_events else None
        timeline_list = sorted(list(global_timeline.values()), key=lambda x: x["f"])

        save_meta(run_dir, {
            "file": file.filename,
            "analyzed_at": datetime.now().isoformat(timespec="seconds"),
            "model": MODEL_LABEL,
            "label": "shoplifting" if is_shoplift else "normal",
            "score_max": round(global_max_score, 4),
            "total_persons_tracked": total_persons,
            "n_events": len(all_events),
            "email_sent": email_sent,
            "events": all_events,
        })

        return {
            "model": "v1.4",
            "model_label": MODEL_LABEL,
            "label": "shoplifting" if is_shoplift else "normal",
            "verdict": (f"⚠ {len(all_events)} HÀNH VI GIẤU HÀNG (Đã vẽ lên video)" if is_shoplift else "✅ Bình thường"),
            "score_max": round(global_max_score, 4),
            "threshold": THRESHOLD,
            "n_events": len(all_events),
            "events": all_events,
            "peak_frame": top["peak_frame"] if top else None,
            "peak_time": top["peak_time"] if top else 0.0,
            "n_frames": n_frames,
            "total_persons_tracked": total_persons,
            "n_windows": total_windows_processed,
            "fps": round(fps, 3),
            "timeline": timeline_list,
            "playback_url": playback_url,
            "snapshot_url": top["snapshot_url"] if top else None,
            "snapshot_frame": top["peak_frame"] if top else None,
            "email_sent": email_sent,
            "recipient_email": RECIPIENT_EMAIL if email_sent else None,
        }
    finally:
        for p_file in (tmp_upload, tmp_rendered):
            if p_file.exists():
                p_file.unlink()


@app.get("/")
def root():
    return {"message": "HCMUTE Abnormal Behavior API with Rendering Enabled", "model": MODEL_LABEL}