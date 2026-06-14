"""
Bước 1 — Trích xuất 17 keypoint COCO từ video bằng YOLOv8s-pose.

Đây là bước đầu của pipeline data preprocessing.

Input:
    data/raw_videos/{normal, shoplifting}/*.mp4 (.avi cũng được)

Output:
    data/raw_keypoints/{normal, shoplifting}/*.csv  (mỗi file 34 cột x,y)

Cách chạy:
    python scripts/01_extract_keypoints_from_video.py

Tác giả gốc: Hảo (extract_data.py), tổ chức lại bởi mhhai.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def extract_keypoints_from_videos(
    yolo_path: str,
    input_root: str,
    output_root: str,
    conf_threshold: float = 0.5,
    labels: tuple = ('normal', 'shoplifting'),
):
    """Duyệt từng video, trích keypoints, lưu CSV."""
    print(f"[1/4] Loading YOLO: {yolo_path}")
    model = YOLO(yolo_path)

    total_videos = 0
    total_frames = 0

    for label in labels:
        video_dir = os.path.join(input_root, label)
        csv_dir = os.path.join(output_root, label)
        os.makedirs(csv_dir, exist_ok=True)

        if not os.path.exists(video_dir):
            print(f"  ⚠ Bỏ qua: {video_dir} không tồn tại")
            continue

        for vname in sorted(os.listdir(video_dir)):
            if not vname.lower().endswith(('.mp4', '.avi', '.mov')):
                continue
            total_videos += 1
            vpath = os.path.join(video_dir, vname)
            cap = cv2.VideoCapture(vpath)
            kps = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                results = model.predict(frame, verbose=False, conf=conf_threshold)
                frame_data = np.zeros(34, dtype=np.float32)  # default
                for r in results:
                    if r.keypoints is not None and len(r.keypoints.data) > 0:
                        kp = r.keypoints.xyn[0].cpu().numpy()
                        if kp.shape == (17, 2):
                            frame_data = kp.flatten()
                        break
                kps.append(frame_data)
                total_frames += 1
            cap.release()

            csv_name = os.path.splitext(vname)[0] + ".csv"
            out_path = os.path.join(csv_dir, csv_name)
            pd.DataFrame(kps).to_csv(out_path, index=False)
            print(f"  ✓ {label}/{vname} → {len(kps)} frame → {csv_name}")

    print(f"\n[Done] Processed {total_videos} videos, {total_frames} frames.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", type=str,
                        default=str(ROOT / "models" / "checkpoints" / "yolov8s-pose.pt"))
    parser.add_argument("--input", type=str,
                        default=str(ROOT / "data" / "raw_videos"),
                        help="Folder chứa subfolder normal/ và shoplifting/")
    parser.add_argument("--output", type=str,
                        default=str(ROOT / "data" / "raw_keypoints"))
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    extract_keypoints_from_videos(
        yolo_path=args.yolo,
        input_root=args.input,
        output_root=args.output,
        conf_threshold=args.conf,
    )


if __name__ == "__main__":
    main()
