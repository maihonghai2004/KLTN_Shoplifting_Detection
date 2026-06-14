"""
Trích xuất 56 đặc trưng hình học từ 17 keypoint COCO.

Đây là logic gốc của Hảo trong filter_data(1).py, được tổ chức lại để
tái sử dụng từ nhiều nơi trong project (training + inference).

56 đặc trưng = 34 relative_kp + 20 bone_vectors + 2 hand-hip distances.
"""
from __future__ import annotations

import numpy as np

# 10 cặp khớp liên kết kinematic (theo paper Hảo cite)
BONE_PAIRS = [
    (5, 7), (7, 9),       # Tay trái
    (6, 8), (8, 10),      # Tay phải
    (11, 13), (13, 15),   # Chân trái
    (12, 14), (14, 16),   # Chân phải
    (5, 6),               # Vai-vai
    (11, 12),             # Hông-hông
]


def extract_geometric_features(kp_raw: np.ndarray) -> np.ndarray:
    """
    Convert 17 keypoint thô thành 56 đặc trưng hình học.

    Args:
        kp_raw: ndarray shape (17, 2) hoặc (34,) — toạ độ chuẩn hoá xyn từ YOLO-Pose.

    Returns:
        ndarray shape (56,) gồm:
            - 34 dim: relative_kp = kp - hip_midpoint
            - 20 dim: bone vectors của 10 cặp khớp
            - 2 dim: distance(left_wrist, left_hip), distance(right_wrist, right_hip)
    """
    kp = kp_raw.reshape(17, 2)
    feats = []

    # A. Toạ độ tương đối (centered at hip midpoint)
    hip_center = (kp[11] + kp[12]) / 2.0
    relative = kp - hip_center
    feats.extend(relative.flatten())  # 34

    # B. Bone vectors
    for start, end in BONE_PAIRS:
        feats.extend(kp[end] - kp[start])  # 20

    # C. Khoảng cách cổ tay - hông
    feats.append(np.linalg.norm(kp[9] - kp[11]))    # cổ tay trái - hông trái
    feats.append(np.linalg.norm(kp[10] - kp[12]))   # cổ tay phải - hông phải

    return np.array(feats, dtype=np.float32)  # 56
