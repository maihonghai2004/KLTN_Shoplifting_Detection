"""
Trích 56 đặc trưng hình học từ 17 keypoint COCO.
(Chuyển từ dự án KLTN sang để backend ShopliftingDetection chạy độc lập.)

56 = 34 relative_kp + 20 bone vectors + 2 hand-hip distances.
"""
from __future__ import annotations

import numpy as np

# 10 cặp khớp liên kết kinematic
BONE_PAIRS = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (11, 12),
]


def extract_geometric_features(kp_raw: np.ndarray) -> np.ndarray:
    """(17,2) hoặc (34,) keypoint chuẩn hoá -> vector 56 đặc trưng."""
    kp = kp_raw.reshape(17, 2)
    feats = []

    # A. Toạ độ tương đối quanh trung điểm hông
    hip_center = (kp[11] + kp[12]) / 2.0
    feats.extend((kp - hip_center).flatten())          # 34

    # B. Vector xương
    for start, end in BONE_PAIRS:
        feats.extend(kp[end] - kp[start])              # 20

    # C. Khoảng cách cổ tay - hông
    feats.append(np.linalg.norm(kp[9] - kp[11]))       # cổ tay trái - hông trái
    feats.append(np.linalg.norm(kp[10] - kp[12]))      # cổ tay phải - hông phải

    return np.array(feats, dtype=np.float32)           # 56
