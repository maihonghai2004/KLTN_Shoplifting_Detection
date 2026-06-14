"""Shared fixtures cho unit tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_keypoints_17x2():
    """Keypoints chuẩn hoá cho người đứng bình thường (COCO-17)."""
    return np.array([
        [0.50, 0.15],  # 0  nose
        [0.48, 0.13],  # 1  left_eye
        [0.52, 0.13],  # 2  right_eye
        [0.45, 0.14],  # 3  left_ear
        [0.55, 0.14],  # 4  right_ear
        [0.42, 0.25],  # 5  left_shoulder
        [0.58, 0.25],  # 6  right_shoulder
        [0.38, 0.38],  # 7  left_elbow
        [0.62, 0.38],  # 8  right_elbow
        [0.35, 0.50],  # 9  left_wrist
        [0.65, 0.50],  # 10 right_wrist
        [0.44, 0.55],  # 11 left_hip
        [0.56, 0.55],  # 12 right_hip
        [0.43, 0.72],  # 13 left_knee
        [0.57, 0.72],  # 14 right_knee
        [0.43, 0.90],  # 15 left_ankle
        [0.57, 0.90],  # 16 right_ankle
    ], dtype=np.float32)


@pytest.fixture
def concealing_keypoints_17x2():
    """Keypoints khi wrists gần hips (tư thế cất giấu hàng)."""
    return np.array([
        [0.50, 0.15],  # 0  nose
        [0.48, 0.13],  # 1  left_eye
        [0.52, 0.13],  # 2  right_eye
        [0.45, 0.14],  # 3  left_ear
        [0.55, 0.14],  # 4  right_ear
        [0.42, 0.25],  # 5  left_shoulder
        [0.58, 0.25],  # 6  right_shoulder
        [0.38, 0.38],  # 7  left_elbow
        [0.62, 0.38],  # 8  right_elbow
        [0.45, 0.56],  # 9  left_wrist  ← gần left_hip
        [0.57, 0.56],  # 10 right_wrist ← gần right_hip
        [0.44, 0.55],  # 11 left_hip
        [0.56, 0.55],  # 12 right_hip
        [0.43, 0.72],  # 13 left_knee
        [0.57, 0.72],  # 14 right_knee
        [0.43, 0.90],  # 15 left_ankle
        [0.57, 0.90],  # 16 right_ankle
    ], dtype=np.float32)
