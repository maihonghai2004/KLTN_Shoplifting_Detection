"""
One-Euro Filter [Casiez, Roussel & Vogel, ACM CHI 2012] cho làm mượt keypoint.

Lý do dùng: YOLO-Pose dự đoán độc lập từng frame → keypoint rung (jitter) quanh
vị trí thật. Jitter này làm vận tốc cổ tay tính ra nhiễu → suspicion score nhảy
loạn, demo trông giật. One-Euro là bộ lọc thông thấp *thích nghi*:
    - Khi khớp đứng yên (vận tốc thấp) → cutoff thấp → lọc MẠNH → hết rung.
    - Khi khớp di chuyển nhanh (vận tốc cao) → cutoff cao → lọc NHẸ → không trễ.
Nhờ vậy vừa mượt vừa không thêm độ trễ đáng kể → phù hợp realtime.

Tham chiếu:
    G. Casiez, N. Roussel, D. Vogel, "1€ Filter: A Simple Speed-based Low-pass
    Filter for Noisy Input in Interactive Systems," ACM CHI 2012, pp. 2527-2530.

Dùng:
    smoother = KeypointSmoother(num_joints=17, freq=12.0,
                               min_cutoff=0.5, beta=0.5)
    kp_smooth = smoother(kp)        # kp: (17, 2) chuẩn hoá [0,1]
    smoother.reset()                # khi chuyển video mới
"""
from __future__ import annotations

import math

import numpy as np


def _smoothing_alpha(cutoff: float, freq: float) -> float:
    """Hệ số alpha của bộ lọc thông thấp bậc 1 theo tần số cutoff."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    te = 1.0 / freq
    return 1.0 / (1.0 + tau / te)


class OneEuroFilter:
    """One-Euro filter cho MỘT tín hiệu vô hướng (scalar)."""

    def __init__(self,
                 freq: float = 12.0,
                 min_cutoff: float = 0.5,
                 beta: float = 0.5,
                 d_cutoff: float = 1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0

    def __call__(self, x: float) -> float:
        x = float(x)
        if self._x_prev is None:
            # Lần đầu: không có vận tốc → trả về nguyên giá trị
            self._x_prev = x
            self._dx_prev = 0.0
            return x

        # 1) Ước lượng đạo hàm (vận tốc) rồi lọc nó
        dx = (x - self._x_prev) * self.freq
        a_d = _smoothing_alpha(self.d_cutoff, self.freq)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # 2) Cutoff thích nghi theo độ lớn vận tốc
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_alpha(cutoff, self.freq)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class KeypointSmoother:
    """
    Bộ lọc One-Euro cho cả khung xương (num_joints, 2).

    Mỗi toạ độ (x, y) của mỗi khớp có một filter riêng (tổng num_joints*2 filter).
    Toạ độ = 0 (khớp bị che / chưa carry-forward) được BỎ QUA để không kéo lệch
    bộ lọc; chỉ làm mượt toạ độ hợp lệ.
    """

    def __init__(self,
                 num_joints: int = 17,
                 freq: float = 12.0,
                 min_cutoff: float = 0.5,
                 beta: float = 0.5,
                 d_cutoff: float = 1.0):
        self.num_joints = num_joints
        self._fx = [OneEuroFilter(freq, min_cutoff, beta, d_cutoff)
                    for _ in range(num_joints)]
        self._fy = [OneEuroFilter(freq, min_cutoff, beta, d_cutoff)
                    for _ in range(num_joints)]

    def reset(self) -> None:
        for f in self._fx:
            f.reset()
        for f in self._fy:
            f.reset()

    def __call__(self, kp: np.ndarray) -> np.ndarray:
        """kp: (num_joints, 2) → trả về (num_joints, 2) đã làm mượt."""
        out = kp.copy().astype(np.float32)
        for j in range(self.num_joints):
            x, y = float(kp[j, 0]), float(kp[j, 1])
            # Bỏ qua khớp = 0 (không hợp lệ) để không nhiễm bộ lọc
            if x == 0.0 and y == 0.0:
                continue
            out[j, 0] = self._fx[j](x)
            out[j, 1] = self._fy[j](y)
        return out
