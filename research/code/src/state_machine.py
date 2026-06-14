"""
Rule-based State Machine cho hành vi cất giấu hàng hóa.

4 trạng thái:
    BROWSING → PICKING → CONCEALING → EXITING

Vì hiện chưa có bbox bag/pocket riêng, ta dùng heuristic dựa skeleton:
    - Cổ tay gần hông (dist < threshold): có thể đang đưa hàng vào túi quần
    - Tốc độ cổ tay giảm đột ngột: có thể đang dừng để giấu
    - Đếm số frame liên tiếp trong mỗi state để chuyển trạng thái
"""
from __future__ import annotations

from collections import deque
from enum import Enum

import numpy as np


class State(Enum):
    BROWSING = 0
    PICKING = 1
    CONCEALING = 2
    EXITING = 3


class StateMachine:
    """Heuristic state machine — không thay thế ML model, chỉ bổ trợ giải thích."""

    def __init__(self,
                 dist_threshold: float = 0.06,        # cổ tay gần hông (chuẩn hoá [0,1])
                 velocity_threshold: float = 0.01,     # vận tốc thấp = dừng
                 min_frames_for_transition: int = 6):
        self.dt = dist_threshold
        self.vt = velocity_threshold
        self.min_f = min_frames_for_transition
        self.state = State.BROWSING
        self.state_counter = 0
        self.wrist_history = deque(maxlen=10)  # để tính velocity

    def _wrist_to_hip_dist(self, kp):
        """Tính khoảng cách trung bình 2 cổ tay đến 2 hông."""
        wL, wR = kp[9], kp[10]
        hL, hR = kp[11], kp[12]
        d1 = np.linalg.norm(wL - hL)
        d2 = np.linalg.norm(wR - hR)
        return (d1 + d2) / 2.0

    def _wrist_velocity(self):
        """Vận tốc trung bình của cổ tay trong vài frame gần đây."""
        if len(self.wrist_history) < 2:
            return 1.0
        diffs = []
        for i in range(1, len(self.wrist_history)):
            diffs.append(np.linalg.norm(self.wrist_history[i] - self.wrist_history[i-1]))
        return float(np.mean(diffs))

    def update(self, keypoints) -> dict:
        """
        Cập nhật state machine với 1 frame keypoints.

        Returns:
            {
                'state': State, 'p_rule': float [0,1],
                'wrist_dist': float, 'wrist_vel': float,
            }
        """
        if keypoints is None:
            # Không có skeleton → reset về BROWSING
            self.state = State.BROWSING
            self.state_counter = 0
            return {'state': self.state, 'p_rule': 0.0,
                    'wrist_dist': None, 'wrist_vel': None}

        # Track cổ tay trung bình
        wrist_mid = (keypoints[9] + keypoints[10]) / 2.0
        self.wrist_history.append(wrist_mid)

        dist = self._wrist_to_hip_dist(keypoints)
        vel = self._wrist_velocity()

        # Heuristic transition
        prev_state = self.state
        if dist < self.dt:
            # Cổ tay gần hông → có thể đang vào trạng thái CONCEALING
            if self.state in (State.BROWSING, State.PICKING):
                self.state_counter += 1
                if self.state_counter >= self.min_f:
                    self.state = State.CONCEALING
                    self.state_counter = 0
            elif self.state == State.CONCEALING:
                pass  # giữ
        else:
            # Cổ tay xa hông
            if self.state == State.CONCEALING:
                self.state_counter += 1
                if self.state_counter >= self.min_f:
                    self.state = State.EXITING
                    self.state_counter = 0
            elif self.state == State.EXITING:
                # Sau khi EXITING một thời gian, quay về BROWSING
                self.state_counter += 1
                if self.state_counter >= 30:
                    self.state = State.BROWSING
                    self.state_counter = 0
            elif vel > self.vt:
                # Tay đang chuyển động → PICKING
                if self.state == State.BROWSING:
                    self.state_counter += 1
                    if self.state_counter >= self.min_f:
                        self.state = State.PICKING
                        self.state_counter = 0

        if self.state != prev_state:
            self.state_counter = 0

        # Convert state thành rule score
        rule_score = {
            State.BROWSING: 0.0,
            State.PICKING: 0.3,
            State.CONCEALING: 0.85,
            State.EXITING: 1.0,
        }[self.state]

        return {
            'state': self.state,
            'p_rule': rule_score,
            'wrist_dist': dist,
            'wrist_vel': vel,
        }

    def reset(self):
        self.state = State.BROWSING
        self.state_counter = 0
        self.wrist_history.clear()
