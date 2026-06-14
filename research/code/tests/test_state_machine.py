"""Tests cho StateMachine (FSM 4 trạng thái)."""
from __future__ import annotations

import numpy as np
import pytest

from src.state_machine import StateMachine, State


class TestStateMachine:
    """Kiểm tra FSM transitions và rule scores."""

    def test_initial_state_is_browsing(self):
        fsm = StateMachine()
        assert fsm.state == State.BROWSING

    def test_none_keypoints_returns_browsing(self):
        """Không có skeleton → reset về BROWSING, p_rule=0."""
        fsm = StateMachine()
        result = fsm.update(None)
        assert result['state'] == State.BROWSING
        assert result['p_rule'] == 0.0
        assert result['wrist_dist'] is None
        assert result['wrist_vel'] is None

    def test_browsing_rule_score_is_zero(self, sample_keypoints_17x2):
        """Tư thế bình thường → BROWSING, p_rule=0."""
        fsm = StateMachine()
        result = fsm.update(sample_keypoints_17x2)
        assert result['p_rule'] == 0.0

    def test_transition_to_concealing(self, concealing_keypoints_17x2):
        """Wrists gần hips liên tục >= min_frames → chuyển sang CONCEALING."""
        fsm = StateMachine(dist_threshold=0.06, min_frames_for_transition=6)
        for _ in range(20):
            result = fsm.update(concealing_keypoints_17x2)
        assert result['state'] == State.CONCEALING
        assert result['p_rule'] == 0.85

    def test_concealing_to_exiting(self, concealing_keypoints_17x2, sample_keypoints_17x2):
        """Sau CONCEALING, wrists xa hips liên tục → EXITING."""
        fsm = StateMachine(dist_threshold=0.06, min_frames_for_transition=6)
        # Đưa vào CONCEALING
        for _ in range(20):
            fsm.update(concealing_keypoints_17x2)
        assert fsm.state == State.CONCEALING
        # Wrists xa hips → EXITING
        for _ in range(20):
            result = fsm.update(sample_keypoints_17x2)
        assert result['state'] == State.EXITING
        assert result['p_rule'] == 1.0

    def test_reset_returns_to_browsing(self, concealing_keypoints_17x2):
        """reset() phải đưa FSM về trạng thái ban đầu."""
        fsm = StateMachine(min_frames_for_transition=6)
        for _ in range(20):
            fsm.update(concealing_keypoints_17x2)
        assert fsm.state != State.BROWSING
        fsm.reset()
        assert fsm.state == State.BROWSING
        assert fsm.state_counter == 0

    def test_wrist_dist_is_computed(self, sample_keypoints_17x2):
        """wrist_dist phải được tính và > 0."""
        fsm = StateMachine()
        result = fsm.update(sample_keypoints_17x2)
        assert result['wrist_dist'] is not None
        assert result['wrist_dist'] > 0
