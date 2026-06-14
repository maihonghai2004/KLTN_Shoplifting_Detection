"""Tests cho extract_geometric_features (56 đặc trưng hình học)."""
from __future__ import annotations

import numpy as np
import pytest

from src.data_preprocessing.geometric_features import extract_geometric_features, BONE_PAIRS


class TestExtractGeometricFeatures:
    """Kiểm tra tính đúng đắn của 56 geometric features."""

    def test_output_shape_from_17x2(self, sample_keypoints_17x2):
        """Input (17, 2) → output (56,)."""
        feats = extract_geometric_features(sample_keypoints_17x2)
        assert feats.shape == (56,)
        assert feats.dtype == np.float32

    def test_output_shape_from_flat_34(self, sample_keypoints_17x2):
        """Input (34,) (flattened) → output (56,)."""
        flat = sample_keypoints_17x2.flatten()
        feats = extract_geometric_features(flat)
        assert feats.shape == (56,)

    def test_relative_coords_centered_at_hip(self, sample_keypoints_17x2):
        """34 chiều đầu = keypoints relative tới hip midpoint.
        Hip midpoint relative phải = 0."""
        feats = extract_geometric_features(sample_keypoints_17x2)
        # hip_center = (kp[11] + kp[12]) / 2
        hip_center = (sample_keypoints_17x2[11] + sample_keypoints_17x2[12]) / 2.0
        # relative hip_left (index 11) → feats[22:24], hip_right (index 12) → feats[24:26]
        rel_hip_left = feats[22:24]
        rel_hip_right = feats[24:26]
        # Trung bình 2 hip relative phải ≈ 0
        np.testing.assert_allclose((rel_hip_left + rel_hip_right) / 2, [0, 0], atol=1e-6)

    def test_hand_hip_distances_non_negative(self, sample_keypoints_17x2):
        """2 chiều cuối (hand-hip distances) phải >= 0."""
        feats = extract_geometric_features(sample_keypoints_17x2)
        assert feats[54] >= 0  # dist(left_wrist, left_hip)
        assert feats[55] >= 0  # dist(right_wrist, right_hip)

    def test_all_same_keypoints_gives_zero_relative(self):
        """Nếu tất cả keypoints cùng vị trí → relative coords = 0, bone vectors = 0."""
        kp_same = np.full((17, 2), 0.5, dtype=np.float32)
        feats = extract_geometric_features(kp_same)
        # 34 relative coords = 0 (vì mọi điểm = hip_center)
        np.testing.assert_allclose(feats[:34], 0, atol=1e-7)
        # 20 bone vectors = 0 (vì mọi điểm giống nhau)
        np.testing.assert_allclose(feats[34:54], 0, atol=1e-7)
        # 2 hand-hip distances = 0
        np.testing.assert_allclose(feats[54:56], 0, atol=1e-7)

    def test_bone_pairs_count(self):
        """Phải có đúng 10 bone pairs → 20 chiều bone vectors."""
        assert len(BONE_PAIRS) == 10

    def test_concealing_has_small_hand_hip_dist(self, concealing_keypoints_17x2):
        """Tư thế concealing: wrist gần hip → hand-hip distance nhỏ."""
        feats = extract_geometric_features(concealing_keypoints_17x2)
        # Khoảng cách wrist-hip khi concealing phải < 0.05
        assert feats[54] < 0.05
        assert feats[55] < 0.05

    def test_normal_has_larger_hand_hip_dist(self, sample_keypoints_17x2):
        """Tư thế bình thường: wrist xa hip → hand-hip distance lớn hơn."""
        feats = extract_geometric_features(sample_keypoints_17x2)
        # Khoảng cách wrist-hip khi bình thường phải > 0.05
        assert feats[54] > 0.05
        assert feats[55] > 0.05
