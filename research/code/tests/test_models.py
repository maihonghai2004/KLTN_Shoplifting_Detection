"""Tests cho model architectures (ST-GCN, MSG3D) và graph utilities."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.stgcn import (
    build_adjacency, build_adjacency_msg3d,
    V_JOINTS, COCO_17_EDGES,
)

# PyTorch là optional — skip tests nếu không có
torch = pytest.importorskip("torch")
from src.models.stgcn import STGCN, MSG3D


class TestGraphAdjacency:
    """Kiểm tra ma trận kề COCO-17."""

    def test_adjacency_shape(self):
        """build_adjacency() → (3, 17, 17): 3 partitions."""
        A = build_adjacency()
        assert A.shape == (3, V_JOINTS, V_JOINTS)

    def test_identity_partition_symmetric(self):
        """Partition 0 (Identity) phải đối xứng."""
        A = build_adjacency()
        np.testing.assert_allclose(A[0], A[0].T, atol=1e-6)

    def test_adjacency_msg3d_shape(self):
        """build_adjacency_msg3d() → (4, 17, 17): 4 partitions."""
        A = build_adjacency_msg3d()
        assert A.shape == (4, V_JOINTS, V_JOINTS)

    def test_adjacency_msg3d_identity_symmetric(self):
        """Partition 0 (Identity) trong MSG3D phải đối xứng."""
        A = build_adjacency_msg3d()
        np.testing.assert_allclose(A[0], A[0].T, atol=1e-6)

    def test_edges_count(self):
        """COCO-17 có 18 cạnh."""
        assert len(COCO_17_EDGES) == 18


class TestSTGCN:
    """Kiểm tra ST-GCN forward pass."""

    def test_forward_shape(self):
        """Input (1, 3, 90, 17) → output (1, 2)."""
        model = STGCN(in_ch=3, n_classes=2)
        model.eval()
        x = torch.randn(1, 3, 90, 17)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)

    def test_forward_batch(self):
        """Batch size > 1: (4, 3, 90, 17) → (4, 2)."""
        model = STGCN(in_ch=3, n_classes=2)
        model.eval()
        x = torch.randn(4, 3, 90, 17)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 2)

    def test_output_is_logits(self):
        """Output là logits (chưa softmax), không bị clip [0,1]."""
        model = STGCN()
        model.eval()
        x = torch.randn(1, 3, 90, 17)
        with torch.no_grad():
            out = model(x)
        # Logits có thể âm hoặc > 1
        assert out.requires_grad is False


class TestMSG3D:
    """Kiểm tra MS-G3D forward pass."""

    def test_forward_shape(self):
        """Input (1, 3, 90, 17) → output (1, 2)."""
        model = MSG3D(in_ch=3, n_classes=2)
        model.eval()
        x = torch.randn(1, 3, 90, 17)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)

    def test_forward_batch(self):
        """Batch size > 1: (2, 3, 90, 17) → (2, 2)."""
        model = MSG3D(in_ch=3, n_classes=2)
        model.eval()
        x = torch.randn(2, 3, 90, 17)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2)
