"""
ST-GCN / MS-G3D model definitions cho skeleton-based action recognition.

Kiến trúc:
    - STGCN: 3-block Spatial-Temporal Graph Convolutional Network (V2.0)
    - MSG3D: 4-block Multi-Scale Graph 3D variant (V2.4)

Tham chiếu:
    - Yan et al., "Spatial Temporal Graph Convolutional Networks
      for Skeleton-Based Action Recognition", AAAI 2018
    - Chen et al., "Channel-wise Topology Refinement Graph Convolution
      for Skeleton-Based Action Recognition", ICCV 2021
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# =====================================================================
# COCO-17 graph definition
# =====================================================================
COCO_17_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 0), (6, 0),
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
V_JOINTS = 17
CENTER_JOINT = 0
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


# =====================================================================
# Graph utilities
# =====================================================================
def _bfs_dist(adj, root):
    V = adj.shape[0]
    dist = -np.ones(V, dtype=np.int32)
    dist[root] = 0
    fr = [root]
    while fr:
        nf = []
        for u in fr:
            for v in range(V):
                if adj[u, v] > 0 and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    nf.append(v)
        fr = nf
    if (dist == -1).any():
        dist[dist == -1] = dist.max() + 1
    return dist


def _norm(A):
    D = A.sum(0)
    Di = np.zeros_like(D, dtype=np.float32)
    Di[D > 0] = D[D > 0] ** -0.5
    return np.diag(Di) @ A @ np.diag(Di)


def build_adjacency():
    """3 partition adjacency: Identity + Centripetal + Centrifugal."""
    A = np.zeros((V_JOINTS, V_JOINTS), dtype=np.float32)
    for i, j in COCO_17_EDGES:
        A[i, j] = 1; A[j, i] = 1
    I = np.eye(V_JOINTS, dtype=np.float32)
    dist = _bfs_dist(A, CENTER_JOINT)
    cp = np.zeros_like(A); cf = np.zeros_like(A)
    for i, j in COCO_17_EDGES:
        if dist[i] == dist[j]:
            cp[i, j] = 1; cp[j, i] = 1
        elif dist[i] > dist[j]:
            cp[i, j] = 1; cf[j, i] = 1
        else:
            cf[i, j] = 1; cp[j, i] = 1
    return np.stack([_norm(I), _norm(cp), _norm(cf)], axis=0)


def build_adjacency_msg3d():
    """4 partition: 3 (1-hop ST-GCN) + 1 (2-hop disentangled) — cho MS-G3D V2.4."""
    base = build_adjacency()                       # (3, V, V)
    A = np.zeros((V_JOINTS, V_JOINTS), dtype=np.float32)
    for i, j in COCO_17_EDGES:
        A[i, j] = 1; A[j, i] = 1
    I = np.eye(V_JOINTS, dtype=np.float32)
    A2 = ((A @ A > 0).astype(np.float32)) - (A + I)
    A2[A2 < 0] = 0
    A2 -= np.diag(np.diag(A2))
    return np.concatenate([base, _norm(A2)[None]], axis=0)   # (4, V, V)


# =====================================================================
# PyTorch model components
# =====================================================================
if _HAS_TORCH:
    class _CTRGC(nn.Module):
        def __init__(self, ic, oc, A):
            super().__init__()
            self.K = A.shape[0]
            self.register_buffer("A", torch.from_numpy(A).float())
            self.conv = nn.Conv2d(ic, oc * self.K, 1)
            mid = max(oc // 8, 8)
            self.th = nn.Conv2d(ic, mid, 1)
            self.ph = nn.Conv2d(ic, mid, 1)
            self.al = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            N, _, T, V = x.shape
            y = self.conv(x); oc = y.shape[1] // self.K
            y = y.view(N, self.K, oc, T, V)
            o = torch.einsum("nkctv,kvw->nkctw", y, self.A)
            th = self.th(x).mean(2); ph = self.ph(x).mean(2)
            off = torch.tanh(torch.einsum("ncv,ncw->nvw", th, ph))
            o = o + self.al * torch.einsum("nkctv,nvw->nkctw", y, off)
            return o.sum(1)


    class _TC(nn.Module):
        def __init__(self, ic, oc, k=9, st=1):
            super().__init__()
            self.c = nn.Conv2d(ic, oc, (k, 1), padding=((k - 1) // 2, 0), stride=(st, 1))
            self.b = nn.BatchNorm2d(oc)

        def forward(self, x): return self.b(self.c(x))


    class _Blk(nn.Module):
        def __init__(self, ic, oc, A, st=1, dp=0.2):
            super().__init__()
            self.sp = _CTRGC(ic, oc, A)
            self.bn = nn.BatchNorm2d(oc)
            self.tc = _TC(oc, oc, 9, st)
            self.r = nn.ReLU(inplace=True)
            self.dp = nn.Dropout2d(dp)
            self.res = (nn.Identity() if (ic == oc and st == 1)
                        else nn.Sequential(nn.Conv2d(ic, oc, 1, stride=(st, 1)),
                                           nn.BatchNorm2d(oc)))

        def forward(self, x):
            res = self.res(x)
            y = self.r(self.bn(self.sp(x)))
            y = self.dp(self.tc(y))
            return self.r(y + res)


    class STGCN(nn.Module):
        """ST-GCN 3-block (V2.0)."""
        def __init__(self, in_ch=3, n_classes=2, channels=(64, 128, 256), dropout=0.5):
            super().__init__()
            A = build_adjacency()
            c1, c2, c3 = channels
            self.dbn = nn.BatchNorm1d(in_ch * V_JOINTS)
            self.b1 = _Blk(in_ch, c1, A, 1, 0.2)
            self.b2 = _Blk(c1, c2, A, 2, 0.2)
            self.b3 = _Blk(c2, c3, A, 2, 0.2)
            self.dp = nn.Dropout(dropout)
            self.fc = nn.Linear(c3, n_classes)

        def forward(self, x):
            N, C, T, V = x.shape
            xb = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
            xb = self.dbn(xb)
            x = xb.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()
            x = self.b1(x); x = self.b2(x); x = self.b3(x)
            x = x.mean(dim=(2, 3))
            return self.fc(self.dp(x))


    class MSG3D(nn.Module):
        """MS-G3D 4-block multi-scale (V2.4)."""
        def __init__(self, in_ch=3, n_classes=2, channels=(64, 128, 256, 256), dropout=0.5):
            super().__init__()
            A = build_adjacency_msg3d()
            c1, c2, c3, c4 = channels
            self.dbn = nn.BatchNorm1d(in_ch * V_JOINTS)
            self.b1 = _Blk(in_ch, c1, A, 1, 0.2)
            self.b2 = _Blk(c1, c2, A, 2, 0.2)
            self.b3 = _Blk(c2, c3, A, 2, 0.3)
            self.b4 = _Blk(c3, c4, A, 1, 0.3)
            self.dp = nn.Dropout(dropout)
            self.fc = nn.Linear(c4, n_classes)

        def forward(self, x):
            N, C, T, V = x.shape
            xb = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
            xb = self.dbn(xb)
            x = xb.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()
            x = self.b1(x); x = self.b2(x); x = self.b3(x); x = self.b4(x)
            x = x.mean(dim=(2, 3))
            return self.fc(self.dp(x))


    def load_skeleton_model(path, device="cpu"):
        """Load checkpoint, auto-detect STGCN (3 block) hay MSG3D (4 block)."""
        sd = torch.load(path, map_location=device)
        is_msg3d = any(k.startswith("b4.") for k in sd.keys())
        model = (MSG3D() if is_msg3d else STGCN()).to(device)
        model.load_state_dict(sd)
        model.eval()
        return model, ("MSG3D (V2.4)" if is_msg3d else "STGCN (V2.0)")
