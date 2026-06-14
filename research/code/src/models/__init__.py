"""Model definitions cho skeleton-based action recognition."""
from .stgcn import (
    COCO_17_EDGES, V_JOINTS, CENTER_JOINT, FLIP_IDX,
    build_adjacency, build_adjacency_msg3d,
)

try:
    from .stgcn import STGCN, MSG3D, load_skeleton_model
except ImportError:
    pass  # PyTorch not installed

__all__ = [
    "COCO_17_EDGES", "V_JOINTS", "CENTER_JOINT", "FLIP_IDX",
    "build_adjacency", "build_adjacency_msg3d",
    "STGCN", "MSG3D", "load_skeleton_model",
]
