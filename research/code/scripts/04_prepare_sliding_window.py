"""
Bước 4 — Tạo dataset bằng sliding window từ CSV.

Hỗ trợ cả 34 feat (raw) và 56 feat (geometric).

Input:
    data/clean_csv_56feat/{normal, shoplifting}/*.csv  (mặc định)
    Hoặc data/clean_csv_34feat/...

Output:
    data/processed_npy/X_data_56feat.npy   (N, 90, 56)
    data/processed_npy/y_data.npy           (N,)
    data/processed_npy/clip_ids.npy         (N,)  — id video gốc của mỗi sample
                                                    ★ QUAN TRỌNG: chống data leakage

Tác giả gốc: Hảo (prepare_data.py / prepare_data(1).py), bổ sung clip_ids để fix leakage.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def prepare_sliding_window(
    csv_root: str,
    output_dir: str,
    seq_length: int = 90,
    step: int = 30,
    n_features_expected: int = 56,
    suffix: str = "56feat",
):
    X, y, clip_ids = [], [], []

    for label_name in ('normal', 'shoplifting'):
        label_val = 0 if label_name == 'normal' else 1
        folder = os.path.join(csv_root, label_name)
        if not os.path.exists(folder):
            print(f"  ⚠ Bỏ qua {folder}")
            continue

        files = sorted(glob.glob(os.path.join(folder, "*.csv")))
        print(f"--- {label_name}: {len(files)} files ---")
        for fp in files:
            data = pd.read_csv(fp).values
            if data.shape[1] != n_features_expected:
                print(f"  ⚠ Skip {os.path.basename(fp)}: shape {data.shape}")
                continue
            cid = f"{label_name}/{os.path.basename(fp)}"
            for i in range(0, len(data) - seq_length + 1, step):
                X.append(data[i: i + seq_length])
                y.append(label_val)
                clip_ids.append(cid)

    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int64')
    clip_ids = np.array(clip_ids)

    os.makedirs(output_dir, exist_ok=True)
    x_path = os.path.join(output_dir, f"X_data_{suffix}.npy")
    y_path = os.path.join(output_dir, "y_data.npy")
    c_path = os.path.join(output_dir, "clip_ids.npy")
    np.save(x_path, X)
    np.save(y_path, y)
    np.save(c_path, clip_ids)

    print(f"\n[Done]")
    print(f"  X: {X.shape}  → {x_path}")
    print(f"  y: {y.shape}  → {y_path}")
    print(f"  clip_ids: {clip_ids.shape}, unique={len(np.unique(clip_ids))}  → {c_path}")

    u, c = np.unique(y, return_counts=True)
    for ui, ci in zip(u, c):
        print(f"    class {int(ui)}: {ci} ({100*ci/len(y):.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default=str(ROOT / "data" / "clean_csv_56feat"),
                        help="Folder chứa CSV đã làm sạch")
    parser.add_argument("--output", type=str,
                        default=str(ROOT / "data" / "processed_npy"))
    parser.add_argument("--seq", type=int, default=90)
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument("--features", type=int, default=56,
                        help="Số feature expect trong CSV (34 hoặc 56)")
    parser.add_argument("--suffix", type=str, default="56feat",
                        help="Hậu tố file X_data (e.g., '56feat' hoặc '34feat')")
    args = parser.parse_args()

    prepare_sliding_window(
        csv_root=args.input,
        output_dir=args.output,
        seq_length=args.seq,
        step=args.step,
        n_features_expected=args.features,
        suffix=args.suffix,
    )


if __name__ == "__main__":
    main()
