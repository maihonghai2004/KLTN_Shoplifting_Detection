"""
Bước 3 — Trích xuất 56 đặc trưng hình học từ CSV 34 cột.

Input:
    data/raw_keypoints/{normal, shoplifting}/*.csv  (34 cột toạ độ thô)

Output:
    data/clean_csv_56feat/{normal, shoplifting}/*.csv  (56 đặc trưng)

Logic:
    - 34 chiều: relative_kp với hip_midpoint
    - 20 chiều: bone vectors của 10 cặp khớp
    - 2 chiều: distance(wrist, hip) trái + phải

Đồng thời lọc file có frame mất pose (giống bước 2).

Tác giả gốc: Hảo (filter_data(1).py).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_preprocessing import extract_geometric_features


def filter_and_extract(source_root: str, dest_root: str):
    n_total = 0; n_processed = 0; n_skipped = 0

    for label in os.listdir(source_root):
        src = os.path.join(source_root, label)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(dest_root, label)
        os.makedirs(dst, exist_ok=True)
        print(f"\n--- {label} ---")

        for fname in sorted(os.listdir(src)):
            if not fname.endswith('.csv'):
                continue
            n_total += 1
            fp = os.path.join(src, fname)
            df = pd.read_csv(fp)
            if (df == 0).all(axis=1).any():
                n_skipped += 1
                continue

            # Trích 56 đặc trưng cho từng frame
            raw_data = df.values  # (T, 34)
            enhanced = np.stack([extract_geometric_features(row) for row in raw_data])

            out_df = pd.DataFrame(enhanced)
            out_df.to_csv(os.path.join(dst, fname), index=False)
            n_processed += 1
            if n_processed % 20 == 0:
                print(f"  ✓ {n_processed} files processed...")

    print(f"\n[Done] Total: {n_total}, processed: {n_processed}, skipped: {n_skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default=str(ROOT / "data" / "raw_keypoints"))
    parser.add_argument("--output", type=str,
                        default=str(ROOT / "data" / "clean_csv_56feat"))
    args = parser.parse_args()
    filter_and_extract(args.input, args.output)


if __name__ == "__main__":
    main()
