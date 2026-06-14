"""
Bước 2 — Lọc các file CSV "sạch" (không có frame nào toàn 0).

Input:
    data/raw_keypoints/{normal, shoplifting}/*.csv

Output:
    data/clean_csv_34feat/{normal, shoplifting}/*.csv  (chỉ chứa file sạch, 34 cột)

Tác giả gốc: Hảo (filter_data.py).
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def filter_clean_csv(source_root: str, dest_root: str):
    n_total = 0; n_copied = 0; n_skipped = 0

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
            # Frame "bị mất pose" = toàn cột = 0
            has_zero_frame = (df == 0).all(axis=1).any()
            if has_zero_frame:
                n_skipped += 1
                print(f"  ✗ Bỏ qua: {fname} (có frame mất pose)")
            else:
                shutil.copy2(fp, os.path.join(dst, fname))
                n_copied += 1

    print(f"\n[Done] Total: {n_total}, sạch: {n_copied}, bỏ qua: {n_skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default=str(ROOT / "data" / "raw_keypoints"))
    parser.add_argument("--output", type=str,
                        default=str(ROOT / "data" / "clean_csv_34feat"))
    args = parser.parse_args()
    filter_clean_csv(args.input, args.output)


if __name__ == "__main__":
    main()
