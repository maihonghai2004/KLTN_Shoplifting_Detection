"""
Bước 5 — Train lại model baseline LSTM-CNN Hybrid của Hảo trên máy local.

Đây là phiên bản script (không phải notebook) của KLTN.ipynb gốc.

Input:
    data/processed_npy/X_data_56feat.npy   (hoặc 34feat)
    data/processed_npy/y_data.npy

Output:
    models/checkpoints/hao_baseline_lstm_cnn_<feat>.h5
    logs/training_hao_baseline_<timestamp>.csv

Cách chạy:
    # Train 56 feat (giống model gốc Hảo)
    python scripts/05_train_baseline_hao.py --features 56

    # Train 34 feat
    python scripts/05_train_baseline_hao.py --features 34
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.behavior import build_hao_lstm_cnn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=int, choices=[34, 56], default=56)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--split", type=str, choices=["random", "by_clip"], default="by_clip",
                        help="random = giống Hảo gốc (có leakage), by_clip = đúng (chống leakage)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--npy_dir", type=str,
                        default=str(ROOT / "data" / "processed_npy"))
    parser.add_argument("--out_dir", type=str,
                        default=str(ROOT / "models" / "checkpoints"))
    parser.add_argument("--log_dir", type=str,
                        default=str(ROOT / "logs"))
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Load data
    suffix = f"{args.features}feat"
    x_path = os.path.join(args.npy_dir, f"X_data_{suffix}.npy")
    y_path = os.path.join(args.npy_dir, "y_data.npy")
    if not os.path.exists(x_path):
        print(f"❌ Không thấy {x_path}. Chạy `python scripts/04_prepare_sliding_window.py --features {args.features}` trước.")
        sys.exit(1)

    X = np.load(x_path)
    y = np.load(y_path)
    print(f"X: {X.shape}, y: {y.shape}")

    # Split
    if args.split == "random":
        print("[Split] RANDOM (có nguy cơ data leakage do sliding window)")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                    random_state=args.seed, stratify=y)
    else:
        clip_path = os.path.join(args.npy_dir, "clip_ids.npy")
        if not os.path.exists(clip_path):
            print(f"❌ Cần {clip_path} để split by clip.")
            sys.exit(1)
        clip_ids = np.load(clip_path)
        print(f"[Split] BY CLIP (GroupShuffleSplit), clips unique = {len(np.unique(clip_ids))}")
        gss = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=args.seed)
        idx_tr, idx_te = next(gss.split(X, y, groups=clip_ids))
        X_tr, X_te = X[idx_tr], X[idx_te]
        y_tr, y_te = y[idx_tr], y[idx_te]
        # Verify
        assert not (set(clip_ids[idx_tr]) & set(clip_ids[idx_te]))
        print(f"  ✓ Không có clip chung giữa train/test")
    print(f"Train: {X_tr.shape[0]}, Test: {X_te.shape[0]}")

    # Build model
    model = build_hao_lstm_cnn(input_shape=(X.shape[1], X.shape[2]),
                                num_classes=2, learning_rate=args.lr)
    model.summary()

    ts = time.strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(args.out_dir, f"hao_baseline_{suffix}_{args.split}_{ts}.h5")
    csv_path = os.path.join(args.log_dir, f"training_hao_baseline_{suffix}_{args.split}_{ts}.csv")

    callbacks = [
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        CSVLogger(csv_path),
    ]

    print(f"\n[Training] checkpoint → {ckpt_path}")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        verbose=2,
    )

    # Final eval
    y_pred = model.predict(X_te, verbose=0).argmax(1)
    print("\n[Test classification report]")
    print(classification_report(y_te, y_pred, target_names=['Normal', 'Shoplifting'], digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_te, y_pred))
    print(f"F1-macro: {f1_score(y_te, y_pred, average='macro'):.4f}")
    print(f"\n✓ Model saved → {ckpt_path}")
    print(f"✓ Log saved   → {csv_path}")


if __name__ == "__main__":
    main()
