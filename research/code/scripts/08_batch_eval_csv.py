"""
Batch evaluation OFFLINE - chay model tren NHIEU clip CSV va tinh do chinh xac.

Muc dich: kiem tra model du doan dung/sai tren tung clip (sanity-check), dung
keypoint tu CSV da extract (KHONG chay YOLO realtime -> tranh distribution mismatch
giua version YOLO, cho score chinh xac khop luc train/eval).

CANH BAO HOC THUAT:
    Mac dinh chay tren TAT CA clip, GOM CA clip da train -> do chinh xac BI "AO"
    (cao gia) do data leakage. Con so nay CHI de sanity-check, KHONG dua vao bao cao.
    So lieu hop le = split-by-clip multi-seed (notebook V1.4). Dung --split test
    de chi danh gia clip held-out.

Cach chay:
    python scripts/08_batch_eval_csv.py
    python scripts/08_batch_eval_csv.py --threshold 0.5 --agg mean
    python scripts/08_batch_eval_csv.py --split test --seed 42
    python scripts/08_batch_eval_csv.py --out outputs/eval_v14_allclips.csv
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

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

from src.behavior.keras_v14 import (_build_v14_bilstm, _focal_loss_dummy,
                                    _load_v14_from_keras3_zip)

try:
    from tensorflow.keras.models import load_model
    _HAS_TF = True
except ImportError:
    _HAS_TF = False


def load_v14(path):
    """Load V1.4 .keras bang loader 3 tang GIONG HET RealtimePipeline.

    v1.4_best.keras luu bang Keras 3, env local la Keras 2 (TF 2.15) -> load_model
    truc tiep / load_weights naive se fail. Tang 2 (zip loader) doc model.weights.h5
    trong file .keras roi gan tay vao model Keras 2.
    """
    if not _HAS_TF:
        sys.exit("Can TensorFlow de load V1.4 (.keras). pip install tensorflow")
    if not os.path.exists(path):
        sys.exit("Khong thay model: " + str(path))
    try:
        return load_model(path, custom_objects={'f': _focal_loss_dummy(),
                                                 'loss_fn': _focal_loss_dummy()},
                          compile=False)
    except Exception as e:
        print("[V1.4] direct load fail (" + str(e) + "); thu Keras 3 zip loader")
        try:
            return _load_v14_from_keras3_zip(path)
        except Exception as e2:
            print("[V1.4] zip loader fail (" + str(e2) + "); fallback load_weights")
            m = _build_v14_bilstm(input_shape=(90, 56))
            m.load_weights(path)
            return m


def list_clips(csv_dir: Path):
    clips = []
    for lbl_name, lbl in [("normal", 0), ("shoplifting", 1)]:
        d = csv_dir / lbl_name
        if not d.exists():
            continue
        for fp in sorted(d.glob("*.csv")):
            clips.append((lbl_name + "/" + fp.name, lbl, fp))
    return clips


def make_windows(arr, seq, step):
    T = arr.shape[0]
    if T < seq:
        pad = np.repeat(arr[-1:], seq - T, axis=0)
        return np.concatenate([arr, pad], axis=0)[None]
    wins = [arr[i:i + seq] for i in range(0, T - seq + 1, step)]
    return np.asarray(wins, dtype=np.float32)


def stratified_test_clips(clips, seed):
    from sklearn.model_selection import StratifiedKFold
    y = np.array([c[1] for c in clips])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(len(y)), y))
    _, test_idx = folds[seed % 5]
    return [clips[i] for i in test_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "v1.4_best.keras"))
    ap.add_argument("--csv_dir", default=str(ROOT / "data" / "clean_csv_56feat"))
    ap.add_argument("--seq", type=int, default=90)
    ap.add_argument("--step", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--agg", choices=["mean", "max"], default="mean")
    ap.add_argument("--split", choices=["all", "test"], default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    clips = list_clips(csv_dir)
    if not clips:
        sys.exit("Khong thay CSV nao trong " + str(csv_dir) + "/(normal|shoplifting)")

    if args.split == "test":
        clips = stratified_test_clips(clips, args.seed)
        print("[Split] test split (seed=" + str(args.seed) + "): " + str(len(clips)) + " clip")
    else:
        print("=" * 72)
        print("CANH BAO --split all: GOM CA CLIP DA TRAIN -> do chinh xac bi 'ao' (leakage).")
        print("    Chi dung de sanity-check. KHONG dua so nay vao bao cao/bao ve.")
        print("=" * 72)

    print("[Model] " + str(args.model))
    model = load_v14(args.model)
    print("[Eval ] " + str(len(clips)) + " clip | seq=" + str(args.seq)
          + " step=" + str(args.step) + " agg=" + args.agg
          + " threshold=" + str(args.threshold) + "\n")

    rows = []
    for clip_id, y_true, fp in clips:
        arr = pd.read_csv(fp).values.astype(np.float32)
        if arr.shape[1] != 56:
            print("  [SKIP] " + clip_id + ": " + str(arr.shape[1]) + " cot (can 56)")
            continue
        wins = make_windows(arr, args.seq, args.step)
        probs = model.predict(wins, verbose=0)[:, 1]
        score = float(probs.mean() if args.agg == "mean" else probs.max())
        y_pred = int(score >= args.threshold)
        rows.append({"clip": clip_id, "true": y_true, "pred": y_pred,
                     "score": round(score, 4),
                     "score_mean": round(float(probs.mean()), 4),
                     "score_max": round(float(probs.max()), 4),
                     "n_windows": len(wins),
                     "correct": int(y_pred == y_true)})
        flag = "OK " if y_pred == y_true else "ERR"
        print("  [" + flag + "] " + clip_id.ljust(32) + " true=" + str(y_true)
              + " pred=" + str(y_pred) + " score=" + ("%.3f" % score)
              + " (" + str(len(wins)) + "w)")

    df = pd.DataFrame(rows)
    yt = df["true"].values
    yp = df["pred"].values

    print("\n" + "=" * 72)
    print("KET QUA - V1.4 LSTM-CNN | " + str(len(df)) + " clip | split="
          + args.split + " | agg=" + args.agg)
    print("=" * 72)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    print("Confusion matrix [hang=thuc, cot=du doan], lop [normal, shoplifting]:")
    print("              pred_normal  pred_shoplift")
    print("  normal   :  " + str(cm[0, 0]).rjust(10) + "  " + str(cm[0, 1]).rjust(12))
    print("  shoplift :  " + str(cm[1, 0]).rjust(10) + "  " + str(cm[1, 1]).rjust(12))
    print("\n  Accuracy        : " + ("%.4f" % accuracy_score(yt, yp)))
    print("  Precision(macro): " + ("%.4f" % precision_score(yt, yp, average="macro", zero_division=0)))
    print("  Recall   (macro): " + ("%.4f" % recall_score(yt, yp, average="macro", zero_division=0)))
    print("  F1-macro        : " + ("%.4f" % f1_score(yt, yp, average="macro", zero_division=0)))
    print("  Sai             : " + str(int((df["correct"] == 0).sum())) + "/" + str(len(df)) + " clip")
    print("\n" + classification_report(yt, yp, target_names=["normal", "shoplifting"], zero_division=0))

    wrong = df[df["correct"] == 0]
    if len(wrong):
        print("Clip DU DOAN SAI:")
        for _, r in wrong.iterrows():
            kind = "FN (bo sot trom)" if r["true"] == 1 else "FP (bao nham)"
            print("  ERR " + str(r["clip"]).ljust(32) + " " + kind.ljust(18)
                  + " score=" + ("%.3f" % r["score"]))

    if args.split == "all":
        print("\nNhac lai: so tren bi 'ao' do gom clip train. So HOP LE = notebook V1.4 (split-by-clip).")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        print("\n[Saved] bao cao per-clip -> " + args.out)


if __name__ == "__main__":
    main()
