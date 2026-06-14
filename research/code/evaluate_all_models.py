"""
Đánh giá TOÀN BỘ model KLTN trên cùng một tập dữ liệu để so sánh cái nào chuẩn nhất.

Models so sánh:
    - Hao baseline   (LSTM-CNN .h5, 56 feat)
    - V1.4           (BiLSTM-CNN .keras, 56 feat)
    - V2.0 ST-GCN    (.pt, skeleton graph)
    - V2.4 MS-G3D    (.pt, multi-scale graph)
    - Ensemble V2.3  = 0.5*V1.4 + 0.5*V2.0
    - Ensemble alt   = 0.5*V1.4 + 0.5*V2.4

Tái dùng loader + kiến trúc gốc trong src/ của KLTN (không train lại).
Đầu vào: keypoint CSV 56-feat đã gán nhãn (data/clean_csv_56feat/{normal,shoplifting}).

Chạy bằng venv backend (đã có TF 2.15 + torch + sklearn):
    "<root>/backend/.venv/Scripts/python.exe" research/code/evaluate_all_models.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# --- Paths (độc lập — dùng bản copy trong research/code) -----------------
RESEARCH = Path(__file__).resolve().parent            # research/code
MODELS_DIR = RESEARCH / "models"                      # hao / v1.4 / v2.0 / v2.4
CSV_DIR = RESEARCH / "data" / "clean_csv_56feat"

sys.path.insert(0, str(RESEARCH))                     # để import src.*

import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold

from src.models.stgcn import load_skeleton_model
from src.behavior import load_hao_checkpoint
from src.behavior.keras_v14 import (_build_v14_bilstm, _focal_loss_dummy,
                                    _load_v14_from_keras3_zip)
from tensorflow.keras.models import load_model

SEQ, STEP = 90, 30
SEEDS = [42, 123, 7, 2024, 999]


# --- Model loaders -------------------------------------------------------
def load_keras_v14(path):
    try:
        return load_model(path, custom_objects={'f': _focal_loss_dummy(),
                                                 'loss_fn': _focal_loss_dummy()},
                          compile=False)
    except Exception:
        try:
            return _load_v14_from_keras3_zip(path)
        except Exception:
            m = _build_v14_bilstm(input_shape=(SEQ, 56))
            m.load_weights(path)
            return m


# --- Data ----------------------------------------------------------------
def list_clips():
    clips = []
    for name, lbl in [("normal", 0), ("shoplifting", 1)]:
        d = CSV_DIR / name
        if d.exists():
            for fp in sorted(d.glob("*.csv")):
                clips.append((f"{name}/{fp.name}", lbl, fp))
    return clips


def make_windows(arr):
    T = arr.shape[0]
    if T < SEQ:
        pad = np.repeat(arr[-1:], SEQ - T, axis=0)
        return np.concatenate([arr, pad], axis=0)[None]
    return np.asarray([arr[i:i + SEQ] for i in range(0, T - SEQ + 1, STEP)],
                      dtype=np.float32)


# --- Per-window probability (P(shoplifting)) for each base model ---------
def probs_keras(model, wins):
    return model.predict(wins, verbose=0)[:, 1]


def probs_stgcn(model, wins):
    n = wins.shape[0]
    x34 = wins[:, :, :34].reshape(n, SEQ, 17, 2)
    x = np.concatenate([x34, np.ones((n, SEQ, 17, 1), np.float32)], -1).transpose(0, 3, 1, 2)
    with torch.no_grad():
        logits = model(torch.from_numpy(x.astype(np.float32)))
        return F.softmax(logits, -1)[:, 1].cpu().numpy()


# --- Metrics helper ------------------------------------------------------
def metrics(y_true, y_pred):
    return {
        "Acc": accuracy_score(y_true, y_pred),
        "Prec": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Rec": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def main():
    print(f"[Data] {CSV_DIR}")
    print(f"[Models] {MODELS_DIR}\n")
    clips = list_clips()
    print(f"[Clips] {len(clips)} ("
          f"{sum(1 for c in clips if c[1]==0)} normal / "
          f"{sum(1 for c in clips if c[1]==1)} shoplifting)\n")

    print("Loading models...")
    hao = load_hao_checkpoint(str(MODELS_DIR / "hao_baseline_lstm_cnn_56feat.h5"), (SEQ, 56))
    v14 = load_keras_v14(str(MODELS_DIR / "v1.4_best.keras"))
    v20, a20 = load_skeleton_model(str(MODELS_DIR / "v2.0_best.pt"), "cpu")
    v24, a24 = load_skeleton_model(str(MODELS_DIR / "v2.4_best.pt"), "cpu")
    print(f"  hao .h5 | v1.4 .keras | {a20} | {a24}\n")

    # Precompute per-clip score for each model (mean over windows).
    y_true = np.array([c[1] for c in clips])
    base = {"Hao": [], "V1.4": [], "V2.0_STGCN": [], "V2.4_MSG3D": []}
    ens_v20, ens_v24 = [], []

    print("Scoring clips...")
    for i, (cid, lbl, fp) in enumerate(clips):
        arr = pd.read_csv(fp).values.astype(np.float32)
        if arr.shape[1] != 56:
            print(f"  [skip] {cid}: {arr.shape[1]} cols"); continue
        wins = make_windows(arr)
        p_hao = probs_keras(hao, wins)
        p_v14 = probs_keras(v14, wins)
        p_v20 = probs_stgcn(v20, wins)
        p_v24 = probs_stgcn(v24, wins)
        base["Hao"].append(p_hao.mean())
        base["V1.4"].append(p_v14.mean())
        base["V2.0_STGCN"].append(p_v20.mean())
        base["V2.4_MSG3D"].append(p_v24.mean())
        ens_v20.append((0.5 * p_v14 + 0.5 * p_v20).mean())
        ens_v24.append((0.5 * p_v14 + 0.5 * p_v24).mean())
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(clips)}")

    scores = {
        "Hao baseline": np.array(base["Hao"]),
        "V1.4 BiLSTM-CNN": np.array(base["V1.4"]),
        "V2.0 ST-GCN": np.array(base["V2.0_STGCN"]),
        "V2.4 MS-G3D": np.array(base["V2.4_MSG3D"]),
        "Ens V2.3 (V1.4+V2.0)": np.array(ens_v20),
        "Ens (V1.4+V2.4)": np.array(ens_v24),
    }

    # ---- (A) ALL clips (leaky / optimistic) ----
    def table(get_idx):
        rows = []
        for name, sc in scores.items():
            idx = get_idx()
            yt = y_true[idx]; yp = (sc[idx] >= 0.5).astype(int)
            m = metrics(yt, yp)
            rows.append((name, m))
        return rows

    def print_table(title, rows):
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)
        print(f"{'Model':<24}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}")
        print("-" * 78)
        best = max(rows, key=lambda r: r[1]["F1"])
        for name, m in rows:
            star = "  <== best F1" if name == best[0] else ""
            print(f"{name:<24}{m['Acc']:>8.3f}{m['Prec']:>8.3f}{m['Rec']:>8.3f}{m['F1']:>8.3f}{star}")

    all_idx = np.arange(len(clips))
    print_table("(A) TOAN BO CLIP  —  optimistic (gom ca clip train, chi de sanity-check)",
                table(lambda: all_idx))

    # ---- (B) Held-out test split, multi-seed (it leak hon) ----
    print("\n" + "=" * 78)
    print("(B) HELD-OUT TEST  —  StratifiedKFold split-by-clip, trung binh 5 seed")
    print("=" * 78)
    print(f"{'Model':<24}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}")
    print("-" * 78)
    agg = {name: [] for name in scores}
    for name, sc in scores.items():
        for seed in SEEDS:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            _, test_idx = list(skf.split(np.zeros(len(y_true)), y_true))[seed % 5]
            yt = y_true[test_idx]; yp = (sc[test_idx] >= 0.5).astype(int)
            agg[name].append(metrics(yt, yp))
    rows_b = []
    for name in scores:
        ms = agg[name]
        mean = {k: np.mean([m[k] for m in ms]) for k in ["Acc", "Prec", "Rec", "F1"]}
        std_f1 = np.std([m["F1"] for m in ms])
        rows_b.append((name, mean, std_f1))
    best_b = max(rows_b, key=lambda r: r[1]["F1"])
    for name, m, sf in rows_b:
        star = "  <== best" if name == best_b[0] else ""
        print(f"{name:<24}{m['Acc']:>8.3f}{m['Prec']:>8.3f}{m['Rec']:>8.3f}{m['F1']:>8.3f}  (±{sf:.3f}F1){star}")

    # Save per-clip scores for the web/report.
    out = pd.DataFrame({"clip": [c[0] for c in clips], "true": y_true})
    for name, sc in scores.items():
        out[name] = sc
    out_path = RESEARCH.parent / "results" / "model_comparison_scores.csv"
    out.to_csv(out_path, index=False)
    print(f"\n[Saved] per-clip scores -> {out_path}")
    print("\nGoi y: con so (B) it leak hon (A); de bao cao chinh thuc dung multi-seed notebook.")


if __name__ == "__main__":
    main()
