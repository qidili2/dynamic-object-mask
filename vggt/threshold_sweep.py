#!/usr/bin/env python3
"""
Sweep fixed thresholds for the formula:
  score = (1 - norm(q_deep)) * (norm(k_shallow) + 2*norm(qv_shallow) + 2*norm(kv_shallow))

where:
  q_deep    = mean of query layers [11, 16, 17, 18, 23]
  k_shallow = mean of key   layers [2, 4, 7]
  qv_shallow= mean of query_var layers [2, 4, 7]
  kv_shallow= mean of key_var   layers [2, 4, 7]

Evaluates at patch level vs DAVIS GT masks.
Reports F1, precision, recall for each threshold.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")

NPY_ROOT = Path("/mnt/data0/andy/vggt/cross_attn_out_qksum_sample_window_10_allLayer")
GT_ROOT  = Path("/mnt/data0/andy/Easi3R/DAVIS/Annotations/480p")
EPS      = 1e-8

Q_LAYERS  = [11, 16, 17, 18, 23]   # deep query
K_LAYERS  = [2, 4, 7]              # shallow key / var


def norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + EPS)


def load_mean_layers(npy_dir: Path, frame_id: int, side: str,
                     layers: list, var: bool = False) -> np.ndarray | None:
    suffix = f"_{side}_var" if var else f"_{side}"
    mats = []
    for lid in layers:
        p = npy_dir / f"frame_{frame_id:05d}_layer{lid}{suffix}.npy"
        if p.exists():
            mats.append(np.load(p).astype(np.float32))
    if not mats:
        return None
    return np.stack(mats, 0).mean(0)


_gt_cache: dict = {}

def load_gt_patch(seq: str, frame_id: int, H_p: int, W_p: int) -> np.ndarray | None:
    key = (seq, frame_id)
    if key in _gt_cache:
        return _gt_cache[key]
    p = GT_ROOT / seq / f"{frame_id:05d}.png"
    if not p.exists():
        _gt_cache[key] = None
        return None
    gt = np.array(Image.open(p))
    from PIL import Image as PILImage
    gt_small = np.array(PILImage.fromarray(gt).resize((W_p, H_p),
                                                       PILImage.NEAREST))
    binary = (gt_small > 0).astype(np.float32)
    _gt_cache[key] = binary
    return binary


def main():
    seqs = sorted(p.name for p in NPY_ROOT.iterdir() if p.is_dir())
    print(f"Found {len(seqs)} sequences")

    all_scores = []
    all_labels = []

    for seq in seqs:
        npy_dir = NPY_ROOT / seq / "npy"
        if not npy_dir.exists():
            continue

        # find frame ids
        frame_ids = sorted({
            int(m.group(1))
            for f in npy_dir.iterdir()
            if (m := re.match(r"frame_(\d{5})_", f.name))
        })

        for fid in frame_ids:
            # load components
            q  = load_mean_layers(npy_dir, fid, 'query', Q_LAYERS, var=False)
            k  = load_mean_layers(npy_dir, fid, 'key',   K_LAYERS, var=False)
            qv = load_mean_layers(npy_dir, fid, 'query', K_LAYERS, var=True)
            kv = load_mean_layers(npy_dir, fid, 'key',   K_LAYERS, var=True)

            if q is None or k is None or qv is None or kv is None:
                continue

            H_p, W_p = q.shape
            gt = load_gt_patch(seq, fid, H_p, W_p)
            if gt is None:
                continue
            # skip frames with no foreground or all foreground
            fg_ratio = gt.mean()
            if fg_ratio == 0 or fg_ratio == 1:
                continue

            score = (1.0 - norm01(q)) * (norm01(k) + 2.0 * norm01(qv) + 2.0 * norm01(kv))
            # clip & renorm to [0,1]
            score = np.clip(score, 0.0, None)
            score = score / (score.max() + EPS)

            all_scores.append(score.ravel())
            all_labels.append(gt.ravel())

    scores_arr = np.concatenate(all_scores)
    labels_arr = np.concatenate(all_labels).astype(int)

    print(f"Total patches: {len(scores_arr):,}  FG ratio: {labels_arr.mean():.3f}")
    auc = roc_auc_score(labels_arr, scores_arr)
    print(f"ROC-AUC: {auc:.4f}\n")

    # threshold sweep
    thresholds = np.arange(0.05, 0.96, 0.025)
    rows = []
    for t in thresholds:
        pred = (scores_arr >= t).astype(int)
        f1   = f1_score(labels_arr, pred, zero_division=0)
        prec = precision_score(labels_arr, pred, zero_division=0)
        rec  = recall_score(labels_arr, pred, zero_division=0)
        rows.append({"threshold": round(float(t), 4),
                     "f1": f1, "precision": prec, "recall": rec})

    df = pd.DataFrame(rows)
    best = df.loc[df.f1.idxmax()]
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nBest threshold: {best.threshold:.4f}  F1={best.f1:.4f}  "
          f"Prec={best.precision:.4f}  Rec={best.recall:.4f}")

    out = Path("/mnt/data0/andy/vggt/attn_analysis/threshold_sweep.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
