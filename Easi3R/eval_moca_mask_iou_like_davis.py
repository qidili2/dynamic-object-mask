#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MoCA eval (DAVIS-like CSV outputs), using mask-IoU against GT masks.

Inputs:
  --gt_mask_root    GT masks root: gt_root/seq/00000.png ...
  --pred_mask_root  Pred masks root: pred_root/seq/<frame><pred_suffix> OR <frame>.png

Evaluation protocol:
  - Iterate ONLY frames that exist in GT (GT-driven).
  - For each GT frame, find pred in this priority:
        1) pred_root/seq/{frame}{pred_suffix}   (e.g. 00000_dynamic_mask_60.png)
        2) pred_root/seq/{frame}.png
     If both missing -> IoU = 0 (count missing)
  - If pred exists but empty -> IoU = 0 (count empty)
  - Masks can be grayscale/multi-value: foreground is pixel > 0

Outputs (in --results_path, default pred_root):
  - global_results.csv
  - per-sequence_results.csv
"""

import argparse
from pathlib import Path
from time import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_seq_frames(mask_root: Path) -> Dict[str, List[str]]:
    """
    Return dict: seq -> sorted list of frame stems (e.g., "00000")
    Only considers files directly under seq folder, suffix in IMG_EXTS.
    """
    seq_frames: Dict[str, List[str]] = {}
    if not mask_root.exists():
        raise FileNotFoundError(f"mask_root not found: {mask_root}")

    for seq_dir in sorted([p for p in mask_root.iterdir() if p.is_dir()]):
        frames = []
        for f in seq_dir.iterdir():
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                frames.append(f.stem)
        if frames:
            try:
                frames = sorted(frames, key=lambda s: int(s))
            except Exception:
                frames = sorted(frames)
            seq_frames[seq_dir.name] = frames
    return seq_frames


def load_mask01(p: Path) -> Tuple[bool, np.ndarray]:
    """
    Returns (ok, mask01).
    ok=False if file missing.
    mask01 is uint8 {0,1}.
    """
    if not p.exists():
        return False, np.zeros((0, 0), dtype=np.uint8)
    m = np.array(Image.open(p))
    if m.ndim == 3:
        m = np.any(m > 0, axis=2).astype(np.uint8)
    else:
        m = (m > 0).astype(np.uint8)
    return True, m.astype(np.uint8)


def mask_iou(gt01: np.ndarray, pr01: np.ndarray) -> float:
    """
    IoU on binary masks.
    If shapes differ, compute on overlap area by cropping to min(H,W).
    """
    if gt01.size == 0 or pr01.size == 0:
        return 0.0

    H = min(gt01.shape[0], pr01.shape[0])
    W = min(gt01.shape[1], pr01.shape[1])
    gt = gt01[:H, :W].astype(bool)
    pr = pr01[:H, :W].astype(bool)

    inter = np.logical_and(gt, pr).sum()
    union = np.logical_or(gt, pr).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def recall_at_threshold(vals: np.ndarray, thr: float) -> float:
    if vals.size == 0:
        return 0.0
    return float((vals >= thr).mean())


def decay(vals: np.ndarray) -> float:
    """mean(first 20%) - mean(last 20%)"""
    if vals.size == 0:
        return 0.0
    n = vals.size
    k = max(1, int(round(0.2 * n)))
    return float(vals[:k].mean() - vals[-k:].mean())


def resolve_pred_path(pred_root: Path, seq: str, frame: str, pred_suffix: str) -> Path:
    """
    Priority:
      1) {frame}{pred_suffix}  (if pred_suffix not empty)
      2) {frame}.png
    """
    if pred_suffix:
        p1 = pred_root / seq / f"{frame}{pred_suffix}"
        if p1.exists():
            return p1
    p2 = pred_root / seq / f"{frame}.png"
    return p2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_mask_root", type=str, required=True,
                    help="GT mask root: gt_root/seq/00000.png ...")
    ap.add_argument("--pred_mask_root", type=str, required=True,
                    help="Pred mask root: pred_root/seq/<frame><pred_suffix> or <frame>.png")
    ap.add_argument("--pred_suffix", type=str, default="_dynamic_mask_60.png",
                    help='Preferred pred filename suffix, e.g. "_dynamic_mask_60.png". '
                         'Pred path will try {frame}{pred_suffix} first. '
                         'Set "" to disable.')
    ap.add_argument("--results_path", type=str, default="",
                    help="Where to save csv. Default = pred_mask_root")
    ap.add_argument("--seq_list", type=str, default="",
                    help="Optional comma-separated seq names")
    ap.add_argument("--iou_thr", type=float, default=0.5,
                    help="Recall threshold (default 0.5)")
    args = ap.parse_args()

    t0 = time()

    gt_root = Path(args.gt_mask_root)
    pr_root = Path(args.pred_mask_root)
    out_dir = Path(args.results_path) if args.results_path else pr_root
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_map = list_seq_frames(gt_root)
    if not gt_map:
        raise RuntimeError(f"No sequences/frames found under gt_mask_root: {gt_root}")

    wanted = None
    if args.seq_list.strip():
        wanted = set([s.strip() for s in args.seq_list.split(",") if s.strip()])

    per_rows = []
    all_vals = []

    total_frames = 0
    missing_pred = 0
    empty_pred = 0
    missing_seq_pred = 0

    for seq, frames in gt_map.items():
        if wanted is not None and seq not in wanted:
            continue

        pr_seq_dir = pr_root / seq
        if not pr_seq_dir.exists():
            missing_seq_pred += 1

        vals = []
        seq_missing_or_empty = 0

        for fr in frames:
            gt_p = gt_root / seq / f"{fr}.png"
            pr_p = resolve_pred_path(pr_root, seq, fr, args.pred_suffix)

            ok_gt, gt01 = load_mask01(gt_p)
            if not ok_gt:
                continue  # should not happen since GT-driven listing

            ok_pr, pr01 = load_mask01(pr_p)
            if not ok_pr:
                iou = 0.0
                missing_pred += 1
                seq_missing_or_empty += 1
            else:
                if pr01.size == 0 or (pr01.ndim == 2 and pr01.sum() == 0):
                    empty_pred += 1
                    seq_missing_or_empty += 1
                    iou = 0.0
                else:
                    iou = mask_iou(gt01, pr01)

            vals.append(iou)

        vals_np = np.array(vals, dtype=np.float32)
        n = len(vals)
        if n == 0:
            continue

        total_frames += n
        all_vals.append(vals_np)

        mean_iou = float(vals_np.mean())
        rec = recall_at_threshold(vals_np, args.iou_thr)
        dec = decay(vals_np)
        miss_rate = float(seq_missing_or_empty / n)

        per_rows.append([seq, n, seq_missing_or_empty, miss_rate, mean_iou, rec, dec])

    if not per_rows:
        raise RuntimeError("No sequences evaluated. Check gt_mask_root / seq_list.")

    per_df = pd.DataFrame(
        per_rows,
        columns=["Sequence", "NumFrames(GT)", "MissingOrEmptyPred", "MissingRate",
                 "IoU-Mean", f"IoU-Recall@{args.iou_thr}", "IoU-Decay"],
    ).sort_values("Sequence")

    all_np = np.concatenate(all_vals, axis=0) if len(all_vals) else np.zeros((0,), dtype=np.float32)
    g_mean = float(all_np.mean()) if all_np.size else 0.0
    g_rec = recall_at_threshold(all_np, args.iou_thr) if all_np.size else 0.0
    g_dec = decay(all_np) if all_np.size else 0.0
    g_miss_rate = float((missing_pred + empty_pred) / total_frames) if total_frames > 0 else 1.0

    g_df = pd.DataFrame(
        [[g_mean, g_rec, g_dec, total_frames, missing_pred, empty_pred, g_miss_rate, missing_seq_pred, args.pred_suffix]],
        columns=["IoU-Mean", f"IoU-Recall@{args.iou_thr}", "IoU-Decay",
                 "TotalFrames(GT)", "MissingPredFrames", "EmptyPredFrames",
                 "MissingRate", "MissingPredSeqs", "PredSuffixPriority"],
    )

    global_csv = out_dir / "global_results.csv"
    per_csv = out_dir / "per-sequence_results.csv"
    g_df.to_csv(global_csv, index=False, float_format="%.4f")
    per_df.to_csv(per_csv, index=False, float_format="%.4f")

    print("Evaluating MoCA (mask IoU, GT-driven)...")
    print(f"GT root      : {gt_root}")
    print(f"Pred root    : {pr_root}")
    print(f"Pred suffix  : {args.pred_suffix!r} (priority)")
    print(f"Save to      : {out_dir}")
    print(f"Global saved : {global_csv}")
    print(f"Per-seq saved: {per_csv}")
    print("--------------------------- Global results ---------------------------")
    print(g_df.to_string(index=False))
    print(f"\nTotal time: {time() - t0:.2f}s")


if __name__ == "__main__":
    main()
