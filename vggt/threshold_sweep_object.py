#!/usr/bin/env python3
"""
Sweep fixed thresholds at object level to find best mean IoU across all sequences.

Requires intermediates saved by vggt_sam2_mask.py:
  <sam2_out>/<seq>/groups_hr.npy   [B, H, W] int64 region IDs
  <sam2_out>/<seq>/obj_ids.npy     [N_obj] int64
  <sam2_out>/<seq>/obj_attn.npy    [N_obj] float32 attention score (HIGH = dynamic)

GT masks: /mnt/data0/andy/Easi3R/DAVIS/Annotations/480p/<seq>/<XXXXX>.png

Usage:
  python threshold_sweep_object.py --sam2_out <dir> [--thresholds 0.05,0.1,...] [--gt_root <dir>]
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")

GT_ROOT = Path("/mnt/data0/andy/Easi3R/DAVIS/Annotations/480p")


def load_gt(gt_root: Path, seq: str, frame_id: int, H: int, W: int) -> np.ndarray | None:
    p = gt_root / seq / f"{frame_id:05d}.png"
    if not p.exists():
        return None
    gt = np.array(Image.open(p).resize((W, H), Image.NEAREST))
    return (gt > 0).astype(bool)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def eval_seq_fixed(seq_out: Path, gt_root: Path, thresholds: np.ndarray) -> dict | None:
    """Return {threshold -> mean_iou} for one sequence using fixed threshold."""
    groups_path = seq_out / "groups_hr.npy"
    ids_path    = seq_out / "obj_ids.npy"
    attn_path   = seq_out / "obj_attn.npy"

    if not (groups_path.exists() and ids_path.exists() and attn_path.exists()):
        return None

    groups_hr = np.load(groups_path)   # [B, H, W] int64
    obj_ids   = np.load(ids_path)      # [N_obj]
    obj_attn  = np.load(attn_path)     # [N_obj]
    id_to_attn = dict(zip(obj_ids.tolist(), obj_attn.tolist()))

    B, H, W = groups_hr.shape
    seq = seq_out.name

    gts = [load_gt(gt_root, seq, fi, H, W) for fi in range(B)]
    if not any(g is not None for g in gts):
        return None

    result = {}
    for t in thresholds:
        dynamic_ids = {oid for oid, a in id_to_attn.items() if a > t}
        frame_ious = []
        for fi in range(B):
            gt = gts[fi]
            if gt is None:
                continue
            pred = np.zeros((H, W), dtype=bool)
            for oid in dynamic_ids:
                pred |= (groups_hr[fi] == oid)
            frame_ious.append(iou(pred, gt))
        result[t] = float(np.mean(frame_ious)) if frame_ious else 0.0

    return result


def eval_seq_topk(seq_out: Path, gt_root: Path, thresholds: np.ndarray) -> dict | None:
    """Return {k_pct -> mean_iou} for one sequence using top-K% method."""
    groups_path = seq_out / "groups_hr.npy"
    ids_path    = seq_out / "obj_ids.npy"
    attn_path   = seq_out / "obj_attn.npy"

    if not (groups_path.exists() and ids_path.exists() and attn_path.exists()):
        return None

    groups_hr = np.load(groups_path)   # [B, H, W] int64
    obj_ids   = np.load(ids_path)      # [N_obj]
    obj_attn  = np.load(attn_path)     # [N_obj]

    B, H, W = groups_hr.shape
    seq = seq_out.name
    n_obj = len(obj_ids)

    # sort objects by score descending
    sorted_oids = [oid for oid, _ in sorted(
        zip(obj_ids.tolist(), obj_attn.tolist()), key=lambda x: x[1], reverse=True
    )]

    gts = [load_gt(gt_root, seq, fi, H, W) for fi in range(B)]
    if not any(g is not None for g in gts):
        return None

    result = {}
    for k_pct in thresholds:
        n_dynamic = max(1, round(n_obj * k_pct / 100.0))
        dynamic_ids = set(sorted_oids[:n_dynamic])
        frame_ious = []
        for fi in range(B):
            gt = gts[fi]
            if gt is None:
                continue
            pred = np.zeros((H, W), dtype=bool)
            for oid in dynamic_ids:
                pred |= (groups_hr[fi] == oid)
            frame_ious.append(iou(pred, gt))
        result[k_pct] = float(np.mean(frame_ious)) if frame_ious else 0.0

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2_out", required=True,
                        help="Directory containing per-sequence subdirs")
    parser.add_argument("--gt_root", default=str(GT_ROOT))
    parser.add_argument("--method", choices=["fixed", "topk"], default="fixed",
                        help="Thresholding method: 'fixed' (score > t) or 'topk' (top K%% objects)")
    parser.add_argument("--step", type=float, default=0.025,
                        help="Fixed threshold grid step (default 0.025)")
    parser.add_argument("--topk", type=float, default=25.0,
                        help="Top-K%% to use when method=topk (default 25.0)")
    args = parser.parse_args()

    sam2_out = Path(args.sam2_out)
    gt_root  = Path(args.gt_root)

    seq_dirs = sorted(d for d in sam2_out.iterdir()
                      if d.is_dir() and (d / "groups_hr.npy").exists())
    print(f"Found {len(seq_dirs)} sequences with intermediates")

    if not seq_dirs:
        print("No sequences found. Re-run vggt_sam2_mask.py first to generate intermediates.")
        return

    if args.method == "topk":
        # top-K% grid: fine-grained at low end, coarser at high end
        thresholds = np.unique(np.array(
            list(range(1, 6)) +        # 1-5%
            list(range(5, 55, 5)) +    # 5-50%
            list(range(60, 110, 10)),  # 60-100%
            dtype=float
        ))
        t_label = "top_k_pct"
        eval_fn = eval_seq_topk
        fmt_fn  = lambda t: f"top={t:.0f}%"
    else:
        thresholds = np.arange(0.05, 0.96, args.step)
        t_label = "threshold"
        eval_fn = eval_seq_fixed
        fmt_fn  = lambda t: f"t={t:.3f}"

    print(f"Method: {args.method}  grid size: {len(thresholds)}")

    # per-seq results
    seq_results = {}
    for sd in seq_dirs:
        r = eval_fn(sd, gt_root, thresholds)
        if r is not None:
            seq_results[sd.name] = r
            best_t = max(r, key=r.get)
            print(f"  {sd.name:<25}  best={fmt_fn(best_t)}  iou={r[best_t]:.4f}")
        else:
            print(f"  {sd.name:<25}  [skip - no GT or missing files]")

    if not seq_results:
        print("No valid sequences.")
        return

    # aggregate
    rows = []
    for t in thresholds:
        ious = [seq_results[s][t] for s in seq_results if t in seq_results[s]]
        rows.append({
            t_label:      round(float(t), 4),
            "mean_iou":   float(np.mean(ious)),
            "median_iou": float(np.median(ious)),
            "n_seqs":     len(ious),
        })

    df = pd.DataFrame(rows).sort_values("mean_iou", ascending=False)
    best = df.iloc[0]

    print(f"\n{'='*55}")
    if args.method == "topk":
        print(f"Best top-K%: {best[t_label]:.1f}%  mean_IoU={best.mean_iou:.4f}  "
              f"median_IoU={best.median_iou:.4f}")
        # also report fixed top-K% result
        topk_row = df[df[t_label] == args.topk]
        if not topk_row.empty:
            r = topk_row.iloc[0]
            print(f"At top-{args.topk:.0f}%:   mean_IoU={r.mean_iou:.4f}  "
                  f"median_IoU={r.median_iou:.4f}")
    else:
        print(f"Best threshold: {best[t_label]:.4f}  mean_IoU={best.mean_iou:.4f}  "
              f"median_IoU={best.median_iou:.4f}")
    print(f"{'='*55}\n")

    df_sorted = df.sort_values(t_label)
    print(df_sorted.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    suffix = "topk" if args.method == "topk" else "fixed"
    out = sam2_out / f"threshold_sweep_object_{suffix}.csv"
    df_sorted.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
