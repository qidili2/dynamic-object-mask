#!/usr/bin/env python3
"""
Standalone threshold sweep at object level.

For each sequence:
  1. Load VGGT layer npy -> compute formula score [B, Ht, Wt]
  2. Run SAM2 tracking -> get group_ids [B, H, W]
  3. Downsample group_ids to token grid
  4. Region mean-pool score per object -> {obj_id: score}
  5. Threshold -> binary mask per frame
  6. Compute IoU vs DAVIS GT

Sweeps fixed thresholds and reports mean IoU across all sequences.

Usage:
  python sweep_object_threshold.py \
      --davis_root /mnt/data0/andy/Easi3R/DAVIS/JPEGImages/480p \
      --attn_root  /mnt/data0/andy/vggt/cross_attn_out_qksum_sample_window_10_allLayer \
      [--seqs blackswan,bmx-trees,...] \
      [--step 0.025]
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ── import helpers from vggt_sam2_mask.py ────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from vggt_sam2_mask import (
    load_seq_images,
    load_query_layer_maps,
    load_key_layer_maps,
    load_var_layer_maps,
    build_dynamic_token_map,
    generate_region_groups_with_tracking,
    downsample_groups_to_tokens,
    compute_object_global_attention,
)
from PIL import Image

GT_ROOT = Path("/mnt/data0/andy/Easi3R/DAVIS/Annotations/480p")

EPS = 1e-8
Q_LAYERS = [11, 16, 17, 18, 23]   # deep query
K_LAYERS = [2, 4, 7]              # shallow key / var


# ─────────────────────────────────────────────────────────────────────────────

def load_gt(seq: str, frame_id: int, H: int, W: int, gt_root: Path) -> np.ndarray | None:
    p = gt_root / seq / f"{frame_id:05d}.png"
    if not p.exists():
        return None
    gt = np.array(Image.open(p).resize((W, H), Image.NEAREST))
    return (gt > 0).astype(bool)


def iou_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def eval_seq(seq_dir: Path, attn_root: Path, gt_root: Path,
             thresholds: np.ndarray, device: str,
             resolution: int = 518) -> dict | None:
    """
    Returns {threshold: mean_iou_over_frames} for one sequence, or None on failure.
    """
    seq = seq_dir.name
    npy_dir = attn_root / seq / "npy"
    if not npy_dir.exists():
        print(f"  [{seq}] npy dir not found, skipping")
        return None

    # ── load images ──────────────────────────────────────────────────────────
    imgs, _ = load_seq_images(seq_dir, resolution)
    if not imgs:
        print(f"  [{seq}] no images, skipping")
        return None
    B = len(imgs)
    H, W = imgs[0].shape[:2]

    # ── load npy and build score map ─────────────────────────────────────────
    q_maps  = load_query_layer_maps(npy_dir, B, layers=Q_LAYERS)
    k_maps  = load_key_layer_maps(npy_dir,   B, layers=K_LAYERS)
    qv_maps = load_var_layer_maps(npy_dir,   B, side='query', layers=K_LAYERS)
    kv_maps = load_var_layer_maps(npy_dir,   B, side='key',   layers=K_LAYERS)

    # fallback if deep-layer query missing
    if all(m is None for m in q_maps):
        from vggt_sam2_mask import load_attn_maps
        q_maps = load_attn_maps(npy_dir, B)

    if all(m is None for m in q_maps):
        print(f"  [{seq}] no query maps, skipping")
        return None

    first = next(m for m in q_maps if m is not None)
    Ht, Wt = first.shape

    if all(m is None for m in k_maps):
        k_maps = None
    if k_maps is not None and (all(m is None for m in qv_maps) or
                                all(m is None for m in kv_maps)):
        qv_maps = kv_maps = None

    attn_tensor = build_dynamic_token_map(
        q_maps, Ht, Wt, key_maps=k_maps, qv_maps=qv_maps, kv_maps=kv_maps
    ).to(device)  # [B, Ht, Wt]

    # ── SAM2 tracking → group IDs ─────────────────────────────────────────────
    print(f"  [{seq}] running SAM2 tracking ...")
    try:
        region_groups, _ = generate_region_groups_with_tracking(
            imgs, device, vis_dir=None
        )
    except Exception as e:
        print(f"  [{seq}] SAM2 failed: {e}")
        return None

    # ── downsample group IDs to token grid ───────────────────────────────────
    groups_hr    = torch.stack(region_groups, dim=0).to(device)      # [B, H, W]
    groups_token = downsample_groups_to_tokens(groups_hr, Ht, Wt)    # [B, Ht, Wt]

    # ── per-object mean score ─────────────────────────────────────────────────
    obj_global = compute_object_global_attention(attn_tensor, groups_token)
    if not obj_global:
        print(f"  [{seq}] no objects found")
        return None

    # ── load GT ───────────────────────────────────────────────────────────────
    gts = [load_gt(seq, fi, H, W, gt_root) for fi in range(B)]
    if all(g is None for g in gts):
        print(f"  [{seq}] no GT found, skipping")
        return None

    # ── top-K% sweep ─────────────────────────────────────────────────────────
    groups_hr_np = groups_hr.cpu().numpy()   # [B, H, W]
    n_obj = len(obj_global)
    # sort objects by score descending once
    sorted_oids = sorted(obj_global, key=obj_global.__getitem__, reverse=True)

    result = {}
    for k_pct in thresholds:          # thresholds here = top-K% grid
        n_dynamic = max(1, round(n_obj * k_pct / 100.0))
        dynamic_ids = set(sorted_oids[:n_dynamic])
        frame_ious = []
        for fi in range(B):
            gt = gts[fi]
            if gt is None:
                continue
            pred = np.zeros((H, W), dtype=bool)
            for oid in dynamic_ids:
                pred |= (groups_hr_np[fi] == oid)
            frame_ious.append(iou_binary(pred, gt))
        result[k_pct] = float(np.mean(frame_ious)) if frame_ious else 0.0

    best_k = max(result, key=result.get)
    print(f"  [{seq}] done  best_top={best_k:.1f}%  iou={result[best_k]:.4f}  "
          f"objects={n_obj}")
    return result


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--davis_root", required=True)
    parser.add_argument("--attn_root",  required=True)
    parser.add_argument("--gt_root",    default=str(GT_ROOT))
    parser.add_argument("--seqs",       default=None,
                        help="Comma-separated sequence names; default=all")
    parser.add_argument("--step",       type=float, default=0.025)
    parser.add_argument("--resolution", type=int,   default=518)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    davis_root = Path(args.davis_root)
    attn_root  = Path(args.attn_root)
    gt_root    = Path(args.gt_root)

    if args.seqs:
        seq_dirs = [davis_root / s.strip() for s in args.seqs.split(",")]
    else:
        seq_dirs = sorted(d for d in davis_root.iterdir() if d.is_dir())

    # top-K% grid: 1%, 2%, ..., 5%, 10%, 15%, ..., 50%, 60%, 70%, 80%, 90%, 100%
    thresholds = np.array(
        list(range(1, 6)) +            # 1-5%  fine-grained (few objects = dynamic)
        list(range(5, 55, 5)) +        # 5-50% step 5
        list(range(60, 110, 10)),      # 60-100% step 10
        dtype=float
    )
    thresholds = np.unique(thresholds)
    print(f"Sweeping top-K% grid ({thresholds.tolist()}) "
          f"over {len(seq_dirs)} sequences on {args.device}")

    seq_results = {}
    for sd in seq_dirs:
        if not sd.exists():
            print(f"  [{sd.name}] dir not found, skipping")
            continue
        r = eval_seq(sd, attn_root, gt_root, thresholds,
                     device=args.device, resolution=args.resolution)
        if r is not None:
            seq_results[sd.name] = r

    if not seq_results:
        print("No valid sequences processed.")
        return

    # ── aggregate ─────────────────────────────────────────────────────────────
    rows = []
    for t in thresholds:
        ious = [seq_results[s][t] for s in seq_results]
        rows.append({
            "top_k_pct":   float(t),
            "mean_iou":    float(np.mean(ious)),
            "median_iou":  float(np.median(ious)),
            "n_seqs":      len(ious),
        })

    df = pd.DataFrame(rows)
    best = df.loc[df.mean_iou.idxmax()]

    print(f"\n{'='*60}")
    print(f"Best top-K%    : {best.top_k_pct:.1f}%")
    print(f"Mean   IoU     : {best.mean_iou:.4f}")
    print(f"Median IoU     : {best.median_iou:.4f}")
    print(f"Sequences      : {int(best.n_seqs)}")
    print(f"{'='*60}\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    out = Path(args.attn_root) / "topk_sweep_object.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
