#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare TextRegion (TR) vs DAVIS GT using ONLY the first frame of each sequence.

- IoU (binary, unchanged logic)
- GT coverage percentage: |GT ∩ TR| / |GT|
- TR is assumed to be a binary white-foreground mask (*_bin.png)
- GT is a color indexed mask; foreground is any nonzero pixel

For each sequence:
  - Pick the lexicographically first TR frame "*_bin.png"
  - Map to GT "<seq>/<sameframe>.png"
  - Resize TR -> GT if needed (nearest neighbor)
  - Compute IoU and coverage
  - Write one CSV row per sequence

Usage:
  python compare_firstframe_tr_vs_gt.py \
    --tr_dir /mnt/data0/andy/Easi3R/third_party/TextRegion/davis_finallabel_out2 \
    --gt_dir /mnt/data0/andy/Easi3R/DAVIS/Annotations/480p \
    --out_csv firstframe_tr_vs_gt.csv
"""

import argparse, sys, csv
from pathlib import Path
from typing import List
import numpy as np

try:
    import cv2
except Exception:
    print("ERROR: OpenCV (cv2) is required. Install with `pip install opencv-python`.", file=sys.stderr)
    raise

def read_binary_mask(path: Path) -> np.ndarray:
    """Return boolean mask where foreground is any nonzero pixel."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (img.astype(np.uint8) > 0)

def resize_to(mask: np.ndarray, target_shape) -> np.ndarray:
    h, w = target_shape
    out = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return out.astype(bool)

def iou_bool(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    uni = np.logical_or(a, b).sum(dtype=np.int64)
    return 1.0 if uni == 0 else float(inter) / float(uni)

def discover_sequences(tr_root: Path) -> List[Path]:
    """Return list of per-sequence folders under TR root that contain *_bin.png."""
    # Find all *_bin.png, group by parent folder
    frames = sorted(tr_root.rglob("*_bin.png"))
    seq_dirs = {}
    for p in frames:
        seq_dirs.setdefault(p.parent, []).append(p)
    # Keep only folders with at least one frame
    return [(seq, files) for seq, files in seq_dirs.items()]

def map_tr_to_gt(tr_png: Path, gt_root: Path) -> Path:
    seq = tr_png.parent.name
    frame = tr_png.name.replace("_bin.png", ".png")
    return gt_root / seq / frame

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tr_dir", default= "/mnt/data0/andy/Easi3R/third_party/TextRegion/davis_fingrainmos_label_out")
    ap.add_argument("--gt_dir", default= "/mnt/data0/andy/Easi3R/DAVIS/Annotations/480p")
    ap.add_argument("--out_csv", default="firstframe_tr_vs_gt.csv")
    args = ap.parse_args()

    tr_root = Path(args.tr_dir)
    gt_root = Path(args.gt_dir)
    if not tr_root.exists() or not gt_root.exists():
        print(f"ERROR: path not found.\n  TR: {tr_root}\n  GT: {gt_root}", file=sys.stderr)
        sys.exit(1)

    seq_entries = discover_sequences(tr_root)
    if not seq_entries:
        print(f"ERROR: No '*_bin.png' found under {tr_root}", file=sys.stderr)
        sys.exit(1)

    rows = []
    low_cov_seqs = []
    iou_sum = 0.0
    cov_sum = 0.0
    n = 0

    for seq_dir, files in sorted(seq_entries, key=lambda kv: kv[0].name):
        # Pick first frame (lexicographically)
        tr_png = sorted(files)[0]
        gt_png = map_tr_to_gt(tr_png, gt_root)

        if not gt_png.exists():
            # Try fallback: search by filename within the sequence folder
            candidates = list((gt_root / seq_dir.name).rglob(tr_png.name.replace("_bin.png", ".png")))
            if candidates:
                gt_png = sorted(candidates, key=lambda p: len(p.parts))[0]
            else:
                print(f"[WARN] Missing GT for {tr_png}")
                continue

        try:
            tr = read_binary_mask(tr_png)
            gt = read_binary_mask(gt_png)
        except Exception as e:
            print(f"[WARN] Read failed for sequence {seq_dir.name}: {e}")
            continue

        if tr.shape != gt.shape:
            tr = resize_to(tr, gt.shape)

        i = iou_bool(tr, gt)
        gt_pix = int(gt.sum(dtype=np.int64))
        inter = int(np.logical_and(gt, tr).sum(dtype=np.int64))
        cov = (inter / gt_pix) if gt_pix > 0 else 1.0  # coverage = |GT ∩ TR| / |GT|
        if cov < 0.5:
            low_cov_seqs.append(seq_dir.name)

        rows.append({
            "sequence": seq_dir.name,
            "tr_frame": str(tr_png),
            "gt_frame": str(gt_png),
            "H": gt.shape[0], "W": gt.shape[1],
            "IoU": round(i, 6),
            "GT_Recall": round(cov, 6),
            "GT_pixels": gt_pix,
            "TR_pixels": int(tr.sum(dtype=np.int64)),
            "Intersection_pixels": inter
        })

        iou_sum += i
        cov_sum += cov
        n += 1

    if n == 0:
        print("ERROR: No sequences evaluated.", file=sys.stderr)
        sys.exit(1)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sequence","tr_frame","gt_frame","H","W","IoU","GT_Recall","GT_pixels","TR_pixels","Intersection_pixels"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("=== First-frame Summary ===")
    print(f"Sequences evaluated: {n}")
    print(f"Mean IoU:           {iou_sum/n:.6f}")
    print(f"Mean GT Recall:   {cov_sum/n:.6f}")
    print("=== Low GT Recall (<50%) ===")
    if low_cov_seqs:
        for name in sorted(set(low_cov_seqs)):
            print(name)
        print(f"Total: {len(set(low_cov_seqs))}")
    else:
        print("None")
    print(f"CSV: {out_csv}")

if __name__ == "__main__":
    main()
