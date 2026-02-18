#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import cv2
import numpy as np

def read_bin_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"cannot read: {path}")
    return (m > 0).astype(np.uint8)

def resize_nn(mask01: np.ndarray, H: int, W: int) -> np.ndarray:
    if mask01.shape == (H, W):
        return mask01
    return cv2.resize(mask01.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

def list_png_stems(dir_path: Path):
    return sorted([p.stem for p in dir_path.glob("*.png")])

def check_one_seq(gt_seq_dir: Path, pred_seq_dir: Path, verbose=False):
    gt_frames = list_png_stems(gt_seq_dir)
    if len(gt_frames) == 0:
        return None

    # pred masks can be in pred_seq_dir/masks/*.png or directly pred_seq_dir/*.png
    pred_masks_dir = pred_seq_dir / "masks"
    if pred_masks_dir.exists():
        pred_root = pred_masks_dir
    else:
        pred_root = pred_seq_dir

    bad = []
    overlaps = []
    worst = None  # (overlap_ratio, stem)

    for stem in gt_frames:
        gt_path = gt_seq_dir / f"{stem}.png"
        pred_path = pred_root / f"{stem}.png"
        if not pred_path.exists():
            # 如果你希望 pred 缺失也算 bad，可以在这改
            continue

        gt_fg = read_bin_mask(gt_path)          # DAVIS: fg is >0
        pred_bg = read_bin_mask(pred_path)      # your output: bg is >0

        H, W = gt_fg.shape
        pred_bg = resize_nn(pred_bg, H, W)

        inter = (pred_bg & gt_fg).sum()
        fg_area = int(gt_fg.sum())
        # 以 GT 前景面积为分母：看有多少前景被误标成背景
        ratio = (inter / fg_area) if fg_area > 0 else 0.0

        overlaps.append(ratio)

        if inter > 0:
            bad.append((stem, int(inter), fg_area, ratio))
            if worst is None or ratio > worst[0]:
                worst = (ratio, stem)

        if verbose and inter > 0:
            print(f"  [BAD] {stem}: overlap={inter} fg_area={fg_area} ratio={ratio:.4f}")

    if len(overlaps) == 0:
        return {
            "num_gt": len(gt_frames),
            "num_checked": 0,
            "num_bad": 0,
            "mean_ratio": 0.0,
            "max_ratio": 0.0,
            "worst_frame": None,
            "bad_list": []
        }

    mean_ratio = float(np.mean(overlaps))
    max_ratio = float(np.max(overlaps))
    worst_frame = worst[1] if worst else None

    return {
        "num_gt": len(gt_frames),
        "num_checked": len(overlaps),
        "num_bad": len(bad),
        "mean_ratio": mean_ratio,
        "max_ratio": max_ratio,
        "worst_frame": worst_frame,
        "bad_list": bad
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_root", type=str, required=True,
                    help="DAVIS Annotations root, e.g. /.../DAVIS/Annotations/480p")
    ap.add_argument("--pred_root", type=str, required=True,
                    help="Your output root, e.g. /.../davis_background_track")
    ap.add_argument("--seq", type=str, default="",
                    help="Optional: single sequence name, e.g. bear. Empty = all seqs in gt_root.")
    ap.add_argument("--print_bad_frames", action="store_true",
                    help="Print per-frame violations.")
    ap.add_argument("--topk", type=int, default=10,
                    help="Show top-k worst frames per seq when printing bad frames.")
    args = ap.parse_args()

    gt_root = Path(args.gt_root)
    pred_root = Path(args.pred_root)
    assert gt_root.exists(), f"gt_root not found: {gt_root}"
    assert pred_root.exists(), f"pred_root not found: {pred_root}"

    if args.seq:
        seqs = [args.seq]
    else:
        seqs = sorted([p.name for p in gt_root.iterdir() if p.is_dir()])

    total_bad = 0
    total_checked = 0

    for s in seqs:
        gt_seq = gt_root / s
        pred_seq = pred_root / s
        if not gt_seq.exists():
            continue
        if not pred_seq.exists():
            print(f"[SKIP] pred missing for seq={s}: {pred_seq}")
            continue

        res = check_one_seq(gt_seq, pred_seq, verbose=args.print_bad_frames)
        if res is None:
            print(f"[SKIP] no GT frames for seq={s}")
            continue

        total_bad += res["num_bad"]
        total_checked += res["num_checked"]

        print(f"[SEQ] {s}: checked={res['num_checked']}/{res['num_gt']}  "
              f"bad_frames={res['num_bad']}  mean_overlap={res['mean_ratio']:.4f}  "
              f"max_overlap={res['max_ratio']:.4f}  worst={res['worst_frame']}")

        if args.print_bad_frames and res["num_bad"] > 0:
            # sort by ratio desc
            bad_sorted = sorted(res["bad_list"], key=lambda x: x[3], reverse=True)[:args.topk]
            print("  top bad frames:")
            for stem, inter, fg_area, ratio in bad_sorted:
                print(f"    {stem}: overlap={inter} fg_area={fg_area} ratio={ratio:.4f}")

    print(f"\n[ALL] total_checked={total_checked} total_bad_frames={total_bad}")

if __name__ == "__main__":
    main()
