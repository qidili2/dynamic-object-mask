#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MoCA mask-vs-annotation bbox IoU evaluation.

- Reads VIA-style annotation CSV (like the one you provided).
- For each annotated frame, loads corresponding grayscale mask PNG:
    {mask_root}/{seq}/{frame}.png
  where seq & frame are derived from annotation file_list like:
    /arabian_horn_viper/00000.jpg  -> seq=arabian_horn_viper, frame=00000
- Computes bbox from mask as the tight bounding box of all non-zero pixels
  (because mask is grayscale with multiple values).
- Computes IoU between (mask bbox) and (annotation bbox).
- Automatically skips missing sequences (no seq folder under mask_root) and missing frames.

Outputs:
  - per-frame results CSV
  - per-sequence summary CSV
  - a short text summary printed to stdout

Usage example:
  python eval_moca_bbox_iou.py \
    --ann_csv /path/to/annotations.csv \
    --mask_root /path/to/MoCA/Annotations \
    --out_csv /path/to/out/frame_iou.csv

Notes:
  - Annotation bbox comes from spatial_coordinates: [2, x, y, w, h] (RECTANGLE).
  - Mask bbox uses pixel coordinates (x_min, y_min, x_max, y_max) where x_max/y_max are exclusive.
"""

import os
import ast
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from PIL import Image


def _find_data_start_and_header(csv_path: Path) -> Tuple[int, List[str]]:
    """
    VIA export has comment lines starting with '#'.
    It often includes: "# CSV_HEADER = metadata_id,file_list,flags,temporal_coordinates,spatial_coordinates,metadata,,,"

    Returns:
      data_start_line_idx: 0-based line index of the first data row (not starting with '#')
      headers: list of column names we will read from the CSV
    """
    headers = None
    data_start = None
    with csv_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if s.startswith("# CSV_HEADER"):
                # e.g. "# CSV_HEADER = metadata_id,file_list,flags,temporal_coordinates,spatial_coordinates,metadata,,,"
                if "=" in s:
                    after = s.split("=", 1)[1].strip()
                    # remove trailing commas and empty columns
                    cols = [c.strip() for c in after.split(",") if c.strip() != ""]
                    if cols:
                        headers = cols
            if data_start is None and (not s.startswith("#")) and s != "":
                data_start = i
                break
    if data_start is None:
        raise RuntimeError(f"Cannot find data rows in {csv_path}")
    if headers is None:
        # fallback for common VIA format
        headers = [
            "metadata_id",
            "file_list",
            "flags",
            "temporal_coordinates",
            "spatial_coordinates",
            "metadata",
        ]
    return data_start, headers


def load_annotations(csv_path: Path) -> pd.DataFrame:
    """
    Load and parse annotations:
      columns used: file_list, spatial_coordinates
    Output dataframe columns:
      seq, frame, gt_x, gt_y, gt_w, gt_h
    """
    data_start, headers = _find_data_start_and_header(csv_path)
    ncols = len(headers)

    df = pd.read_csv(csv_path, skiprows=data_start, header=None, usecols=list(range(ncols)))
    df.columns = headers[:ncols]

    # keep rows that have spatial_coordinates
    if "file_list" not in df.columns or "spatial_coordinates" not in df.columns:
        raise RuntimeError(f"CSV does not contain required columns. Found: {list(df.columns)}")

    df = df[["file_list", "spatial_coordinates"]].dropna()

    # parse spatial_coordinates: "[2,x,y,w,h]"
    def parse_spatial(s: str):
        try:
            arr = ast.literal_eval(s)
            if not (isinstance(arr, (list, tuple)) and len(arr) >= 5):
                return None
            if int(arr[0]) != 2:
                return None
            x, y, w, h = float(arr[1]), float(arr[2]), float(arr[3]), float(arr[4])
            return x, y, w, h
        except Exception:
            return None

    parsed = df["spatial_coordinates"].map(parse_spatial)
    df = df[parsed.notna()].copy()
    df[["gt_x", "gt_y", "gt_w", "gt_h"]] = pd.DataFrame(parsed.dropna().tolist(), index=df.index)

    # derive seq and frame from file_list: "/seq_name/00000.jpg"
    def parse_file_list(p: str):
        p = str(p).lstrip("/")  # remove leading '/'
        parts = Path(p).parts
        if len(parts) < 2:
            return None, None
        seq = parts[0]
        frame = Path(parts[-1]).stem  # "00000"
        return seq, frame

    seq_frame = df["file_list"].map(parse_file_list)
    df[["seq", "frame"]] = pd.DataFrame(seq_frame.tolist(), index=df.index)
    df = df.dropna(subset=["seq", "frame"])

    # If multiple bboxes exist for same (seq,frame), merge them by union bbox.
    # (More robust than picking one row.)
    def union_group(g: pd.DataFrame) -> pd.Series:
        x1 = g["gt_x"].min()
        y1 = g["gt_y"].min()
        x2 = (g["gt_x"] + g["gt_w"]).max()
        y2 = (g["gt_y"] + g["gt_h"]).max()
        return pd.Series({"gt_x": x1, "gt_y": y1, "gt_w": x2 - x1, "gt_h": y2 - y1})

    merged = df.groupby(["seq", "frame"], as_index=False).apply(union_group).reset_index(drop=True)
    return merged[["seq", "frame", "gt_x", "gt_y", "gt_w", "gt_h"]]


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    mask: HxW grayscale or HxWxC. Non-zero pixels are treated as foreground.
    Returns bbox in (x1, y1, x2, y2) with x2/y2 exclusive, or None if no foreground.
    """
    if mask.ndim == 3:
        mask = mask[..., 0]
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


def bbox_xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return x, y, x + w, y + h


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_csv", type=str, required=True, help="annotation CSV path (VIA export)")
    ap.add_argument("--mask_root", type=str, required=True, help="root dir that contains seq folders of PNG masks")
    ap.add_argument("--out_csv", type=str, required=True, help="output per-frame CSV")
    ap.add_argument("--out_seq_csv", type=str, default=None, help="output per-seq summary CSV (default: beside out_csv)")
    ap.add_argument("--mask_ext", type=str, default=".png", help="mask extension (default .png)")
    ap.add_argument("--verbose", action="store_true", help="print skipped sequence names")
    args = ap.parse_args()

    ann_csv = Path(args.ann_csv)
    mask_root = Path(args.mask_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_seq_csv = Path(args.out_seq_csv) if args.out_seq_csv else out_csv.with_name(out_csv.stem + "_per_seq.csv")

    ann = load_annotations(ann_csv)

    # pre-check missing seq
    seqs = sorted(ann["seq"].unique().tolist())
    missing_seqs = [s for s in seqs if not (mask_root / s).exists()]
    present_seqs = [s for s in seqs if (mask_root / s).exists()]

    if args.verbose and missing_seqs:
        print(f"[WARN] Missing {len(missing_seqs)} sequences under mask_root, will skip. Examples: {missing_seqs[:10]}")

    rows = []
    missing_frame_cnt = 0
    empty_mask_cnt = 0
    eval_cnt = 0

    for _, r in ann.iterrows():
        seq = r["seq"]
        frame = r["frame"]
        gt_xyxy = bbox_xywh_to_xyxy(r["gt_x"], r["gt_y"], r["gt_w"], r["gt_h"])

        seq_dir = mask_root / seq
        if not seq_dir.exists():
            # auto skip missing sequence
            rows.append({
                "seq": seq, "frame": frame,
                "status": "missing_seq",
                "iou": np.nan,
                "gt_x": r["gt_x"], "gt_y": r["gt_y"], "gt_w": r["gt_w"], "gt_h": r["gt_h"],
            })
            continue

        mask_path = seq_dir / f"{frame}{args.mask_ext}"
        if not mask_path.exists():
            missing_frame_cnt += 1
            rows.append({
                "seq": seq, "frame": frame,
                "status": "missing_frame",
                "iou": np.nan,
                "gt_x": r["gt_x"], "gt_y": r["gt_y"], "gt_w": r["gt_w"], "gt_h": r["gt_h"],
            })
            continue

        mask = np.array(Image.open(mask_path))
        pred_bbox = mask_to_bbox(mask)
        if pred_bbox is None:
            empty_mask_cnt += 1
            rows.append({
                "seq": seq, "frame": frame,
                "status": "empty_mask",
                "iou": 0.0,
                "gt_x": r["gt_x"], "gt_y": r["gt_y"], "gt_w": r["gt_w"], "gt_h": r["gt_h"],
                "pred_x1": np.nan, "pred_y1": np.nan, "pred_x2": np.nan, "pred_y2": np.nan,
            })
            continue

        pred_xyxy = tuple(float(v) for v in pred_bbox)
        iou = iou_xyxy(gt_xyxy, pred_xyxy)
        eval_cnt += 1
        rows.append({
            "seq": seq, "frame": frame,
            "status": "ok",
            "iou": iou,
            "gt_x": r["gt_x"], "gt_y": r["gt_y"], "gt_w": r["gt_w"], "gt_h": r["gt_h"],
            "pred_x1": pred_xyxy[0], "pred_y1": pred_xyxy[1], "pred_x2": pred_xyxy[2], "pred_y2": pred_xyxy[3],
        })

    res = pd.DataFrame(rows)
    res.to_csv(out_csv, index=False)

    ok = res[res["status"].isin(["ok", "empty_mask"])].copy()  # count empty_mask as evaluated IoU=0
    # if you prefer excluding empty_mask from mean, change filter to status=="ok"
    per_seq = ok.groupby("seq", as_index=False).agg(
        n=("iou", "count"),
        mean_iou=("iou", "mean"),
        median_iou=("iou", "median"),
    ).sort_values("mean_iou", ascending=False)
    per_seq.to_csv(out_seq_csv, index=False)

    overall_mean = float(ok["iou"].mean()) if len(ok) else float("nan")
    overall_median = float(ok["iou"].median()) if len(ok) else float("nan")

    print("==== MoCA BBox IoU Evaluation ====")
    print(f"ann frames (unique seq-frame): {len(ann)}")
    print(f"missing sequences: {len(missing_seqs)}")
    print(f"missing frames: {missing_frame_cnt}")
    print(f"empty masks (no foreground): {empty_mask_cnt}")
    print(f"evaluated frames (ok+empty_mask): {len(ok)}")
    print(f"overall mean IoU: {overall_mean:.4f}")
    print(f"overall median IoU: {overall_median:.4f}")
    print(f"saved per-frame CSV: {out_csv}")
    print(f"saved per-seq   CSV: {out_seq_csv}")


if __name__ == "__main__":
    main()
