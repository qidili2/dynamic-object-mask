#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MoCA: bbox -> crop -> SAM2 -> full-size mask PNG (checkpoint-only interface)
Hybrid prompt strategy (generally better): POINT(center) first, fallback to BOX if point result is bad.

You provide:
  --ann_csv       /path/to/annotations.csv
  --image_root    /path/to/MoCA/JPEGImages
  --sam2_ckpt     /path/to/sam2*.pt
  --out_mask_root /path/to/output_masks

This script will:
  1) parse MoCA annotations.csv -> (seq, frame, union_bbox)
  2) for each annotated frame:
       - crop around bbox with context (CROP_PAD_RATIO)
       - build prompt box in crop coords (BOX_PAD_RATIO)
       - run SAM2 with center-point prompt
       - if mask is too small / empty / bbox-iou too low -> fallback to box prompt
       - (optional) morphology close/dilate
       - paste crop mask back to full-size black canvas and save:
           {out_mask_root}/{seq}/{frame}.png  (binary 0/255 by default)
  3) automatically skip missing seq dirs / missing frames

USAGE:
python gen_sam2_masks_from_bboxes_checkpoint_only.py \
  --ann_csv /mnt/data0/andy/SegAnyMo/MoCA/Annotations/annotations.csv \
  --image_root /mnt/data0/andy/SegAnyMo/MoCA/JPEGImages \
  --sam2_ckpt /mnt/data0/andy/Easi3R/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --out_mask_root /mnt/data0/andy/SegAnyMo/MoCA/Annotations \
  --device cuda --fp16 --overwrite --verbose

Notes:
- We resolve the SAM2 yaml config from the installed `sam2` python package (pkg://sam2 hydra search path).
- If your cluster doesn't expose GPU VRAM classads, pin machines in condor requirements.
"""

import ast
import argparse
import importlib
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from PIL import Image

# -------------------------------
# IMPORTANT KNOBS (edit here)
# -------------------------------
CROP_PAD_RATIO = 0.35      # expand bbox for crop (context). bigger => more room to grow
BOX_PAD_RATIO  = 0.20      # expand the box prompt (bigger mask)
PICK_STRATEGY  = "largest" # "largest" or "best_score"
MORPH_CLOSE_K  = 9         # 0 disables. bigger => fill holes / connect
DILATE_K       = 5         # 0 disables. bigger => grow mask outward
SAVE_MASK_255  = True      # output PNG values: {0,255}. (False => {0,1})

# Prefer SAM2.1 configs if ckpt name contains "sam2.1"
PREFER_SAM2_1_IF_POSSIBLE = True

# ---- Hybrid strategy thresholds (point -> fallback to box) ----
MIN_MASK_AREA_PIX   = 200        # point-mask pixel count too small => fallback
MIN_MASK_AREA_RATIO = 0.001      # point-mask / crop_area too small => fallback
MIN_BBOX_IOU        = 0.10       # mask_bbox IoU with prompt_box too low => fallback


def _find_data_start_and_header(csv_path: Path) -> Tuple[int, List[str]]:
    headers = None
    data_start = None
    with csv_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if s.startswith("# CSV_HEADER") and "=" in s:
                after = s.split("=", 1)[1].strip()
                cols = [c.strip() for c in after.split(",") if c.strip() != ""]
                if cols:
                    headers = cols
            if data_start is None and (not s.startswith("#")) and s != "":
                data_start = i
                break
    if data_start is None:
        raise RuntimeError(f"Cannot find data rows in {csv_path}")
    if headers is None:
        headers = ["metadata_id", "file_list", "flags", "temporal_coordinates", "spatial_coordinates", "metadata"]
    return data_start, headers


def load_annotations(csv_path: Path) -> pd.DataFrame:
    """
    Return unique (seq, frame) with union bbox if multiple bboxes exist in same frame.
    Expect spatial_coordinates like "[2, x, y, w, h]".
    """
    data_start, headers = _find_data_start_and_header(csv_path)
    ncols = len(headers)

    df = pd.read_csv(csv_path, skiprows=data_start, header=None, usecols=list(range(ncols)))
    df.columns = headers[:ncols]
    df = df[["file_list", "spatial_coordinates"]].dropna()

    def parse_spatial(s: str):
        try:
            arr = ast.literal_eval(s)
            if not (isinstance(arr, (list, tuple)) and len(arr) >= 5 and int(arr[0]) == 2):
                return None
            return float(arr[1]), float(arr[2]), float(arr[3]), float(arr[4])
        except Exception:
            return None

    parsed = df["spatial_coordinates"].map(parse_spatial)
    df = df[parsed.notna()].copy()
    df[["x", "y", "w", "h"]] = pd.DataFrame(parsed.dropna().tolist(), index=df.index)

    def parse_file_list(p: str):
        p = str(p).lstrip("/")
        parts = Path(p).parts
        if len(parts) < 2:
            return None, None, None
        seq = parts[0]
        fname = parts[-1]
        frame = Path(fname).stem
        ext = Path(fname).suffix.lower()
        return seq, frame, ext

    sfx = df["file_list"].map(parse_file_list)
    df[["seq", "frame", "img_ext"]] = pd.DataFrame(sfx.tolist(), index=df.index)
    df = df.dropna(subset=["seq", "frame"])

    # Union bbox without groupby.apply (avoid pandas FutureWarning)
    df["x2"] = df["x"] + df["w"]
    df["y2"] = df["y"] + df["h"]

    agg = df.groupby(["seq", "frame"], as_index=False).agg(
        x=("x", "min"),
        y=("y", "min"),
        x2=("x2", "max"),
        y2=("y2", "max"),
        img_ext=("img_ext", lambda s: s.mode().iloc[0] if len(s.mode()) else ".jpg"),
    )
    agg["w"] = agg["x2"] - agg["x"]
    agg["h"] = agg["y2"] - agg["y"]
    return agg[["seq", "frame", "x", "y", "w", "h", "img_ext"]]


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def expand_xywh(x: float, y: float, w: float, h: float, pad_ratio: float):
    pad_x = w * pad_ratio
    pad_y = h * pad_ratio
    return x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y


def xywh_to_xyxy(x: float, y: float, w: float, h: float):
    return x, y, x + w, y + h


def ensure_binary_mask(m: np.ndarray) -> np.ndarray:
    return (m > 0).astype(np.uint8)


def maybe_morph(mask01: np.ndarray, morph_close: int, dilate: int) -> np.ndarray:
    if morph_close <= 0 and dilate <= 0:
        return mask01
    try:
        import cv2
    except Exception:
        print("[WARN] cv2 not available; skipping morphology.")
        return mask01

    m = mask01.astype(np.uint8)
    if morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        m = cv2.dilate(m, k, iterations=1)
    return (m > 0).astype(np.uint8)


def mask_bbox_xyxy(mask01: np.ndarray):
    """Return bbox (x1,y1,x2,y2) in crop pixel coords; x2,y2 are exclusive. None if empty."""
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return (x1, y1, x2, y2)


def bbox_iou_xyxy(a, b) -> float:
    """IoU of two xyxy boxes where x2,y2 are exclusive."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter) / float(union + 1e-6)


def infer_model_size_from_ckpt(ckpt_path: Path) -> str:
    name = ckpt_path.name.lower()
    if "hiera_l" in name or "large" in name or "_l." in name:
        return "l"
    if "hiera_b" in name or "base" in name or "_b." in name:
        return "b"
    if "hiera_s" in name or "small" in name or "_s." in name:
        return "s"
    if "hiera_t" in name or "tiny" in name or "_t." in name:
        return "t"
    return "l"


def resolve_sam2_config_by_ckpt(ckpt_path: Path, model_size: str) -> Tuple[Path, str]:
    """
    Return:
      cfg_path: absolute path to yaml
      cfg_name: hydra config_name relative to pkg://sam2 (MUST start with "configs/...")

    Hydra search path typically includes:
      provider=main, path=pkg://sam2

    We search under:
      <sam2_pkg_dir>/configs/**.yaml
    and prefer:
      configs/sam2.1/*  if ckpt contains "sam2.1"
      configs/sam2/*    otherwise
    """
    try:
        sam2_pkg = importlib.import_module("sam2")
    except Exception as e:
        raise RuntimeError(
            "Cannot import sam2. Please install it (e.g., in SAM2 repo: `pip install -e .`)."
        ) from e

    pkg_dir = Path(sam2_pkg.__file__).resolve().parent
    cfg_root = pkg_dir / "configs"
    if not cfg_root.exists():
        raise FileNotFoundError(f"Cannot find sam2 configs directory: {cfg_root}")

    want_21 = ("sam2.1" in ckpt_path.name.lower()) and PREFER_SAM2_1_IF_POSSIBLE
    size_key = f"hiera_{model_size}"

    yamls = sorted(cfg_root.rglob("*.yaml"))
    if not yamls:
        raise FileNotFoundError(f"No yaml configs found under: {cfg_root}")

    def score(p: Path) -> int:
        s = 0
        rel = str(p.relative_to(pkg_dir)).replace("\\", "/")  # e.g. configs/sam2/sam2_hiera_l.yaml
        low = rel.lower()

        # prefer correct branch
        if want_21:
            if "/sam2.1/" in low:
                s += 50
            if "/sam2/" in low and "/sam2.1/" not in low:
                s -= 10
        else:
            if "/sam2/" in low and "/sam2.1/" not in low:
                s += 50
            if "/sam2.1/" in low:
                s -= 10

        # match size
        if size_key in low:
            s += 30

        # small bonuses
        if "sam2" in p.name.lower():
            s += 3
        return s

    best = sorted(yamls, key=score, reverse=True)[0]
    cfg_name = str(best.relative_to(pkg_dir)).replace("\\", "/")  # MUST be like "configs/...yaml"
    return best, cfg_name


class Sam2Wrapper:
    def __init__(self, cfg_name: str, ckpt_path: Path, device="cuda", fp16=False):
        self.device = device
        self.fp16 = fp16
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as e:
            raise RuntimeError(
                "Failed to import SAM2 runtime modules.\n"
                "Expected:\n"
                "  from sam2.build_sam import build_sam2\n"
                "  from sam2.sam2_image_predictor import SAM2ImagePredictor\n"
            ) from e

        self.torch = __import__("torch")

        # Hydra compose wants a config_name IN the search path (pkg://sam2),
        # so pass something like "configs/sam2/sam2_hiera_l" (or with .yaml).
        cfg_try = [cfg_name, cfg_name.replace(".yaml", "")]
        last_err = None
        for cn in cfg_try:
            try:
                self.model = build_sam2(config_file=cn, ckpt_path=str(ckpt_path), device=device)
                last_err = None
                break
            except Exception as e:
                last_err = e
        if last_err is not None:
            raise last_err

        self.predictor = SAM2ImagePredictor(self.model)

    def predict_mask_from_point(self, image_rgb: np.ndarray, point_xy: np.ndarray, pick: str = "largest") -> np.ndarray:
        torch = self.torch
        point_coords = point_xy.reshape(1, 2).astype(np.float32)
        point_labels = np.array([1], dtype=np.int64)

        with torch.inference_mode():
            if self.fp16 and self.device.startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.float16):
                    self.predictor.set_image(image_rgb)
                    masks, scores, _ = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True,
                    )
            else:
                self.predictor.set_image(image_rgb)
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )

        if masks is None or len(masks) == 0:
            return np.zeros(image_rgb.shape[:2], dtype=np.uint8)

        if pick == "largest":
            areas = np.array([m.astype(np.uint8).sum() for m in masks], dtype=np.int64)
            idx = int(areas.argmax())
        else:
            idx = int(np.array(scores).argmax())
        return masks[idx].astype(np.uint8)

    def predict_mask_from_box(self, image_rgb: np.ndarray, box_xyxy: np.ndarray, pick: str = "largest") -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            if self.fp16 and self.device.startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.float16):
                    self.predictor.set_image(image_rgb)
                    masks, scores, _ = self.predictor.predict(
                        box=box_xyxy[None, :],
                        multimask_output=True,
                    )
            else:
                self.predictor.set_image(image_rgb)
                masks, scores, _ = self.predictor.predict(
                    box=box_xyxy[None, :],
                    multimask_output=True,
                )

        if masks is None or len(masks) == 0:
            return np.zeros(image_rgb.shape[:2], dtype=np.uint8)

        if pick == "largest":
            areas = np.array([m.astype(np.uint8).sum() for m in masks], dtype=np.int64)
            idx = int(areas.argmax())
        else:
            idx = int(np.array(scores).argmax())
        return masks[idx].astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_csv", type=str, required=True)
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--sam2_ckpt", type=str, required=True)
    ap.add_argument("--out_mask_root", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--model_size",
        type=str,
        default="auto",
        choices=["auto", "t", "s", "b", "l"],
        help="Override model size if auto-infer fails. Default auto.",
    )
    args = ap.parse_args()

    ann = load_annotations(Path(args.ann_csv))
    image_root = Path(args.image_root)
    out_root = Path(args.out_mask_root)
    out_root.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.sam2_ckpt)
    ms = infer_model_size_from_ckpt(ckpt_path) if args.model_size == "auto" else args.model_size

    cfg_path, cfg_name = resolve_sam2_config_by_ckpt(ckpt_path, ms)
    if args.verbose:
        print(f"[INFO] Using SAM2 model_size={ms}")
        print(f"[INFO] cfg_name={cfg_name}")
        print(f"[INFO] cfg_path={cfg_path}")
        print(f"[INFO] ckpt={ckpt_path}")

    sam2 = Sam2Wrapper(cfg_name=cfg_name, ckpt_path=ckpt_path, device=args.device, fp16=args.fp16)

    present_seq_dirs = {p.name for p in image_root.iterdir() if p.is_dir()}
    missing_seq = 0
    missing_frame = 0
    processed = 0
    saved = 0
    fallback_count = 0

    for _, r in ann.iterrows():
        if args.limit > 0 and processed >= args.limit:
            break

        seq = r["seq"]
        frame = r["frame"]
        img_ext = r["img_ext"] if isinstance(r["img_ext"], str) else ".jpg"

        if seq not in present_seq_dirs:
            missing_seq += 1
            continue

        img_path = image_root / seq / f"{frame}{img_ext}"
        if not img_path.exists():
            for ext_try in [".jpg", ".jpeg", ".png"]:
                alt = image_root / seq / f"{frame}{ext_try}"
                if alt.exists():
                    img_path = alt
                    break
            if not img_path.exists():
                missing_frame += 1
                continue

        out_path = out_root / seq / f"{frame}.png"
        if out_path.exists() and (not args.overwrite):
            processed += 1
            continue

        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        x, y, w, h = float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])

        # crop bbox (padded)
        cx, cy, cw, ch = expand_xywh(x, y, w, h, CROP_PAD_RATIO)
        cx1, cy1, cx2, cy2 = xywh_to_xyxy(cx, cy, cw, ch)
        ix1 = clamp(int(np.floor(cx1)), 0, W)
        iy1 = clamp(int(np.floor(cy1)), 0, H)
        ix2 = clamp(int(np.ceil(cx2)), 0, W)
        iy2 = clamp(int(np.ceil(cy2)), 0, H)
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        crop = img.crop((ix1, iy1, ix2, iy2))
        crop_np = np.array(crop)
        Hc, Wc = crop_np.shape[:2]

        # prompt box (expanded) in crop coords
        bx, by, bw, bh = expand_xywh(x, y, w, h, BOX_PAD_RATIO)
        bx1, by1, bx2, by2 = xywh_to_xyxy(bx, by, bw, bh)

        pbx1 = clamp(int(np.floor(bx1 - ix1)), 0, Wc)
        pby1 = clamp(int(np.floor(by1 - iy1)), 0, Hc)
        pbx2 = clamp(int(np.ceil(bx2 - ix1)), 0, Wc)
        pby2 = clamp(int(np.ceil(by2 - iy1)), 0, Hc)
        if pbx2 <= pbx1 or pby2 <= pby1:
            pbx1, pby1, pbx2, pby2 = 0, 0, Wc, Hc

        prompt_box = (int(pbx1), int(pby1), int(pbx2), int(pby2))
        box_np = np.array([float(pbx1), float(pby1), float(pbx2), float(pby2)], dtype=np.float32)

        # ---- 1) point prompt at bbox center ----
        center_x = x + 0.5 * w
        center_y = y + 0.5 * h
        pcx = clamp(int(round(center_x - ix1)), 0, Wc - 1)
        pcy = clamp(int(round(center_y - iy1)), 0, Hc - 1)
        point = np.array([float(pcx), float(pcy)], dtype=np.float32)

        pred_crop = sam2.predict_mask_from_point(crop_np, point, pick=PICK_STRATEGY)
        pred01 = ensure_binary_mask(pred_crop)

        # quality check BEFORE morph
        area = int(pred01.sum())
        area_ratio = area / float(Hc * Wc + 1e-6)
        mb = mask_bbox_xyxy(pred01)

        ok = True
        if mb is None or area < MIN_MASK_AREA_PIX or area_ratio < MIN_MASK_AREA_RATIO:
            ok = False
        else:
            iou = bbox_iou_xyxy(mb, prompt_box)
            if iou < MIN_BBOX_IOU:
                ok = False

        # ---- 2) fallback to box prompt if needed ----
        if not ok:
            fallback_count += 1
            pred_crop = sam2.predict_mask_from_box(crop_np, box_np, pick=PICK_STRATEGY)
            pred01 = ensure_binary_mask(pred_crop)

        # postprocess
        pred01 = maybe_morph(pred01, MORPH_CLOSE_K, DILATE_K)

        # paste back to full size
        full = np.zeros((H, W), dtype=np.uint8)
        full[iy1:iy2, ix1:ix2] = pred01[: (iy2 - iy1), : (ix2 - ix1)]

        if SAVE_MASK_255:
            full = (full > 0).astype(np.uint8) * 255

        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(full).save(out_path)

        processed += 1
        saved += 1
        if args.verbose and processed % 50 == 0:
            print(
                f"[INFO] processed={processed} saved={saved} "
                f"(fallback={fallback_count}, missing_seq={missing_seq}, missing_frame={missing_frame})"
            )

    print("==== SAM2 mask generation done ====")
    print(f"total annotated frames: {len(ann)}")
    print(f"processed (saved or existed): {processed}")
    print(f"saved newly: {saved}")
    print(f"fallback count (point->box): {fallback_count}")
    print(f"skipped missing seq: {missing_seq}")
    print(f"skipped missing frame: {missing_frame}")
    print(f"output mask root: {out_root}")


if __name__ == "__main__":
    main()
