#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MoCA: bbox -> crop -> SAM2 -> full-size mask PNG (checkpoint-only interface)

You ONLY provide:
  --sam2_ckpt  /path/to/sam2_*.pt

This script will:
  1) infer which SAM2 YAML config to use from the checkpoint filename (or you can override)
  2) locate that YAML inside your installed `sam2` package (sam2/configs/sam2/*.yaml)
  3) run SAM2 with an expanded crop + expanded box prompt to deliberately produce a "bigger mask"
  4) paste the crop mask back into a full-size black canvas and save:
        {out_mask_root}/{seq}/{frame}.png

Then you can directly reuse your existing bbox-IoU eval script by pointing its --mask_root
to out_mask_root.

------------------------------------------------------------
IMPORTANT KNOBS (edit here if you want, no CLI needed)
------------------------------------------------------------
CROP_PAD_RATIO = 0.35   # expand bbox for crop (context). bigger => more room to grow
BOX_PAD_RATIO  = 0.20   # expand the box prompt (bigger mask)
PICK_STRATEGY  = "largest"  # "largest" or "best_score" from SAM2 multimask outputs
MORPH_CLOSE_K  = 9      # 0 disables. bigger => fill holes / connect
DILATE_K       = 5      # 0 disables. bigger => grow mask outward

SAVE_MASK_255  = True   # output PNG values: {0,255}. (Set False to save {0,1})

------------------------------------------------------------
USAGE
------------------------------------------------------------
python gen_sam2_masks_from_bboxes_checkpoint_only.py \
  --ann_csv /path/to/annotations.csv \
  --image_root /path/to/MoCA/JPEGImages \
  --sam2_ckpt /path/to/sam2_hiera_l.pt \
  --out_mask_root /path/to/masks_sam2_from_bbox \
  --device cuda --fp16 \
  --verbose

Optional override if auto-infer fails:
  --model_size l   # one of: t, s, b, l

Notes:
- Automatically skips missing seq dirs and missing frames.
- Mask foreground is defined as >0 in SAM2 output. We save binary masks.

"""

import ast
import argparse
import importlib
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

# -------------------------------
# Editable "important parameters"
# -------------------------------
CROP_PAD_RATIO = 0.35
BOX_PAD_RATIO  = 0.20
PICK_STRATEGY  = "largest"   # "largest" or "best_score"
MORPH_CLOSE_K  = 9
DILATE_K       = 5
SAVE_MASK_255  = True


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

    def union_group(g: pd.DataFrame) -> pd.Series:
        x1 = g["x"].min()
        y1 = g["y"].min()
        x2 = (g["x"] + g["w"]).max()
        y2 = (g["y"] + g["h"]).max()
        img_ext = g["img_ext"].mode().iloc[0] if len(g["img_ext"].mode()) else ".jpg"
        return pd.Series({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "img_ext": img_ext})

    merged = df.groupby(["seq", "frame"], as_index=False).apply(union_group).reset_index(drop=True)
    return merged[["seq", "frame", "x", "y", "w", "h", "img_ext"]]


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def expand_xywh(x: float, y: float, w: float, h: float, pad_ratio: float):
    pad_x = w * pad_ratio
    pad_y = h * pad_ratio
    return x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y


def xywh_to_xyxy(x: float, y: float, w: float, h: float):
    return x, y, x + w, y + h


def ensure_binary_mask(m: np.ndarray) -> np.ndarray:
    """Returns uint8 {0,1} mask."""
    return (m > 0).astype(np.uint8)


def maybe_morph(mask01: np.ndarray, morph_close: int, dilate: int) -> np.ndarray:
    """
    Optional post-processing (expects 0/1 uint8):
      - close
      - dilate
    """
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


def infer_model_size_from_ckpt(ckpt_path: Path) -> Optional[str]:
    name = ckpt_path.name.lower()
    # common patterns
    if "hiera_l" in name or "large" in name or "_l." in name:
        return "l"
    if "hiera_b" in name or "base" in name or "_b." in name:
        return "b"
    if "hiera_s" in name or "small" in name or "_s." in name:
        return "s"
    if "hiera_t" in name or "tiny" in name or "_t." in name:
        return "t"
    return None


def resolve_sam2_config_path(model_size: str) -> Path:
    """
    Locate YAML under installed sam2 package:
      sam2/configs/sam2/sam2_hiera_{size}.yaml
    """
    try:
        sam2_pkg = importlib.import_module("sam2")
    except Exception as e:
        raise RuntimeError("Cannot import sam2. Please install it (e.g., `pip install -e .` in SAM2 repo).") from e

    pkg_dir = Path(sam2_pkg.__file__).resolve().parent
    cfg = pkg_dir / "configs" / "sam2" / f"sam2_hiera_{model_size}.yaml"
    if cfg.exists():
        return cfg

    # Fallback search: find any yaml that matches model_size
    cfg_dir = pkg_dir / "configs" / "sam2"
    if cfg_dir.exists():
        candidates = sorted(cfg_dir.glob(f"*{model_size}*.yaml"))
        if candidates:
            return candidates[0]

    raise FileNotFoundError(
        f"Cannot find SAM2 config yaml inside installed package.\n"
        f"Tried: {cfg}\n"
        f"Package dir: {pkg_dir}\n"
        "Tip: install from the official repo with `pip install -e .` so configs are present."
    )


class Sam2Wrapper:
    def __init__(self, cfg_path: Path, ckpt_path: Path, device: str = "cuda", fp16: bool = False):
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
        # build_sam2 expects config file name or path, depending on repo version.
        # Passing absolute path is the most robust.
        self.model = build_sam2(config_file=str(cfg_path), ckpt_path=str(ckpt_path), device=device)
        self.predictor = SAM2ImagePredictor(self.model)

    def predict_mask_from_box(self, image_rgb: np.ndarray, box_xyxy: np.ndarray, pick: str = "largest") -> np.ndarray:
        torch = self.torch
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

    # optional override
    ap.add_argument("--model_size", type=str, default="auto", choices=["auto", "t", "s", "b", "l"],
                    help="Override model size if auto-infer fails. Default auto.")

    args = ap.parse_args()

    ann = load_annotations(Path(args.ann_csv))
    image_root = Path(args.image_root)
    out_root = Path(args.out_mask_root)
    out_root.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.sam2_ckpt)

    if args.model_size == "auto":
        ms = infer_model_size_from_ckpt(ckpt_path) or "l"
    else:
        ms = args.model_size

    cfg_path = resolve_sam2_config_path(ms)
    if args.verbose:
        print(f"[INFO] Using SAM2 model_size={ms} cfg={cfg_path} ckpt={ckpt_path}")

    sam2 = Sam2Wrapper(cfg_path=cfg_path, ckpt_path=ckpt_path, device=args.device, fp16=args.fp16)

    present_seq_dirs = {p.name for p in image_root.iterdir() if p.is_dir()}
    missing_seq = 0
    missing_frame = 0
    processed = 0

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

        box = np.array([float(pbx1), float(pby1), float(pbx2), float(pby2)], dtype=np.float32)

        pred_crop = sam2.predict_mask_from_box(crop_np, box, pick=PICK_STRATEGY)
        pred01 = ensure_binary_mask(pred_crop)
        pred01 = maybe_morph(pred01, MORPH_CLOSE_K, DILATE_K)

        full = np.zeros((H, W), dtype=np.uint8)
        full[iy1:iy2, ix1:ix2] = pred01[: (iy2 - iy1), : (ix2 - ix1)]

        if SAVE_MASK_255:
            full = (full > 0).astype(np.uint8) * 255

        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(full).save(out_path)

        processed += 1
        if args.verbose and processed % 50 == 0:
            print(f"[INFO] processed={processed} (missing_seq={missing_seq}, missing_frame={missing_frame})")

    print("==== SAM2 mask generation done ====")
    print(f"total annotated frames: {len(ann)}")
    print(f"processed (saved or existed): {processed}")
    print(f"skipped missing seq: {missing_seq}")
    print(f"skipped missing frame: {missing_frame}")
    print(f"output mask root: {out_root}")


if __name__ == "__main__":
    main()
