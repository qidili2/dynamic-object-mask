#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TextRegion (two-group labels) + SAM2 video tracking, BUT track BACKGROUND only.

Pipeline per sequence:
1) Run TextRegionSegmenter.py on first frame (subprocess) with labels = MOVING_LABELS + BACKGROUND_LABELS
2) Load TextRegion outputs: pred.npy (+ optional union.npy)
3) Build background init mask using two-group logic:
     - mode="union_not_moving" (default, robust): BG = union & (NOT moving)
       (optionally OR bg_labels in)
     - mode="bg_labels": BG = union & bg_label_regions
4) Optional mild cleanup (keep more background): closing + fill holes, no CC selection
5) Run SAM2 video predictor to propagate mask over video frames
6) Save per-frame masks (png+npy) and tracks.json

Works for:
- single seq: --frames_dir <seq_folder>
- dataset:   --dataset_root <root_with_seq_folders> [--seq_list list.txt]
"""

import os
import sys
import json
import glob
import time
import yaml
import shutil
import argparse
import subprocess
from pathlib import Path

import numpy as np
import cv2


# -------------------------
# Two-group label settings
# -------------------------
MOVING_LABELS = ["moving object", "human", "animal", "vehicle", "sport"]

BACKGROUND_LABELS = [
    "ground", "road", "sidewalk", "floor",
    "grass", "tree", "bush", "plant",
    "water", "river", "lake", "sea",
    "sky", "cloud",
    "wall", "building", "house", "window", "door",
    "mountain", "rock", "sand", "bridge", "fence", "rail"
]

# If none of BACKGROUND_LABELS hit, you can still get BG via union_not_moving.
BACKGROUND_FALLBACK = ["background", "scene", "outdoor", "indoor"]


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def list_frames(frames_dir: Path):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    frames = []
    for ext in exts:
        frames += glob.glob(str(frames_dir / ext))
    frames = sorted(frames)
    return [Path(p) for p in frames]


def write_yaml_for_image(img_path: Path, yml_path: Path, labels):
    data = {str(img_path): {"label": labels}}
    yml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def run_textregion_segmenter(
    textregion_repo: Path,
    image_path: Path,
    yml_path: Path,
    points_per_side=12,
    resize_method="multi_resolution",
    crop_size=336,
    dtype="bf16",
    sam2_ckpt: Path | None = None,
):
    """
    Run TextRegionSegmenter.py in a subprocess to avoid Hydra conflicts in the current process.
    """
    seg_py = textregion_repo / "TextRegionSegmenter.py"
    assert seg_py.exists(), f"TextRegionSegmenter.py not found: {seg_py}"

    cmd = [
        sys.executable, str(seg_py),
        "--image_list", str(image_path),
        "--image_query_cfg", str(yml_path),
        "--resize_method", str(resize_method),
        "--crop_size", str(crop_size),
        "--points_per_side", str(points_per_side),
        "--dtype", str(dtype),
        "--viz_regions", "True",
        "--dump_region_labels", "True",
    ]
    if sam2_ckpt is not None and sam2_ckpt.exists():
        cmd += ["--sam2_checkpoint", str(sam2_ckpt)]

    print("[TextRegion RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(textregion_repo))


def load_textregion_pred_and_union(textregion_repo: Path, seq_name: str, stem: str):
    """
    TextRegion outputs under <repo>/outputs/<seq>/<stem>_pred.npy and optional <stem>_union.npy
    """
    out_dir = textregion_repo / "outputs" / seq_name
    pred_path = out_dir / f"{stem}_pred.npy"
    union_path = out_dir / f"{stem}_union.npy"

    if not pred_path.exists():
        raise FileNotFoundError(f"TextRegion pred not found: {pred_path}")

    pred = np.load(pred_path).astype(np.int32)

    union = None
    if union_path.exists():
        union = np.load(union_path)
        union = (union > 0).astype(np.uint8)

    return pred, union


def _norm_list(xs):
    return [str(x).strip().lower() for x in xs]


def build_background_mask_two_groups(pred: np.ndarray,
                                     labels_in_yaml,
                                     union: np.ndarray | None,
                                     mode: str = "union_not_moving",
                                     or_bg_labels: bool = True):
    """
    Build BACKGROUND init mask using two-group label logic.

    mode:
      - "bg_labels":        BG = union & (pred in BACKGROUND_LABELS)
      - "union_not_moving": BG = union & (pred NOT in MOVING_LABELS)   (recommended)

    or_bg_labels:
      - if True and mode=="union_not_moving": BG = BG | (bg_labels & union)
        (helps when moving is weak/empty and bg labels cover important areas)
    """
    labels_low = _norm_list(labels_in_yaml)
    label_to_id = {name: i for i, name in enumerate(labels_low)}

    moving_ids = [label_to_id[n.lower()] for n in MOVING_LABELS if n.lower() in label_to_id]
    bg_ids = [label_to_id[n.lower()] for n in BACKGROUND_LABELS if n.lower() in label_to_id]

    # fallback bg keywords
    if not bg_ids:
        bg_ids = [label_to_id[n.lower()] for n in BACKGROUND_FALLBACK if n.lower() in label_to_id]

    moving = np.isin(pred, moving_ids).astype(np.uint8) if moving_ids else np.zeros_like(pred, dtype=np.uint8)
    bg_by_label = np.isin(pred, bg_ids).astype(np.uint8) if bg_ids else np.zeros_like(pred, dtype=np.uint8)

    if union is not None:
        if union.shape != pred.shape:
            union = cv2.resize(union.astype(np.uint8), (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        u = (union > 0).astype(np.uint8)
    else:
        u = np.ones_like(pred, dtype=np.uint8)

    if mode == "bg_labels":
        bg = (bg_by_label & u).astype(np.uint8)

    elif mode == "union_not_moving":
        bg = ((1 - moving) & u).astype(np.uint8)
        if or_bg_labels and bg_by_label is not None:
            bg = (bg | (bg_by_label & u)).astype(np.uint8)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return bg


def clean_bg_mask_keep_more(mask01: np.ndarray,
                            close_ksize: int = 11,
                            fill_holes: bool = True,
                            min_area: int = 0):
    """
    Keep more background:
      - NO CC selection
      - Optional closing (connect/smooth)
      - Optional fill holes
      - Optional remove tiny speckles only (min_area=0 disables)
    """
    m = (mask01 > 0).astype(np.uint8)

    if close_ksize and close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

    if fill_holes:
        h, w = m.shape
        flood = m.copy()
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, mask, (0, 0), 1)          # fill outside
        holes = (flood == 0).astype(np.uint8)          # holes are inside zeros
        m = (m | holes).astype(np.uint8)

    if min_area and min_area > 0:
        num, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros_like(m)
        for lab in range(1, num):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area >= min_area:
                keep[cc == lab] = 1
        m = keep

    return m


def mask_to_bbox(mask01: np.ndarray):
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1, y1]


def make_clean_frames_dir(frames, clean_dir: Path):
    """
    Create canonical frame dir for SAM2:
      _clean_frames/00000.jpg ...  (symlink/copy)

    Stabilizes internal reading/sorting and avoids surprises.
    """
    clean_dir.mkdir(parents=True, exist_ok=True)
    for p in clean_dir.glob("*"):
        p.unlink()

    kept = 0
    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] unreadable frame, skip: {fp}")
            continue
        dst = clean_dir / f"{i:05d}.jpg"
        try:
            os.symlink(str(fp), str(dst))
        except Exception:
            shutil.copy2(str(fp), str(dst))
        kept += 1

    if kept == 0:
        raise RuntimeError(f"No readable frames for {frames[0].parent}")
    return clean_dir


def build_sam2_video_predictor(sam2_repo: Path, cfg_name: str, ckpt_path: Path, device="cuda"):
    """
    Build SAM2 video predictor from sam2_repo (TextRegion root is OK if it contains sam2/).

    cfg_name should be config NAME: e.g., sam2_hiera_l
    (Do NOT pass absolute yaml path; config name avoids Hydra global init mess.)
    """
    sys.path.insert(0, str(sam2_repo))
    try:
        from sam2.build_sam import build_sam2_video_predictor  # type: ignore
    except Exception as ex:
        raise RuntimeError(f"Cannot import sam2.build_sam from sam2_repo={sam2_repo}\nError: {ex}")

    cfg = cfg_name
    if cfg.endswith(".yaml") or cfg.endswith(".yml"):
        cfg = Path(cfg).stem

    predictor = build_sam2_video_predictor(cfg, str(ckpt_path), device=device)
    return predictor


def _torch_to_numpy(x):
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().to("cpu").numpy()
    except Exception:
        pass
    return x


def _to_numpy_mask_2d(mk):
    mk = _torch_to_numpy(mk)
    mk = np.asarray(mk)
    mk = np.squeeze(mk)
    while mk.ndim > 2:
        mk = mk[0]
        mk = np.squeeze(mk)
    if mk.ndim != 2:
        raise RuntimeError(f"Mask shape is not 2D after processing: {mk.shape}")
    mk = (mk > 0).astype(np.uint8)
    mk = np.ascontiguousarray(mk)
    return mk


def sam2_propagate_masks(predictor, frames_dir: Path, init_mask01: np.ndarray, out_dir: Path, obj_id: int = 1):
    out_masks_dir = out_dir / "masks"
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    state = predictor.init_state(video_path=str(frames_dir))
    predictor.add_new_mask(state, frame_idx=0, obj_id=obj_id, mask=(init_mask01 > 0))

    tracks = {}
    for out in predictor.propagate_in_video(state):
        if not isinstance(out, (list, tuple)):
            raise RuntimeError(f"Unexpected propagate output type: {type(out)}; value={out}")

        if len(out) == 2:
            frame_idx, masks = out
            obj_ids = [obj_id]
        elif len(out) == 3:
            frame_idx, obj_ids, masks = out
        elif len(out) == 4:
            frame_idx, obj_ids, masks, _ = out
        else:
            raise RuntimeError(f"Unexpected propagate output length: {len(out)}; value={out}")

        obj_ids = _torch_to_numpy(obj_ids)
        if isinstance(obj_ids, np.ndarray):
            obj_ids_list = obj_ids.tolist()
        elif isinstance(obj_ids, (list, tuple)):
            obj_ids_list = list(obj_ids)
        else:
            obj_ids_list = [obj_id]

        k = obj_ids_list.index(obj_id) if obj_id in obj_ids_list else 0

        # do NOT wrap masks with np.asarray before indexing if it may be a CUDA tensor
        if isinstance(masks, (list, tuple)):
            mk = masks[k]
        else:
            # masks is tensor/ndarray
            try:
                import torch
                if torch.is_tensor(masks):
                    mk = masks[k] if masks.ndim >= 3 else masks
                else:
                    mk = masks
            except Exception:
                mk = masks

        mk = _to_numpy_mask_2d(mk)

        png_path = out_masks_dir / f"{int(frame_idx):05d}.png"
        npy_path = out_masks_dir / f"{int(frame_idx):05d}.npy"

        ok = cv2.imwrite(str(png_path), mk * 255)
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed: {png_path} shape={mk.shape} dtype={mk.dtype}")

        np.save(npy_path, mk)
        tracks[f"{int(frame_idx):05d}"] = {"bbox": mask_to_bbox(mk), "area": int(mk.sum())}

    with open(out_dir / "tracks.json", "w") as f:
        json.dump({"frames_dir": str(frames_dir), "obj_id": obj_id, "tracks": tracks}, f, indent=2)

    print(f"[SAM2] done. saved masks to {out_masks_dir}")


def process_one_seq(args, frames_dir: Path):
    frames = list_frames(frames_dir)
    if not frames:
        eprint(f"[SKIP] no frames found: {frames_dir}")
        return

    seq_name = frames_dir.name
    frame0 = frames[0]
    stem0 = frame0.stem

    out_dir = Path(args.out_root) / seq_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) TextRegion on first frame (subprocess)
    textregion_repo = Path(args.textregion_repo)
    sam2_ckpt = Path(args.sam2_ckpt)

    # two-group labels: moving + bg
    labels = list(dict.fromkeys(MOVING_LABELS + BACKGROUND_LABELS + BACKGROUND_FALLBACK))
    yml_path = out_dir / "_textregion_two_groups.yaml"
    write_yaml_for_image(frame0, yml_path, labels)

    run_textregion_segmenter(
        textregion_repo=textregion_repo,
        image_path=frame0,
        yml_path=yml_path,
        points_per_side=args.points_per_side,
        resize_method=args.resize_method,
        crop_size=args.crop_size,
        dtype=args.dtype,
        sam2_ckpt=sam2_ckpt,
    )

    # 2) Load pred/union
    pred, union = load_textregion_pred_and_union(textregion_repo, seq_name, stem0)

    # align pred to union or image size
    if union is not None:
        target_h, target_w = union.shape[:2]
    else:
        im0 = cv2.imread(str(frame0), cv2.IMREAD_COLOR)
        target_h, target_w = im0.shape[:2]
    if pred.shape != (target_h, target_w):
        pred = cv2.resize(pred.astype(np.int32), (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(np.int32)

    # 3) Build BACKGROUND init mask using two-group logic
    bg0 = build_background_mask_two_groups(
        pred=pred,
        labels_in_yaml=labels,
        union=union,
        mode=args.bg_mode,
        or_bg_labels=(not args.no_or_bg_labels),
    )

    # 4) Optional mild cleanup (keeps more bg; no CC selection)
    if not args.no_bg_cleanup:
        bg0 = clean_bg_mask_keep_more(
            bg0,
            close_ksize=args.close_ksize,
            fill_holes=(not args.no_fill_holes),
            min_area=args.min_area,
        )

    # save init outputs
    init_dir = out_dir / "init"
    init_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(init_dir / "00000_bg_init.png"), bg0 * 255)
    np.save(init_dir / "00000_bg_init.npy", bg0)

    im0 = cv2.imread(str(frame0), cv2.IMREAD_COLOR)
    if im0 is not None and im0.shape[:2] != bg0.shape:
        im0 = cv2.resize(im0, (bg0.shape[1], bg0.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = im0.copy()
    tint = np.zeros_like(overlay); tint[..., 2] = 255
    ys, xs = np.where(bg0 > 0)
    overlay[ys, xs] = (args.overlay_alpha * tint[ys, xs] + (1 - args.overlay_alpha) * overlay[ys, xs]).astype(np.uint8)
    cv2.imwrite(str(init_dir / "00000_bg_overlay.png"), overlay)

    print(f"[BG] init mask area={int(bg0.sum())} bbox={mask_to_bbox(bg0)} saved to {init_dir}")

    # 5) Clean frames dir for SAM2
    clean_video_dir = make_clean_frames_dir(frames, out_dir / "_clean_frames")

    # 6) SAM2 video predictor + propagate
    predictor = build_sam2_video_predictor(
        sam2_repo=Path(args.sam2_repo),
        cfg_name=args.sam2_model_cfg,
        ckpt_path=sam2_ckpt,
        device=args.device,
    )

    sam2_propagate_masks(
        predictor=predictor,
        frames_dir=clean_video_dir,
        init_mask01=bg0,
        out_dir=out_dir,
        obj_id=args.obj_id,
    )


def main():
    p = argparse.ArgumentParser()

    # Input
    p.add_argument("--frames_dir", type=str, default="", help="Single sequence frames directory")
    p.add_argument("--dataset_root", type=str, default="", help="Dataset root that contains multiple seq dirs")
    p.add_argument("--seq_list", type=str, default="", help="Optional txt list of seq names (one per line)")

    # Output
    p.add_argument("--out_root", type=str, required=True)

    # Repos / ckpt
    p.add_argument("--textregion_repo", type=str, required=True)
    p.add_argument("--sam2_repo", type=str, required=True)
    p.add_argument("--sam2_model_cfg", type=str, required=True, help="Config name, e.g., sam2_hiera_l")
    p.add_argument("--sam2_ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")

    # TextRegion params
    p.add_argument("--points_per_side", type=int, default=12)
    p.add_argument("--resize_method", type=str, default="multi_resolution")
    p.add_argument("--crop_size", type=int, default=336)
    p.add_argument("--dtype", type=str, default="bf16")

    # Background build options
    p.add_argument("--bg_mode", type=str, default="union_not_moving",
                   choices=["union_not_moving", "bg_labels"],
                   help="How to define background from two groups.")
    p.add_argument("--no_or_bg_labels", action="store_true",
                   help="Disable OR(bg_labels) in union_not_moving mode (default: enabled).")

    # BG cleanup (keep more)
    p.add_argument("--no_bg_cleanup", action="store_true",
                   help="Disable background cleanup (closing/fill holes).")
    p.add_argument("--close_ksize", type=int, default=11)
    p.add_argument("--no_fill_holes", action="store_true")
    p.add_argument("--min_area", type=int, default=0,
                   help="Remove tiny speckles only; 0 disables.")

    p.add_argument("--overlay_alpha", type=float, default=0.45)

    # SAM2 obj id
    p.add_argument("--obj_id", type=int, default=1)

    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Determine sequences
    seq_dirs = []
    if args.frames_dir:
        seq_dirs = [Path(args.frames_dir)]
    else:
        dataset_root = Path(args.dataset_root)
        assert dataset_root.exists(), f"dataset_root not found: {dataset_root}"

        if args.seq_list:
            seqs = [x.strip() for x in Path(args.seq_list).read_text().splitlines() if x.strip()]
            seq_dirs = [dataset_root / s for s in seqs]
        else:
            seq_dirs = [p for p in sorted(dataset_root.iterdir()) if p.is_dir()]

    print(f"[INFO] total seqs = {len(seq_dirs)}")

    for i, sd in enumerate(seq_dirs):
        try:
            print(f"\n==== [{i+1}/{len(seq_dirs)}] seq: {sd} ====")
            process_one_seq(args, sd)
        except Exception as ex:
            eprint(f"[ERROR] seq {sd.name}: {ex}")

    print("[DONE]")


if __name__ == "__main__":
    main()
