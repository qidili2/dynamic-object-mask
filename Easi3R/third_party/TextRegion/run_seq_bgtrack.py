#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, glob, subprocess, yaml
from pathlib import Path
import numpy as np
import cv2

# -------------------- USER CONFIG (bear) --------------------
FRAMES_DIR = Path("/mnt/data0/andy/Easi3R/DAVIS/JPEGImages/480p/bear")
OUT_ROOT   = Path("/mnt/data0/andy/Easi3R/third_party/TextRegion/davis_background_track_bear_debug")

TEXTREGION_REPO = Path("/mnt/data0/andy/Easi3R/third_party/TextRegion")
SAM2_REPO       = Path("/mnt/data0/andy/Easi3R/third_party/TextRegion")  # contains sam2/
SAM2_CFG_NAME   = "sam2_hiera_l"
SAM2_CKPT       = Path("/mnt/data0/andy/Easi3R/third_party/sam2/checkpoints/sam2.1_hiera_large.pt")

# TextRegion params
POINTS_PER_SIDE = 12
RESIZE_METHOD   = "multi_resolution"
CROP_SIZE       = 336
DTYPE           = "bf16"

# BG definition (adaptive)
BG_MIN_COVER     = 0.15   # bg_labels < 15% 才允许扩张
MOVING_MIN_COVER = 0.003  # moving < 0.3% 禁止扩张(避免全白退化)

# BG cleanup (keep more bg, no CC selection)
DO_CLEANUP  = True
CLOSE_KSIZE = 11
FILL_HOLES  = False
MIN_AREA    = 0

OVERLAY_ALPHA = 0.45
DEVICE = "cuda"
OBJ_ID = 1

# -------------------- LABELS (two groups) --------------------
MOVING_LABELS = ["moving object", "human", "animal", "vehicle", "sport"]

BACKGROUND_LABELS = [
    "ground", "road", "sidewalk", "floor",
    "grass", "tree", "bush", "plant",
    "water", "river", "lake", "sea",
    "sky", "cloud",
    "wall", "building", "house", "window", "door",
    "mountain", "rock", "sand", "bridge", "fence", "rail"
]
BACKGROUND_FALLBACK = ["background", "scene", "outdoor", "indoor"]


def list_frames(frames_dir: Path):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    frames = []
    for ext in exts:
        frames += glob.glob(str(frames_dir / ext))
    return [Path(p) for p in sorted(frames)]


def write_yaml_for_image(img_path: Path, yml_path: Path, labels):
    data = {str(img_path): {"label": labels}}
    yml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def run_textregion(image_path: Path, yml_path: Path):
    seg_py = TEXTREGION_REPO / "TextRegionSegmenter.py"
    assert seg_py.exists(), seg_py
    cmd = [
        sys.executable, str(seg_py),
        "--image_list", str(image_path),
        "--image_query_cfg", str(yml_path),
        "--resize_method", RESIZE_METHOD,
        "--crop_size", str(CROP_SIZE),
        "--points_per_side", str(POINTS_PER_SIDE),
        "--dtype", DTYPE,
        "--viz_regions", "True",
        "--dump_region_labels", "True",
        "--sam2_checkpoint", str(SAM2_CKPT),
    ]
    print("[TextRegion RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(TEXTREGION_REPO))


def load_textregion_outputs(seq_name: str, stem: str):
    out_dir = TEXTREGION_REPO / "outputs" / seq_name
    pred = np.load(out_dir / f"{stem}_pred.npy").astype(np.int32)
    union_path = out_dir / f"{stem}_union.npy"
    union = None
    if union_path.exists():
        union = (np.load(union_path) > 0).astype(np.uint8)
    return pred, union


def _norm_list(xs):
    return [str(x).strip().lower() for x in xs]


def build_bg_and_moving_adaptive(pred: np.ndarray, labels_in_yaml, union: np.ndarray | None):
    labels_low = _norm_list(labels_in_yaml)
    label_to_id = {name: i for i, name in enumerate(labels_low)}

    moving_ids = [label_to_id[n.lower()] for n in MOVING_LABELS if n.lower() in label_to_id]
    bg_ids     = [label_to_id[n.lower()] for n in BACKGROUND_LABELS if n.lower() in label_to_id]
    if not bg_ids:
        bg_ids = [label_to_id[n] for n in BACKGROUND_FALLBACK if n in label_to_id]

    moving = np.isin(pred, moving_ids).astype(np.uint8) if moving_ids else np.zeros_like(pred, dtype=np.uint8)
    bg_by_label = np.isin(pred, bg_ids).astype(np.uint8) if bg_ids else np.zeros_like(pred, dtype=np.uint8)

    if union is not None:
        if union.shape != pred.shape:
            union = cv2.resize(union.astype(np.uint8), (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        u = (union > 0).astype(np.uint8)
    else:
        u = np.ones_like(pred, dtype=np.uint8)

    bg = (bg_by_label & u).astype(np.uint8)

    bg_cover = float(bg.sum()) / float(bg.size)
    mv_cover = float(moving.sum()) / float(moving.size)
    print(f"[DBG] moving_ids={len(moving_ids)} bg_ids={len(bg_ids)} mv_cover={mv_cover:.3f} bg_cover={bg_cover:.3f}")

    # If moving too weak -> forbid expanding with union_not_moving (avoids full-white bg)
    if mv_cover < MOVING_MIN_COVER:
        return bg, moving

    # If bg too small -> expand with not_moving in union
    if bg_cover < BG_MIN_COVER:
        bg_expand = ((1 - moving) & u).astype(np.uint8)
        bg = (bg | bg_expand).astype(np.uint8)

    return bg, moving


def clean_bg(mask01: np.ndarray):
    m = (mask01 > 0).astype(np.uint8)
    if CLOSE_KSIZE and CLOSE_KSIZE > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KSIZE, CLOSE_KSIZE))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    if FILL_HOLES:
        h, w = m.shape
        flood = m.copy()
        tmp = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, tmp, (0, 0), 1)
        holes = (flood == 0).astype(np.uint8)
        m = (m | holes).astype(np.uint8)
    if MIN_AREA and MIN_AREA > 0:
        num, cc, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros_like(m)
        for lab in range(1, num):
            if int(stats[lab, cv2.CC_STAT_AREA]) >= MIN_AREA:
                keep[cc == lab] = 1
        m = keep
    return m


def make_clean_frames_dir(frames, clean_dir: Path):
    clean_dir.mkdir(parents=True, exist_ok=True)
    for p in clean_dir.glob("*"):
        p.unlink()
    for i, fp in enumerate(frames):
        dst = clean_dir / f"{i:05d}.jpg"
        try:
            os.symlink(str(fp), str(dst))
        except Exception:
            import shutil
            shutil.copy2(str(fp), str(dst))
    return clean_dir


def build_sam2_video_predictor():
    sys.path.insert(0, str(SAM2_REPO))
    from sam2.build_sam import build_sam2_video_predictor
    return build_sam2_video_predictor(SAM2_CFG_NAME, str(SAM2_CKPT), device=DEVICE)


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
        raise RuntimeError(f"mask not 2D: {mk.shape}")
    return (mk > 0).astype(np.uint8)


def mask_to_bbox(mask01):
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def sam2_track_bg(predictor, video_dir: Path, init_bg01: np.ndarray, out_dir: Path):
    out_masks = out_dir / "masks"
    out_masks.mkdir(parents=True, exist_ok=True)

    state = predictor.init_state(video_path=str(video_dir))
    predictor.add_new_mask(state, frame_idx=0, obj_id=OBJ_ID, mask=(init_bg01 > 0))

    tracks = {}

    for out in predictor.propagate_in_video(state):
        if len(out) == 2:
            frame_idx, masks = out
            obj_ids = [OBJ_ID]
        elif len(out) == 3:
            frame_idx, obj_ids, masks = out
        else:
            frame_idx, obj_ids, masks, _ = out

        obj_ids = _torch_to_numpy(obj_ids)
        if isinstance(obj_ids, np.ndarray):
            obj_ids = obj_ids.tolist()
        if not isinstance(obj_ids, (list, tuple)):
            obj_ids = [OBJ_ID]
        k = obj_ids.index(OBJ_ID) if OBJ_ID in obj_ids else 0

        if isinstance(masks, (list, tuple)):
            mk = masks[k]
        else:
            try:
                import torch
                if torch.is_tensor(masks) and masks.ndim >= 3:
                    mk = masks[k]
                else:
                    mk = masks
            except Exception:
                mk = masks

        mk = _to_numpy_mask_2d(mk)

        cv2.imwrite(str(out_masks / f"{int(frame_idx):05d}.png"), mk * 255)
        np.save(out_masks / f"{int(frame_idx):05d}.npy", mk)
        tracks[f"{int(frame_idx):05d}"] = {"area": int(mk.sum()), "bbox": mask_to_bbox(mk)}

    with open(out_dir / "tracks.json", "w") as f:
        json.dump(tracks, f, indent=2)

    print(f"[SAM2] saved {len(tracks)} masks to {out_masks}")


def main():
    frames = list_frames(FRAMES_DIR)
    assert len(frames) > 0, f"No frames found: {FRAMES_DIR}"

    seq_name = FRAMES_DIR.name
    frame0 = frames[0]
    stem0 = frame0.stem

    out_dir = OUT_ROOT / seq_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) YAML with two groups
    labels = list(dict.fromkeys(MOVING_LABELS + BACKGROUND_LABELS + BACKGROUND_FALLBACK))
    yml = out_dir / "_textregion_two_groups.yaml"
    write_yaml_for_image(frame0, yml, labels)

    # 2) TextRegion run
    run_textregion(frame0, yml)

    # 3) load pred/union
    pred, union = load_textregion_outputs(seq_name, stem0)

    # align pred to union size (or image size)
    if union is not None:
        H, W = union.shape[:2]
    else:
        im0 = cv2.imread(str(frame0))
        H, W = im0.shape[:2]
    if pred.shape != (H, W):
        pred = cv2.resize(pred.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)

    # 4) build bg + moving debug (adaptive)
    bg0, mv0 = build_bg_and_moving_adaptive(pred, labels, union)

    # optional cleanup (keep more)
    if DO_CLEANUP:
        bg0 = clean_bg(bg0)

    # 5) save init debug
    init_dir = out_dir / "init"
    init_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(init_dir / "00000_bg_init.png"), bg0 * 255)
    np.save(init_dir / "00000_bg_init.npy", bg0)

    cv2.imwrite(str(init_dir / "00000_moving_debug.png"), mv0 * 255)
    np.save(init_dir / "00000_moving_debug.npy", mv0)

    # overlay BG on frame0 (red)
    im0 = cv2.imread(str(frame0))
    if im0.shape[:2] != bg0.shape:
        im0 = cv2.resize(im0, (bg0.shape[1], bg0.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = im0.copy()
    tint = np.zeros_like(overlay); tint[..., 2] = 255
    ys, xs = np.where(bg0 > 0)
    overlay[ys, xs] = (OVERLAY_ALPHA * tint[ys, xs] + (1 - OVERLAY_ALPHA) * overlay[ys, xs]).astype(np.uint8)
    cv2.imwrite(str(init_dir / "00000_bg_overlay.png"), overlay)

    # stats to verify "full white" vs "mostly white"
    print("[STAT] bg unique:", np.unique(bg0), "bg_white_ratio:", float((bg0 > 0).mean()))
    print("[STAT] mv unique:", np.unique(mv0), "mv_white_ratio:", float((mv0 > 0).mean()))

    # 6) build clean frames dir for SAM2
    clean_dir = make_clean_frames_dir(frames, out_dir / "_clean_frames")

    # 7) SAM2 propagate background
    predictor = build_sam2_video_predictor()
    sam2_track_bg(predictor, clean_dir, bg0, out_dir)

    print("[DONE] output:", out_dir)


if __name__ == "__main__":
    main()
