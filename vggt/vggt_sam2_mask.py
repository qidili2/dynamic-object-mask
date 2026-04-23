#!/usr/bin/env python3
"""
Generate dynamic masks using VGGT cross-frame attention + Easi3R-style region tracking.

Pipeline (aligned with Easi3R generate_region_groups_with_tracking + get_motion_mask_from_attns):
  1. Load VGGT cross-frame attention maps -> build [B, Ht, Wt] dynamic token map
       dynamic_score = 1 - norm(mean_attn)   (HIGH value = dynamic)
  2. Generate region groups via SAM2:
       - Frame 0: SAM2ImagePredictor with 16x16 grid -> non-overlapping region proposals
       - All frames: SAM2 video predictor tracking -> per-frame [H,W] int group_ids
  3. Downsample group_ids to token grid: [B, H, W] -> [B, Ht, Wt]
  4. Compute object-level global attention: per-obj mean over token positions and frames
  5. adaptive_multiotsu_variance threshold on all object attention values
  6. Mark dynamic_object_ids (attention > threshold)
  7. Build per-frame binary masks from dynamic_object_ids in high-res group_ids
  8. Save masks + visualizations

Usage:
  # Step 1: generate VGGT attention npy files
  python viz_cross_frame_attn.py --davis_root /path/to/DAVIS --output_dir cross_attn_out

  # Step 2: generate dynamic masks
  python vggt_sam2_mask.py \\
      --davis_root /mnt/data0/andy/Easi3R/DAVIS/JPEGImages/480p \\
      --attn_root  cross_attn_out \\
      --output_dir vggt_sam2_out
"""

import os
import sys
import uuid
import shutil
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from typing import List, Optional, Tuple, Dict
from skimage.filters import threshold_multiotsu

sys.path.insert(0, str(Path(__file__).parent))

# SAM2 — point to Easi3R's third_party/sam2
_EASI3R_ROOT    = Path(__file__).parent.parent / "Easi3R"
sys.path.insert(0, str(_EASI3R_ROOT / "third_party" / "sam2"))

SAM2_CHECKPOINT = str(_EASI3R_ROOT / "third_party/sam2/checkpoints/sam2.1_hiera_large.pt")
MODEL_CFG       = "configs/sam2.1/sam2.1_hiera_l.yaml"

COLLECT_LAYERS = [4, 11, 17, 23]


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_seq_images(seq_dir: Path, resolution: int):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted(p for p in seq_dir.iterdir() if p.suffix.lower() in exts)
    imgs = [
        np.array(Image.open(p).convert("RGB").resize((resolution, resolution), Image.BILINEAR))
        for p in paths
    ]
    return imgs, paths


def load_attn_maps(npy_dir: Path, n_frames: int,
                   layers: List[int] = COLLECT_LAYERS) -> List[Optional[np.ndarray]]:
    """
    Load per-frame query mean attention maps saved by viz_cross_frame_attn.py.
    Prefers frame_XXXXX_avg_query.npy (new naming), falls back to
    frame_XXXXX_avg.npy (old naming), then per-layer files.
    Returns list[n_frames] of [H_p, W_p] float32, or None if missing.
    """
    maps = []
    for i in range(n_frames):
        # New naming first, then legacy
        for avg_name in (f"frame_{i:05d}_avg_query.npy", f"frame_{i:05d}_avg.npy"):
            avg_path = npy_dir / avg_name
            if avg_path.exists():
                maps.append(np.load(avg_path).astype(np.float32))
                break
        else:
            layer_maps = []
            for lid in layers:
                for layer_name in (f"frame_{i:05d}_layer{lid}_query.npy",
                                   f"frame_{i:05d}_layer{lid}.npy"):
                    p = npy_dir / layer_name
                    if p.exists():
                        layer_maps.append(np.load(p).astype(np.float32))
                        break
            maps.append(np.stack(layer_maps, 0).mean(0) if layer_maps else None)
    return maps


def load_query_layer_maps(npy_dir: Path, n_frames: int,
                          layers: List[int] = (11, 16, 17, 18, 23)) -> List[Optional[np.ndarray]]:
    """
    Load per-frame query attention maps for specified layers and return their mean.
    Deep layers [11,16,17,18,23] carry the strongest (1-q) dynamic signal.
    Returns list[n_frames] of [H_p, W_p] float32, or None if all layers missing.
    """
    maps = []
    for i in range(n_frames):
        layer_maps = []
        for lid in layers:
            p = npy_dir / f"frame_{i:05d}_layer{lid}_query.npy"
            if p.exists():
                layer_maps.append(np.load(p).astype(np.float32))
        maps.append(np.stack(layer_maps, 0).mean(0) if layer_maps else None)
    return maps


def load_key_layer_maps(npy_dir: Path, n_frames: int,
                        layers: List[int] = (2, 4, 7)) -> List[Optional[np.ndarray]]:
    """
    Load per-frame key attention maps for multiple layers and return their mean.
    Shallow layers [2,4,7] carry the strongest key signal.
    Returns list[n_frames] of [H_p, W_p] float32, or None if all layers missing.
    """
    maps = []
    for i in range(n_frames):
        layer_maps = []
        for lid in layers:
            p = npy_dir / f"frame_{i:05d}_layer{lid}_key.npy"
            if p.exists():
                layer_maps.append(np.load(p).astype(np.float32))
        maps.append(np.stack(layer_maps, 0).mean(0) if layer_maps else None)
    return maps


def load_var_layer_maps(npy_dir: Path, n_frames: int, side: str,
                        layers: List[int] = (2, 4, 7)) -> List[Optional[np.ndarray]]:
    """
    Load per-frame query_var or key_var maps for specified layers and return their mean.
    side: 'query' or 'key'
    Returns list[n_frames] of [H_p, W_p] float32, or None if all layers missing.
    """
    maps = []
    for i in range(n_frames):
        layer_maps = []
        for lid in layers:
            p = npy_dir / f"frame_{i:05d}_layer{lid}_{side}_var.npy"
            if p.exists():
                layer_maps.append(np.load(p).astype(np.float32))
        maps.append(np.stack(layer_maps, 0).mean(0) if layer_maps else None)
    return maps


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def _resize_to(x: np.ndarray, H: int, W: int) -> np.ndarray:
    if x.shape == (H, W):
        return x
    return cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR)


def build_dynamic_token_map(attn_maps: List[Optional[np.ndarray]],
                             H_token: int, W_token: int,
                             key_maps: Optional[List[Optional[np.ndarray]]] = None,
                             qv_maps: Optional[List[Optional[np.ndarray]]] = None,
                             kv_maps: Optional[List[Optional[np.ndarray]]] = None,
                             ) -> torch.Tensor:
    """
    Convert raw VGGT attention maps to a dynamic token map.

    Semantics (HIGH = dynamic):
      Full formula:  (1 - norm(qm)) * norm(km) * norm(qv) * norm(kv)
        all four maps averaged over the same 4 layers [4, 11, 17, 23]
      Fallback (if var maps absent): (1 - norm(qm)) * norm(km)
      Fallback (if key maps absent): 1 - norm(qm)

    Returns [B, H_token, W_token] float32 tensor, values in [0, 1].
    Missing frames are filled with zeros (treated as static/unknown).
    """
    frames = []
    n = len(attn_maps)
    for i in range(n):
        q  = attn_maps[i]
        k  = key_maps[i] if key_maps is not None else None
        qv = qv_maps[i]  if qv_maps  is not None else None
        kv = kv_maps[i]  if kv_maps  is not None else None

        if q is None and k is None:
            frames.append(np.zeros((H_token, W_token), dtype=np.float32))
            continue

        if q is not None and k is not None:
            q_n = _norm01(_resize_to(q, H_token, W_token))
            k_n = _norm01(_resize_to(k, H_token, W_token))
            if qv is not None and kv is not None:
                qv_n = _norm01(_resize_to(qv, H_token, W_token))
                kv_n = _norm01(_resize_to(kv, H_token, W_token))
                dyn = (1.0 - q_n) * k_n * qv_n * kv_n
                dyn = dyn / (dyn.max() + 1e-8)   # renorm to [0,1]
            else:
                dyn = (1.0 - q_n) * k_n
        elif q is not None:
            dyn = 1.0 - _norm01(_resize_to(q, H_token, W_token))
        else:
            dyn = 1.0 - _norm01(_resize_to(k, H_token, W_token))

        frames.append(dyn.astype(np.float32))
    return torch.from_numpy(np.stack(frames, axis=0))   # [B, Ht, Wt]


# ─────────────────────────────────────────────────────────────────────────────
# SAM2 dtype helpers  (ported from Easi3R base_opt.py)
# ─────────────────────────────────────────────────────────────────────────────

def _convert_to_float32(model):
    """Convert all parameters and buffers of a model to float32."""
    for name, param in model.named_parameters(recurse=False):
        if param.dtype in (torch.bfloat16, torch.float16):
            param.data = param.data.float()
    for name, buf in model.named_buffers(recurse=False):
        if buf.dtype in (torch.bfloat16, torch.float16):
            model.register_buffer(name, buf.float())
    for child in model.children():
        _convert_to_float32(child)
    return model


def _convert_inference_state_dtype(state):
    """Recursively convert bfloat16/float16 tensors in SAM2 inference_state to float32."""
    if not isinstance(state, dict):
        return
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.dtype in (torch.bfloat16, torch.float16):
            state[k] = v.float()
        elif isinstance(v, dict):
            _convert_inference_state_dtype(v)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, torch.Tensor) and item.dtype in (torch.bfloat16, torch.float16):
                    v[i] = item.float()
                elif isinstance(item, dict):
                    _convert_inference_state_dtype(item)


# ─────────────────────────────────────────────────────────────────────────────
# Region generation helpers  (ported from Easi3R base_opt.py)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_grid_masks(predictor, H: int, W: int,
                          grid: int, select_mode: str = "largest") -> List[dict]:
    """Sample a grid of points, run SAM2ImagePredictor, collect masks."""
    masks_raw = []
    ys = np.linspace(0, H - 1, grid, dtype=np.int32)
    xs = np.linspace(0, W - 1, grid, dtype=np.int32)
    for y in ys:
        for x in xs:
            pts  = np.array([[x, y]], dtype=np.float32)
            lbls = np.array([1],      dtype=np.int32)
            out  = predictor.predict(point_coords=pts, point_labels=lbls,
                                     multimask_output=True)
            masks = (out[0] if isinstance(out, tuple) else out["masks"]).astype(bool)
            areas = masks.reshape(masks.shape[0], -1).sum(1)
            if select_mode == "largest":
                k = int(np.argmax(areas))
            else:   # smallest
                k = int(np.argmin(areas))
            masks_raw.append({"segmentation": masks[k], "area": int(areas[k])})
    return masks_raw


def _deduplicate_masks(masks: List[dict], iou_threshold: float = 0.8) -> List[dict]:
    masks = sorted(masks, key=lambda d: d["area"], reverse=True)
    kept = []
    for d in masks:
        m = d["segmentation"]
        dup = any(
            (m & k["segmentation"]).sum() / ((m | k["segmentation"]).sum() + 1e-8) > iou_threshold
            for k in kept
        )
        if not dup:
            kept.append(d)
    return kept


def generate_initial_regions(img_rgb: np.ndarray, device: str,
                              min_size: int = 500, grid: int = 16,
                              fill_background: bool = True,
                              small_min_size: int = 100,
                              small_grid: int = 32) -> List[dict]:
    """
    Ported from Easi3R generate_whole_masks_from_grid.
    Returns list of dicts with keys: 'segmentation' [H,W] bool, 'area', 'obj_id'.
    Regions are non-overlapping (later assignments don't overwrite earlier ones).
    """
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    H, W, _ = img_rgb.shape

    # Build image predictor
    with torch.cuda.amp.autocast(enabled=False):
        sam2 = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=device)
        sam2 = _convert_to_float32(sam2).to(device).eval()
        predictor = SAM2ImagePredictor(sam2)
        predictor.set_image(img_rgb)

    # --- Round 1: large masks ---
    large_masks = _generate_grid_masks(predictor, H, W, grid, select_mode="largest")
    large_masks = [m for m in large_masks if m["area"] >= min_size]
    large_masks = _deduplicate_masks(large_masks, iou_threshold=0.8)
    print(f"  [SAM2-grid] {len(large_masks)} large masks after dedup")

    # Non-overlapping assignment
    group_ids    = np.full((H, W), fill_value=-1, dtype=np.int32)
    group_ctr    = 1
    final_masks  = []

    for d in large_masks:
        free      = group_ids == -1
        to_assign = d["segmentation"] & free
        if to_assign.sum() < min_size:
            continue
        group_ids[to_assign] = group_ctr
        final_masks.append({"segmentation": to_assign, "area": int(to_assign.sum()),
                             "obj_id": group_ctr, "level": "large"})
        group_ctr += 1

    # --- Round 2: fill background with small masks ---
    if fill_background and (group_ids == -1).sum() / (H * W) > 0.05:
        small_masks = _generate_grid_masks(predictor, H, W, small_grid, select_mode="smallest")
        small_masks = [m for m in small_masks if m["area"] >= small_min_size]
        small_masks = _deduplicate_masks(small_masks, iou_threshold=0.7)
        small_masks.sort(key=lambda d: d["area"], reverse=True)
        print(f"  [SAM2-grid] {len(small_masks)} small masks for background fill")

        for d in small_masks:
            free      = group_ids == -1
            to_assign = d["segmentation"] & free
            if to_assign.sum() < small_min_size:
                continue
            group_ids[to_assign] = group_ctr
            final_masks.append({"segmentation": to_assign, "area": int(to_assign.sum()),
                                 "obj_id": group_ctr, "level": "small"})
            group_ctr += 1

    # Remaining pixels -> background (0)
    group_ids[group_ids == -1] = 0

    del sam2, predictor
    torch.cuda.empty_cache()

    print(f"  [SAM2-grid] total {len(final_masks)} regions "
          f"(large={sum(1 for m in final_masks if m['level']=='large')}, "
          f"small={sum(1 for m in final_masks if m['level']=='small')})")
    return final_masks


# ─────────────────────────────────────────────────────────────────────────────
# Region group tracking  (ported from Easi3R generate_region_groups_with_tracking)
# ─────────────────────────────────────────────────────────────────────────────

def _mask_to_points_and_box(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, None, None
    box    = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
    points = np.array([[int(xs.mean()), int(ys.mean())]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    return points, labels, box


def generate_region_groups_with_tracking(
    imgs: List[np.ndarray],
    device: str,
    min_size: int = 500,
    max_objects: int = 30,
    vis_dir: Optional[Path] = None,
) -> Tuple[List[torch.Tensor], Dict]:
    """
    Ported from Easi3R BasePCOptimizer.generate_region_groups_with_tracking.

    Step 1: Generate initial masks on frame 0 via SAM2 grid sampling.
    Step 2: Seed SAM2 video predictor with those masks (box + centroid prompt).
    Step 3: Propagate tracking through all frames.
    Step 4: Convert to list[B] of [H,W] int64 tensors (0=background, 1..K=objects).

    Returns (region_groups, video_segments).
    """
    from sam2.build_sam import build_sam2_video_predictor

    n = len(imgs)
    H, W = imgs[0].shape[:2]
    original_cwd = os.getcwd()

    # --- Step 1: initial masks on frame 0 ---
    print(f"  [SAM→SAM2] generating initial regions on frame 0 ...")
    init_masks = generate_initial_regions(imgs[0], device, min_size=min_size)
    init_masks = init_masks[:max_objects]
    print(f"  [SAM→SAM2] {len(init_masks)} initial regions selected")

    # --- Step 2: save frames to temp dir (SAM2 video predictor API) ---
    tmp_dir = f"/dev/shm/vggt_sam2_{uuid.uuid4().hex[:8]}" \
              if os.path.exists("/dev/shm") \
              else f"/tmp/vggt_sam2_{uuid.uuid4().hex[:8]}"
    os.makedirs(tmp_dir, exist_ok=True)
    for i, img in enumerate(imgs):
        cv2.imwrite(os.path.join(tmp_dir, f"{i:06d}.jpg"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Visualisation setup
    if vis_dir is not None:
        (vis_dir / "groups").mkdir(parents=True, exist_ok=True)
        (vis_dir / "overlays").mkdir(parents=True, exist_ok=True)

    region_groups  = []
    video_segments = {}

    try:
        # --- Step 3: build predictor and seed ---
        with torch.cuda.amp.autocast(enabled=False):
            predictor = build_sam2_video_predictor(MODEL_CFG, SAM2_CHECKPOINT, device=device)
            predictor = _convert_to_float32(predictor).to(device).eval()
            inference_state = predictor.init_state(video_path=tmp_dir)
            predictor.reset_state(inference_state)
            _convert_inference_state_dtype(inference_state)

        added_objects = []
        for obj_id, mask_dict in enumerate(init_masks, start=1):
            mask   = mask_dict["segmentation"].astype(bool)
            points, labels, box = _mask_to_points_and_box(mask)
            if points is None:
                continue
            try:
                with torch.cuda.amp.autocast(enabled=False):
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=obj_id,
                        points=points,
                        labels=labels,
                        box=box,
                    )
                added_objects.append(obj_id)
            except Exception as e:
                print(f"  [SAM→SAM2] failed to add object {obj_id}: {e}")

        print(f"  [SAM→SAM2] seeded {len(added_objects)} objects, propagating ...")

        # --- Step 4: propagate ---
        try:
            with torch.cuda.amp.autocast(enabled=False):
                for out_frame_idx, out_obj_ids, out_mask_logits in \
                        predictor.propagate_in_video(inference_state):
                    if out_frame_idx >= n:
                        break
                    frame_masks = {}
                    for i, oid in enumerate(out_obj_ids):
                        if oid not in added_objects:
                            continue
                        logits = out_mask_logits[i]
                        if logits.dtype in (torch.bfloat16, torch.float16):
                            logits = logits.float()
                        while logits.dim() > 2:
                            logits = logits.squeeze(0)
                        mask = (logits > 0.0).cpu().numpy()
                        if mask.ndim == 2 and mask.shape == (H, W):
                            frame_masks[oid] = mask
                    video_segments[out_frame_idx] = frame_masks
                    if out_frame_idx % 10 == 0:
                        print(f"  [SAM→SAM2] frame {out_frame_idx}/{n}, "
                              f"{len(frame_masks)} objects")
        except Exception as e:
            print(f"  [SAM→SAM2] tracking failed: {e}")
            for fi in range(n):
                video_segments.setdefault(fi, {})

        # --- Step 5: convert to group_ids tensors ---
        _COLORS = np.array([
            [0,0,0],[31,119,180],[255,127,14],[44,160,44],[214,39,40],
            [148,103,189],[140,86,75],[227,119,194],[127,127,127],
            [188,189,34],[23,190,207],[174,199,232],[255,187,120],
            [152,223,138],[255,152,150],[197,176,213],[196,156,148],
            [247,182,210],[199,199,199],[219,219,141],[158,218,229],
        ], dtype=np.uint8)

        for fi in range(n):
            gids = np.zeros((H, W), dtype=np.int32)
            for oid, mask in video_segments.get(fi, {}).items():
                if mask.shape == (H, W):
                    gids[mask] = oid
            g_tensor = torch.from_numpy(gids.astype(np.int64)).to(device)
            region_groups.append(g_tensor)

            if vis_dir is not None:
                # Coloured group map
                cmap = np.zeros((H, W, 3), dtype=np.uint8)
                for gid in np.unique(gids):
                    cmap[gids == gid] = _COLORS[gid % len(_COLORS)]
                bgr = cv2.cvtColor(cmap, cv2.COLOR_RGB2BGR)
                txt = f"Frame {fi}/{n} | objs: {len(video_segments.get(fi,{}))}"
                cv2.putText(bgr, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(bgr, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255,255,255), 1, cv2.LINE_AA)
                cv2.imwrite(str(vis_dir / "groups" / f"frame_{fi:04d}.png"), bgr)

                img_bgr = cv2.cvtColor(imgs[fi], cv2.COLOR_RGB2BGR)
                ov = cv2.addWeighted(img_bgr, 0.6, cmap, 0.4, 0)
                cv2.putText(ov, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255,255,255), 1, cv2.LINE_AA)
                cv2.imwrite(str(vis_dir / "overlays" / f"frame_{fi:04d}.png"), ov)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if os.getcwd() != original_cwd:
            os.chdir(original_cwd)
        try:
            del predictor
        except Exception:
            pass
        torch.cuda.empty_cache()

    print(f"  [SAM→SAM2] done. {len(region_groups)} frames, "
          f"{len(added_objects)} objects tracked")
    return region_groups, video_segments


# ─────────────────────────────────────────────────────────────────────────────
# Token-level region operations  (ported from Easi3R base_opt.py)
# ─────────────────────────────────────────────────────────────────────────────

def downsample_groups_to_tokens(groups_bhw: torch.Tensor,
                                 Ht: int, Wt: int) -> torch.Tensor:
    """
    Nearest-neighbour downsample [B, H, W] int64 group_ids to [B, Ht, Wt].
    Ported from Easi3R _downsample_groups_to_tokens.
    """
    x  = groups_bhw.unsqueeze(1).float()          # [B,1,H,W]
    ds = F.interpolate(x, size=(Ht, Wt), mode="nearest").squeeze(1).long()
    return ds   # [B, Ht, Wt]


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive multi-Otsu threshold  (ported from Easi3R base_opt.py)
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_multiotsu_variance(values, verbose: bool = False,
                                 max_classes: int = 3) -> float:
    """
    Ported from Easi3R adaptive_multiotsu_variance.
    Selects best K (2..max_classes) by inter-class variance / sqrt(K).
    Returns the last (highest) threshold of the best split.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.5

    n = arr.size
    if n < max_classes:
        max_classes = max(2, int(n))

    uniques = np.unique(arr)
    if uniques.size == 1:
        return float(uniques[0])

    max_classes_eff = int(max(2, min(max_classes, uniques.size)))
    best_score      = -np.inf
    best_threshold  = None
    best_k          = None

    for k in range(2, max_classes_eff + 1):
        try:
            thresholds = threshold_multiotsu(arr, classes=k)
        except Exception as e:
            if verbose:
                print(f"  [AMOV] K={k} failed: {e}")
            continue
        if thresholds is None or len(thresholds) != k - 1:
            continue
        if not np.all(np.diff(thresholds) > 0):
            continue

        regions    = np.digitize(arr, bins=thresholds, right=False)
        means      = []
        empty_flag = False
        for i in range(k):
            cls_vals = arr[regions == i]
            if cls_vals.size == 0:
                empty_flag = True
                break
            means.append(cls_vals.mean())
        if empty_flag:
            continue

        score = np.var(np.array(means)) / np.sqrt(k)
        if score > best_score:
            best_score     = score
            best_threshold = float(thresholds[-1])
            best_k         = k

    if best_threshold is None:
        fallback = float(np.median(arr))
        if verbose:
            print(f"  [AMOV] all K failed -> median fallback {fallback:.6f}")
        return fallback

    if verbose:
        print(f"  [AMOV] best K={best_k}, threshold={best_threshold:.6f}")
    return best_threshold


# ─────────────────────────────────────────────────────────────────────────────
# Object-level dynamic detection  (ported from Easi3R get_motion_mask_from_attns)
# ─────────────────────────────────────────────────────────────────────────────

def compute_object_global_attention(
    attn_tensor: torch.Tensor,   # [B, Ht, Wt]  HIGH = dynamic
    groups_token: torch.Tensor,  # [B, Ht, Wt]  int64
    top_region_pct: float = 5.0,
) -> Dict[int, float]:
    """
    For each object id (non-zero), compute mean attention across all frames and
    all token positions where that object is present.

    Returns dict {obj_id: mean_dynamic_attention}.
    """
    B    = attn_tensor.shape[0]
    dev  = attn_tensor.device
    attn = attn_tensor.to(dev)
    grp  = groups_token.to(dev)

    # Collect all object ids across all frames
    all_ids: set = set()
    for b in range(B):
        uniq = torch.unique(grp[b])
        all_ids.update(uniq[uniq != 0].cpu().tolist())
    all_ids = sorted(all_ids)

    print(f"  [Object-Attn] {len(all_ids)} unique objects across {B} frames")

    TOP_PCT = top_region_pct / 100.0

    obj_global: Dict[int, float] = {}
    for oid in all_ids:
        vals = []
        for b in range(B):
            mask_t = grp[b] == oid   # [Ht, Wt] bool
            n = int(mask_t.sum().item())
            if n == 0:
                continue
            token_vals = attn[b][mask_t]                       # [n]
            k = max(1, round(n * TOP_PCT))
            top_vals, _ = torch.topk(token_vals, k)
            vals.append(top_vals.mean().item())
        obj_global[oid] = float(np.mean(vals)) if vals else 0.0

    return obj_global


def decide_dynamic_objects(obj_global: Dict[int, float],
                             groups_hr: torch.Tensor,     # [B, H, W]
                             H_img: int, W_img: int,
                             B: int,
                             method: str = "multiotsu",
                             topk_pct: float = 25.0) -> set:
    """
    Select dynamic objects from per-object attention scores.

    method='multiotsu' (default):
      1. Compute threshold via adaptive_multiotsu_variance.
      2. Mark objects above threshold as dynamic.
      3. Fallback: single object -> mark dynamic.
      4. Fallback: dynamic area < 0.2% -> re-threshold.

    method='topk':
      Take the top topk_pct% of objects by score as dynamic (min 1).

    Returns set of dynamic object ids.
    """
    global_att_values = np.array(list(obj_global.values()), dtype=np.float32)
    if len(global_att_values) == 0:
        return set()

    all_oids = set(obj_global.keys())
    n_obj = len(all_oids)
    sorted_objs = sorted(obj_global.items(), key=lambda x: x[1], reverse=True)

    if method == "topk":
        n_dynamic = max(1, round(n_obj * topk_pct / 100.0))
        dynamic_ids = {oid for oid, _ in sorted_objs[:n_dynamic]}
        print(f"  [Object-Attn] top-{topk_pct:.0f}%: {n_dynamic}/{n_obj} objects dynamic")
        print("  [Object-Attn] top-5 objects:")
        for oid, att in sorted_objs[:5]:
            tag = "DYNAMIC" if oid in dynamic_ids else "static"
            print(f"    obj {oid:3d}: attn={att:.4f}  [{tag}]")
        return dynamic_ids

    # --- multiotsu ---
    threshold = adaptive_multiotsu_variance(global_att_values)
    print(f"  [Object-Attn] threshold = {threshold:.4f}")

    # Single-object fallback
    if n_obj == 1:
        oid = next(iter(all_oids))
        print(f"  [Object-Attn] single object {oid} -> marked dynamic (fallback)")
        return {oid}

    dynamic_ids = {oid for oid, a in obj_global.items() if a > threshold}
    print(f"  [Object-Attn] {len(dynamic_ids)}/{n_obj} objects dynamic")

    print("  [Object-Attn] top-5 objects:")
    for oid, att in sorted_objs[:5]:
        tag = "DYNAMIC" if oid in dynamic_ids else "static"
        print(f"    obj {oid:3d}: attn={att:.4f}  [{tag}]")

    # Small-area fallback: if < 0.2% pixels dynamic, re-threshold on all objects
    MIN_MASK_RATIO = 0.002
    if len(dynamic_ids) > 0:
        total_px = float(H_img * W_img)
        ratios   = []
        for b in range(B):
            dyn_px = sum((groups_hr[b] == oid).sum().item() for oid in dynamic_ids)
            ratios.append(dyn_px / (total_px + 1e-6))
        avg_ratio = float(np.mean(ratios))
    else:
        avg_ratio = 0.0

    if avg_ratio < MIN_MASK_RATIO:
        print(f"  [Object-Attn] avg dynamic area {avg_ratio*100:.3f}% < 0.2%; "
              "re-thresholding on all objects ...")
        threshold2   = adaptive_multiotsu_variance(global_att_values)
        dynamic_ids  = {oid for oid, a in obj_global.items() if a > threshold2}
        print(f"  [Object-Attn] after re-threshold: {len(dynamic_ids)}/{n_obj}")

    return dynamic_ids


def build_masks_from_dynamic_objects(
    region_groups: List[torch.Tensor],   # [B] of [H,W] int64
    dynamic_ids: set,
    H: int, W: int,
) -> List[np.ndarray]:
    """
    For each frame, union the high-res regions of all dynamic objects.
    Returns list[B] of [H, W] bool numpy arrays.
    """
    masks = []
    for g in region_groups:
        m = torch.zeros(H, W, dtype=torch.bool, device=g.device)
        for oid in dynamic_ids:
            m |= (g == oid)
        masks.append(m.cpu().numpy())
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def save_region_vis(imgs: List[np.ndarray],
                    region_groups: List[torch.Tensor],
                    region_dir: Path):
    """Save per-frame PNG with each SAM2 region in a distinct colour."""
    region_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for i, (img, g) in enumerate(zip(imgs, region_groups)):
        gids   = g.cpu().numpy()
        canvas = img.copy()
        for oid in np.unique(gids):
            if oid == 0:
                continue
            color = rng.integers(60, 240, size=3, dtype=np.uint8)
            mask  = gids == oid
            canvas[mask] = (canvas[mask].astype(np.uint16) // 2
                            + color // 2).astype(np.uint8)
        Image.fromarray(canvas).save(region_dir / f"{i:05d}.png")


def save_vis(imgs: List[np.ndarray],
             attn_maps_hr: List[np.ndarray],   # [H,W] dynamic score, high=dynamic
             region_groups: List[torch.Tensor],
             final_masks: List[np.ndarray],
             vis_dir: Path,
             dynamic_ids: set):
    """
    Five-panel visualisation per frame:
      [original | dynamic score heatmap | region mean-pooled score | region group colours | final dynamic mask]
    """
    vis_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # Pre-build colour map for region groups (consistent across frames)
    all_oids = set()
    for g in region_groups:
        all_oids.update(g.cpu().numpy().ravel().tolist())
    all_oids.discard(0)
    oid_color = {oid: rng.integers(60, 240, size=3, dtype=np.uint8)
                 for oid in sorted(all_oids)}

    H, W = imgs[0].shape[:2]

    for i, (img, dyn, g, fm) in enumerate(zip(imgs, attn_maps_hr, region_groups, final_masks)):
        fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

        # Panel 0: original
        axes[0].imshow(img)
        axes[0].set_title(f"frame {i:05d}", fontsize=8)
        axes[0].axis("off")

        # Panel 1: dynamic score heatmap (high = dynamic)
        axes[1].imshow(dyn, cmap="inferno", vmin=0, vmax=1)
        axes[1].set_title("dynamic score\n(VGGT, high=dynamic)", fontsize=8)
        axes[1].axis("off")

        # Panel 2: regional mean-pooled score
        gids   = g.cpu().numpy()
        pooled = np.zeros((H, W), dtype=np.float32)
        for oid in np.unique(gids):
            if oid == 0:
                continue
            mask = gids == oid
            vals = dyn[mask]
            k = max(1, round(len(vals) * 0.05))
            pooled[mask] = np.partition(vals, -k)[-k:].mean()
        axes[2].imshow(pooled, cmap="inferno", vmin=0, vmax=1)
        axes[2].set_title("region top-5% score\n(high=dynamic)", fontsize=8)
        axes[2].axis("off")

        # Panel 3: region groups (each region coloured; dynamic ones marked)
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        for oid in np.unique(gids):
            if oid == 0:
                continue
            canvas[gids == oid] = oid_color.get(oid, np.array([128, 128, 128],
                                                                dtype=np.uint8))
        axes[3].imshow(canvas)
        axes[3].set_title(f"region groups\n(dynamic ids: {sorted(dynamic_ids)[:6]})",
                          fontsize=7)
        axes[3].axis("off")

        # Panel 4: final mask overlay
        axes[4].imshow(img)
        axes[4].imshow(fm.astype(np.uint8) * 180, cmap="Reds", alpha=0.5, vmin=0, vmax=255)
        ratio = fm.mean() * 100
        axes[4].set_title(f"dynamic mask\n({ratio:.1f}% pixels)", fontsize=8)
        axes[4].axis("off")

        fig.tight_layout()
        fig.savefig(vis_dir / f"{i:05d}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Per-sequence entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_sequence(seq_dir: Path, attn_root: Path, output_root: Path, args, device: str):
    seq_name = seq_dir.name
    print(f"\n{'='*60}\n  seq: {seq_name}")

    imgs, _ = load_seq_images(seq_dir, args.resolution)
    if not imgs:
        print("  no images, skipping")
        return
    B = len(imgs)
    H, W = imgs[0].shape[:2]
    print(f"  {B} frames at {H}x{W}")

    # ── 1. Load VGGT attention maps ──────────────────────────────────────────
    npy_dir = attn_root / seq_name / "npy"
    if not npy_dir.exists():
        print(f"  [warn] npy dir not found: {npy_dir}  (run viz_cross_frame_attn.py first)")
        return

    # score = (1 - norm(qm)) * norm(km) * norm(qv) * norm(kv)
    # query_mean / query_var: deep layers [11,18,23]
    # key_mean   / key_var:   shallow layers [2,4,7]
    attn_maps = load_query_layer_maps(npy_dir, B, layers=[11, 18, 23])
    q_missing = sum(1 for m in attn_maps if m is None)
    if q_missing == B:
        print("  [warn] no deep-layer query npy found -> fallback to all-layer avg")
        attn_maps = load_attn_maps(npy_dir, B)
        q_missing = sum(1 for m in attn_maps if m is None)
    if q_missing:
        print(f"  [warn] {q_missing}/{B} frames missing query map -> zero-filled")

    key_maps = load_key_layer_maps(npy_dir, B, layers=[2, 4, 7])
    key_missing = sum(1 for m in key_maps if m is None)
    if key_missing == B:
        print(f"  [warn] no key-layer[2,4,7] npy found -> fallback to query-only score")
        key_maps = None
    elif key_missing > 0:
        print(f"  [warn] {key_missing}/{B} frames missing key-layer[2,4,7] -> per-frame fallback")

    qv_maps = load_var_layer_maps(npy_dir, B, side='query', layers=[11, 18, 23])
    kv_maps = load_var_layer_maps(npy_dir, B, side='key',   layers=[2, 4, 7])
    qv_missing = sum(1 for m in qv_maps if m is None)
    kv_missing = sum(1 for m in kv_maps if m is None)
    if qv_missing == B or kv_missing == B:
        print(f"  [warn] var maps not found (qv_miss={qv_missing}, kv_miss={kv_missing}) -> no var term")
        qv_maps = kv_maps = None
    else:
        print(f"  var maps: qv_missing={qv_missing}/{B}, kv_missing={kv_missing}/{B}")

    # Infer token grid from first non-None map
    first_map = next((m for m in attn_maps if m is not None), None)
    if first_map is None:
        print("  [error] no attn maps found, skipping")
        return
    Ht, Wt = first_map.shape
    has_var = qv_maps is not None and kv_maps is not None
    has_key = key_maps is not None
    if has_var:
        score_desc = "(1-norm(qm)) * norm(km) * norm(qv) * norm(kv)"
    elif has_key:
        score_desc = "(1-norm(qm)) * norm(km)"
    else:
        score_desc = "1 - norm(qm)"
    print(f"  [Debug] attention tensor shape: [{B}, {Ht}, {Wt}]  score={score_desc}")

    # ── 2. Build dynamic token map  [B, Ht, Wt], HIGH = dynamic ────────────
    attn_tensor = build_dynamic_token_map(
        attn_maps, Ht, Wt, key_maps=key_maps, qv_maps=qv_maps, kv_maps=kv_maps
    ).to(device)
    print(f"  [Debug] attn_tensor: {attn_tensor.shape}  "
          f"min={attn_tensor.min():.3f} max={attn_tensor.max():.3f}")

    # ── 3. Easi3R-style region group tracking ───────────────────────────────
    seq_out   = output_root / seq_name
    seq_out.mkdir(parents=True, exist_ok=True)
    groups_vis_dir = seq_out / "region_groups"

    region_groups, _ = generate_region_groups_with_tracking(
        imgs, device,
        min_size=args.min_region_size,
        max_objects=args.max_objects,
        vis_dir=groups_vis_dir,
    )
    print(f"  [Debug] region_groups: {len(region_groups)} frames, "
          f"shape {region_groups[0].shape}")

    # ── 4. Downsample group_ids to token grid ───────────────────────────────
    groups_hr    = torch.stack(region_groups, dim=0).to(device)   # [B, H, W]
    groups_token = downsample_groups_to_tokens(groups_hr, Ht, Wt) # [B, Ht, Wt]
    n_unique     = int(torch.unique(groups_token[groups_token != 0]).numel())
    print(f"  [Debug] groups_token: {groups_token.shape}  unique objects: {n_unique}")

    # ── 5. Object-level global attention ────────────────────────────────────
    obj_global = compute_object_global_attention(attn_tensor, groups_token,
                                                  top_region_pct=args.top_region_pct)
    print(f"  [Debug] object attention range: "
          f"[{min(obj_global.values()):.4f}, {max(obj_global.values()):.4f}]"
          if obj_global else "  [Debug] no objects found")

    # ── 6. Adaptive multi-Otsu threshold -> dynamic object ids ──────────────
    dynamic_ids = decide_dynamic_objects(
        obj_global, groups_hr, H, W, B,
        method=args.threshold_method, topk_pct=args.topk_pct,
    )
    print(f"  [Debug] dynamic_object_ids: {sorted(dynamic_ids)}")

    # ── 7. Build per-frame binary masks ─────────────────────────────────────
    final_masks = build_masks_from_dynamic_objects(region_groups, dynamic_ids, H, W)

    # Print per-frame stats (first 5 frames)
    for fi in range(min(5, B)):
        ratio = final_masks[fi].mean() * 100
        print(f"  [Debug] frame {fi:05d}: {ratio:.2f}% dynamic pixels")

    # ── 8. Save results ─────────────────────────────────────────────────────
    # Binary masks directly under <seq_out>/
    for i, mask in enumerate(final_masks):
        Image.fromarray((mask * 255).astype(np.uint8)).save(seq_out / f"{i:05d}.png")
    print(f"  saved {B} masks -> {seq_out}")

    # SAM2 region colouring (per-frame PNG)
    save_region_vis(imgs, region_groups, seq_out / "sam2_regions")
    print(f"  saved region vis -> {seq_out / 'sam2_regions'}")

    # Build full-res dynamic score maps for visualisation
    attn_hr = [
        cv2.resize(attn_tensor[b].cpu().numpy(), (W, H), interpolation=cv2.INTER_LINEAR)
        for b in range(B)
    ]

    save_vis(imgs, attn_hr, region_groups, final_masks,
             seq_out / "vis", dynamic_ids)
    print(f"  saved vis -> {seq_out / 'vis'}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="VGGT cross-frame attn + Easi3R-style SAM2 region tracking -> dynamic masks"
    )
    p.add_argument("--davis_root",       type=str, required=True,
                   help="DAVIS JPEGImages root, e.g. .../480p")
    p.add_argument("--attn_root",        type=str, required=True,
                   help="output_dir used in viz_cross_frame_attn.py (contains <seq>/npy/)")
    p.add_argument("--output_dir",       type=str, default="vggt_sam2_out")
    p.add_argument("--seq_list",         type=str, nargs="+", default=None,
                   help="Sequences to process: a .txt file path (one name per line) "
                        "OR inline names e.g. 'bear bikepacking camel' "
                        "(default: all sub-dirs)")
    p.add_argument("--resolution",       type=int, default=518,
                   help="Square resize resolution (must match viz script, default 518)")
    p.add_argument("--min_region_size",  type=int, default=500,
                   help="Minimum pixel area for initial SAM2 region proposals (default 500)")
    p.add_argument("--max_objects",      type=int, default=30,
                   help="Maximum number of objects to track (default 30)")
    p.add_argument("--threshold_method", type=str, default="multiotsu",
                   choices=["multiotsu", "topk"],
                   help="Method to select dynamic objects: 'multiotsu' (adaptive, default) "
                        "or 'topk' (top K%% by score)")
    p.add_argument("--topk_pct",         type=float, default=25.0,
                   help="Top-K%% threshold when --threshold_method=topk (default 25.0)")
    p.add_argument("--top_region_pct",   type=float, default=5.0,
                   help="Top-K%% of tokens used for region attention pooling (default 5.0; "
                        "100.0 = mean pooling)")
    return p.parse_args()


def _resolve_seq_list(seq_list_arg):
    if seq_list_arg is None:
        return None
    if len(seq_list_arg) == 1 and Path(seq_list_arg[0]).is_file():
        lines = Path(seq_list_arg[0]).read_text().splitlines()
        return [l.strip() for l in lines if l.strip()]
    return seq_list_arg


def main():
    args        = parse_args()
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # SAM2 requires running from the sam2 package root so Hydra can find configs
    os.chdir(str(_EASI3R_ROOT))

    davis_root  = Path(args.davis_root)
    attn_root   = Path(args.attn_root)
    output_root = Path(args.output_dir)

    seq_names = _resolve_seq_list(args.seq_list)
    if seq_names is not None:
        seq_dirs = [davis_root / s for s in seq_names]
    else:
        seq_dirs = sorted(d for d in davis_root.iterdir() if d.is_dir())

    for seq_dir in seq_dirs:
        if not seq_dir.is_dir():
            print(f"  [warn] not found: {seq_dir}")
            continue
        run_sequence(seq_dir, attn_root, output_root, args, device)

    print(f"\ndone -> {output_root}/")


if __name__ == "__main__":
    main()
