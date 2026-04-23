#!/usr/bin/env python3
"""
demo_attn_analysis.py  –  Comprehensive VGGT attention analysis demo.

Analyses VGGT self-attention in four directions, all from a single forward pass:

  1. cross_block       Direction 1 – Global attention cross-frame sub-block analysis.
                       Extracts A[i→j] sub-blocks from global self-attention.
                       Closest VGGT analogue to Easi3R cross-view attention.

  2. frame_vs_global   Direction 2 – Compare frame-attention vs global-attention.
                       Separate heatmaps show which patches are "interesting"
                       within-frame vs across all frames.

  3. first_frame_asym  Direction 3 – First-frame-anchored asymmetric statistics.
                       VGGT outputs geometry in frame-0 coordinates, so we define
                       a pseudo-reference asymmetry: A[k→0] vs A[0→k].
                       This is NOT native cross-attention; it is a manually defined
                       asymmetry inspired by the first-frame anchoring in VGGT.

  4. residual          Direction 4 – Feature-update / residual magnitude.
                       ||x_after_block − x_before_block||₂ per patch token.
                       More robust than raw attention weights for dynamic-region signals.

Usage example:
    python demo_attn_analysis.py \\
        --seq_dir /path/to/frames \\
        --out_root ./attn_out \\
        --num_frames 8 \\
        --layers 4 8 16 23 \\
        --mode cross_block frame_vs_global first_frame_asym residual \\
        --save_npy

Output structure:
    out_root/
      overlays/
        frame_000/
          dir1_cross_frame_layerXX.png
          dir2_frame_vs_global_layerXX.png
          dir3_asym_k_to_0_layerXX.png
          dir4_residual_layerXX.png
      summaries/
          layer_XX_pair_matrix.png
          layer_XX_stats.json
      tensors/
          stats.npz               (if --save_npy)
      index.html
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from vggt.models.vggt import VGGT
from vggt.attn_utils import GlobalAttnCapture, FrameAttnCapture, BlockResidualCapture


# ─────────────────────────────────────────────────────────────────────────────
# Utility: image loading & preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def load_images(seq_dir: str, num_frames: int, stride: int, resolution: int
                ) -> Tuple[List[np.ndarray], torch.Tensor]:
    """
    Load frames from seq_dir, subsample, resize, and return:
        images_np   list of [H, W, 3] uint8 numpy arrays  (for visualization)
        images_t    torch.Tensor [1, S, 3, H, W] float in [0, 1]
    """
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = sorted(p for p in Path(seq_dir).iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise RuntimeError(f"No images found in {seq_dir}")

    paths = paths[::stride][:num_frames]
    print(f"  Loaded {len(paths)} frames from {seq_dir}")

    # resolution must be divisible by 14 (patch_size)
    res = (resolution // 14) * 14

    images_np = []
    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((res, res), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        images_np.append(arr)
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        tensors.append(t)

    images_t = torch.stack(tensors, dim=0).unsqueeze(0)  # [1, S, 3, H, W]
    return images_np, images_t


# ─────────────────────────────────────────────────────────────────────────────
# Utility: visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def heatmap_rgb(data: np.ndarray, cmap: str = "inferno") -> np.ndarray:
    """Convert 2-D float array to [H, W, 3] uint8 using colormap."""
    rgba = matplotlib.colormaps[cmap](norm01(data))
    return (rgba[..., :3] * 255).astype(np.uint8)


def overlay(img: np.ndarray, hmap: np.ndarray, cmap: str = "inferno",
            alpha: float = 0.55) -> np.ndarray:
    """Blend heatmap [H_p, W_p] over image [H, W, 3].  Resizes hmap if needed."""
    H, W = img.shape[:2]
    hmap_up = np.array(
        Image.fromarray(heatmap_rgb(hmap, cmap)).resize((W, H), Image.BILINEAR)
    )
    return (alpha * hmap_up.astype(float) + (1 - alpha) * img.astype(float)).astype(np.uint8)


def save_panel(rows: List[List[np.ndarray]],
               titles: List[List[str]],
               row_labels: List[str],
               path: str,
               suptitle: str = "",
               dpi: int = 130):
    """
    Save a grid of images.
        rows[r][c]   = [H, W, 3] uint8 image for row r, col c
        titles[r][c] = caption string
        row_labels   = label for each row (left margin text)
    """
    R = len(rows)
    C = max(len(r) for r in rows)
    fig, axes = plt.subplots(R, C, figsize=(3 * C, 3.2 * R), squeeze=False)
    for r in range(R):
        for c in range(C):
            ax = axes[r, c]
            if c < len(rows[r]):
                ax.imshow(rows[r][c])
                ax.set_title(titles[r][c], fontsize=7)
            ax.axis("off")
        axes[r, 0].set_ylabel(row_labels[r], fontsize=8, rotation=0,
                               labelpad=55, va='center')
    if suptitle:
        fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def patch_to_grid(vec: np.ndarray, H_p: int, W_p: int) -> np.ndarray:
    """Reshape flat patch vector [Pp] to spatial grid [H_p, W_p]."""
    assert vec.shape[0] == H_p * W_p, f"Expected {H_p*W_p} patches, got {vec.shape[0]}"
    return vec.reshape(H_p, W_p)


# ─────────────────────────────────────────────────────────────────────────────
# Direction 1 : Global attention cross-frame sub-block analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_direction1(gcap: GlobalAttnCapture,
                   images_np: List[np.ndarray],
                   H_p: int, W_p: int,
                   out_root: str,
                   selected_layers: List[int],
                   save_npy: bool) -> Dict:
    """
    Direction 1: Global attention cross-frame sub-block analysis.

    For each selected layer and each frame j, visualises:
      - cross-frame received attention: mean over i≠j of recv[i→j]
      - intra-frame attention:          recv[j→j]
      - cross / (cross + intra) ratio   (attention fraction going across frames)
      - per-source heatmap grid         (recv[i→j] for all i)

    Also saves a frame-pair summary matrix (S×S grid of mean attentions).

    Interpretation guide
    ─────────────────────
    High cross-frame attention on a patch → that patch is "queried" by other
    frames, suggesting it is distinctive or geometrically informative.
    High variance of recv[i→j] across i → that patch in frame j is attended
    differently by different source frames → possible dynamic or occluded region.
    """
    S = len(images_np)
    over_dir = os.path.join(out_root, "overlays")
    sum_dir = os.path.join(out_root, "summaries")
    os.makedirs(sum_dir, exist_ok=True)

    # For JSON stats and optional npy save
    stats_all = {}
    npy_data  = {}

    for lid in selected_layers:
        layer_stats = {}

        # ── frame-pair matrix: mean recv_patch[i→j] scalar ──────────────
        # recv_patch sums attention from ALL frame-i tokens (cam+reg+patch)
        # to each frame-j PATCH token, then takes the spatial mean.
        pair_matrix = np.zeros((S, S))
        for i in range(S):
            for j in range(S):
                rv = gcap.recv_patch(lid, i, j)
                if rv is not None:
                    pair_matrix[i, j] = rv.mean()

        fig, ax = plt.subplots(figsize=(max(4, S), max(4, S)))
        im = ax.imshow(pair_matrix, cmap="viridis", vmin=0)
        ax.set_xticks(range(S)); ax.set_xticklabels([f"f{j}" for j in range(S)], fontsize=8)
        ax.set_yticks(range(S)); ax.set_yticklabels([f"f{i}" for i in range(S)], fontsize=8)
        ax.set_xlabel("Destination frame j  (patch keys)", fontsize=8)
        ax.set_ylabel("Source frame i  (all queries: cam+reg+patch)", fontsize=8)
        ax.set_title(f"[Dir1] Cross-frame attention matrix  layer {lid}\n"
                     f"full softmax · all query types · values = mean recv_patch", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.04)
        matrix_path = os.path.join(sum_dir, f"layer_{lid:02d}_pair_matrix.png")
        fig.savefig(matrix_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        layer_stats["pair_matrix_path"] = matrix_path
        layer_stats["pair_matrix"] = pair_matrix.tolist()

        # ── per-frame heatmaps + camera-attention strip ───────────────────
        for j in range(S):
            img = images_np[j]

            # Cross-frame received: mean over all i≠j
            # Uses ALL query token types (camera, register, patch) from frame i
            cross_maps = []
            for i in range(S):
                if i == j:
                    continue
                rv = gcap.recv_patch(lid, i, j)
                if rv is not None:
                    cross_maps.append(patch_to_grid(rv, H_p, W_p))

            if not cross_maps:
                continue

            cross_mean = np.mean(np.stack(cross_maps, 0), 0)  # [H_p, W_p]
            cross_var  = np.var( np.stack(cross_maps, 0), 0)

            intra_rv  = gcap.recv_patch(lid, j, j)
            intra_map = patch_to_grid(intra_rv, H_p, W_p) if intra_rv is not None \
                        else np.zeros((H_p, W_p))

            total     = cross_mean + intra_map
            ratio     = cross_mean / (total + 1e-8)
            dyn_score = norm01(cross_var) * (1.0 - norm01(cross_mean))

            fdir = os.path.join(over_dir, f"frame_{j:03d}")
            os.makedirs(fdir, exist_ok=True)

            rows = [[img, overlay(img, cross_mean), overlay(img, intra_map),
                     overlay(img, ratio, "RdYlGn"), overlay(img, dyn_score, "hot")]]
            ttl  = [[f"frame {j}", "cross-frame recv\n(mean i≠j, all query types)",
                     "intra-frame recv\n(j→j, all query types)",
                     "cross / total ratio", "dynamic score\n(var×(1−mean))"]]
            path = os.path.join(fdir, f"dir1_cross_frame_layer{lid:02d}.png")
            save_panel(rows, ttl, [f"layer {lid}"], path,
                       suptitle=f"[Dir1] Cross-frame attention  frame {j}  layer {lid}\n"
                                 f"(full softmax, all query types: camera+register+patch)")

            # Per-source breakdown
            if len(cross_maps) <= 8:
                src_row = [img] + [overlay(img, m) for m in cross_maps]
                src_ttl = [f"frame {j}"] + [f"recv from f{i}" for i in range(S) if i != j]
                path2 = os.path.join(fdir, f"dir1_per_source_layer{lid:02d}.png")
                save_panel([src_row], [src_ttl], ["sources"], path2,
                           suptitle=f"[Dir1] Per-source recv  frame {j}  layer {lid}")

            # ── Camera-token attention row (bonus, only available with full attn) ──
            # cam_row[j]: where the camera token of frame j attends across all S*P tokens.
            # Reshape: for each source frame i, extract its patch portion → [H_p, W_p]
            cam_row = gcap.cam_row(lid, j)   # [N] = [S*P]
            if cam_row is not None:
                P_tok = gcap.P
                ps    = gcap.ps
                cam_patch_maps = []
                for i in range(S):
                    cam_to_fi = cam_row[i*P_tok + ps:(i+1)*P_tok]  # [Pp]
                    cam_patch_maps.append(patch_to_grid(cam_to_fi, H_p, W_p))

                cam_row_imgs = [img] + [overlay(images_np[i], cam_patch_maps[i], "plasma")
                                         for i in range(S)]
                cam_ttl = [f"cam-tok frame {j}"] + [f"→ frame {i} patches" for i in range(S)]
                path3 = os.path.join(fdir, f"dir1_cam_attn_layer{lid:02d}.png")
                save_panel([cam_row_imgs], [cam_ttl], [f"cam-row f{j}"], path3,
                           suptitle=f"[Dir1] Camera-token attention  frame {j}  layer {lid}\n"
                                     f"where does frame-{j} camera token attend?")

            if save_npy:
                npy_data[f"dir1_l{lid}_f{j}_cross_mean"] = cross_mean
                npy_data[f"dir1_l{lid}_f{j}_cross_var"]  = cross_var
                npy_data[f"dir1_l{lid}_f{j}_intra"]      = intra_map
                npy_data[f"dir1_l{lid}_f{j}_dyn_score"]  = dyn_score
                if cam_row is not None:
                    npy_data[f"dir1_l{lid}_f{j}_cam_row"] = cam_row

        stats_all[f"layer_{lid:02d}"] = layer_stats

    return {"stats": stats_all, "npy_data": npy_data}


# ─────────────────────────────────────────────────────────────────────────────
# Direction 2 : Frame-attention vs global-attention comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_direction2(gcap: GlobalAttnCapture,
                   fcap: FrameAttnCapture,
                   images_np: List[np.ndarray],
                   H_p: int, W_p: int,
                   out_root: str,
                   selected_layers: List[int],
                   save_npy: bool) -> Dict:
    """
    Direction 2: Frame-attention vs global-attention comparison.

    For each selected layer and each frame:
      frame_score  = received attention in frame-wise attention blocks
      global_score = total received attention in global attention blocks
                     (sum of recv[i→j] over all i, including i=j)
      diff         = global_score - frame_score   (what global adds over frame)

    Interpretation guide
    ─────────────────────
    If diff is large on certain patches → those patches benefit from cross-frame
    context in global attention.
    If frame_score ≈ global_score → that region's representation is mostly
    determined by within-frame features.
    Dynamic objects tend to show larger global-vs-frame difference because other
    frames provide disambiguating context.
    """
    S = len(images_np)
    npy_data = {}
    over_dir = os.path.join(out_root, "overlays")

    for lid in selected_layers:
        for s in range(S):
            img = images_np[s]

            # Frame-attn: received patch attention from within-frame block
            # Uses ALL query types (camera+register+patch) → full P×P attn, patch-col sum
            f_scores = fcap.recv_patch(lid, s)    # [Pp] or None
            if f_scores is None:
                continue
            f_map = patch_to_grid(f_scores, H_p, W_p)   # [H_p, W_p]

            # Global score for frame s: sum received attention from all source frames
            # Uses ALL query types from each source frame (full softmax)
            g_maps = []
            for i in range(S):
                rv = gcap.recv_patch(lid, i, s)
                if rv is not None:
                    g_maps.append(patch_to_grid(rv, H_p, W_p))
            if not g_maps:
                continue
            g_map = np.sum(np.stack(g_maps, 0), 0)  # [H_p, W_p]

            # Normalise each to [0,1] before difference (different scales otherwise)
            f_n = norm01(f_map)
            g_n = norm01(g_map)
            diff_map = g_n - f_n          # positive: global > frame
            diff_abs  = np.abs(diff_map)

            fdir = os.path.join(over_dir, f"frame_{s:03d}")
            os.makedirs(fdir, exist_ok=True)

            rows = [[
                img,
                overlay(img, f_map,    "Blues"),
                overlay(img, g_map,    "Oranges"),
                overlay(img, diff_map, "RdBu_r"),
                overlay(img, diff_abs, "hot"),
            ]]
            ttl = [[
                f"frame {s}",
                f"frame-attn score (layer {lid})",
                f"global-attn score (layer {lid})",
                "global − frame (normalised)",
                "|global − frame|",
            ]]
            path = os.path.join(fdir, f"dir2_frame_vs_global_layer{lid:02d}.png")
            save_panel(rows, ttl, [f"layer {lid}"], path,
                       suptitle=f"[Dir2] Frame vs Global attention  frame {s}  layer {lid}")

            if save_npy:
                npy_data[f"dir2_l{lid}_f{s}_frame"] = f_map
                npy_data[f"dir2_l{lid}_f{s}_global"] = g_map
                npy_data[f"dir2_l{lid}_f{s}_diff"]  = diff_map

    return {"npy_data": npy_data}


# ─────────────────────────────────────────────────────────────────────────────
# Direction 3 : First-frame-anchored asymmetric statistics
# ─────────────────────────────────────────────────────────────────────────────

def run_direction3(gcap: GlobalAttnCapture,
                   images_np: List[np.ndarray],
                   H_p: int, W_p: int,
                   out_root: str,
                   selected_layers: List[int],
                   save_npy: bool) -> Dict:
    """
    Direction 3: First-frame-anchored asymmetric statistics.

    VGGT outputs geometry in the coordinate system of frame 0 (the reference frame).
    Although VGGT backbone uses self-attention (not cross-attention), we can manually
    define a pseudo-reference/pseudo-source asymmetry analogous to Easi3R:

        A[k→0]: attention from frame k patches TO frame 0 patches
                = how much frame k "looks at" frame 0
        A[0→k]: attention from frame 0 patches TO frame k patches
                = how much frame 0 "looks at" frame k
        diff  : A[k→0] − A[0→k]
                > 0 on patches where frame k queries frame 0 more than vice versa

    This is NOT native cross-attention.  It is a manually defined asymmetry
    inspired by the first-frame coordinate anchoring of VGGT.

    Interpretation guide
    ─────────────────────
    Asymmetric regions (large |diff|) may correspond to scene content that is
    visible in frame 0 but occluded / out-of-frame in frame k, or vice versa.
    Dynamic objects that have moved between frame 0 and frame k may appear as
    regions where A[0→k] is high (frame 0 trying to find the object in k).
    """
    S = len(images_np)
    npy_data = {}
    over_dir = os.path.join(out_root, "overlays")

    if S < 2:
        print("  [Dir3] Need ≥2 frames for first-frame asymmetry, skipping.")
        return {"npy_data": {}}

    for lid in selected_layers:
        # Aggregate statistics for frame 0 as reference
        k_to_0_all = []   # [k, H_p, W_p] for k > 0
        zero_to_k_all = []

        for k in range(1, S):
            # recv_patch uses ALL query token types (cam+reg+patch) from the source frame
            # and full global softmax, so the asymmetry is well-defined.
            k0  = gcap.recv_patch(lid, k, 0)   # all frame-k tokens → frame-0 patches
            z_k = gcap.recv_patch(lid, 0, k)   # all frame-0 tokens → frame-k patches

            if k0 is None or z_k is None:
                continue

            k0_map  = patch_to_grid(k0,  H_p, W_p)
            zk_map  = patch_to_grid(z_k, H_p, W_p)
            diff_map = norm01(k0_map) - norm01(zk_map)

            k_to_0_all.append(k0_map)
            zero_to_k_all.append(zk_map)

            fdir = os.path.join(over_dir, f"frame_{k:03d}")
            os.makedirs(fdir, exist_ok=True)

            rows = [[
                images_np[k],
                overlay(images_np[k], k0_map,  "Purples"),
                overlay(images_np[0], zk_map,  "Greens"),
                overlay(images_np[k], diff_map, "RdBu_r"),
                overlay(images_np[k], np.abs(diff_map), "hot"),
            ]]
            ttl = [[
                f"frame {k}",
                f"A[{k}→0] recv on frame 0  (layer {lid})",
                f"A[0→{k}] recv on frame {k}  (layer {lid})",
                f"A[{k}→0] − A[0→{k}]  (norm diff)",
                f"|asym| magnitude",
            ]]
            path = os.path.join(fdir, f"dir3_asym_k{k}_to_ref_layer{lid:02d}.png")
            save_panel(rows, ttl, [f"layer {lid}"], path,
                       suptitle=f"[Dir3] First-frame asymmetry  frame {k} vs 0  layer {lid}\n"
                                 f"(pseudo-reference asymmetry, NOT native cross-attention)")

            if save_npy:
                npy_data[f"dir3_l{lid}_k{k}_to0"] = k0_map
                npy_data[f"dir3_l{lid}_0_to_k{k}"] = zk_map
                npy_data[f"dir3_l{lid}_k{k}_diff"]  = diff_map

        # Aggregate summary for frame 0: mean over all k>0
        if k_to_0_all:
            mean_k_to_0  = np.mean(np.stack(k_to_0_all, 0), 0)
            mean_0_to_k  = np.mean(np.stack(zero_to_k_all, 0), 0)
            std_k_to_0   = np.std( np.stack(k_to_0_all, 0), 0)
            agg_diff     = norm01(mean_k_to_0) - norm01(mean_0_to_k)

            f0dir = os.path.join(over_dir, "frame_000")
            os.makedirs(f0dir, exist_ok=True)

            rows = [[
                images_np[0],
                overlay(images_np[0], mean_k_to_0,  "Purples"),
                overlay(images_np[0], mean_0_to_k,  "Greens"),
                overlay(images_np[0], std_k_to_0,   "YlOrRd"),
                overlay(images_np[0], agg_diff,      "RdBu_r"),
            ]]
            ttl = [[
                "frame 0 (reference)",
                "mean A[k→0]  (all k>0)",
                "mean A[0→k]  (all k>0)",
                "std A[k→0]   (consistency)",
                "mean(A[k→0]) − mean(A[0→k])",
            ]]
            path = os.path.join(f0dir, f"dir3_frame0_aggregate_layer{lid:02d}.png")
            save_panel(rows, ttl, [f"layer {lid}"], path,
                       suptitle=f"[Dir3] Frame-0 reference aggregate  layer {lid}")

            if save_npy:
                npy_data[f"dir3_l{lid}_mean_k_to_0"] = mean_k_to_0
                npy_data[f"dir3_l{lid}_mean_0_to_k"] = mean_0_to_k
                npy_data[f"dir3_l{lid}_std_k_to_0"]  = std_k_to_0

    return {"npy_data": npy_data}


# ─────────────────────────────────────────────────────────────────────────────
# Direction 4 : Residual / feature-update magnitude
# ─────────────────────────────────────────────────────────────────────────────

def run_direction4(rcap: BlockResidualCapture,
                   images_np: List[np.ndarray],
                   H_p: int, W_p: int,
                   out_root: str,
                   selected_layers: List[int],
                   save_npy: bool) -> Dict:
    """
    Direction 4: Feature-update / residual magnitude visualisation.

    ||x_after_block − x_before_block||₂ per patch token (attn + MLP combined).

    For each selected layer and each frame:
      frame_res   = residual magnitude in frame_blocks[layer]
      global_res  = residual magnitude in global_blocks[layer]
      diff        = global_res − frame_res   (extra update from cross-frame block)

    Also saves layer-wise mean/std curves and a comparison between frame and global
    residuals averaged over all patches.

    Interpretation guide
    ─────────────────────
    Large residual norm → the block is significantly updating the token's
    representation.  Dynamic / inconsistent patches are expected to have larger
    global-block residuals (because cross-frame context forces a bigger update).
    Comparing frame_res vs global_res directly shows which patches benefit most
    from cross-frame information.
    """
    S = len(images_np)
    npy_data = {}
    over_dir = os.path.join(out_root, "overlays")
    sum_dir  = os.path.join(out_root, "summaries")
    os.makedirs(sum_dir, exist_ok=True)

    # Layer-level summary: mean residual per layer (scalar per frame)
    layer_ids = sorted(selected_layers)
    frame_layer_mean  = {}   # lid -> [S] mean residual norm
    global_layer_mean = {}

    for lid in layer_ids:
        fr = rcap.frame_residuals(lid)   # [S, Pp] or None
        gr = rcap.global_residuals(lid)  # [S, Pp] or None

        if fr is not None:
            frame_layer_mean[lid]  = fr.mean(axis=1)   # [S]
        if gr is not None:
            global_layer_mean[lid] = gr.mean(axis=1)   # [S]

        if fr is None and gr is None:
            continue

        for s in range(S):
            img = images_np[s]

            fr_map = patch_to_grid(fr[s], H_p, W_p) if fr is not None else np.zeros((H_p, W_p))
            gr_map = patch_to_grid(gr[s], H_p, W_p) if gr is not None else np.zeros((H_p, W_p))
            diff_map = norm01(gr_map) - norm01(fr_map)  # global extra update

            fdir = os.path.join(over_dir, f"frame_{s:03d}")
            os.makedirs(fdir, exist_ok=True)

            rows = [[
                img,
                overlay(img, fr_map,   "Blues"),
                overlay(img, gr_map,   "Oranges"),
                overlay(img, diff_map, "RdBu_r"),
            ]]
            ttl = [[
                f"frame {s}",
                f"frame-block residual (layer {lid})",
                f"global-block residual (layer {lid})",
                "global − frame residual (norm)",
            ]]
            path = os.path.join(fdir, f"dir4_residual_layer{lid:02d}.png")
            save_panel(rows, ttl, [f"layer {lid}"], path,
                       suptitle=f"[Dir4] Residual magnitude  frame {s}  layer {lid}\n"
                                 f"residual = ||x_after_block − x_before||₂  (attn+MLP)")

            if save_npy:
                npy_data[f"dir4_l{lid}_f{s}_frame"]  = fr_map
                npy_data[f"dir4_l{lid}_f{s}_global"] = gr_map
                npy_data[f"dir4_l{lid}_f{s}_diff"]   = diff_map

    # ── Layer-mean curve plot ────────────────────────────────────────────────
    if frame_layer_mean or global_layer_mean:
        fig, axes = plt.subplots(1, S, figsize=(4 * S, 4), sharey=True, squeeze=False)
        for s in range(S):
            ax = axes[0, s]
            if frame_layer_mean:
                lids_f = sorted(frame_layer_mean.keys())
                ax.plot(lids_f, [frame_layer_mean[l][s] for l in lids_f],
                        'b-o', markersize=4, label="frame block")
            if global_layer_mean:
                lids_g = sorted(global_layer_mean.keys())
                ax.plot(lids_g, [global_layer_mean[l][s] for l in lids_g],
                        'r-o', markersize=4, label="global block")
            ax.set_title(f"frame {s}", fontsize=8)
            ax.set_xlabel("layer", fontsize=7)
            if s == 0:
                ax.set_ylabel("mean residual norm", fontsize=7)
                ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle("[Dir4] Per-layer mean residual norm per frame", fontsize=9)
        fig.tight_layout()
        curve_path = os.path.join(sum_dir, "dir4_layer_residual_curves.png")
        fig.savefig(curve_path, bbox_inches="tight", dpi=120)
        plt.close(fig)

    return {"npy_data": npy_data}


# ─────────────────────────────────────────────────────────────────────────────
# HTML index generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_html(out_root: str, images_np: List[np.ndarray]):
    """
    Generate a simple browsable HTML index of all output images.
    Organised by frame directory, listing every PNG found.
    """
    S = len(images_np)
    over_dir = os.path.join(out_root, "overlays")
    sum_dir  = os.path.join(out_root, "summaries")

    lines = ["<html><head><meta charset='utf-8'>",
             "<style>body{font-family:monospace;background:#111;color:#eee}",
             "h1,h2,h3{color:#9cf} img{margin:4px;border:1px solid #444}",
             "section{border-bottom:1px solid #333;padding:8px 0}</style></head><body>",
             "<h1>VGGT Attention Analysis</h1>"]

    # ── Summaries section ────────────────────────────────────────────────────
    lines.append("<h2>Summaries</h2>")
    sum_path = os.path.relpath(sum_dir, out_root)
    for fname in (sorted(os.listdir(sum_dir)) if os.path.isdir(sum_dir) else []):
        if fname.endswith(".png"):
            rel = os.path.join(sum_path, fname)
            lines.append(f"<section><h3>{fname}</h3>"
                         f"<img src='{rel}' style='max-width:900px'></section>")

    # ── Per-frame section ────────────────────────────────────────────────────
    lines.append("<h2>Per-frame Overlays</h2>")
    for s in range(S):
        fdir = os.path.join(over_dir, f"frame_{s:03d}")
        if not os.path.isdir(fdir):
            continue
        lines.append(f"<h3>Frame {s:03d}</h3><section>")
        for fname in sorted(os.listdir(fdir)):
            if not fname.endswith(".png"):
                continue
            rel = os.path.relpath(os.path.join(fdir, fname), out_root)
            lines.append(f"<div style='display:inline-block;vertical-align:top;margin:6px'>"
                         f"<div style='font-size:11px;color:#9cf'>{fname}</div>"
                         f"<img src='{rel}' style='max-width:700px'></div>")
        lines.append("</section>")

    lines.append("</body></html>")

    html_path = os.path.join(out_root, "index.html")
    with open(html_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [html] {html_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="VGGT attention analysis demo (4 directions)"
    )
    p.add_argument("--seq_dir",    required=True,
                   help="Folder of input images (sorted alphabetically)")
    p.add_argument("--out_root",   default="./attn_analysis_out",
                   help="Output root directory")
    p.add_argument("--num_frames", type=int, default=8,
                   help="Maximum number of frames to use")
    p.add_argument("--frame_stride", type=int, default=1,
                   help="Stride when loading frames")
    p.add_argument("--layers",     type=int, nargs="+", default=[4, 8, 16, 23],
                   help="Layer indices to analyse (0-23)")
    p.add_argument("--heads",      type=int, nargs="+", default=None,
                   help="Head indices to use (default: all 16 heads)")
    p.add_argument("--mode",       type=str, nargs="+",
                   default=["cross_block", "frame_vs_global",
                            "first_frame_asym", "residual"],
                   choices=["cross_block", "frame_vs_global",
                            "first_frame_asym", "residual"],
                   help="Which analysis directions to run")
    p.add_argument("--model_path", type=str, default=None,
                   help="Local model path or HF hub id (default: facebook/VGGT-1B)")
    p.add_argument("--resolution", type=int, default=518,
                   help="Image resolution (will be rounded to nearest multiple of 14)")
    p.add_argument("--save_npy",   action="store_true",
                   help="Save raw statistics as tensors/stats.npz")
    # full_softmax is now always True (the new implementation always uses proper global softmax)
    # kept as a no-op flag for backward compatibility with old command lines
    p.add_argument("--full_softmax", action="store_true",
                   help="(no-op) Full softmax is now always used.")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device={device}")

    # ── Load model ────────────────────────────────────────────────────────────
    hub_id = args.model_path or "facebook/VGGT-1B"
    print(f"[main] loading model from {hub_id} …")
    model = VGGT.from_pretrained(hub_id).to(device).eval()
    agg   = model.aggregator
    print("[main] model loaded")

    # ── Load images ───────────────────────────────────────────────────────────
    print(f"[main] loading images from {args.seq_dir} …")
    images_np, images_t = load_images(
        args.seq_dir, args.num_frames, args.frame_stride, args.resolution
    )
    S = len(images_np)
    images_t = images_t.to(device)

    # ── Derive token dimensions ───────────────────────────────────────────────
    res  = (args.resolution // 14) * 14    # effective resolution
    H_p  = res // agg.patch_size           # patch grid height
    W_p  = res // agg.patch_size           # patch grid width
    Pp   = H_p * W_p                       # patch tokens per frame
    ps   = agg.patch_start_idx             # = 5  (1 cam + 4 reg)
    P    = ps + Pp                         # total tokens per frame

    print(f"[main] S={S} frames  H_p={H_p} W_p={W_p}  Pp={Pp}  "
          f"P={P}  patch_start={ps}")
    print(f"[main] analysing layers: {args.layers}")
    print(f"[main] modes: {args.mode}")

    # Validate layer indices
    bad = [l for l in args.layers if l < 0 or l >= agg.depth]
    if bad:
        raise ValueError(f"Layer indices out of range [0, {agg.depth-1}]: {bad}")

    os.makedirs(args.out_root, exist_ok=True)

    # ── Build capture contexts based on selected modes ────────────────────────
    need_global   = any(m in args.mode for m in
                        ["cross_block", "frame_vs_global", "first_frame_asym"])
    need_frame    = "frame_vs_global" in args.mode
    need_residual = "residual" in args.mode

    gcap = GlobalAttnCapture(agg, args.layers, S, P, ps,
                             selected_heads=args.heads) if need_global else None
    fcap = FrameAttnCapture(agg, args.layers, S, P, ps,
                            selected_heads=args.heads) if need_frame else None
    rcap = BlockResidualCapture(agg, args.layers, S, P, ps) if need_residual else None

    # ── Single forward pass with all captures active ──────────────────────────
    print("[main] running forward pass …")

    def run_forward():
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                agg(images_t)

    # Nest active context managers
    active = [c for c in [gcap, fcap, rcap] if c is not None]
    if not active:
        print("[main] no capture contexts active, nothing to do.")
        return

    # Enter all contexts
    for c in active:
        c.__enter__()
    try:
        run_forward()
    finally:
        for c in reversed(active):
            c.__exit__(None, None, None)

    print("[main] forward pass complete")

    # ── Run analysis directions ────────────────────────────────────────────────
    all_npy = {}

    if "cross_block" in args.mode:
        print("[main] Direction 1: cross-frame sub-block analysis …")
        r = run_direction1(gcap, images_np, H_p, W_p,
                           args.out_root, args.layers, args.save_npy)
        all_npy.update(r["npy_data"])

        # Save layer-pair stats JSON
        sum_dir = os.path.join(args.out_root, "summaries")
        os.makedirs(sum_dir, exist_ok=True)
        for lid in args.layers:
            js = {}
            for i in range(S):
                for j in range(S):
                    rv = gcap.recv_patch(lid, i, j)
                    if rv is not None:
                        js[f"{i}_{j}_mean"] = float(rv.mean())
                        js[f"{i}_{j}_max"]  = float(rv.max())
            with open(os.path.join(sum_dir, f"layer_{lid:02d}_stats.json"), "w") as f:
                json.dump(js, f, indent=2)

    if "frame_vs_global" in args.mode:
        print("[main] Direction 2: frame-attention vs global-attention …")
        r = run_direction2(gcap, fcap, images_np, H_p, W_p,
                           args.out_root, args.layers, args.save_npy)
        all_npy.update(r["npy_data"])

    if "first_frame_asym" in args.mode:
        print("[main] Direction 3: first-frame-anchored asymmetric statistics …")
        r = run_direction3(gcap, images_np, H_p, W_p,
                           args.out_root, args.layers, args.save_npy)
        all_npy.update(r["npy_data"])

    if "residual" in args.mode:
        print("[main] Direction 4: residual magnitude analysis …")
        r = run_direction4(rcap, images_np, H_p, W_p,
                           args.out_root, args.layers, args.save_npy)
        all_npy.update(r["npy_data"])

    # ── Save raw tensors ───────────────────────────────────────────────────────
    if args.save_npy and all_npy:
        ten_dir = os.path.join(args.out_root, "tensors")
        os.makedirs(ten_dir, exist_ok=True)
        npy_path = os.path.join(ten_dir, "stats.npz")
        np.savez_compressed(npy_path, **all_npy)
        print(f"  [tensors] {npy_path}  ({len(all_npy)} arrays)")

    # ── Generate HTML index ────────────────────────────────────────────────────
    generate_html(args.out_root, images_np)

    print(f"\n[main] done — outputs in {args.out_root}/")
    print(f"         browse: {os.path.join(args.out_root, 'index.html')}")


if __name__ == "__main__":
    main()
