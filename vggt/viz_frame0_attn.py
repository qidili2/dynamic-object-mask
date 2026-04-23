#!/usr/bin/env python3
"""
Visualize per-reference-frame cross-attention from frame 0 to every other frame.

For each sequence:
  - Fix frame 0 as the query
  - Run VGGT with [frame_0, ref_1, ref_2, ..., ref_K] in one forward pass
    (refs sampled with --ref_stride, up to --max_refs at a time)
  - For each ref frame j, extract A_g[frame0_patches, frame_j_patches]
    -> sum over key dim -> [H_p, W_p] attention heatmap
  - Save one figure per layer group: rows=layers, cols=ref frames

Output under --output_dir/<seq>/:
  frame0_attn/layer{L}.png     per-layer: frame0 -> each ref
  frame0_attn/avg.png          layer-averaged

Usage:
  python viz_frame0_attn.py \\
      --davis_root /mnt/data0/andy/Easi3R/DAVIS/JPEGImages/480p \\
      --output_dir frame0_attn_out \\
      --seqs blackswan bmx-trees \\
      --layers 4 11 17 23 \\
      --ref_stride 1 \\
      --max_refs 8
"""

import sys
import argparse
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from vggt.models.vggt import VGGT
from vggt.attn_utils import GlobalAttnCapture

COLLECT_LAYERS = [4, 11, 17, 23]


# ─────────────────────────────────────────────────────────────────────────────
# Image I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_seq_paths(seq_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in seq_dir.iterdir() if p.suffix.lower() in exts)


def load_image(path: Path, resolution: int) -> np.ndarray:
    return np.array(
        Image.open(path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    )


def frames_to_tensor(frames: List[np.ndarray], device) -> torch.Tensor:
    t = [torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames]
    return torch.stack(t).unsqueeze(0).to(device)   # [1, S, 3, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# Attention extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_frame0_camera_per_ref(
    model,
    images_tensor: torch.Tensor,   # [1, S, 3, H, W]  frame0 at index 0
    layers: List[int],
) -> Dict[int, List[np.ndarray]]:
    """
    Token layout per frame (from attn_utils.py):
      index 0       : camera token  (1 token)
      index 1..ps-1 : register tokens (ps-1 = 4 tokens, default)
      index ps..P-1 : patch tokens   (Pp = H_p * W_p tokens)

    This function extracts the attention of frame-0's CAMERA TOKEN (index 0)
    as the query, attending to each ref frame's PATCH TOKENS as keys.

    Returns per_ref_maps: Dict[layer_id -> List[np.ndarray [H_p, W_p]]]
      per_ref_maps[lid][r] = attention from frame-0 camera token
                             to each patch token of ref-frame r.
                             (r=0 → frames[1], r=1 → frames[2], ...)
    Values are norm01 within each (layer, ref) map.
    """
    agg = model.aggregator
    _, S, _, H, W = images_tensor.shape
    ps  = agg.patch_start_idx   # = 1 (camera) + 4 (register) = 5
    H_p = H // agg.patch_size
    W_p = W // agg.patch_size
    P   = ps + H_p * W_p       # total tokens per frame

    gcap = GlobalAttnCapture(agg, selected_layers=layers, S=S, P=P, patch_start=ps)

    with gcap, torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            agg(images_tensor)

    def _norm01(x: np.ndarray) -> np.ndarray:
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-8)

    per_ref_maps: Dict[int, List[np.ndarray]] = {}

    for lid in layers:
        A_g = gcap.mean_attn(lid)   # [S*P, S*P] float32 numpy
        if A_g is None:
            continue

        # Camera token of frame 0 is at global index 0*P + 0 = 0
        cam_row = A_g[0, :]   # [S*P]  attention from frame-0 camera to all tokens

        ref_maps = []
        for j in range(1, S):
            # patch tokens of ref frame j: global indices [j*P+ps .. (j+1)*P)
            ref_patch_attn = cam_row[j * P + ps : (j + 1) * P]   # [Pp]
            ref_maps.append(_norm01(ref_patch_attn).reshape(H_p, W_p))

        per_ref_maps[lid] = ref_maps

    return per_ref_maps


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def to_heatmap(data: np.ndarray, size=None) -> np.ndarray:
    rgba = cm.get_cmap("inferno")(norm01(data))
    rgb  = (rgba[..., :3] * 255).astype(np.uint8)
    if size is not None:
        rgb = np.array(Image.fromarray(rgb).resize(size, Image.BILINEAR))
    return rgb


def overlay_heatmap(img: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend heatmap over RGB image."""
    H, W = img.shape[:2]
    heat_rgb = to_heatmap(heat, size=(W, H))
    return (img.astype(np.float32) * (1 - alpha) +
            heat_rgb.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)


def make_figure(
    frame0: np.ndarray,
    ref_frames: List[np.ndarray],
    ref_ids: List[int],
    per_ref_maps: Dict[int, List[np.ndarray]],
    layers: List[int],
    seq_name: str,
    batch_label: str = "",
    overlay: bool = False,
) -> plt.Figure:
    """
    Layout:
      Row 0:               frame 0 image  |  ref frame images  (RGB)
      Row 1..len(layers):  heatmap or overlay of per-ref attention per layer
      Row -1:              layer-averaged heatmap
    """
    avail = [lid for lid in layers if lid in per_ref_maps and per_ref_maps[lid]]
    n_refs = len(ref_frames)
    ncols  = 1 + n_refs             # col 0 = frame0, cols 1..n_refs = refs
    nrows  = 1 + len(avail) + 1     # header + per-layer + avg

    H, W = frame0.shape[:2]
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.5 * ncols, 2.5 * nrows),
                             squeeze=False)

    # ── Row 0: RGB images ────────────────────────────────────────────────────
    axes[0, 0].imshow(frame0)
    axes[0, 0].set_title("frame 0 (query)", fontsize=7)
    axes[0, 0].axis("off")
    for k, (rf, rid) in enumerate(zip(ref_frames, ref_ids)):
        axes[0, k + 1].imshow(rf)
        axes[0, k + 1].set_title(f"ref {rid:05d}", fontsize=7)
        axes[0, k + 1].axis("off")

    # ── Rows 1..len(avail): per-layer heatmaps ───────────────────────────────
    for row, lid in enumerate(avail, start=1):
        axes[row, 0].imshow(frame0)
        axes[row, 0].set_title(f"L{lid} query", fontsize=7)
        axes[row, 0].axis("off")

        ref_maps = per_ref_maps[lid]   # List[H_p, W_p], len = n_refs
        for k in range(n_refs):
            ax = axes[row, k + 1]
            if k < len(ref_maps):
                vis = (overlay_heatmap(ref_frames[k], ref_maps[k])
                       if overlay else to_heatmap(ref_maps[k], size=(W, H)))
                ax.imshow(vis)
            ax.set_title(f"L{lid}→ref{ref_ids[k]:05d}", fontsize=6)
            ax.axis("off")

    # ── Last row: layer-averaged ──────────────────────────────────────────────
    avg_row = 1 + len(avail)
    axes[avg_row, 0].imshow(frame0)
    axes[avg_row, 0].set_title("avg (query)", fontsize=7)
    axes[avg_row, 0].axis("off")

    for k in range(n_refs):
        ax = axes[avg_row, k + 1]
        per_layer = [per_ref_maps[lid][k]
                     for lid in avail
                     if k < len(per_ref_maps[lid])]
        if per_layer:
            avg_map = np.stack(per_layer, 0).mean(0)
            vis = (overlay_heatmap(ref_frames[k], avg_map)
                   if overlay else to_heatmap(avg_map, size=(W, H)))
            ax.imshow(vis)
        ax.set_title(f"avg→ref{ref_ids[k]:05d}", fontsize=6)
        ax.axis("off")

    title = f"{seq_name}  frame0→refs  {batch_label}"
    fig.suptitle(title, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Per-sequence processing
# ─────────────────────────────────────────────────────────────────────────────

def run_sequence(model, seq_dir: Path, args, device):
    seq = seq_dir.name
    print(f"\n{'='*60}\n  seq: {seq}")

    all_paths = load_seq_paths(seq_dir)
    if not all_paths:
        print("  no images, skipping")
        return
    n = len(all_paths)
    if n < 2:
        print("  only 1 frame, skipping")
        return

    out_dir = Path(args.output_dir) / seq / "frame0_attn"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame0 = load_image(all_paths[0], args.resolution)

    # all ref frame indices (skip frame 0)
    all_ref_ids = list(range(1, n, args.ref_stride))
    print(f"  {n} frames  ref_stride={args.ref_stride}  "
          f"total_refs={len(all_ref_ids)}  batch_size={args.max_refs}")

    # accumulate maps across batches: lid -> list of (ref_id, map)
    collected: Dict[int, List[tuple]] = {lid: [] for lid in args.layers}

    # process refs in batches of max_refs
    for batch_start in range(0, len(all_ref_ids), args.max_refs):
        batch_ref_ids = all_ref_ids[batch_start: batch_start + args.max_refs]
        frames = [frame0] + [load_image(all_paths[i], args.resolution)
                             for i in batch_ref_ids]
        tensor = frames_to_tensor(frames, device)

        try:
            per_ref_maps = extract_frame0_camera_per_ref(model, tensor, args.layers)
        except Exception:
            print(f"  [ERROR] batch {batch_ref_ids}: extraction failed")
            traceback.print_exc()
            continue

        for lid, ref_maps in per_ref_maps.items():
            for ref_id, rmap in zip(batch_ref_ids, ref_maps):
                collected[lid].append((ref_id, rmap))

        # ── save per-batch figure ─────────────────────────────────────────────
        avail = [lid for lid in args.layers
                 if lid in per_ref_maps and per_ref_maps[lid]]
        if not avail:
            continue

        ref_frames_batch = frames[1:]
        fig = make_figure(
            frame0, ref_frames_batch, batch_ref_ids,
            per_ref_maps, args.layers,
            seq_name=seq,
            batch_label=f"refs {batch_ref_ids[0]:05d}-{batch_ref_ids[-1]:05d}",
            overlay=not args.no_overlay,
        )
        fig_path = (out_dir /
                    f"batch_{batch_ref_ids[0]:05d}_{batch_ref_ids[-1]:05d}.png")
        fig.savefig(fig_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fig_path.name}  refs={batch_ref_ids}")

    # ── save combined all-refs figure (one column per ref) ───────────────────
    avail_lids = [lid for lid in args.layers if collected.get(lid)]
    if not avail_lids:
        print("  no maps collected")
        return

    all_ref_ids_sorted = sorted({rid for rid, _ in collected[avail_lids[0]]})
    # build per_ref_maps dict for all refs
    full_maps: Dict[int, List[np.ndarray]] = {}
    for lid in avail_lids:
        ordered = sorted(collected[lid], key=lambda x: x[0])
        full_maps[lid] = [m for _, m in ordered]

    # load all ref images (may be many — use thumbnail size for combined figure)
    ref_frames_all = [load_image(all_paths[i], args.resolution)
                      for i in all_ref_ids_sorted]

    # if too many refs, downsample columns for the combined figure
    MAX_COLS = 20
    if len(all_ref_ids_sorted) > MAX_COLS:
        step = len(all_ref_ids_sorted) // MAX_COLS
        sel  = list(range(0, len(all_ref_ids_sorted), step))[:MAX_COLS]
        ref_ids_plot  = [all_ref_ids_sorted[i] for i in sel]
        ref_imgs_plot = [ref_frames_all[i] for i in sel]
        full_maps_plot = {lid: [full_maps[lid][i] for i in sel]
                          for lid in avail_lids}
    else:
        ref_ids_plot   = all_ref_ids_sorted
        ref_imgs_plot  = ref_frames_all
        full_maps_plot = full_maps

    fig_all = make_figure(
        frame0, ref_imgs_plot, ref_ids_plot,
        full_maps_plot, args.layers,
        seq_name=seq,
        batch_label="all refs",
        overlay=not args.no_overlay,
    )
    all_path = out_dir / "all_refs.png"
    fig_all.savefig(all_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig_all)
    print(f"  saved combined figure -> {all_path}")

    # ── optionally save per-ref npy ───────────────────────────────────────────
    if args.save_npy:
        npy_dir = out_dir / "npy"
        npy_dir.mkdir(exist_ok=True)
        for lid in avail_lids:
            for ref_id, rmap in collected[lid]:
                np.save(npy_dir / f"ref_{ref_id:05d}_layer{lid}.npy", rmap)
        # layer-averaged per ref
        for ref_id in all_ref_ids_sorted:
            layers_for_ref = [full_maps[lid][all_ref_ids_sorted.index(ref_id)]
                              for lid in avail_lids]
            avg = np.stack(layers_for_ref, 0).mean(0)
            np.save(npy_dir / f"ref_{ref_id:05d}_avg.npy", avg)
        print(f"  saved npy -> {npy_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize frame-0 cross-frame attention to every other frame"
    )
    p.add_argument("--davis_root",  required=True,
                   help="Root dir containing one sub-dir per sequence")
    p.add_argument("--output_dir",  default="frame0_attn_out")
    p.add_argument("--seqs",        nargs="+", default=None,
                   help="Sequence names or a single .txt file (one name per line); "
                        "default: all sub-dirs")
    p.add_argument("--layers",      type=int, nargs="+", default=COLLECT_LAYERS,
                   help="Transformer block indices to capture (default: 4 11 17 23)")
    p.add_argument("--ref_stride",  type=int, default=1,
                   help="Step between reference frames (default 1 = every frame)")
    p.add_argument("--max_refs",    type=int, default=8,
                   help="Max ref frames per forward pass (GPU memory limit)")
    p.add_argument("--resolution",  type=int, default=518,
                   help="Square resize resolution (multiple of 14; default 518)")
    p.add_argument("--model_path",  default=None,
                   help="Local checkpoint path (default: HF facebook/VGGT-1B)")
    p.add_argument("--no_overlay",  action="store_true",
                   help="Show raw heatmap instead of overlay on image")
    p.add_argument("--save_npy",    action="store_true",
                   help="Also save per-ref-frame .npy maps")
    p.add_argument("--dpi",         type=int, default=120)
    return p.parse_args()


def resolve_seqs(davis_root: Path, seqs_arg) -> List[Path]:
    if seqs_arg is None:
        return sorted(d for d in davis_root.iterdir() if d.is_dir())
    if len(seqs_arg) == 1 and Path(seqs_arg[0]).is_file():
        names = [l.strip() for l in Path(seqs_arg[0]).read_text().splitlines()
                 if l.strip()]
        return [davis_root / n for n in names if (davis_root / n).is_dir()]
    return [davis_root / s for s in seqs_arg if (davis_root / s).is_dir()]


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if args.model_path:
        model = VGGT.from_pretrained(args.model_path)
    else:
        model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.to(device).eval()
    print("model loaded")

    davis_root = Path(args.davis_root)
    seq_dirs   = resolve_seqs(davis_root, args.seqs)
    print(f"{len(seq_dirs)} sequences to process")

    for seq_dir in seq_dirs:
        try:
            run_sequence(model, seq_dir, args, device)
        except Exception:
            print(f"\n[ERROR] seq {seq_dir.name} failed:")
            traceback.print_exc()

    print(f"\ndone -> {args.output_dir}/")


if __name__ == "__main__":
    main()
