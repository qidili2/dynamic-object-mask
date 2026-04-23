#!/usr/bin/env python3
"""
Visualize cross-camera attention asymmetry as a dynamic signal.

For each ref frame j, compute per-patch:
    diff_j = norm(cam_frame0 -> patches_ref_j)
           - norm(cam_ref_j   -> patches_frame0)

Positive diff: frame0's camera attends to position in ref_j MORE than
               ref_j's camera attends to the same position in frame0
               → asymmetric attention → candidate dynamic region

Output under --output_dir/<seq>/cam_diff/:
  all_refs.png    rows = layers + avg,  cols = ref frames

Usage:
  python viz_cam_attn_var.py \\
      --davis_root /mnt/data0/andy/Easi3R/DAVIS/JPEGImages/480p \\
      --output_dir cam_diff_out \\
      --seqs blackswan bmx-trees \\
      --layers 4 11 17 23 \\
      --ref_stride 2 \\
      --max_refs 6
"""

import sys
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

DEFAULT_LAYERS = [4, 11, 17, 23]


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

def extract_cam_diff(
    model,
    images_tensor: torch.Tensor,   # [1, S, 3, H, W]  frame0 at index 0
    layers: List[int],
) -> Tuple[Dict[int, List[np.ndarray]], int, int]:
    """
    For each ref frame j (j = 1 .. S-1), compute per-layer:

        diff_j[h,w] = norm(cam_frame0 → patch(h,w) of ref_j)
                    - norm(cam_ref_j  → patch(h,w) of frame0)

    Returns:
        diff_maps : Dict[layer_id -> List[[H_p, W_p]]]  len = S-1
        H_p, W_p  : patch grid size
    """
    agg = model.aggregator
    _, S, _, H, W = images_tensor.shape
    ps  = agg.patch_start_idx
    H_p = H // agg.patch_size
    W_p = W // agg.patch_size
    P   = ps + H_p * W_p

    gcap = GlobalAttnCapture(agg, selected_layers=layers, S=S, P=P, patch_start=ps)

    with gcap, torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            agg(images_tensor)

    def _norm01(x: np.ndarray) -> np.ndarray:
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-8)

    diff_maps: Dict[int, List[np.ndarray]] = {}

    for lid in layers:
        A_g = gcap.mean_attn(lid)   # [S*P, S*P]
        if A_g is None:
            continue

        diffs = []
        for j in range(1, S):
            # cam_frame0 → patches of ref_j
            cam0_row   = A_g[0, :]                           # [S*P]
            attn_0_j   = cam0_row[j*P + ps : (j+1)*P]       # [Pp]

            # cam_ref_j → patches of frame0
            cam_j_row  = A_g[j*P, :]                         # [S*P]
            attn_j_0   = cam_j_row[ps : P]                   # [Pp]  (frame0 patches)

            diff = _norm01(attn_0_j) - _norm01(attn_j_0)    # [Pp] in [-1, 1]
            diffs.append(diff.reshape(H_p, W_p))

        diff_maps[lid] = diffs

    return diff_maps, H_p, W_p


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
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


def make_figure(
    frame0: np.ndarray,
    ref_frames: List[np.ndarray],
    ref_ids: List[int],
    diff_maps: Dict,        # lid -> List[[H_p, W_p]] OR 'avg' key
    layers: List[int],
    seq_name: str,
) -> plt.Figure:
    """
    Layout:
      col 0        : frame 0 (fixed reference)
      col 1..n_refs: one column per ref frame
      row 0        : RGB images
      row 1..L     : per-layer diff map  (red = cam0 attends more, blue = cam_j attends more)
      row -1       : layer-averaged diff
    """
    avail = [lid for lid in layers if lid in diff_maps and diff_maps[lid]]
    if 'avg' in diff_maps:
        row_keys = avail + ['avg']
    else:
        row_keys = avail

    n_refs = len(ref_frames)
    ncols  = 1 + n_refs
    nrows  = 1 + len(row_keys)
    H, W   = frame0.shape[:2]

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.6 * ncols, 2.6 * nrows),
                             squeeze=False)

    # ── row 0: RGB ────────────────────────────────────────────────────────────
    axes[0, 0].imshow(frame0)
    axes[0, 0].set_title("frame 0 (view1)", fontsize=7)
    axes[0, 0].axis("off")
    for k, (rf, rid) in enumerate(zip(ref_frames, ref_ids)):
        axes[0, 1 + k].imshow(rf)
        axes[0, 1 + k].set_title(f"ref {rid:05d}", fontsize=7)
        axes[0, 1 + k].axis("off")

    # ── data rows ─────────────────────────────────────────────────────────────
    for row, lid in enumerate(row_keys, start=1):
        lbl   = f"L{lid}" if lid != 'avg' else "avg"
        maps  = diff_maps[lid]   # List[[H_p, W_p]]

        axes[row, 0].imshow(frame0)
        axes[row, 0].set_title(f"{lbl}\n(view1 ref)", fontsize=7)
        axes[row, 0].axis("off")

        for k, (rf, rid) in enumerate(zip(ref_frames, ref_ids)):
            ax = axes[row, 1 + k]
            if k < len(maps):
                ax.imshow(to_heatmap(maps[k], size=(W, H)))
            ax.set_title(f"{lbl} ref{rid:05d}", fontsize=6)
            ax.axis("off")

    fig.suptitle(
        f"{seq_name}  —  norm(cam0→patches_j) − norm(cam_j→patches_0)  (inferno, bright=prominent)",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Per-sequence processing
# ─────────────────────────────────────────────────────────────────────────────

def run_sequence(model, seq_dir: Path, args, device):
    seq = seq_dir.name
    print(f"\n{'='*60}\n  seq: {seq}")

    all_paths = load_seq_paths(seq_dir)
    if len(all_paths) < 2:
        print("  <2 frames, skipping")
        return

    n = len(all_paths)
    out_dir = Path(args.output_dir) / seq / "cam_diff"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame0 = load_image(all_paths[0], args.resolution)
    all_ref_ids = list(range(1, n, args.ref_stride))
    print(f"  {n} frames  stride={args.ref_stride}  refs={len(all_ref_ids)}")

    # collected[lid] = list of (ref_id, diff_map)
    collected: Dict = {lid: [] for lid in args.layers}

    for batch_start in range(0, len(all_ref_ids), args.max_refs):
        batch_ids = all_ref_ids[batch_start: batch_start + args.max_refs]
        frames    = [frame0] + [load_image(all_paths[i], args.resolution) for i in batch_ids]
        tensor    = frames_to_tensor(frames, device)

        try:
            per_ref_diffs, H_p, W_p = extract_cam_diff(model, tensor, args.layers)
        except Exception:
            print(f"  [ERROR] batch {batch_ids}")
            traceback.print_exc()
            continue

        for lid, maps in per_ref_diffs.items():
            for ref_id, dmap in zip(batch_ids, maps):
                collected[lid].append((ref_id, dmap))

        print(f"  batch refs {batch_ids[0]:05d}–{batch_ids[-1]:05d} done")

    # ── build full diff_maps dict (sorted by ref_id) + avg ───────────────────
    avail_lids = [lid for lid in args.layers if collected.get(lid)]
    if not avail_lids:
        print("  no maps collected")
        return

    # cap columns for readability
    MAX_COLS = 12
    all_rid_sorted = sorted({rid for lid in avail_lids for rid, _ in collected[lid]})
    step = max(1, len(all_rid_sorted) // MAX_COLS)
    sel_rids = all_rid_sorted[::step][:MAX_COLS]

    rid_to_idx = {rid: i for i, rid in enumerate(all_rid_sorted)}

    diff_maps: Dict = {}
    avg_stack = []
    for lid in avail_lids:
        ordered = sorted(collected[lid], key=lambda x: x[0])
        all_maps = [m for _, m in ordered]
        # select subset for display
        sel_maps = [all_maps[rid_to_idx[r]] for r in sel_rids
                    if r in rid_to_idx and rid_to_idx[r] < len(all_maps)]
        diff_maps[lid] = sel_maps
        avg_stack.append(np.stack(all_maps, 0).mean(0))   # mean diff across all refs

    diff_maps['avg'] = [np.stack(avg_stack, 0).mean(0)] * len(sel_rids)  # same avg for all cols

    sel_ref_frames = [load_image(all_paths[r], args.resolution) for r in sel_rids]

    fig = make_figure(
        frame0, sel_ref_frames, sel_rids,
        diff_maps, args.layers,
        seq_name=seq,
    )
    fig_path = out_dir / "all_refs.png"
    fig.savefig(fig_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {fig_path}")

    if args.save_npy:
        npy_dir = out_dir / "npy"
        npy_dir.mkdir(exist_ok=True)
        for lid in avail_lids:
            for rid, dmap in collected[lid]:
                np.save(npy_dir / f"ref_{rid:05d}_layer{lid}_diff.npy", dmap)
        print(f"  saved npy -> {npy_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Camera-token attention asymmetry: cam0→patchJ  minus  camJ→patch0"
    )
    p.add_argument("--davis_root",  required=True)
    p.add_argument("--output_dir",  default="cam_diff_out")
    p.add_argument("--seqs",        nargs="+", default=None,
                   help="Sequence names or a .txt file; default: all sub-dirs")
    p.add_argument("--layers",      type=int, nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--ref_stride",  type=int, default=2)
    p.add_argument("--max_refs",    type=int, default=6,
                   help="Max ref frames per forward pass (GPU memory limit)")
    p.add_argument("--resolution",  type=int, default=518)
    p.add_argument("--model_path",  default=None)
    p.add_argument("--save_npy",    action="store_true")
    p.add_argument("--dpi",         type=int, default=120)
    return p.parse_args()


def resolve_seqs(davis_root: Path, seqs_arg) -> List[Path]:
    if seqs_arg is None:
        return sorted(d for d in davis_root.iterdir() if d.is_dir())
    if len(seqs_arg) == 1 and Path(seqs_arg[0]).is_file():
        names = [l.strip() for l in Path(seqs_arg[0]).read_text().splitlines() if l.strip()]
        return [davis_root / n for n in names if (davis_root / n).is_dir()]
    return [davis_root / s for s in seqs_arg if (davis_root / s).is_dir()]


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = (VGGT.from_pretrained(args.model_path) if args.model_path
             else VGGT.from_pretrained("facebook/VGGT-1B"))
    model = model.to(device).eval()
    print("model loaded")

    davis_root = Path(args.davis_root)
    seq_dirs   = resolve_seqs(davis_root, args.seqs)
    print(f"{len(seq_dirs)} sequences to process")

    for seq_dir in seq_dirs:
        try:
            run_sequence(model, seq_dir, args, device)
        except Exception:
            print(f"\n[ERROR] seq {seq_dir.name}:")
            traceback.print_exc()

    print(f"\ndone -> {args.output_dir}/")


if __name__ == "__main__":
    main()
