    #!/usr/bin/env python3
"""
Demo: Visualize VGGT global cross-frame attention as an analog to Easi3R's cross-attention
dynamic masks.

Processes the full video by sliding a window of `window_size` frames through all frames.
Each window runs one VGGT aggregator forward pass; dynamic scores are accumulated
per global frame index and averaged when windows overlap.

Output per sequence  (in  attn_vis/<seq>/):
  window_000.png, window_001.png …  – per-window grid: original | mean_recv | dynamic_score
  dynamic_all.png                   – all frames tiled (global accumulation)
  cross_attn_matrix_w000.png        – full [Pp, Pp] attention matrix for pair (0,1) of each window
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from pathlib import Path
from types import MethodType

sys.path.insert(0, str(Path(__file__).parent))
from vggt.models.vggt import VGGT


# ─────────────────────────────────────────────────────────────────────────────
# DAVIS paths & default sequence list
# ─────────────────────────────────────────────────────────────────────────────

DAVIS_ROOT = "/mnt/data0/andy/Easi3R/DAVIS/JPEGImages/480p"

SEQ_LIST = [
    "blackswan",
    "breakdance",
    "camel",
    "car-roundabout",
    "dog",
    "horsejump-high",
    "parkour",
    "motocross-jump",
]

OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "attn_vis")


# ─────────────────────────────────────────────────────────────────────────────
# Attention extraction
# ─────────────────────────────────────────────────────────────────────────────

def _make_patched_forward(S: int, patch_start: int, store: dict, store_full_last: bool):
    """
    Replacement forward() for a global block's Attention module.

    Stores per-pair:
      recv_i_j       : received-attention per patch in frame j from frame i,
                       shape [B, P_patch], accumulated (sum) across collected layers.
      recv_i_j_n     : layer count for averaging.
      full_0_1       : head-averaged full [B, Pp_0, Pp_1] – last collected layer only.
    """
    def forward(self, x, pos=None, return_cam_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if return_cam_attn:
            q_cam = q[:, :, 0:1, :] * self.scale
            attn_cam = torch.matmul(q_cam, k.transpose(-2, -1)).softmax(dim=-1)
            attn_cam = self.attn_drop(attn_cam)
            x_full = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            x_full = x_full.transpose(1, 2).reshape(B, N, C)
            return self.proj_drop(self.proj(x_full)), attn_cam.detach()

        P = N // S
        q_s = q * self.scale   # [B, H, S*P, D]

        with torch.no_grad():
            for i in range(S):
                for j in range(S):
                    if i == j:
                        continue

                    qi = q_s[:, :, i * P + patch_start:(i + 1) * P, :]   # [B, H, Pp, D]
                    kj = k[:,  :, j * P + patch_start:(j + 1) * P, :]    # [B, H, Pp, D]

                    attn_ij = (qi.float() @ kj.float().transpose(-2, -1)).softmax(dim=-1)  # [B,H,Pp,Pp]

                    # received attention per patch in frame j, averaged over heads → [B, Pp]
                    recv = attn_ij.sum(dim=-2).mean(dim=1).cpu()

                    rkey = f"recv_{i}_{j}"
                    if rkey not in store:
                        store[rkey] = recv
                        store[rkey + "_n"] = 1
                    else:
                        store[rkey] = store[rkey] + recv
                        store[rkey + "_n"] += 1

                    if store_full_last and i == 0 and j == 1:
                        store["full_0_1"] = attn_ij.mean(dim=1).cpu()   # [B, Pp, Pp]

        x_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        x_out = x_out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x_out))

    return forward


def extract_cross_frame_attention(model, images_tensor: torch.Tensor, collect_last_n: int = 8):
    """
    Run VGGT aggregator on one window [B, S, 3, H, W] and collect cross-frame attention.

    Returns:
        store  – accumulated attention stats for this window
        meta   – dict(S, patch_start, H_patches, W_patches)
    """
    agg = model.aggregator
    S = images_tensor.shape[1]
    patch_start = agg.patch_start_idx   # 1 camera + 4 register = 5
    depth = agg.depth

    B, _, _, H, W = images_tensor.shape
    H_patches = H // agg.patch_size
    W_patches = W // agg.patch_size

    collect_set = set(range(depth - collect_last_n, depth))

    store = {}
    saved_forwards = {}

    for lid, block in enumerate(agg.global_blocks):
        if lid in collect_set:
            is_last = (lid == depth - 1)
            new_fwd = _make_patched_forward(S, patch_start, store, store_full_last=is_last)
            saved_forwards[lid] = block.attn.forward
            block.attn.forward = MethodType(new_fwd, block.attn)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            agg(images_tensor)

    for lid, fwd in saved_forwards.items():
        agg.global_blocks[lid].attn.forward = fwd

    meta = dict(S=S, patch_start=patch_start, H_patches=H_patches, W_patches=W_patches)
    return store, meta


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic map computation
# ─────────────────────────────────────────────────────────────────────────────

def norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def compute_dynamic_maps(store: dict, meta: dict):
    """
    From one window's store, return per-frame-in-window:
        mean_recv     [H_p, W_p]  – mean received attention
        var_recv      [H_p, W_p]  – variance across source frames
        dynamic_score [H_p, W_p]  – var * (1 - norm(mean))
    """
    S = meta["S"]
    H_p, W_p = meta["H_patches"], meta["W_patches"]
    results = {}

    for j in range(S):
        per_source = []
        for i in range(S):
            if i == j:
                continue
            key = f"recv_{i}_{j}"
            if key not in store:
                continue
            n = store[key + "_n"]
            per_source.append((store[key] / n)[0].numpy().reshape(H_p, W_p))

        if not per_source:
            continue

        stacked  = np.stack(per_source, axis=0)   # [num_src, H_p, W_p]
        mean_m   = stacked.mean(axis=0)
        var_m    = stacked.var(axis=0)

        results[j] = dict(
            mean_recv=mean_m,
            var_recv=var_m,
            dynamic_score=norm01(var_m) * (1.0 - norm01(mean_m)),
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Visualization  (pure heatmaps, no overlay)
# ─────────────────────────────────────────────────────────────────────────────

def _heatmap_img(data: np.ndarray, cmap: str = "inferno") -> np.ndarray:
    """Convert a 2-D float array to an RGB uint8 image via colormap."""
    normed = norm01(data)
    rgba = matplotlib.colormaps[cmap](normed)   # [H, W, 4]
    return (rgba[..., :3] * 255).astype(np.uint8)


def save_window_figure(images_np, dynamic_maps, meta, store,
                       output_dir: str, window_idx: int, global_frame_ids):
    """
    Save one PNG for a window:
      row 0: original frames
      row 1: mean received attention heatmap
      row 2: dynamic score heatmap
    """
    os.makedirs(output_dir, exist_ok=True)
    S = meta["S"]
    H_p, W_p = meta["H_patches"], meta["W_patches"]

    nrows, ncols = 3, S
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                             squeeze=False)

    for j in range(S):
        gid = global_frame_ids[j]
        axes[0, j].imshow(images_np[j])
        axes[0, j].set_title(f"frame {gid:04d}", fontsize=8)
        axes[0, j].axis("off")

        if j not in dynamic_maps:
            axes[1, j].axis("off")
            axes[2, j].axis("off")
            continue

        dm = dynamic_maps[j]
        axes[1, j].imshow(_heatmap_img(dm["mean_recv"], "viridis"))
        axes[1, j].set_title("mean recv attn", fontsize=7)
        axes[1, j].axis("off")

        axes[2, j].imshow(_heatmap_img(dm["dynamic_score"], "inferno"))
        axes[2, j].set_title("dynamic score", fontsize=7)
        axes[2, j].axis("off")

    fig.suptitle(f"Window {window_idx:03d}  frames {global_frame_ids[0]}–{global_frame_ids[-1]}",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(output_dir, f"window_{window_idx:03d}.png")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"    [vis] {path}")

    # ── Cross-attention matrix for pair (0, 1) ────────────────────────────────
    if "full_0_1" in store:
        full = store["full_0_1"][0].numpy()   # [Pp_0, Pp_1]
        step = max(1, H_p // 24)
        fig2, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(full[::step, ::step], cmap="viridis", aspect="auto")
        ax.set_xlabel(f"Key patches – frame {global_frame_ids[1]} (every {step})", fontsize=7)
        ax.set_ylabel(f"Query patches – frame {global_frame_ids[0]} (every {step})", fontsize=7)
        ax.set_title(f"Cross-attn matrix  w{window_idx:03d}  frames "
                     f"{global_frame_ids[0]}→{global_frame_ids[1]}", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        path2 = os.path.join(output_dir, f"cross_attn_matrix_w{window_idx:03d}.png")
        fig2.savefig(path2, bbox_inches="tight", dpi=120)
        plt.close(fig2)


def save_global_figure(all_images_np, global_scores, meta, output_dir: str):
    """
    Tile ALL frames from the sequence in a single figure:
      row 0: originals  |  row 1: dynamic score  (capped at 8 per row for readability)
    """
    os.makedirs(output_dir, exist_ok=True)
    n = len(all_images_np)
    cols = min(n, 8)
    rows_per_chunk = 2   # original + dynamic
    chunks = (n + cols - 1) // cols

    fig, axes = plt.subplots(chunks * rows_per_chunk, cols,
                             figsize=(3 * cols, 3 * chunks * rows_per_chunk),
                             squeeze=False)

    for idx in range(n):
        chunk = idx // cols
        col   = idx % cols
        r0    = chunk * rows_per_chunk

        axes[r0, col].imshow(all_images_np[idx])
        axes[r0, col].set_title(f"frame {idx:04d}", fontsize=7)
        axes[r0, col].axis("off")

        if idx in global_scores:
            axes[r0 + 1, col].imshow(_heatmap_img(global_scores[idx], "inferno"))
        axes[r0 + 1, col].set_title("dynamic score", fontsize=7)
        axes[r0 + 1, col].axis("off")

    # Hide empty cells
    for idx in range(n, chunks * cols):
        chunk = idx // cols
        col   = idx % cols
        r0    = chunk * rows_per_chunk
        for r in range(rows_per_chunk):
            axes[r0 + r, col].axis("off")

    fig.suptitle("Dynamic scores – full sequence", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(output_dir, "dynamic_all.png")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  [vis] global → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_all_frames(image_dir: str, resolution: int):
    """Load every image in a directory (sorted), return list of [H,W,3] uint8 arrays."""
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise RuntimeError(f"No images found in {image_dir}")
    arrays = [
        np.array(Image.open(p).convert("RGB").resize((resolution, resolution), Image.BILINEAR))
        for p in paths
    ]
    return arrays   # list of [H, W, 3] uint8


def frames_to_tensor(frames_np, device):
    """Convert a list of [H,W,3] uint8 arrays to [1, S, 3, H, W] float tensor in [0,1]."""
    tensors = [torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0 for arr in frames_np]
    return torch.stack(tensors, dim=0).unsqueeze(0).to(device)   # [1, S, 3, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# Per-sequence windowed processing
# ─────────────────────────────────────────────────────────────────────────────

def run_sequence(model, seq_dir: str, seq_name: str, args, device):
    print(f"\n{'='*60}")
    print(f"  Sequence: {seq_name}")
    print(f"{'='*60}")

    all_frames = load_all_frames(seq_dir, args.resolution)
    n_total = len(all_frames)
    ws = args.window_size
    stride = args.window_stride if args.window_stride > 0 else ws

    print(f"  {n_total} frames  window={ws}  stride={stride}  "
          f"resolution={args.resolution}")

    out_dir = os.path.join(args.output_dir, seq_name)
    os.makedirs(out_dir, exist_ok=True)

    # global_scores[frame_idx] accumulates dynamic_score across windows that include it
    global_score_sum   = {}   # frame_idx -> np.ndarray [H_p, W_p]
    global_score_count = {}   # frame_idx -> int

    window_starts = list(range(0, n_total - ws + 1, stride))
    # ensure last window is included if there's a remainder
    if (n_total - ws) % stride != 0 and n_total >= ws:
        window_starts.append(n_total - ws)
    # deduplicate while preserving order
    seen = set()
    window_starts = [x for x in window_starts if not (x in seen or seen.add(x))]

    for w_idx, start in enumerate(window_starts):
        end = start + ws
        win_frames_np = all_frames[start:end]
        global_frame_ids = list(range(start, end))

        print(f"  window {w_idx:03d}  frames {start}–{end-1}", end="  ", flush=True)

        images_tensor = frames_to_tensor(win_frames_np, device)
        store, meta = extract_cross_frame_attention(
            model, images_tensor, collect_last_n=args.collect_last
        )
        H_p, W_p = meta["H_patches"], meta["W_patches"]

        dynamic_maps = compute_dynamic_maps(store, meta)

        # Accumulate into global scores
        for local_j, gid in enumerate(global_frame_ids):
            if local_j not in dynamic_maps:
                continue
            ds = dynamic_maps[local_j]["dynamic_score"]   # [H_p, W_p]
            if gid not in global_score_sum:
                global_score_sum[gid]   = ds.copy()
                global_score_count[gid] = 1
            else:
                global_score_sum[gid]   += ds
                global_score_count[gid] += 1

        save_window_figure(win_frames_np, dynamic_maps, meta, store,
                           out_dir, w_idx, global_frame_ids)

    # Average global scores
    global_scores = {
        gid: global_score_sum[gid] / global_score_count[gid]
        for gid in sorted(global_score_sum)
    }

    save_global_figure(all_frames, global_scores, meta, out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize VGGT global cross-frame attention on DAVIS sequences"
    )
    p.add_argument("--davis_root",    type=str, default=DAVIS_ROOT)
    p.add_argument("--seq_list",      type=str, nargs="+", default=SEQ_LIST)
    p.add_argument("--window_size",   type=int, default=4,
                   help="Number of frames per VGGT forward pass")
    p.add_argument("--window_stride", type=int, default=0,
                   help="Stride between windows (0 = non-overlapping, i.e. = window_size)")
    p.add_argument("--resolution",    type=int, default=518,
                   help="Resize frames to this square resolution (multiple of 14)")
    p.add_argument("--collect_last",  type=int, default=8,
                   help="Collect attention from last N global layers")
    p.add_argument("--output_dir",    type=str, default=OUTPUT_ROOT)
    p.add_argument("--model_path",    type=str, default=None,
                   help="Local checkpoint path (default: download facebook/VGGT-1B)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device={device}")

    if args.model_path:
        model = VGGT.from_pretrained(args.model_path)
    else:
        model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.to(device).eval()
    print("[main] model loaded")

    missing = []
    for seq in args.seq_list:
        seq_dir = os.path.join(args.davis_root, seq)
        if not os.path.isdir(seq_dir):
            print(f"[warn] not found, skipping: {seq_dir}")
            missing.append(seq)
            continue
        run_sequence(model, seq_dir, seq, args, device)

    print(f"\n[main] done – {args.output_dir}/")
    if missing:
        print(f"[warn] skipped: {missing}")


if __name__ == "__main__":
    main()
