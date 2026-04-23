#!/usr/bin/env python3
"""
Visualize VGGT global cross-frame attention over DAVIS-style datasets.

Input layout:
  <davis_root>/
    bear/   00000.png  00001.png  ...
    camel/  00000.png  ...
    ...

For each query frame i:
  - Sample --num_refs reference frames uniformly from the next --future_window frames
  - Run VGGT with [query, ref0, ref1, ref2, ref3]  (S = 1 + num_refs)
  - For the query frame: take its patch tokens as Q, all ref frames' patch tokens as K
  - Use the full global softmax attention (over all S*P tokens)
  - Average the cross-frame attention over K  ->  [H_p, W_p] heatmap per layer

Output under --output_dir/<seq>/:
  cross_frame_mean/frame_XXXXX.png   per-frame grid
  npy/frame_XXXXX_layer{L}.npy       raw attn maps per layer
  npy/frame_XXXXX_avg.npy            layer-averaged attn map

Usage:
  python viz_cross_frame_attn.py --davis_root /mnt/data0/andy/Easi3R/DAVIS/JPEGImages/480p
  python viz_cross_frame_attn.py --davis_root ... --seqs bear camel
"""

import sys
import traceback
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent))
from vggt.models.vggt import VGGT
from vggt.attn_utils import GlobalAttnCapture, FrameAttnCapture


COLLECT_LAYERS = [4, 11, 17, 23]


def list_seq_dirs(davis_root: str) -> List[Path]:
    root = Path(davis_root)
    return sorted(d for d in root.iterdir() if d.is_dir())


def load_seq_paths(seq_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in seq_dir.iterdir() if p.suffix.lower() in exts)


def load_image(path: Path, resolution: int) -> np.ndarray:
    return np.array(
        Image.open(path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    )


def frames_to_tensor(frames: List[np.ndarray], device) -> torch.Tensor:
    t = [torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames]
    return torch.stack(t, dim=0).unsqueeze(0).to(device)   # [1, S, 3, H, W]


def sample_ref_ids(query_idx: int, n_total: int,
                   future_window: int, num_refs: int) -> List[int]:
    """
    Uniformly sample num_refs reference frame indices (excluding query_idx).
    Priority: future frames [query+1, query+future_window].
    If fewer than num_refs future frames exist, pad with past frames
    [max(0, query-future_window), query-1] until num_refs is reached.
    The query frame is always position 0 in the input tensor (not in this list).
    """
    # future pool
    future_end  = min(query_idx + future_window + 1, n_total)
    future_pool = list(range(query_idx + 1, future_end))

    if len(future_pool) >= num_refs:
        indices = np.linspace(0, len(future_pool) - 1, num_refs, dtype=int)
        return [future_pool[j] for j in indices]

    # not enough future frames — pad with past frames
    past_start = max(0, query_idx - future_window)
    past_pool  = list(range(past_start, query_idx))  # ascending

    pool   = future_pool + past_pool   # future first, then past
    needed = num_refs - len(future_pool)
    if len(past_pool) <= needed:
        extra = past_pool
    else:
        idx   = np.linspace(0, len(past_pool) - 1, needed, dtype=int)
        extra = [past_pool[j] for j in idx]

    return future_pool + extra


def extract_cross_frame_maps(model, images_tensor: torch.Tensor,
                              layers: List[int]) -> tuple:
    """
    Returns:
      mean_maps_query: Dict[layer_id -> List[[H_p,W_p]]]
        norm01(sum of global cross-frame patch attention, query-row view).
        HIGH = attends strongly to other frames = STATIC.

      mean_maps_key: Dict[layer_id -> List[[H_p,W_p]]]
        norm01(sum of global cross-frame patch attention, key-col view).
        HIGH = other frames attend strongly to this patch = STATIC.

      diff_maps: Dict[layer_id -> List[[H_p,W_p]]]
        norm01(frame_within_patch_sum - global_crossframe_raw_sum).
        Captures how much more a patch attends locally vs globally.
        HIGH = prefers local frame attention over cross-frame.

      meta: dict with shape metadata.
    """
    agg = model.aggregator
    B, S, _, H, W = images_tensor.shape
    ps  = agg.patch_start_idx
    H_p = H // agg.patch_size
    W_p = W // agg.patch_size
    P   = ps + H_p * W_p
    Pp  = H_p * W_p
    meta = dict(S=S, H_patches=H_p, W_patches=W_p, ps=ps, Pp=Pp, P=P)

    gcap = GlobalAttnCapture(agg, selected_layers=layers, S=S, P=P, patch_start=ps)
    fcap = FrameAttnCapture(agg, selected_layers=layers, S=S, P=P, patch_start=ps)

    with gcap, fcap, torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            agg(images_tensor)

    def _norm01(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-8)

    mean_maps_query: Dict[int, List[np.ndarray]] = {}
    mean_maps_key:   Dict[int, List[np.ndarray]] = {}
    diff_maps:       Dict[int, List[np.ndarray]] = {}

    for lid in layers:
        A_g = gcap.mean_attn(lid)   # [S*P, S*P] float32 numpy
        A_f = fcap.mean_attn(lid)   # [S, P, P]  float32 numpy  (may be None)
        if A_g is None:
            continue
        lq, lk, ld = [], [], []
        for i in range(S):
            # ── Query view: row slice ──────────────────────────────────────
            q_rows = A_g[i*P + ps : (i+1)*P, :]           # [Pp, S*P]
            other_cols_q = np.concatenate([
                q_rows[:, j*P + ps : (j+1)*P]
                for j in range(S) if j != i
            ], axis=1)                                      # [Pp, (S-1)*Pp]
            global_raw = other_cols_q.sum(axis=1)          # [Pp] raw sum, before norm
            lq.append(_norm01(global_raw).reshape(H_p, W_p))

            # ── Key view: column slice ─────────────────────────────────────
            k_cols = A_g[:, i*P + ps : (i+1)*P]           # [S*P, Pp]
            other_rows_k = np.concatenate([
                k_cols[j*P + ps : (j+1)*P, :]
                for j in range(S) if j != i
            ], axis=0)                                      # [(S-1)*Pp, Pp]
            lk.append(_norm01(other_rows_k.sum(axis=0)).reshape(H_p, W_p))

            # ── Diff: frame_within_sum + global_crossframe_raw ────────────
            if A_f is not None:
                # A_f[i, ps:, ps:]: rows=patch queries, cols=patch keys within frame
                frame_within = A_f[i, ps:, ps:].sum(axis=1)   # [Pp]
                ld.append(_norm01(frame_within + global_raw - 1).reshape(H_p, W_p))
            else:
                ld.append(np.zeros((H_p, W_p), dtype=np.float32))

        mean_maps_query[lid] = lq
        mean_maps_key[lid]   = lk
        diff_maps[lid]       = ld

    return mean_maps_query, mean_maps_key, diff_maps, meta


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


def make_frame_figure(query_frame: np.ndarray,
                       ref_frames: List[np.ndarray],
                       mean_maps_query: Dict[int, List[np.ndarray]],
                       mean_maps_key:   Dict[int, List[np.ndarray]],
                       diff_maps:       Dict[int, List[np.ndarray]],
                       layers: List[int],
                       query_id: int,
                       ref_ids: List[int]) -> plt.Figure:
    """
    Layout: 6 rows
      Row 0: query frame | ref0 | ref1 | ref2 | ref3 | (blank)
      Row 1: query L4   | L11  | L17  | L23  | mean_query
      Row 2: key   L4   | L11  | L17  | L23  | mean_key
      Row 3: diff  L4   | L11  | L17  | L23  | mean_diff
      Row 4: query_var L4 | L11 | L17 | L23  | mean_query_var
      Row 5: key_var  L4 | L11 | L17 | L23  | mean_key_var

    Query/Key maps: HIGH = static (strong cross-frame attention).
    Diff map: HIGH = prefers local frame attention over cross-frame.
    Var maps: HIGH = attention value varies strongly across frames.
    """
    H, W   = query_frame.shape[:2]
    avail  = [l for l in layers if l in mean_maps_query]
    ncols  = max(1 + len(ref_frames), len(avail) + 1)
    fig, axes = plt.subplots(6, ncols, figsize=(3 * ncols, 18), squeeze=False)

    # Row 0: original frames
    axes[0, 0].imshow(query_frame)
    axes[0, 0].set_title(f"query {query_id:05d}", fontsize=8)
    axes[0, 0].axis("off")
    for k, (rf, rid) in enumerate(zip(ref_frames, ref_ids), start=1):
        axes[0, k].imshow(rf)
        axes[0, k].set_title(f"ref {rid:05d}", fontsize=8)
        axes[0, k].axis("off")
    for k in range(1 + len(ref_frames), ncols):
        axes[0, k].axis("off")

    # Row 1: query heatmaps
    for k, lid in enumerate(avail):
        axes[1, k].imshow(to_heatmap(mean_maps_query[lid][0], size=(W, H)))
        axes[1, k].set_title(f"query L{lid}", fontsize=8)
        axes[1, k].axis("off")
    if avail:
        avg_q = np.stack([mean_maps_query[lid][0] for lid in avail], 0).mean(0)
        axes[1, len(avail)].imshow(to_heatmap(avg_q, size=(W, H)))
        axes[1, len(avail)].set_title("query mean", fontsize=8)
        axes[1, len(avail)].axis("off")
    for k in range(len(avail) + 1, ncols):
        axes[1, k].axis("off")

    # Row 2: key heatmaps
    for k, lid in enumerate(avail):
        km = mean_maps_key.get(lid)
        if km is not None:
            axes[2, k].imshow(to_heatmap(km[0], size=(W, H)))
        axes[2, k].set_title(f"key L{lid}", fontsize=8)
        axes[2, k].axis("off")
    if avail:
        key_avail = [lid for lid in avail if lid in mean_maps_key]
        if key_avail:
            avg_k = np.stack([mean_maps_key[lid][0] for lid in key_avail], 0).mean(0)
            axes[2, len(avail)].imshow(to_heatmap(avg_k, size=(W, H)))
        axes[2, len(avail)].set_title("key mean", fontsize=8)
        axes[2, len(avail)].axis("off")
    for k in range(len(avail) + 1, ncols):
        axes[2, k].axis("off")

    # Row 3: diff heatmaps
    diff_avail = [lid for lid in avail if lid in diff_maps]
    for k, lid in enumerate(avail):
        dm = diff_maps.get(lid)
        if dm is not None:
            axes[3, k].imshow(to_heatmap(dm[0], size=(W, H)))
        axes[3, k].set_title(f"diff L{lid}", fontsize=8)
        axes[3, k].axis("off")
    if diff_avail:
        avg_d = np.stack([diff_maps[lid][0] for lid in diff_avail], 0).mean(0)
        axes[3, len(avail)].imshow(to_heatmap(avg_d, size=(W, H)))
        axes[3, len(avail)].set_title("diff mean\n(frame−global)", fontsize=8)
        axes[3, len(avail)].axis("off")
    for k in range(len(avail) + 1, ncols):
        axes[3, k].axis("off")

    # Row 4: query variance across frames
    var_q_avail = [lid for lid in avail if len(mean_maps_query[lid]) > 1]
    for k, lid in enumerate(avail):
        if lid in var_q_avail:
            vq = np.stack(mean_maps_query[lid], 0).var(0)
            axes[4, k].imshow(to_heatmap(vq, size=(W, H)))
        axes[4, k].set_title(f"query_var L{lid}", fontsize=8)
        axes[4, k].axis("off")
    if var_q_avail:
        avg_vq = np.stack([np.stack(mean_maps_query[lid], 0).var(0)
                           for lid in var_q_avail], 0).mean(0)
        axes[4, len(avail)].imshow(to_heatmap(avg_vq, size=(W, H)))
        axes[4, len(avail)].set_title("query_var mean", fontsize=8)
        axes[4, len(avail)].axis("off")
    for k in range(len(avail) + 1, ncols):
        axes[4, k].axis("off")

    # Row 5: key variance across frames
    var_k_avail = [lid for lid in avail if lid in mean_maps_key and len(mean_maps_key[lid]) > 1]
    for k, lid in enumerate(avail):
        if lid in var_k_avail:
            vk = np.stack(mean_maps_key[lid], 0).var(0)
            axes[5, k].imshow(to_heatmap(vk, size=(W, H)))
        axes[5, k].set_title(f"key_var L{lid}", fontsize=8)
        axes[5, k].axis("off")
    if var_k_avail:
        avg_vk = np.stack([np.stack(mean_maps_key[lid], 0).var(0)
                           for lid in var_k_avail], 0).mean(0)
        axes[5, len(avail)].imshow(to_heatmap(avg_vk, size=(W, H)))
        axes[5, len(avail)].set_title("key_var mean", fontsize=8)
        axes[5, len(avail)].axis("off")
    for k in range(len(avail) + 1, ncols):
        axes[5, k].axis("off")

    ref_str = " ".join(str(r) for r in ref_ids)
    fig.suptitle(f"frame {query_id:05d}  refs=[{ref_str}]", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Per-sequence processing
# ─────────────────────────────────────────────────────────────────────────────

def run_sequence(model, seq_dir: Path, seq_name: str, args, device):
    print(f"\n{'='*60}")
    print(f"  seq: {seq_name}")

    all_paths = load_seq_paths(seq_dir)
    if not all_paths:
        print("  no images found, skipping")
        return

    n       = len(all_paths)
    vis_dir = Path(args.output_dir) / seq_name / "cross_frame_mean"
    npy_dir = Path(args.output_dir) / seq_name / "npy"
    vis_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {n} frames  future_window={args.future_window}  num_refs={args.num_refs}")

    for query_idx in range(n):
        ref_ids = sample_ref_ids(query_idx, n, args.future_window, args.num_refs)
        if not ref_ids:
            print(f"    frame {query_idx:05d}: no future frames, skipping")
            continue

        global_ids = [query_idx] + ref_ids
        frames     = [load_image(all_paths[i], args.resolution) for i in global_ids]
        tensor     = frames_to_tensor(frames, device)

        try:
            mean_maps_query, mean_maps_key, diff_maps, _ = extract_cross_frame_maps(
                model, tensor, args.layers)
        except Exception:
            print(f"    [ERROR] frame {query_idx:05d}: extract_cross_frame_maps failed:")
            traceback.print_exc()
            continue

        # ── save npy for query frame (local index 0) ─────────────────────────
        avail = [lid for lid in args.layers if lid in mean_maps_query]
        for lid in avail:
            np.save(npy_dir / f"frame_{query_idx:05d}_layer{lid}_query.npy",
                    mean_maps_query[lid][0])
            if lid in mean_maps_key:
                np.save(npy_dir / f"frame_{query_idx:05d}_layer{lid}_key.npy",
                        mean_maps_key[lid][0])
            if lid in diff_maps:
                np.save(npy_dir / f"frame_{query_idx:05d}_layer{lid}_diff.npy",
                        diff_maps[lid][0])
            # variance across all S frames for this layer
            if len(mean_maps_query[lid]) > 1:
                var_q = np.stack(mean_maps_query[lid], 0).var(0)
                np.save(npy_dir / f"frame_{query_idx:05d}_layer{lid}_query_var.npy", var_q)
            if lid in mean_maps_key and len(mean_maps_key[lid]) > 1:
                var_k = np.stack(mean_maps_key[lid], 0).var(0)
                np.save(npy_dir / f"frame_{query_idx:05d}_layer{lid}_key_var.npy", var_k)
        if avail:
            avg_q = np.stack([mean_maps_query[lid][0] for lid in avail], 0).mean(0)
            np.save(npy_dir / f"frame_{query_idx:05d}_avg_query.npy", avg_q)
            key_avail = [lid for lid in avail if lid in mean_maps_key]
            if key_avail:
                avg_k = np.stack([mean_maps_key[lid][0] for lid in key_avail], 0).mean(0)
                np.save(npy_dir / f"frame_{query_idx:05d}_avg_key.npy", avg_k)
            diff_avail = [lid for lid in avail if lid in diff_maps]
            if diff_avail:
                avg_d = np.stack([diff_maps[lid][0] for lid in diff_avail], 0).mean(0)
                np.save(npy_dir / f"frame_{query_idx:05d}_avg_diff.npy", avg_d)
            # layer-averaged variance
            var_q_avail = [lid for lid in avail if len(mean_maps_query[lid]) > 1]
            if var_q_avail:
                avg_var_q = np.stack([np.stack(mean_maps_query[lid], 0).var(0)
                                      for lid in var_q_avail], 0).mean(0)
                np.save(npy_dir / f"frame_{query_idx:05d}_avg_query_var.npy", avg_var_q)
            var_k_avail = [lid for lid in key_avail if len(mean_maps_key[lid]) > 1]
            if var_k_avail:
                avg_var_k = np.stack([np.stack(mean_maps_key[lid], 0).var(0)
                                      for lid in var_k_avail], 0).mean(0)
                np.save(npy_dir / f"frame_{query_idx:05d}_avg_key_var.npy", avg_var_k)

        # ── per-frame visualization ───────────────────────────────────────────
        fig = make_frame_figure(
            frames[0], frames[1:], mean_maps_query, mean_maps_key, diff_maps,
            args.layers, query_idx, ref_ids,
        )
        fig.savefig(vis_dir / f"frame_{query_idx:05d}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        print(f"    frame {query_idx:05d}  refs={ref_ids}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize VGGT cross-frame attention on DAVIS-style sequences"
    )
    p.add_argument("--davis_root",    type=str, required=True,
                   help="Root dir containing one sub-dir per sequence")
    p.add_argument("--seq_list",       type=str, nargs="+", default=None,
                   help="Sequences to process: a .txt file path (one name per line) "
                        "OR inline names e.g. 'bear bikepacking camel' "
                        "(default: all sub-dirs)")
    p.add_argument("--future_window", type=int, default=20,
                   help="Sample refs from the next N frames (default: 20)")
    p.add_argument("--num_refs",      type=int, default=4,
                   help="Number of reference frames to sample (default: 4)")
    p.add_argument("--resolution",    type=int, default=518,
                   help="Square resize resolution (multiple of 14)")
    p.add_argument("--output_dir",    type=str, default="cross_attn_out")
    p.add_argument("--model_path",    type=str, default=None,
                   help="Local checkpoint (default: download facebook/VGGT-1B)")
    p.add_argument("--layers",        type=int, nargs="+", default=COLLECT_LAYERS,
                   help="Global block indices to capture (default: 4 11 17 23)")
    return p.parse_args()


def _resolve_seq_list(seq_list_arg):
    """
    Returns list[str] of sequence names, or None (meaning 'all').
    Accepts:
      None                 -> None (process all)
      ["seqs.txt"]         -> read names from file (one per line)
      ["bear", "camel"]    -> use as-is
    """
    if seq_list_arg is None:
        return None
    if len(seq_list_arg) == 1 and Path(seq_list_arg[0]).is_file():
        lines = Path(seq_list_arg[0]).read_text().splitlines()
        return [l.strip() for l in lines if l.strip()]
    return seq_list_arg


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

    root = Path(args.davis_root)
    seq_names = _resolve_seq_list(args.seq_list)
    if seq_names is not None:
        seq_dirs = [root / s for s in seq_names if (root / s).is_dir()]
    else:
        seq_dirs = list_seq_dirs(args.davis_root)

    for seq_dir in seq_dirs:
        try:
            run_sequence(model, seq_dir, seq_dir.name, args, device)
        except Exception:
            print(f"\n[ERROR] seq {seq_dir.name} failed:")
            traceback.print_exc()

    print(f"\ndone  ->  {args.output_dir}/")


if __name__ == "__main__":
    main()
