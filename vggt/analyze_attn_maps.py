#!/usr/bin/env python3
"""
Analyze VGGT cross-frame attention .npy maps vs DAVIS GT masks.

Directory scanned:
  cross_attn_out_qksum_sample_window_10_allLayer/<seq>/npy/

Filename schema:
  frame_{XXXXX}_layer{N}_{side}.npy        -> layer N, side=query/key/diff, stat=mean
  frame_{XXXXX}_layer{N}_{side}_var.npy    -> layer N, stat=var
  frame_{XXXXX}_avg_{side}.npy             -> averaged across layers, stat=mean
  frame_{XXXXX}_avg_{side}_var.npy         -> averaged across layers, stat=var

GT masks:  /mnt/data0/andy/Easi3R/DAVIS/Annotations/480p/<seq>/<XXXXX>.png
  uint8, non-zero pixels = foreground object.
  Downsampled to patch resolution via nearest-neighbor then binarized.

Outputs (all in --out_dir):
  per_map_metrics.csv
  by_layer_summary.csv
  by_group_summary.csv
  best_layers.csv
  best_map_types.csv
  best_groups.csv
  formula_search_results.csv
  top_formula_summary.txt
  plots/roc_auc_by_maptype.png
  plots/roc_auc_layer_x_maptype.png
  plots/absgap_layer_x_maptype.png
"""

import argparse
import re
import warnings
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (average_precision_score, f1_score,
                             roc_auc_score)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

NPY_ROOT = Path("/mnt/data0/andy/vggt/cross_attn_out_qksum_sample_window_10_allLayer")
GT_ROOT  = Path("/mnt/data0/andy/Easi3R/DAVIS/Annotations/480p")
EPS      = 1e-8

LAYER_GROUPS = {
    "shallow": list(range(1,  9)),
    "mid":     list(range(8, 17)),
    "deep":    list(range(16, 24)),
    "all":     list(range(1, 24)),
}

# formula search coefficients
COEF_GRID = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]


# ─────────────────────────────────────────────────────────────────────────────
# File scanning & parsing
# ─────────────────────────────────────────────────────────────────────────────

# Matches:  frame_00003_layer17_query.npy  OR  frame_00003_layer17_query_var.npy
#           frame_00003_avg_query.npy        OR  frame_00003_avg_query_var.npy
_RE = re.compile(
    r"^frame_(\d{5})_(layer(\d+)|avg)_(query|key|diff)(_var)?\.npy$"
)


def parse_filename(name: str):
    """
    Returns dict with keys: frame_id, layer (int or 'avg'), side, stat, or None.
    stat is 'mean' or 'var'.
    """
    m = _RE.match(name)
    if m is None:
        return None
    frame_id = int(m.group(1))
    layer_str = m.group(2)   # 'layerN' or 'avg'
    layer_num = m.group(3)   # digit string or None
    side      = m.group(4)   # query / key / diff
    var_flag  = m.group(5)   # '_var' or None
    layer = int(layer_num) if layer_num is not None else "avg"
    stat  = "var" if var_flag else "mean"
    return dict(frame_id=frame_id, layer=layer, side=side, stat=stat)


def scan_npy_files(npy_root: Path):
    """Return list of dicts with parsed metadata + file_path + seq."""
    records = []
    for seq_dir in sorted(npy_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        npy_dir = seq_dir / "npy"
        if not npy_dir.is_dir():
            continue
        for p in sorted(npy_dir.iterdir()):
            meta = parse_filename(p.name)
            if meta is None:
                continue
            meta["seq"]       = seq_dir.name
            meta["file_path"] = str(p)
            records.append(meta)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# GT mask loading
# ─────────────────────────────────────────────────────────────────────────────

_gt_cache = {}


def load_gt_patch(seq: str, frame_id: int, H_p: int, W_p: int) -> np.ndarray | None:
    """
    Load DAVIS GT mask for (seq, frame_id), resize to [H_p, W_p] (nearest),
    binarize (nonzero -> 1).  Returns None if file missing.
    Cached.
    """
    key = (seq, frame_id, H_p, W_p)
    if key in _gt_cache:
        return _gt_cache[key]
    gt_path = GT_ROOT / seq / f"{frame_id:05d}.png"
    if not gt_path.exists():
        _gt_cache[key] = None
        return None
    mask = np.array(Image.open(gt_path).resize((W_p, H_p), Image.NEAREST))
    binary = (mask > 0).astype(np.float32)
    _gt_cache[key] = binary
    return binary


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(vals: np.ndarray, gt: np.ndarray) -> dict:
    """
    vals: [H_p, W_p] float, higher = more dynamic
    gt:   [H_p, W_p] binary float (1=foreground)
    Returns dict of metrics.
    """
    v = vals.ravel().astype(np.float64)
    g = gt.ravel().astype(np.int32)

    n_in  = int(g.sum())
    n_out = int((1 - g).sum())

    if n_in == 0 or n_out == 0:
        return dict(in_gt_mean=np.nan, out_gt_mean=np.nan,
                    in_gt_var=np.nan, out_gt_var=np.nan,
                    gap=np.nan, abs_gap=np.nan, ratio=np.nan,
                    cohen_d=np.nan, roc_auc=np.nan, ap=np.nan,
                    n_in=n_in, n_out=n_out)

    in_vals  = v[g == 1]
    out_vals = v[g == 0]

    in_mean  = float(in_vals.mean())
    out_mean = float(out_vals.mean())
    in_var   = float(in_vals.var())
    out_var  = float(out_vals.var())
    gap      = in_mean - out_mean
    abs_gap  = abs(gap)
    ratio    = in_mean / (out_mean + EPS)

    # Cohen's d
    pooled_std = float(np.sqrt((in_var + out_var) / 2 + EPS))
    cohen_d    = gap / pooled_std

    # AUC / AP — higher score should predict foreground
    try:
        roc_auc = float(roc_auc_score(g, v))
        ap      = float(average_precision_score(g, v))
    except Exception:
        roc_auc = np.nan
        ap      = np.nan

    return dict(in_gt_mean=in_mean, out_gt_mean=out_mean,
                in_gt_var=in_var,  out_gt_var=out_var,
                gap=gap, abs_gap=abs_gap, ratio=ratio,
                cohen_d=cohen_d, roc_auc=roc_auc, ap=ap,
                n_in=n_in, n_out=n_out)


# ─────────────────────────────────────────────────────────────────────────────
# Per-map analysis  (Pass 1)
# ─────────────────────────────────────────────────────────────────────────────

def run_per_map(records, out_dir: Path) -> pd.DataFrame:
    rows = []
    total = len(records)
    for idx, rec in enumerate(records):
        if idx % 500 == 0:
            print(f"  per-map {idx}/{total} ...")

        # skip avg-layer entries here (handled in group analysis)
        if rec["layer"] == "avg":
            continue
        # focus on query/key (skip diff for primary analysis)
        if rec["side"] == "diff":
            continue

        vals = np.load(rec["file_path"]).astype(np.float32)
        H_p, W_p = vals.shape
        gt = load_gt_patch(rec["seq"], rec["frame_id"], H_p, W_p)
        if gt is None:
            continue

        m = compute_metrics(vals, gt)
        row = dict(
            seq       = rec["seq"],
            frame_id  = rec["frame_id"],
            layer     = rec["layer"],
            side      = rec["side"],
            stat      = rec["stat"],
            map_type  = f"{rec['side']}_{rec['stat']}",
            file_path = rec["file_path"],
        )
        row.update(m)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_map_metrics.csv", index=False)
    print(f"  saved per_map_metrics.csv  ({len(df)} rows)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Summary by single layer + map type  (Pass 2a)
# ─────────────────────────────────────────────────────────────────────────────

def run_layer_summary(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    agg_cols = ["roc_auc", "ap", "abs_gap", "cohen_d", "gap", "ratio"]
    summary = (df.groupby(["layer", "map_type"])[agg_cols]
                 .mean()
                 .reset_index())
    summary = summary.sort_values("roc_auc", ascending=False)
    summary.to_csv(out_dir / "by_layer_summary.csv", index=False)

    # ranking
    best = summary.drop_duplicates("map_type").sort_values("roc_auc", ascending=False)
    best.to_csv(out_dir / "best_map_types.csv", index=False)

    best_layers = summary.sort_values("roc_auc", ascending=False).head(20)
    best_layers.to_csv(out_dir / "best_layers.csv", index=False)

    print(f"  saved by_layer_summary.csv  best layers:")
    print(best_layers[["layer", "map_type", "roc_auc", "ap", "abs_gap"]].head(10).to_string(index=False))
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Group analysis  (Pass 2b) — average maps across layers, then compute metrics
# ─────────────────────────────────────────────────────────────────────────────

def run_group_analysis(records, out_dir: Path) -> pd.DataFrame:
    """
    For each group × side × stat × (seq, frame):
      1. load each layer's map
      2. average them
      3. compute metrics against GT
    """
    # index records for fast lookup: (seq, frame_id, layer, side, stat) -> file_path
    idx = {}
    for rec in records:
        if rec["layer"] == "avg" or rec["side"] == "diff":
            continue
        key = (rec["seq"], rec["frame_id"], rec["layer"], rec["side"], rec["stat"])
        idx[key] = rec["file_path"]

    # enumerate unique (seq, frame_id) pairs
    pairs = sorted(set((r["seq"], r["frame_id"]) for r in records
                       if r["layer"] != "avg" and r["side"] != "diff"))

    rows = []
    sides = ["query", "key"]
    stats = ["mean", "var"]

    for group_name, group_layers in LAYER_GROUPS.items():
        print(f"  group {group_name} ({len(group_layers)} layers) ...")
        for side in sides:
            for stat in stats:
                map_type = f"{side}_{stat}"
                for seq, frame_id in pairs:
                    layer_maps = []
                    for lid in group_layers:
                        fp = idx.get((seq, frame_id, lid, side, stat))
                        if fp is not None:
                            layer_maps.append(np.load(fp).astype(np.float32))
                    if not layer_maps:
                        continue
                    avg_map = np.stack(layer_maps, 0).mean(0)
                    H_p, W_p = avg_map.shape
                    gt = load_gt_patch(seq, frame_id, H_p, W_p)
                    if gt is None:
                        continue
                    m = compute_metrics(avg_map, gt)
                    row = dict(group=group_name, map_type=map_type,
                               side=side, stat=stat, seq=seq, frame_id=frame_id)
                    row.update(m)
                    rows.append(row)

    df = pd.DataFrame(rows)
    agg_cols = ["roc_auc", "ap", "abs_gap", "cohen_d", "gap", "ratio"]
    summary = (df.groupby(["group", "map_type"])[agg_cols]
                 .mean().reset_index()
                 .sort_values("roc_auc", ascending=False))
    summary.to_csv(out_dir / "by_group_summary.csv", index=False)

    best_groups = summary.sort_values("roc_auc", ascending=False)
    best_groups.to_csv(out_dir / "best_groups.csv", index=False)

    print(f"  saved by_group_summary.csv:")
    print(summary[["group", "map_type", "roc_auc", "ap", "abs_gap"]].to_string(index=False))
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Formula search  (Pass 3)
# ─────────────────────────────────────────────────────────────────────────────

FORMULA_DEFS = [
    # name, lambda (q, k, qv, kv, a, b) -> score
    # -- additive second term (original) --
    ("(1-q)*k",                  lambda q,k,qv,kv,a,b: (1-q)*k),
    ("(1-q)*(k+a*kv)",           lambda q,k,qv,kv,a,b: (1-q)*(k+a*kv)),
    ("(1-q-a*qv)*k",             lambda q,k,qv,kv,a,b: (1-q-a*qv)*k),
    ("(1-q)*k*(1+a*qv)",         lambda q,k,qv,kv,a,b: (1-q)*k*(1+a*qv)),
    ("(1-q)*k*(1+a*kv)",         lambda q,k,qv,kv,a,b: (1-q)*k*(1+a*kv)),
    ("(1-q)*(k+a*qv+b*kv)",      lambda q,k,qv,kv,a,b: (1-q)*(k+a*qv+b*kv)),
    ("(1-q+a*qv)*(k+b*kv)",      lambda q,k,qv,kv,a,b: (1-q+a*qv)*(k+b*kv)),
    # -- multiplicative second term (new) --
    ("(1-q)*k*qv*kv",            lambda q,k,qv,kv,a,b: (1-q)*k*qv*kv),
    ("(1-q)*k*qv",               lambda q,k,qv,kv,a,b: (1-q)*k*qv),
    ("(1-q)*k*kv",               lambda q,k,qv,kv,a,b: (1-q)*k*kv),
    ("(1-q)*(k*qv+a*kv)",        lambda q,k,qv,kv,a,b: (1-q)*(k*qv+a*kv)),
    ("(1-q)*(k*kv+a*qv)",        lambda q,k,qv,kv,a,b: (1-q)*(k*kv+a*qv)),
    ("(1-q)*k*(qv+a*kv)",        lambda q,k,qv,kv,a,b: (1-q)*k*(qv+a*kv)),
    ("(1-q)*k*(qv*kv+a)",        lambda q,k,qv,kv,a,b: (1-q)*k*(qv*kv+a)),
]

# Group to use for formula search: match current script usage in vggt_sam2_mask.py
# query = avg all layers;  key = mean(layers 2,4,7) — but here we use "all" group for both
FORMULA_KEY_LAYERS  = [2, 4, 7]
FORMULA_QUERY_GROUP = "all"


def _load_formula_maps(records):
    """
    For each (seq, frame_id), load:
      q   = mean of all query_mean layers
      k   = mean of key_mean layers [2,4,7]
      qv  = mean of all query_var layers
      kv  = mean of all key_var layers
    Returns list of (vals_dict, gt) pairs.
    """
    idx = {}
    for rec in records:
        if rec["layer"] == "avg" or rec["side"] == "diff":
            continue
        key = (rec["seq"], rec["frame_id"], rec["layer"], rec["side"], rec["stat"])
        idx[key] = rec["file_path"]

    pairs = sorted(set((r["seq"], r["frame_id"]) for r in records
                       if r["layer"] != "avg"))
    all_layers = LAYER_GROUPS["all"]

    results = []
    for seq, frame_id in pairs:
        def _avg(side, stat, layers):
            maps = []
            for lid in layers:
                fp = idx.get((seq, frame_id, lid, side, stat))
                if fp is not None:
                    maps.append(np.load(fp).astype(np.float32))
            return np.stack(maps, 0).mean(0) if maps else None

        q  = _avg("query", "mean", all_layers)
        k  = _avg("key",   "mean", FORMULA_KEY_LAYERS)
        qv = _avg("query", "var",  all_layers)
        kv = _avg("key",   "var",  all_layers)

        if q is None or k is None:
            continue
        if qv is None:
            qv = np.zeros_like(q)
        if kv is None:
            kv = np.zeros_like(k)

        # align shapes (all should be same, but just in case)
        H_p, W_p = q.shape
        gt = load_gt_patch(seq, frame_id, H_p, W_p)
        if gt is None:
            continue

        results.append(dict(q=q, k=k, qv=qv, kv=kv, gt=gt))
    return results


def _eval_formula(fn, a, b, data):
    aucs, aps, f1s = [], [], []
    for d in data:
        score = fn(d["q"], d["k"], d["qv"], d["kv"], a, b).ravel().astype(np.float64)
        g     = d["gt"].ravel().astype(np.int32)
        if g.sum() == 0 or (1-g).sum() == 0:
            continue
        try:
            aucs.append(roc_auc_score(g, score))
            aps.append(average_precision_score(g, score))
            # best F1 over uniform thresholds
            threshs = np.linspace(score.min(), score.max(), 20)
            best_f1 = max(f1_score(g, (score >= t).astype(int), zero_division=0)
                          for t in threshs)
            f1s.append(best_f1)
        except Exception:
            pass
    return (float(np.mean(aucs)) if aucs else np.nan,
            float(np.mean(aps))  if aps  else np.nan,
            float(np.mean(f1s))  if f1s  else np.nan)


def run_formula_search(records, out_dir: Path):
    print("  loading maps for formula search ...")
    data = _load_formula_maps(records)
    print(f"  {len(data)} (seq, frame) pairs loaded")

    rows = []
    n_formulas = len(FORMULA_DEFS)
    for fi, (name, fn) in enumerate(FORMULA_DEFS):
        print(f"  formula {fi+1}/{n_formulas}: {name}")
        needs_ab = "{a" in name or "{b" in name or "a*" in name or "b*" in name
        needs_b  = "b*" in name

        a_grid = COEF_GRID if needs_ab else [0.0]
        b_grid = COEF_GRID if needs_b  else [0.0]

        for a, b in product(a_grid, b_grid):
            auc, ap, f1 = _eval_formula(fn, a, b, data)
            rows.append(dict(formula=name, a=a, b=b,
                             roc_auc=auc, ap=ap, best_f1=f1))

    df = pd.DataFrame(rows)
    df = df.sort_values("roc_auc", ascending=False)
    df.to_csv(out_dir / "formula_search_results.csv", index=False)
    print(f"  saved formula_search_results.csv  ({len(df)} rows)")

    # top per formula
    top = df.groupby("formula").first().reset_index().sort_values("roc_auc", ascending=False)
    lines = ["Top formula per type (best roc_auc):\n"]
    lines.append(top[["formula","a","b","roc_auc","ap","best_f1"]].to_string(index=False))
    lines.append("\n\nOverall top 20:\n")
    lines.append(df.head(20)[["formula","a","b","roc_auc","ap","best_f1"]].to_string(index=False))
    txt = "\n".join(lines)
    (out_dir / "top_formula_summary.txt").write_text(txt)
    print(txt)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(layer_summary: pd.DataFrame, out_dir: Path):
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # 1. Bar plot: average roc_auc by map_type
    mt_auc = (layer_summary.groupby("map_type")["roc_auc"]
              .mean().sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(mt_auc.index, mt_auc.values, color="#4c72b0")
    ax.set_ylabel("mean ROC-AUC")
    ax.set_title("ROC-AUC by map type (averaged across layers & sequences)")
    ax.set_ylim(max(0, mt_auc.min() - 0.05), min(1, mt_auc.max() + 0.05))
    for i, v in enumerate(mt_auc.values):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(plot_dir / "roc_auc_by_maptype.png", dpi=130)
    plt.close(fig)

    # 2 & 3. Heatmaps: layer x map_type
    map_types = sorted(layer_summary["map_type"].unique())
    layers    = sorted(layer_summary["layer"].unique())

    def _heatmap(metric, fname, title):
        mat = np.full((len(layers), len(map_types)), np.nan)
        for r in layer_summary.itertuples():
            ri = layers.index(r.layer)
            ci = map_types.index(r.map_type)
            mat[ri, ci] = getattr(r, metric)

        fig, ax = plt.subplots(figsize=(max(6, len(map_types)*1.5),
                                        max(5, len(layers)*0.35)))
        im = ax.imshow(mat, aspect="auto", cmap="viridis",
                       vmin=np.nanmin(mat), vmax=np.nanmax(mat))
        ax.set_xticks(range(len(map_types)))
        ax.set_xticklabels(map_types, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([str(l) for l in layers], fontsize=7)
        ax.set_xlabel("map type")
        ax.set_ylabel("layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        fig.savefig(plot_dir / fname, dpi=130)
        plt.close(fig)

    _heatmap("roc_auc", "roc_auc_layer_x_maptype.png",
             "ROC-AUC: layer × map_type")
    _heatmap("abs_gap", "absgap_layer_x_maptype.png",
             "|gap|: layer × map_type")
    print(f"  plots saved to {plot_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npy_root", type=str,
                   default=str(NPY_ROOT))
    p.add_argument("--gt_root",  type=str,
                   default=str(GT_ROOT))
    p.add_argument("--out_dir",  type=str,
                   default=str(NPY_ROOT.parent / "attn_analysis"))
    p.add_argument("--skip_formula", action="store_true",
                   help="Skip formula search (faster)")
    p.add_argument("--skip_group",   action="store_true",
                   help="Skip group analysis (faster)")
    p.add_argument("--formula_only", action="store_true",
                   help="Run only formula search (skip per-map, group, plots)")
    return p.parse_args()


def main():
    args = parse_args()
    global NPY_ROOT, GT_ROOT
    NPY_ROOT = Path(args.npy_root)
    GT_ROOT  = Path(args.gt_root)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # ── Scan files ────────────────────────────────────────────────────────────
    print("Scanning npy files ...")
    records = scan_npy_files(NPY_ROOT)
    print(f"  {len(records)} files found across "
          f"{len(set(r['seq'] for r in records))} sequences")

    if args.formula_only:
        # ── Formula search only ───────────────────────────────────────────────
        print("Running formula search only ...")
        run_formula_search(records, out_dir)
    else:
        # ── Per-map metrics ───────────────────────────────────────────────────
        print("Computing per-map metrics ...")
        df_per = run_per_map(records, out_dir)

        # ── Layer summary ─────────────────────────────────────────────────────
        print("Computing by-layer summary ...")
        df_layer = run_layer_summary(df_per, out_dir)

        # ── Group analysis ────────────────────────────────────────────────────
        if not args.skip_group:
            print("Computing by-group analysis ...")
            run_group_analysis(records, out_dir)

        # ── Plots ─────────────────────────────────────────────────────────────
        print("Making plots ...")
        make_plots(df_layer, out_dir)

        # ── Formula search ────────────────────────────────────────────────────
        if not args.skip_formula:
            print("Running formula search ...")
            run_formula_search(records, out_dir)

    print(f"\nDone. Results in {out_dir}/")


if __name__ == "__main__":
    main()
