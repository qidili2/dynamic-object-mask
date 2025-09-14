#!/usr/bin/env python3
"""
Compare two DAVIS per-sequence CSVs (baseline vs improved), compute deltas, and
summarize which sequences improved or regressed, sorted by improvement.

Outputs:
  - diff_by_sequence.csv : B - A diffs per "Sequence"
  - diff_by_base_sequence.csv : diffs grouped by base sequence (e.g., "gold-fish_1..N" -> "gold-fish"),
                                with summed improvements ("总和提升度") and sorted by dJF-Mean desc.
  - summary.txt : quick textual summary
"""
import argparse, os, re
import pandas as pd
import numpy as np

RE_DEFAULT_GROUP = r"^(.*)_(\d+)$"   # group "name_123" -> base "name"

def _find_col(df, candidates):
    cl = [c for c in df.columns]
    low = {c.lower().strip(): c for c in cl}
    for cand in candidates:
        k = cand.lower()
        if k in low:
            return low[k]
    # fuzzy
    for c in cl:
        if any(k in c.lower() for k in candidates):
            return c
    return None

def load_perseq_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    seq_col = _find_col(df, ["sequence","seq"])
    j_col   = _find_col(df, ["j-mean","j_mean","jmean","j"])
    f_col   = _find_col(df, ["f-mean","f_mean","fmean","f"])
    if not (seq_col and j_col and f_col):
        raise ValueError(f"Missing required columns in {path}. Need columns like 'Sequence', 'J-Mean', 'F-Mean'. Got: {list(df.columns)}")
    out = df[[seq_col, j_col, f_col]].copy()
    out.columns = ["Sequence","J-Mean","F-Mean"]
    out["JF-Mean"] = (out["J-Mean"] + out["F-Mean"]) / 2.0
    return out

def base_seq(name: str, pattern: str):
    m = re.match(pattern, str(name)) if pattern else None
    return m.group(1) if m else str(name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_a", required=True, help="Baseline per-sequence CSV path")
    ap.add_argument("--csv_b", required=True, help="Improved per-sequence CSV path")
    ap.add_argument("--out_dir", default=".", help="Output directory")
    ap.add_argument("--group_pattern", default=RE_DEFAULT_GROUP,
                    help="Regex to extract base sequence; default strips trailing _<digits>. Set empty '' to disable.")
    ap.add_argument("--which_is_better", choices=["higher","lower"], default="higher",
                    help="For J/F metrics, higher is better (default). If your metric is 'lower is better', choose 'lower'.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    a = load_perseq_csv(args.csv_a)
    b = load_perseq_csv(args.csv_b)

    m = a.merge(b, on="Sequence", suffixes=("_A","_B"), how="inner")
    if m.empty:
        raise SystemExit("No overlapping 'Sequence' names between the two CSVs. Check inputs.")

    # diffs: B - A (positive means improvement if higher is better)
    for key in ["J-Mean","F-Mean","JF-Mean"]:
        m[f"d{key}"] = m[f"{key}_B"] - m[f"{key}_A"]

    asc = (args.which_is_better == "lower")
    m_sorted = m.sort_values("dJF-Mean", ascending=asc).reset_index(drop=True)

    p_seq = os.path.join(args.out_dir, "diff_by_sequence.csv")
    m_sorted.to_csv(p_seq, index=False)

    # group by base sequence and SUM the deltas (总和提升度)
    if args.group_pattern == "":
        g = (m_sorted.groupby("Sequence")[["dJ-Mean","dF-Mean","dJF-Mean"]]
             .sum()
             .sort_values("dJF-Mean", ascending=asc)
             .reset_index()
             .rename(columns={"Sequence":"base_seq"}))
    else:
        m_sorted["base_seq"] = m_sorted["Sequence"].map(lambda s: base_seq(s, args.group_pattern))
        g = (m_sorted.groupby("base_seq")[["dJ-Mean","dF-Mean","dJF-Mean"]]
             .sum()
             .sort_values("dJF-Mean", ascending=asc)
             .reset_index())

    p_grp = os.path.join(args.out_dir, "diff_by_base_sequence.csv")
    g.to_csv(p_grp, index=False)

    gains = int((g["dJF-Mean"] > 0).sum())
    drops = int((g["dJF-Mean"] < 0).sum())
    same  = int((g["dJF-Mean"] == 0).sum())

    # brief summary
    lines = []
    lines.append(f"[Summary by base sequence] groups: {len(g)}, improved: {gains}, worse: {drops}, same: {same}")
    head = g.head(min(20, len(g)))[["base_seq","dJF-Mean","dJ-Mean","dF-Mean"]]
    tail = g.tail(min(20, len(g)))[["base_seq","dJF-Mean","dJ-Mean","dF-Mean"]].sort_values("dJF-Mean")
    lines.append("\n[Top improvements by base]\n" + head.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("\n[Top drops by base]\n" + tail.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    p_txt = os.path.join(args.out_dir, "summary.txt")
    with open(p_txt, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nSaved:\n  {p_seq}\n  {p_grp}\n  {p_txt}")

if __name__ == "__main__":
    main()
