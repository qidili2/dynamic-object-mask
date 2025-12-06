#!/usr/bin/env python
import os
import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare DAVIS-2017 unsupervised results folder from existing "
            "per-sequence grayscale masks."
        )
    )
    parser.add_argument(
        "--src_root",
        type=str,
        required=True,
        help=(
            "根目录：里面应有多个子文件夹，每个子文件夹名为 seq 名，例如 "
            "blackswan, bmx-trees 等，子文件夹中包含灰度 PNG mask。"
        ),
    )
    parser.add_argument(
        "--dst_root",
        type=str,
        required=True,
        help=(
            "输出根目录，将创建 <dst_root>/<seq>/<frame>.png，"
            "用于 evaluation_method.py --task unsupervised"
        ),
    )
    parser.add_argument(
        "--davis_path",
        type=str,
        required=True,
        help=(
            "DAVIS 根目录，包含 ImageSets/2017/<set>.txt。"
        ),
    )
    parser.add_argument(
        "--set",
        type=str,
        default="val",
        help="使用的 DAVIS 子集 (train, val, test-dev, ...) 默认: val",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".png",
        help="源 mask 的后缀名，默认 .png",
    )
    return parser.parse_args()


def load_seqlist(davis_path: Path, subset: str):
    set_file = davis_path / "ImageSets" / "2017" / f"{subset}.txt"
    if not set_file.is_file():
        raise FileNotFoundError(f"Seqlist file not found: {set_file}")
    with open(set_file, "r") as f:
        seqs = [l.strip() for l in f if l.strip()]
    print(f"[INFO] Loaded {len(seqs)} sequences from {set_file}")
    return seqs


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    davis_path = Path(args.davis_path)
    subset = args.set
    ext = args.ext

    if not src_root.is_dir():
        raise RuntimeError(f"src_root is not a directory: {src_root}")
    dst_root.mkdir(parents=True, exist_ok=True)

    seqlist = load_seqlist(davis_path, subset)

    for seq in seqlist:
        src_seq_dir = src_root / seq
        if not src_seq_dir.is_dir():
            print(f"[WARN] seq '{seq}' not found under {src_root}, skip.")
            continue

        dst_seq_dir = dst_root / seq
        dst_seq_dir.mkdir(parents=True, exist_ok=True)

        # 收集所有灰度图
        files = sorted(
            [p for p in src_seq_dir.iterdir()
             if p.is_file() and p.suffix.lower() == ext.lower()]
        )
        if len(files) == 0:
            print(f"[WARN] no '{ext}' files in {src_seq_dir}, skip.")
            continue

        print(f"[INFO] seq '{seq}': {len(files)} frames.")

        for new_idx, f in enumerate(files):
            # 读取为灰度，确保单通道 uint8
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARN] failed to read {f}, skip.")
                continue

            if img.dtype != np.uint8:
                img = img.astype(np.uint8)

            out_name = f"{new_idx:05d}.png"
            out_path = dst_seq_dir / out_name
            cv2.imwrite(str(out_path), img)

    print("\n[INFO] Done. 你现在可以用这个目录做 DAVIS unsupervised eval，例如：")
    print(
        "  python evaluation_method.py "
        f"--task unsupervised --set {subset} "
        f"--davis_path {davis_path} "
        f"--results_path {dst_root}"
    )


if __name__ == "__main__":
    main()
