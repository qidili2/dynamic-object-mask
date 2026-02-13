#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

IMG_EXTS_DEFAULT = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def has_rsync() -> bool:
    return shutil.which("rsync") is not None


def count_images(seq_dir: Path, exts: List[str]) -> int:
    exts_lower = set(e.lower() for e in exts)
    n = 0
    try:
        for p in seq_dir.iterdir():
            if p.is_file() and p.suffix.lower() in exts_lower:
                n += 1
    except Exception as e:
        raise RuntimeError(f"Failed to read directory: {seq_dir} ({e})")
    return n


def rsync_copy(src_dir: Path, dst_dir: Path) -> None:
    """
    Copy src_dir -> dst_dir using rsync.
    We copy the folder itself (sequence folder) into dst_dir's parent by syncing src_dir/ -> dst_dir/
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync",
        "-a",
        "--delete",   # keep dst in sync if rerun (can be removed if you never want delete)
        "--info=progress2",
        str(src_dir) + "/",  # contents
        str(dst_dir) + "/",  # into
    ]
    subprocess.run(cmd, check=True)


def shutil_copytree(src_dir: Path, dst_dir: Path, overwrite: bool) -> None:
    if dst_dir.exists():
        if not overwrite:
            print(f"[SKIP] exists: {dst_dir}")
            return
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir, copy_function=shutil.copy2, dirs_exist_ok=False)


def main():
    parser = argparse.ArgumentParser(
        description="Split MoCA sequences by image count, 30 sequences per folder, copy into moca1/moca2/..."
    )
    parser.add_argument(
        "--src",
        default="/home/qidili2/dynamic-object-mask/SegAnyMo/MoCA/JPEGImages",
        help="Source JPEGImages directory",
    )
    parser.add_argument(
        "--dst_base",
        default="/home/qidili2/dynamic-object-mask/SegAnyMo/MoCAsplit",
        help="Destination base directory (will create moca1, moca2, ... under it)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=30,
        help="How many sequences per destination folder",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort by image count ascending (default: descending)",
    )
    parser.add_argument(
        "--exts",
        default=",".join(IMG_EXTS_DEFAULT),
        help="Comma-separated list of image extensions to count (default: jpg,jpeg,png,bmp,webp)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination sequence folders (default: skip existing)",
    )
    parser.add_argument(
        "--no_rsync",
        action="store_true",
        help="Do not use rsync even if available; use shutil.copytree",
    )
    args = parser.parse_args()

    src = Path(args.src)
    dst_base = Path(args.dst_base)
    chunk_size = args.chunk_size
    ascending = args.ascending
    overwrite = args.overwrite
    exts = [e.strip() for e in args.exts.split(",") if e.strip()]

    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    if not src.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {src}")

    dst_base.mkdir(parents=True, exist_ok=True)

    # list sequences (subdirectories)
    seq_dirs = [p for p in src.iterdir() if p.is_dir()]
    if not seq_dirs:
        print(f"No sequence folders found under: {src}")
        return

    # count images
    stats: List[Tuple[str, int, Path]] = []
    print(f"Scanning sequences under: {src}")
    for i, seq_dir in enumerate(sorted(seq_dirs, key=lambda x: x.name)):
        n = count_images(seq_dir, exts)
        stats.append((seq_dir.name, n, seq_dir))
        if (i + 1) % 200 == 0:
            print(f"  scanned {i+1}/{len(seq_dirs)} ...")

    # sort by count
    stats.sort(key=lambda x: x[1], reverse=not ascending)

    # decide copy method
    use_rsync = (not args.no_rsync) and has_rsync()
    if use_rsync:
        print("Copy method: rsync")
    else:
        print("Copy method: shutil.copytree (rsync not used/available)")

    # split into chunks and copy
    manifest_path = dst_base / "moca_split_manifest.csv"
    print(f"Writing manifest: {manifest_path}")

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "sequence", "image_count", "src", "dst"])

        total = len(stats)
        for idx, (seq_name, img_count, seq_path) in enumerate(stats):
            group_id = idx // chunk_size + 1
            group_name = f"moca{group_id}"

            # ---- MINIMAL CHANGE START ----
            # Make: dst_base/mocaX/JPEGImages/<seq_name>
            group_dir = dst_base / group_name / "JPEGImages"
            dst_seq_dir = group_dir / seq_name
            group_dir.mkdir(parents=True, exist_ok=True)
            # ---- MINIMAL CHANGE END ----

            if dst_seq_dir.exists() and not overwrite:
                print(f"[{idx+1}/{total}] SKIP exists: {group_name}/JPEGImages/{seq_name} (images={img_count})")
            else:
                print(f"[{idx+1}/{total}] COPY  -> {group_name}/JPEGImages/{seq_name} (images={img_count})")
                if use_rsync:
                    rsync_copy(seq_path, dst_seq_dir)
                else:
                    shutil_copytree(seq_path, dst_seq_dir, overwrite=overwrite)

            writer.writerow([group_name, seq_name, img_count, str(seq_path), str(dst_seq_dir)])

    print("\nDone.")
    print(f"Destination base: {dst_base}")
    print(f"Manifest CSV: {manifest_path}")


if __name__ == "__main__":
    main()
