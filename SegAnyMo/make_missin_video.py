#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

ALL_TXT = Path("/home/qidili2/dynamic-object-mask/SegAnyMo/MoCA/seq_list_all.txt")
LIST_TXT = Path("/home/qidili2/dynamic-object-mask/SegAnyMo/MoCA/seq_list.txt")
JPEG_ROOT = Path("/home/qidili2/dynamic-object-mask/SegAnyMo/MoCA/JPEGImages")
OUT_DIR = Path("/home/qidili2/dynamic-object-mask/SegAnyMo/MoCA/MP4_missing")

FPS = 30
CRF = 18
PRESET = "medium"

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def read_seq_list(p: Path) -> list[str]:
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    seqs = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        seqs.append(line)
    return seqs


def find_frames(seq_dir: Path) -> list[Path]:
    if not seq_dir.is_dir():
        return []
    frames = [p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    frames.sort(key=lambda x: x.name)  # lexicographic; 对 000001.jpg 这类最合适
    return frames


def write_concat_list(frames: list[Path], list_file: Path) -> None:
    # ffmpeg concat demuxer format: each line: file '/abs/path/to/frame.jpg'
    # 绝对路径 + safe 0 最省心
    with list_file.open("w") as f:
        for fr in frames:
            f.write(f"file '{fr.as_posix()}'\n")


def run_ffmpeg_from_list(list_file: Path, out_mp4: Path) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-r", str(FPS),                 # 输入帧率（concat 读图时常用）
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c:v", "libx264",
        "-crf", str(CRF),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def main():
    all_seqs = set(read_seq_list(ALL_TXT))
    list_seqs = set(read_seq_list(LIST_TXT))
    missing = sorted(all_seqs - list_seqs)

    print(f"all: {len(all_seqs)}  list: {len(list_seqs)}  missing: {len(missing)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    failed = []
    skipped = 0
    done = 0

    for i, seq in enumerate(missing, 1):
        seq_dir = JPEG_ROOT / seq
        frames = find_frames(seq_dir)

        if not frames:
            print(f"[{i}/{len(missing)}] {seq}: no frames found in {seq_dir}")
            failed.append(seq)
            continue

        out_mp4 = OUT_DIR / f"{seq}.mp4"
        if out_mp4.exists() and out_mp4.stat().st_size > 0:
            print(f"[{i}/{len(missing)}] {seq}: exists, skip -> {out_mp4}")
            skipped += 1
            continue

        list_file = OUT_DIR / f"{seq}__frames.txt"
        try:
            write_concat_list(frames, list_file)
            print(f"[{i}/{len(missing)}] {seq}: {len(frames)} frames -> {out_mp4}")
            run_ffmpeg_from_list(list_file, out_mp4)
            done += 1
        except subprocess.CalledProcessError as e:
            print(f"[{i}/{len(missing)}] {seq}: ffmpeg failed: {e}")
            failed.append(seq)
        finally:
            # list 文件留着也行；你想清理就取消注释
            # if list_file.exists():
            #     list_file.unlink()
            pass

    print("\n=== Summary ===")
    print(f"done: {done}, skipped: {skipped}, failed: {len(failed)}")
    if failed:
        print("failed seqs:")
        for s in failed:
            print("  ", s)


if __name__ == "__main__":
    main()
