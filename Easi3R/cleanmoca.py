import os
import shutil
import argparse
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def is_seq_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    for f in p.iterdir():
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            return True
    return False

def find_seq_dirs(jpeg_root: Path):
    """
    自动探测 seq 目录层级：
    - 直接是 seq： JPEGImages/<seq>/*.jpg
    - 或有一层分辨率： JPEGImages/480p/<seq>/*.jpg
    """
    if not jpeg_root.exists():
        raise FileNotFoundError(f"JPEGImages not found: {jpeg_root}")

    # 如果 JPEGImages 下直接就是很多 seq 文件夹
    direct = [d for d in jpeg_root.iterdir() if is_seq_dir(d)]
    if direct:
        return sorted(direct)

    # 否则尝试下一层（比如 480p）
    seq_dirs = []
    for sub in jpeg_root.iterdir():
        if not sub.is_dir():
            continue
        for d in sub.iterdir():
            if is_seq_dir(d):
                seq_dirs.append(d)
    return sorted(seq_dirs)

def count_frames(seq_dir: Path) -> int:
    return sum(1 for f in seq_dir.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)

def link_or_copy_tree(src: Path, dst: Path, mode: str):
    """
    mode:
      - copy: 复制所有文件
      - symlink: 对文件创建符号链接（推荐，省空间）
      - hardlink: 硬链接（同一文件系统下省空间）
    """
    dst.mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for fn in files:
            s = Path(root) / fn
            t = dst / rel / fn
            if t.exists():
                continue

            if mode == "copy":
                shutil.copy2(s, t)
            elif mode == "symlink":
                os.symlink(s.resolve(), t)
            elif mode == "hardlink":
                os.link(s, t)
            else:
                raise ValueError(f"Unknown mode: {mode}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moca_root", type=str, required=True,
                    help="MoCA 数据集根目录（里面应包含 JPEGImages/）")
    ap.add_argument("--out_root", type=str, required=True,
                    help="输出目录根（会创建同样的 JPEGImages/... 结构）")
    ap.add_argument("--threshold", type=int, default=100,
                    help="只保留帧数 < threshold 的序列（默认 100）")
    ap.add_argument("--mode", type=str, default="symlink",
                    choices=["copy", "symlink", "hardlink"],
                    help="输出方式：copy/symlink/hardlink（默认 symlink）")
    args = ap.parse_args()

    moca_root = Path(args.moca_root)
    out_root = Path(args.out_root)
    jpeg_root = moca_root / "JPEGImages"

    seq_dirs = find_seq_dirs(jpeg_root)
    if not seq_dirs:
        raise RuntimeError(f"No sequence dirs found under: {jpeg_root}")

    picked = []
    skipped = []

    for sd in seq_dirs:
        n = count_frames(sd)
        if n < args.threshold:
            # 保持相对 JPEGImages 的路径结构
            rel = sd.relative_to(jpeg_root)
            dst = out_root / "JPEGImages" / rel
            link_or_copy_tree(sd, dst, args.mode)
            picked.append((str(rel), n))
        else:
            skipped.append((str(sd.relative_to(jpeg_root)), n))

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "picked_under_threshold.txt", "w", encoding="utf-8") as f:
        for name, n in sorted(picked, key=lambda x: x[0]):
            f.write(f"{name}\t{n}\n")

    print(f"Total seq found: {len(seq_dirs)}")
    print(f"Picked (<{args.threshold}): {len(picked)}  -> list saved to {out_root/'picked_under_threshold.txt'}")
    print(f"Skipped: {len(skipped)}")
    if picked:
        print("Example picked:", picked[:5])

if __name__ == "__main__":
    main()
