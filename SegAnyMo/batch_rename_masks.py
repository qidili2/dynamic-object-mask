import os, re, pathlib

root = "/mnt/data0/andy/Easi3R/results/davis/easi3r_sam2_tracking"  # 你的结果总目录
exts = ("*.png","*.jpg","*.jpeg","*.bmp")

def idx(p):
    # dynamic_mask_123.png -> 123
    m = re.search(r'_(\d+)', p.name)
    if m: return int(m.group(1))
    m = re.search(r'(\d+)', p.name)
    return int(m.group(1)) if m else -1

for seq_dir in pathlib.Path(root).iterdir():
    if seq_dir.is_dir():
        # 找到所有图像文件
        paths = sorted([p for ext in exts for p in seq_dir.glob(ext)], key=idx)
        for i, p in enumerate(paths):
            link = seq_dir / f"{i:05d}.png"  # 零填充文件名
            if not link.exists():
                try:
                    os.symlink(p.name, link)  # 相对软链接，不破坏原文件
                except FileExistsError:
                    pass
        print(f"Processed {seq_dir} with {len(paths)} frames")