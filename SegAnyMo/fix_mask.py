import numpy as np
from pathlib import Path
from PIL import Image
import cv2

davis_root = Path("/mnt/data0/andy/SegAnyMo/DAVIS")
res_root   = Path("/mnt/data0/andy/Easi3R/results/davisunsupervised/easi3r_textregion_finalmovinglabel_refine3_test") 
out_root   = Path("/mnt/data0/andy/SegAnyMo/Easi3rresult/fixed_mask")         

# 1) 不再读 val.txt，而是遍历所有 unsupervised GT 里有的序列
gt_root = davis_root / "Annotations_unsupervised/480p"
seqs = sorted([p.name for p in gt_root.iterdir() if p.is_dir()])

print(f"[INFO] Found {len(seqs)} sequences under {gt_root}")

for seq in seqs:
    gt_dir   = gt_root / seq
    res_dir  = res_root / seq
    out_dir  = out_root / seq
    out_dir.mkdir(parents=True, exist_ok=True)

    if not res_dir.exists():
        print(f"[WARN] prediction dir not found, skip seq: {res_dir}")
        continue

    for gt_file in sorted(gt_dir.glob("*.png")):
        name = gt_file.name

        gt = np.array(Image.open(gt_file))
        H, W = gt.shape[:2]

        pred_file = res_dir / name
        if not pred_file.exists():
            print(f"[WARN] missing pred: {pred_file}")
            continue

        pred = np.array(Image.open(pred_file))
        if pred.ndim == 3:
            pred = pred[:, :, 0]

        if pred.shape[:2] != (H, W):
            print(f"[INFO] resize {seq}/{name}: {pred.shape[:2]} -> {(H, W)}")
            pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)

        Image.fromarray(pred.astype(np.uint8), mode="L").save(out_dir / name)
