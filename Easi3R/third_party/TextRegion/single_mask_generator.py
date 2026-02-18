#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-image TextRegion debugger (stable 2-group setting)

前景标签: MOVING_LABELS 
背景标签: BACKGROUND_LABELS 

作用:
  - 只跑一张图 
  - 自动写 YAML
  - 运行 TextRegionSegmenter.py
  - 输出三件套: *_bin.npy / *_bin.png / *_overlay.png
"""

import sys, subprocess, yaml
from pathlib import Path
import numpy as np
import cv2

# =============================
# ====  固定参数区 ========
# =============================

IMAGE_PATH      = "/mnt/data0/andy/Easi3R/DAVIS/JPEGImages/480p/drift-turn/00000.jpg"
SEGMENTER_PY    = "TextRegionSegmenter.py"
SAM2_CKPT       = "/mnt/data0/andy/Easi3R/third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
YAML_OUT        = "utils/_single_mixlabels.yaml"
OUT_DIR         = "debug_single_mixlabels_out"

# 分割配置
POINTS_PER_SIDE = 12
RESIZE_METHOD   = "multi_resolution"
CROP_SIZE       = 336
DTYPE           = "bf16"
USE_UNION       = True     # 若存在 *_union.npy，用它约束
ALPHA           = 0.45     # overlay 透明度

# ===== 标签配置 =====
MOVING_LABELS = ["moving object","human", "animal", "vehicle", "sport"]
BACKGROUND_LABELS = ["ground", "grass", "water", "tree", "cloud", "sky", "window", "wall"]

# =============================
# ======= 函数部分 ===========
# =============================

def write_yaml(img: Path, yml: Path, labels):
    data = {str(img): {"label": labels}}
    yml.parent.mkdir(parents=True, exist_ok=True)
    with open(yml, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    print(f"[YAML] wrote {yml} with labels {labels}")

def run_segmenter(img: Path, yml: Path):
    cmd = [sys.executable, SEGMENTER_PY,
            "--image_list", str(img),
            "--image_query_cfg", str(yml),
            "--resize_method", RESIZE_METHOD,
            "--crop_size", str(CROP_SIZE),
            "--points_per_side", str(POINTS_PER_SIDE),
            "--dtype", DTYPE,
            "--viz_regions", "True",                # ← 改这里
            "--dump_region_labels", "True"]         # ← 改这里
    
    if Path(SAM2_CKPT).exists():
        cmd += ["--sam2_checkpoint", str(SAM2_CKPT)]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    img = Path(IMAGE_PATH)
    assert img.exists(), f"Image not found: {img}"

    # 1. 构造标签并写 YAML
    labels = MOVING_LABELS + BACKGROUND_LABELS
    yml = Path(YAML_OUT)
    write_yaml(img, yml, labels)

    # 2. 运行分割器
    run_segmenter(img, yml)

    # 3. 读取预测结果
    seq, stem = img.parent.name, img.stem
    op = Path("outputs") / seq
    pred  = np.load(op / f"{stem}_pred.npy").astype(np.int32)
    union = np.load(op / f"{stem}_union.npy").astype(np.uint8) if (op / f"{stem}_union.npy").exists() else None
        
    # —— 尺寸对齐：把 pred 最近邻缩放到 union 的尺寸
    H0, W0 = union.shape[:2]
    if pred.shape != (H0, W0):
        pred = cv2.resize(pred.astype(np.int32), (W0, H0), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    # 之后再做前景判定
    with open(yml, "r") as f:
        labels = list(next(iter(yaml.safe_load(f).values()))["label"])
    labels_low = [str(x).lower() for x in labels]

    moving_names = [n.lower() for n in MOVING_LABELS]
    label_to_id = {name.lower(): i for i, name in enumerate(labels_low)}
    moving_ids = [label_to_id[n] for n in moving_names if n in label_to_id]

    if moving_ids:
        fg = np.isin(pred, moving_ids).astype(np.uint8)
    else:
        # 兜底：非 background 视作前景
        bg_id = label_to_id.get("background", None)
        fg = (pred != bg_id).astype(np.uint8) if bg_id is not None else (pred > 0).astype(np.uint8)

    # 用 union 作为硬约束
    fg = (fg & (union > 0).astype(np.uint8)).astype(np.uint8)
    # 4. 前景 = 属于 MOVING_LABELS 的区域
    with open(yml, "r") as f:
        labels = list(next(iter(yaml.safe_load(f).values()))["label"])
    label_to_id = {name:i for i,name in enumerate(labels)}

    moving_ids = [label_to_id[n] for n in MOVING_LABELS if n in label_to_id]
    fg = np.isin(pred, moving_ids).astype(np.uint8)
    if USE_UNION and union is not None:
        fg = (fg & (union > 0)).astype(np.uint8)

    # 5. 绘制 overlay
    img0 = cv2.imread(str(img), cv2.IMREAD_COLOR)
    if img0 is None:
        raise FileNotFoundError(f"Cannot read image: {img}")
    if img0.shape[:2] != fg.shape:
        img0 = cv2.resize(img0, (fg.shape[1], fg.shape[0]), interpolation=cv2.INTER_NEAREST)
    tint = np.zeros_like(img0); tint[..., 1] = 255  # green
    overlay = img0.copy()
    ys, xs = np.where(fg == 1)
    overlay[ys, xs] = (ALPHA * tint[ys, xs] + (1 - ALPHA) * overlay[ys, xs]).astype(np.uint8)

    # 6. 保存结果
    out_dir = Path(OUT_DIR) / seq
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{stem}_bin.png"), fg * 255)
    np.save(out_dir / f"{stem}_bin.npy", fg)
    cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)

    print(f"[OK] {seq}/{stem}: bin+overlay saved to {out_dir}")
    print(f"[STAT] Foreground pixels = {fg.sum()} / {fg.size}")

if __name__ == "__main__":
    main()
