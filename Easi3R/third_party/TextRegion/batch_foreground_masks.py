#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys, subprocess, yaml, glob
from pathlib import Path
import numpy as np
import cv2

# ===== 标签（与 single 保持一致） =====
MOVING_LABELS = ["moving object","human", "animal", "vehicle", "sport"]
BACKGROUND_LABELS = ["ground", "grass", "water", "tree", "cloud", "sky", "window", "wall"]
# BACKGROUND_LABELS = ["water", "ground", "solid", "plant", "building", "structural", "sky"]

# ===== 公用：写 YAML，所有图使用同一标签表 =====
def build_yaml(image_paths, yaml_path, labels):
    data = {}
    for p in image_paths:
        data[str(p)] = {"label": labels}
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return yaml_path

# ===== 运行 TextRegionSegmenter（保持和 single 一致的参数） =====
def run_segmenter(image_path, yaml_path, points_per_side=24,
                  resize_method="resize", crop_size=448,
                  dtype="bf16", sam2_checkpoint=None):
    cmd = [
        sys.executable, "TextRegionSegmenter.py",
        "--image_list", str(image_path),
        "--image_query_cfg", str(yaml_path),
        "--resize_method", resize_method,
        "--crop_size", str(crop_size),
        "--points_per_side", str(points_per_side),
        "--dtype", dtype,
        # 需要时也可以加入可视化/region导出
        # "--viz_regions", "True",
        # "--dump_region_labels", "True",
    ]
    if sam2_checkpoint:
        cmd += ["--sam2_checkpoint", str(sam2_checkpoint)]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)

# ===== 尺寸对齐：把 A 最近邻缩放到 B 的尺寸 =====
def resize_like(a: np.ndarray, like: np.ndarray) -> np.ndarray:
    H, W = like.shape[:2]
    if a.shape[:2] == (H, W):
        return a
    return cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST)

# ===== 主处理（批量） =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="根目录，例如 /mnt/data0/andy/Easi3R/DAVIS/JPEGImages/480p")
    ap.add_argument("--yaml_out", default="utils/dataset_labels.yaml",
                    help="输出 YAML 路径")
    ap.add_argument("--out_dir", default="batch_outputs",
                    help="二值mask与overlay输出目录")
    ap.add_argument("--points_per_side", type=int, default=24)
    ap.add_argument("--resize_method", choices=["resize","multi_resolution"], default="resize")
    ap.add_argument("--crop_size", type=int, default=448)
    ap.add_argument("--dtype", choices=["fp32","bf16"], default="bf16")
    ap.add_argument("--sam2_checkpoint", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    # 批量挑第一帧 00000.jpg；如需改成 *.jpg 自行调整
    SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

    image_paths = []
    # 遍历每个 sequence 子目录
    for seq_dir in sorted(Path(root).glob("*")):
        if not seq_dir.is_dir():
            continue

        # 收集该 sequence 下所有图片
        imgs = []
        for ext in SUPPORTED_EXTS:
            imgs.extend(sorted(seq_dir.glob(f"*{ext}")))
        if len(imgs) == 0:
            continue

        # 取按文件名排序后的最小文件（一般就是帧号最小的）
        first_img = sorted(imgs)[0]
        image_paths.append(first_img)

    print(f"[Info] Found {len(image_paths)} sequences, first images:")
    for p in image_paths:
        print("  ", p)
        
    if not image_paths:
        print(f"No images like {root}/*/00000.jpg found.")
        return

    # 统一标签表（single 的 MOVING + BACKGROUND）
    labels = MOVING_LABELS + BACKGROUND_LABELS
    yaml_path = build_yaml(image_paths, Path(args.yaml_out), labels)
    print(f"[YAML] wrote {yaml_path} with {len(labels)} labels for {len(image_paths)} images")

    # 预构建一些 lower-case 映射，避免大小写问题
    labels_lc = [s.lower() for s in labels]
    moving_lc = [s.lower() for s in MOVING_LABELS]
    label_to_id = {name.lower(): i for i, name in enumerate(labels_lc)}
    moving_ids = [label_to_id[n] for n in moving_lc if n in label_to_id]
    bg_id = label_to_id.get("background", None)

    for img in image_paths:
        try:
            # 1) 跑分割
            run_segmenter(
                img, yaml_path,
                points_per_side=args.points_per_side,
                resize_method=args.resize_method,
                crop_size=args.crop_size,
                dtype=args.dtype,
                sam2_checkpoint=args.sam2_checkpoint,
            )

            # 2) 回读结果（主程序 pred/union 是原图尺寸）
            seq, stem = img.parent.name, img.stem
            out_seg = Path("outputs") / seq
            pred_path  = out_seg / f"{stem}_pred.npy"
            union_path = out_seg / f"{stem}_union.npy"

            if not pred_path.exists():
                raise FileNotFoundError(f"Missing {pred_path}; make sure TextRegionSegmenter saved it.")

            pred  = np.load(pred_path)   # (H?, W?) uint8
            union = np.load(union_path).astype(np.uint8) if union_path.exists() else None

            # 3) 计算前景（与 single 相同逻辑）
            if len(moving_ids) > 0:
                # 先对 union 对齐（如果存在），避免后续广播问题
                if union is not None:
                    pred = resize_like(pred, union)
                fg = np.isin(pred, moving_ids).astype(np.uint8)
            else:
                # 兜底：非 background == 前景
                if union is not None:
                    pred = resize_like(pred, union)
                if bg_id is not None:
                    fg = (pred != bg_id).astype(np.uint8)
                else:
                    fg = (pred > 0).astype(np.uint8)

            # 4) 用 union 作为硬约束（如果存在），先把 union 对齐到 pred 再相与
            if union is not None:
                union = resize_like(union, pred)
                fg = (fg & (union > 0)).astype(np.uint8)

            # 5) 叠加到原图（按 pred 尺寸对齐）
            img0 = cv2.imread(str(img), cv2.IMREAD_COLOR)  # BGR
            if img0 is None:
                raise FileNotFoundError(f"Cannot read image: {img}")
            img0 = resize_like(img0, pred)

            tint = np.zeros_like(img0); tint[..., 1] = 255  # green
            overlay = img0.copy()
            ys, xs = np.where(fg == 1)
            alpha = 0.45
            overlay[ys, xs] = (alpha * tint[ys, xs] + (1 - alpha) * overlay[ys, xs]).astype(np.uint8)

            # 6) 保存（与 single 同目录结构：batch_outputs/<seq>/）
            out_dir = Path(args.out_dir) / seq
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"{stem}_bin.png"), fg * 255)
            np.save(out_dir / f"{stem}_bin.npy", fg)
            cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)
            print(f"[OK] {seq}/{stem}: bin -> {out_dir/f'{stem}_bin.png'}, overlay -> {out_dir/f'{stem}_overlay.png'}")

        except subprocess.CalledProcessError as e:
            print(f"[ERR] Segmenter failed on {img}: {e}")
        except Exception as e:
            print(f"[ERR] Post-process failed on {img}: {e}")

if __name__ == "__main__":
    main()



# import argparse, sys, subprocess, yaml, glob
# from pathlib import Path
# import numpy as np
# import cv2

# COCO_OBJECT_TXT = "/mnt/data0/andy/Easi3R/third_party/TextRegion/configs/cls_coco_object.txt"
# COCO_STUFF_TXT  = "/mnt/data0/andy/Easi3R/third_party/TextRegion/configs/cls_coco_stuff.txt"

# def _read_label_txt(txt_path: Path):
#     lines = []
#     for raw in Path(txt_path).read_text(encoding="utf-8").splitlines():
#         s = raw.strip()
#         if not s or s.startswith("#"):
#             continue
#         s = s.replace("_", " ").lower()
#         lines.append(s)
#     seen = set(); dedup = []
#     for s in lines:
#         if s not in seen:
#             dedup.append(s); seen.add(s)
#     return dedup

# def load_coco_fg_bg(object_txt: Path, stuff_txt: Path):
#     fg = _read_label_txt(object_txt)
#     bg = _read_label_txt(stuff_txt)
#     fg_set = set(fg)
#     bg = [b for b in bg if b not in fg_set]
#     for bad in ["other","misc","background","unknown"]:
#         fg = [f for f in fg if f != bad]
#         bg = [b for b in bg if b != bad]
#     return fg, bg

# def build_yaml(image_paths, yaml_path, labels):
#     data = {}
#     for p in image_paths:
#         data[str(p)] = {"label": labels}
#     yaml_path = Path(yaml_path)
#     yaml_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(yaml_path, "w") as f:
#         yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
#     return yaml_path

# def run_segmenter(image_path, yaml_path, points_per_side=24,
#                   resize_method="resize", crop_size=448,
#                   dtype="bf16", sam2_checkpoint=None):
#     cmd = [
#         sys.executable, "TextRegionSegmenter.py",
#         "--image_list", str(image_path),
#         "--image_query_cfg", str(yaml_path),
#         "--resize_method", resize_method,
#         "--crop_size", str(crop_size),
#         "--points_per_side", str(points_per_side),
#         "--dtype", dtype,
#         # 可选打开：
#         # "--viz_regions", "True",
#         # "--dump_region_labels", "True",
#     ]
#     if sam2_checkpoint:
#         cmd += ["--sam2_checkpoint", str(sam2_checkpoint)]
#     print("[RUN]", " ".join(cmd))
#     subprocess.run(cmd, check=True)

# def _resize_like(a: np.ndarray, like: np.ndarray):
#     H, W = like.shape[:2]
#     if a.shape[:2] == (H, W):
#         return a
#     return cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST)

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--root", required=True,
#                     help="如 /mnt/data0/andy/Easi3R/DAVIS/JPEGImages/480p")
#     ap.add_argument("--yaml_out", default="utils/_coco_fg_bg.yaml")
#     ap.add_argument("--out_dir", default="batch_coco_fg_bg_out")
#     ap.add_argument("--points_per_side", type=int, default=24)
#     ap.add_argument("--resize_method", choices=["resize","multi_resolution"], default="resize")
#     ap.add_argument("--crop_size", type=int, default=448)
#     ap.add_argument("--dtype", choices=["fp32","bf16"], default="bf16")
#     ap.add_argument("--sam2_checkpoint", default=None)
#     args = ap.parse_args()

#     root = Path(args.root)
#     image_paths = sorted(Path(p) for p in glob.glob(str(root / "*" / "00000.jpg")))
#     if not image_paths:
#         print(f"No images like {root}/*/00000.jpg found.")
#         return

#     # 载入 COCO 前景/背景并写 YAML（全数据集复用同一标签表）
#     fg_list, bg_list = load_coco_fg_bg(COCO_OBJECT_TXT, COCO_STUFF_TXT)
#     labels = fg_list + bg_list
#     yaml_path = build_yaml(image_paths, args.yaml_out, labels)
#     print(f"[YAML] wrote {yaml_path} with {len(labels)} labels (FG {len(fg_list)} + BG {len(bg_list)})")

#     # id 映射（便于前景二值化）
#     labels_lc = [x.lower() for x in labels]
#     name2id = {n:i for i,n in enumerate(labels_lc)}
#     moving_ids = [name2id[n] for n in fg_list if n in name2id]

#     for img in image_paths:
#         try:
#             run_segmenter(
#                 img, yaml_path,
#                 points_per_side=args.points_per_side,
#                 resize_method=args.resize_method,
#                 crop_size=args.crop_size,
#                 dtype=args.dtype,
#                 sam2_checkpoint=args.sam2_checkpoint,
#             )

#             seq, stem = img.parent.name, img.stem
#             out_seg = Path("outputs") / seq
#             pred_path  = out_seg / f"{stem}_pred.npy"
#             union_path = out_seg / f"{stem}_union.npy"

#             if not pred_path.exists():
#                 raise FileNotFoundError(f"Missing {pred_path}")

#             pred  = np.load(pred_path)                             # (H?,W?)  uint8
#             union = np.load(union_path) if union_path.exists() else None

#             # 统一尺寸并得到前景二值
#             if union is not None:
#                 pred = _resize_like(pred, union)
#             fg = np.isin(pred, moving_ids).astype(np.uint8)
#             if union is not None:
#                 union = (union > 0).astype(np.uint8)
#                 union = _resize_like(union, fg)
#                 fg = (fg & union).astype(np.uint8)

#             # 叠 overlay
#             img0 = cv2.imread(str(img), cv2.IMREAD_COLOR)
#             img0 = _resize_like(img0, fg)
#             tint = np.zeros_like(img0); tint[...,1] = 255
#             overlay = img0.copy()
#             ys, xs = np.where(fg == 1)
#             overlay[ys, xs] = (0.45 * tint[ys, xs] + 0.55 * overlay[ys, xs]).astype(np.uint8)

#             out_dir = Path(args.out_dir) / seq
#             out_dir.mkdir(parents=True, exist_ok=True)
#             cv2.imwrite(str(out_dir / f"{stem}_bin.png"), fg * 255)
#             np.save(out_dir / f"{stem}_bin.npy", fg)
#             cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)
#             print(f"[OK] {seq}/{stem}: bin -> {out_dir/f'{stem}_bin.png'}, overlay -> {out_dir/f'{stem}_overlay.png'}")

#         except subprocess.CalledProcessError as e:
#             print(f"[ERR] Segmenter failed on {img}: {e}")
#         except Exception as e:
#             print(f"[ERR] Post-process failed on {img}: {e}")

# if __name__ == "__main__":
#     main()
