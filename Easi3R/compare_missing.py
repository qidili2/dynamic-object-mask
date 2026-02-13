#!/usr/bin/env python3
import os, re, argparse, csv

parser = argparse.ArgumentParser()
parser.add_argument('--label_path', required=True, help='只用来取 sequence 名字（以及可选统计帧数）')
parser.add_argument('--results_path', required=True, help='结果根目录，内含 <sequence>/dynamic_mask_*.png')
parser.add_argument('--seq', default='', help='仅检查某一个序列名（可选）')
args = parser.parse_args()

label_path = os.path.abspath(args.label_path)
results_path = os.path.abspath(args.results_path)

pred_re = re.compile(r'^dynamic_mask_(\d+)\.png$')

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def count_images_anyname(seq_dir: str) -> int:
    """统计 seq_dir 下的图片数量，不要求数字命名"""
    n = 0
    try:
        for f in os.listdir(seq_dir):
            p = os.path.join(seq_dir, f)
            if os.path.isfile(p) and os.path.splitext(f)[1].lower() in IMG_EXTS:
                n += 1
    except FileNotFoundError:
        return 0
    return n

def list_pred_indices(seq_dir: str):
    idx = []
    try:
        for f in os.listdir(seq_dir):
            m = pred_re.match(f)
            if m:
                idx.append(int(m.group(1)))
    except FileNotFoundError:
        return []
    return sorted(idx)

# 序列集合：只看 label_path 下的子目录
if args.seq:
    seqs = [args.seq]
else:
    seqs = sorted([d for d in os.listdir(label_path)
                   if os.path.isdir(os.path.join(label_path, d))])

if not seqs:
    raise RuntimeError(f"label_path 下没有任何 sequence 子目录：{label_path}")

summary_rows = []
any_missing = False

for s in seqs:
    gt_dir = os.path.join(label_path, s)
    pred_dir = os.path.join(results_path, s)

    if not os.path.isdir(gt_dir):
        print(f"[跳过] label_path 下没有该序列目录: {s} -> {gt_dir}")
        continue

    # 1) 先统计 GT 图像数量（不要求数字命名）
    gt_frames = count_images_anyname(gt_dir)

    # 2) 读预测
    if not os.path.isdir(pred_dir):
        print(f"[致命] 结果目录缺失: {s} -> {pred_dir}")
        # 期望帧数如果 gt_frames>0 用它，否则未知就写 -1
        summary_rows.append([s, gt_frames if gt_frames > 0 else -1, 0,
                             (gt_frames if gt_frames > 0 else -1), "MISSING_RESULT_DIR"])
        any_missing = True
        continue

    pred_idx = list_pred_indices(pred_dir)
    pred_set = set(pred_idx)

    # 3) 决定期望范围
    #    - 如果 gt_frames>0: 期望 dynamic_mask_0..gt_frames-1
    #    - 否则：如果有预测，就用 max_pred 推断期望 0..max_pred
    #    - 否则：两边都没东西
    if gt_frames > 0:
        expected_n = gt_frames
        expected = set(range(expected_n))
        expected_note = f"GT(image-count)={gt_frames}"
    elif pred_idx:
        expected_n = max(pred_idx) + 1
        expected = set(range(expected_n))
        expected_note = f"GT=0，用 max_pred 推断 expected_n={expected_n}"
    else:
        print(f"[警告] {s}: GT 目录下没找到图片，预测目录也没有 dynamic_mask_*.png")
        summary_rows.append([s, 0, 0, 0, "EMPTY_BOTH"])
        continue

    missing = sorted(expected - pred_set)
    extra = sorted(pred_set - expected)

    if missing:
        any_missing = True
        print(f"[缺失] {s}: 缺 {len(missing)}/{expected_n} ({expected_note}) 示例:",
              [f"dynamic_mask_{i}.png" for i in missing[:10]])
    else:
        print(f"[OK]   {s}: 覆盖 dynamic_mask_0..dynamic_mask_{expected_n-1} ({expected_note})")

    if extra:
        print(f"[多余] {s}: 存在 {len(extra)} 个越界结果 示例:",
              [f"dynamic_mask_{i}.png" for i in extra[:10]])

    status = "OK"
    if missing and extra:
        status = "MISSING/EXTRA"
    elif missing:
        status = "MISSING"
    elif extra:
        status = "EXTRA"

    summary_rows.append([s, expected_n, len(pred_idx), len(missing), status])

csv_out = os.path.join(results_path, "missing_dynamic_masks_summary.csv")
with open(csv_out, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sequence", "expected_frames", "pred_masks_found", "missing_count", "status"])
    writer.writerows(summary_rows)

print("\n汇总写入：", csv_out)
if any_missing:
    print("提示：按 [缺失]/[致命] 输出优先重跑缺失序列或补齐 dynamic_mask_*.png。")
else:
    print("👍 没发现缺失。")

# --- 只输出有缺失的序列，按 expected_frames(总帧数) 从短到长排序，空格连接 ---
txt_out = os.path.join(results_path, "missing_sequences_sorted_by_length.txt")

# summary_rows: [sequence, expected_frames, pred_masks_found, missing_count, status]
missing_rows = [r for r in summary_rows if "MISSING" in str(r[4])]

# expected_frames 可能为 -1（未知），放到最后
missing_rows.sort(key=lambda r: (r[1] < 0, r[1]))

with open(txt_out, "w") as f:
    f.write(" ".join(r[0] for r in missing_rows))
