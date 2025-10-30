#!/usr/bin/env python3
import os, re, argparse, csv
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--label_path', required=True, help='DAVIS Annotations/480p 路径')
parser.add_argument('--results_path', required=True, help='你的结果根目录，内含若干 <sequence>/dynamic_mask_*.png')
parser.add_argument('--seq', default='', help='仅检查某一个序列名（可选）')
args = parser.parse_args()

label_path = os.path.abspath(args.label_path)
results_path = os.path.abspath(args.results_path)

# DAVIS 的帧名通常是 00000.png 这种格式；我们把它转换成整数 index
gt_num_re = re.compile(r'^(\d+)\.png$')
pred_re   = re.compile(r'^dynamic_mask_(\d+)\.png$')

def list_gt_indices(seq_dir):
    idx = []
    for f in os.listdir(seq_dir):
        m = gt_num_re.match(f)
        if m:
            idx.append(int(m.group(1)))
    return sorted(idx)

def list_pred_indices(seq_dir):
    idx = []
    for f in os.listdir(seq_dir):
        m = pred_re.match(f)
        if m:
            idx.append(int(m.group(1)))
    return sorted(idx)

# 序列集合：以 GT 为准
if args.seq:
    seqs = [args.seq]
else:
    seqs = sorted([d for d in os.listdir(label_path)
                   if os.path.isdir(os.path.join(label_path, d))])

summary_rows = []
any_missing = False

for s in seqs:
    gt_dir   = os.path.join(label_path, s)
    pred_dir = os.path.join(results_path, s)

    if not os.path.isdir(gt_dir):
        print(f"[跳过] GT 不存在: {s}")
        continue

    gt_idx = list_gt_indices(gt_dir)
    if not gt_idx:
        print(f"[警告] GT 空目录或没有数字命名PNG: {s} -> {gt_dir}")
        continue

    if not os.path.isdir(pred_dir):
        print(f"[致命] 结果目录缺失: {s} -> {pred_dir}")
        summary_rows.append([s, len(gt_idx), 0, len(gt_idx), 'MISSING_RESULT_DIR'])
        any_missing = True
        continue

    pred_idx = list_pred_indices(pred_dir)

    # 缺失 = GT 有但预测没有
    missing = [i for i in gt_idx if i not in set(pred_idx)]

    # 额外 = 预测有但 GT 没有（一般是越界或命名问题）
    extra = [i for i in pred_idx if i not in set(gt_idx)]

    # 打印
    if missing:
        any_missing = True
        miss_names = [f"dynamic_mask_{i}.png" for i in missing[:10]]  # 仅预览前10个
        print(f"[缺失] {s}: 缺 {len(missing)}/{len(gt_idx)} 帧  示例: {miss_names}")
    else:
        print(f"[OK]   {s}: 全部 {len(gt_idx)} 帧都有对应 dynamic_mask_*.png")

    if extra:
        print(f"[多余] {s}: 存在 {len(extra)} 个结果无对应GT（示例前10个）：",
              [f"dynamic_mask_{i}.png" for i in extra[:10]])

    summary_rows.append([s, len(gt_idx), len(pred_idx), len(missing),
                         "OK" if not missing else "MISSING" + ("/EXTRA" if extra else "")])

# 写一个汇总CSV到结果根目录
csv_path = os.path.join(results_path, "missing_dynamic_masks_summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sequence", "gt_frames", "pred_masks", "missing_count", "status"])
    writer.writerows(summary_rows)

print("\n汇总写入：", csv_path)
if any_missing:
    print("提示：按上面的 [缺失]/[致命] 输出，优先重跑缺失的序列或补齐命名。")
else:
    print("👍 没发现缺失的 dynamic_mask_*.png（与 GT 帧数一致）。")
