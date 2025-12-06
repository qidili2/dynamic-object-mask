#!/usr/bin/env python
import os
import sys
from time import time
import argparse
import glob   # 新增

import numpy as np
import pandas as pd
from davis2017.evaluation import DAVISEvaluation

default_davis_path = '/mnt/data0/andy/SegAnyMo/DAVIS'

time_start = time()
parser = argparse.ArgumentParser()
parser.add_argument('--davis_path', type=str,
    help='Path to the DAVIS folder containing the JPEGImages, Annotations, '
         'ImageSets, Annotations_unsupervised folders',
    required=False, default=default_davis_path)
parser.add_argument('--set', type=str, help='Subset to evaluate the results', default='val')
parser.add_argument('--task', type=str, help='Task to evaluate the results', default='unsupervised',
                    choices=['semi-supervised', 'unsupervised'])
parser.add_argument('--results_path', type=str, help='Path to the folder containing the sequences folders',
                    required=True)

# === 新增：自定义 seqlist + 跳过没有预测的 seq ===
parser.add_argument('--seq_list', type=str, default=None,
                    help='Optional txt file (one seq per line). If given, only these seqs will be evaluated.')
parser.add_argument('--skip_missing', action='store_true',
                    help='If set, skip sequences that do not have any prediction mask in results_path.')

args, _ = parser.parse_known_args()
csv_name_global = f'global_results-{args.set}.csv'
csv_name_per_sequence = f'per-sequence_results-{args.set}.csv'
csv_name_global_path = os.path.join(args.results_path, csv_name_global)
csv_name_per_sequence_path = os.path.join(args.results_path, csv_name_per_sequence)

# ====== 根据 seq_list 和 results_path 决定要 eval 的 sequences ======

def load_seq_list_from_txt(path):
    with open(path, 'r') as f:
        return [l.strip() for l in f if l.strip()]

def list_pred_seqs(results_root):
    """列出 results_root 下面所有有 mask 文件的 seq 目录"""
    seqs = []
    for name in sorted(os.listdir(results_root)):
        full = os.path.join(results_root, name)
        if not os.path.isdir(full):
            continue
        # 看这个目录里有没有任何图像文件
        mask_files = []
        for pat in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            mask_files.extend(glob.glob(os.path.join(full, pat)))
        if mask_files:
            seqs.append(name)
        else:
            print(f"[WARN] seq '{name}' has no prediction masks, skip.")
    return seqs

# 1) 从 seq_list 或 results_path 取初始列表
if args.seq_list is not None:
    seqs_from_txt = load_seq_list_from_txt(args.seq_list)
    # 如果 skip_missing 开了，就和 results_path 交集；否则就直接用 txt
    if args.skip_missing:
        seqs_with_pred = list_pred_seqs(args.results_path)
        seqs = [s for s in seqs_from_txt if s in seqs_with_pred]
        missing = [s for s in seqs_from_txt if s not in seqs_with_pred]
        for s in missing:
            print(f"[WARN] seq '{s}' in seq_list but no predictions found, skip.")
    else:
        seqs = seqs_from_txt
else:
    # 没有给 seq_list，就用 results_path 下面所有有预测的 seq
    seqs = list_pred_seqs(args.results_path) if args.skip_missing else 'all'

# 注意：如果 seqs 是空 list，就没必要往下跑了
if isinstance(seqs, list) and len(seqs) == 0:
    print("[ERROR] No sequences to evaluate (after filtering). Check seq_list / results_path.")
    sys.exit(1)

# ==========================================================

# Check if the method has been evaluated before, if so read the results, otherwise compute the results
if os.path.exists(csv_name_global_path) and os.path.exists(csv_name_per_sequence_path):
    print('Using precomputed results...')
    table_g = pd.read_csv(csv_name_global_path)
    table_seq = pd.read_csv(csv_name_per_sequence_path)
else:
    print(f'Evaluating sequences for the {args.task} task...')
    print(f"[INFO] Sequences to evaluate: {seqs}")
    # Create dataset and evaluate
    dataset_eval = DAVISEvaluation(
        davis_root=args.davis_path,
        task=args.task,
        gt_set=args.set,
        sequences=seqs    # ★ 这里传入我们过滤好的列表或 'all'
    )
    # ★ 新增：看看 DAVIS 内部最终留下了哪些序列
    effective_seqs = list(dataset_eval.dataset.get_sequences())
    print("[DEBUG] Effective sequences inside DAVIS:", effective_seqs)

    if len(effective_seqs) == 0:
        print("[ERROR] None of the sequences in seq_list appear in DAVIS for set:", args.set)
        sys.exit(1)
    metrics_res = dataset_eval.evaluate(args.results_path)
    J, F = metrics_res['J'], metrics_res['F']

    # Generate dataframe for the general results
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]),
                      np.mean(F["M"]), np.mean(F["R"]), np.mean(F["D"])])
    g_res = np.reshape(g_res, [1, len(g_res)])
    table_g = pd.DataFrame(data=g_res, columns=g_measures)
    with open(csv_name_global_path, 'w') as f:
        table_g.to_csv(f, index=False, float_format="%.3f")
    print(f'Global results saved in {csv_name_global_path}')

    # Generate a dataframe for the per sequence results
    seq_names = list(J['M_per_object'].keys())
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
    with open(csv_name_per_sequence_path, 'w') as f:
        table_seq.to_csv(f, index=False, float_format="%.3f")
    print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

# Print the results (后面原样保留)
sys.stdout.write(f"--------------------------- Global results for {args.set} ---------------------------\n")
print(table_g.to_string(index=False))
sys.stdout.write(f"\n---------- Per sequence results for {args.set} ----------\n")
print(table_seq.to_string(index=False))
total_time = time() - time_start
sys.stdout.write('\nTotal time:' + str(total_time))
