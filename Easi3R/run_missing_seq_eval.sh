#!/bin/bash
set -e

# ====== 配置路径 ======
DAVIS_DIR="/mnt/data0/andy/Easi3R/DAVIS/JPEGImages/480p"
TEXTREGION_DIR="/mnt/data0/andy/Easi3R/third_party/TextRegion/davis_finalmovinglabel_out3"
CHECKPOINT="/mnt/data0/andy/Easi3R/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
OUTPUT_DIR="results/davisunsupervised/easi3r_textregion_finalmovinglabel_refine3_test"

# ====== 日志文件 ======
LOG_FILE="run_missing_seq_eval.log"
> "$LOG_FILE"

echo "[INFO] 开始检查缺失序列..." | tee -a "$LOG_FILE"

# ====== 找出缺失序列 ======
missing_seqs=()
for seq_path in "$DAVIS_DIR"/*; do
    seq_name=$(basename "$seq_path")
    mp4_path="$OUTPUT_DIR/$seq_name/0_dynamic_masks.mp4"

    # 若 mp4 不存在则加入缺失序列列表
    if [ ! -f "$mp4_path" ]; then
        missing_seqs+=("$seq_name")
    fi
done

if [ ${#missing_seqs[@]} -eq 0 ]; then
    echo "[INFO] ✅ 没有缺失序列（所有 0_dynamic_masks.mp4 均存在）。" | tee -a "$LOG_FILE"
    exit 0
fi

echo "[INFO] 共发现 ${#missing_seqs[@]} 个缺失序列：" | tee -a "$LOG_FILE"
printf '  - %s\n' "${missing_seqs[@]}" | tee -a "$LOG_FILE"

# ====== 循环运行每个缺失序列 ======
for seq_name in "${missing_seqs[@]}"; do
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "[INFO] 运行序列: $seq_name" | tee -a "$LOG_FILE"
    echo "[CMD] python launch.py --seq_list $seq_name" | tee -a "$LOG_FILE"

    python launch.py \
        --mode=eval_pose \
        --n_iter 0 \
        --pretrained="$CHECKPOINT" \
        --eval_dataset=davis \
        --output_dir "$OUTPUT_DIR" \
        --use_atten_mask \
        --use_region_pooling \
        --textregion_annotations_dir "$TEXTREGION_DIR" \
        --seq_list "$seq_name" \
        >> "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "[INFO] ✅ 序列 $seq_name 完成。" | tee -a "$LOG_FILE"
    else
        echo "[WARN] ⚠️ 序列 $seq_name 运行失败，请检查日志。" | tee -a "$LOG_FILE"
    fi
done

echo "----------------------------------------" | tee -a "$LOG_FILE"
echo "[INFO] ✅ 所有缺失序列处理完毕！日志见 $LOG_FILE"
