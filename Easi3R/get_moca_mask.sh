#!/bin/bash
set -euo pipefail

# ---- conda (跟你能跑的脚本一致) ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate easi3r

# ---- project dir ----
cd /home/qidili2/dynamic-object-mask/Easi3R

echo "[SAM2-MoCA] host=$(hostname) date=$(date)"
echo "[SAM2-MoCA] python=$(which python)"
echo "[SAM2-MoCA] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true

# =========================
# Paths (按你之前给的 MoCA 路径改成 /home/qidili2 版)
# =========================
ANN_CSV="/home/qidili2/dynamic-object-mask/SegAnyMo/MoCA/Annotations/annotations.csv"
IMAGE_ROOT="/home/qidili2/dynamic-object-mask/SegAnyMo/MoCA/JPEGImages"
OUT_MASK_ROOT="/home/qidili2/dynamic-object-mask/SegAnyMo/MoCA/Annotations"   # 直接存到 Annotations/seq/00000.png
SAM2_CKPT="/home/qidili2/dynamic-object-mask/Easi3R/third_party/sam2/checkpoints/sam2.1_hiera_large.pt"

PY_SCRIPT="gen_sam2_masks_from_bboxes_checkpoint_only.py"

# =========================
# Runtime options
# =========================
DEVICE="cuda"
FP16="--fp16"
OVERWRITE="--overwrite"
VERBOSE="--verbose"
LIMIT=""   # 例如：LIMIT="--limit 200" 只跑前200帧标注

# 推荐的 CUDA allocator 设置（你 submit 里也用过）
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# 日志目录
LOG_DIR="/home/qidili2/dynamic-object-mask/Easi3R/logs/moca_sam2_masks"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/gen_moca_sam2_masks_$(date +%Y%m%d_%H%M%S).log"

echo "[SAM2-MoCA] ann_csv=${ANN_CSV}"
echo "[SAM2-MoCA] image_root=${IMAGE_ROOT}"
echo "[SAM2-MoCA] out_mask_root=${OUT_MASK_ROOT}"
echo "[SAM2-MoCA] sam2_ckpt=${SAM2_CKPT}"
echo "[SAM2-MoCA] log=${LOG_FILE}"

# =========================
# Run
# =========================
python "${PY_SCRIPT}" \
  --ann_csv "${ANN_CSV}" \
  --image_root "${IMAGE_ROOT}" \
  --sam2_ckpt "${SAM2_CKPT}" \
  --out_mask_root "${OUT_MASK_ROOT}" \
  --device "${DEVICE}" \
  ${FP16} \
  ${OVERWRITE} \
  ${VERBOSE} \
  ${LIMIT} \
  2>&1 | tee "${LOG_FILE}"

echo "[SAM2-MoCA] Done."
