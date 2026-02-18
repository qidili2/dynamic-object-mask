#!/bin/bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate easi3r

cd /home/qidili2/dynamic-object-mask/Easi3R

echo "[EASi3R] host=$(hostname) date=$(date)"
echo "[EASi3R] python=$(which python)"
echo "[EASi3R] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true

# DATA_DIR="/home/qidili2/dynamic-object-mask/SegAnyMo/MoCA/JPEGImages"
# OUT_ROOT="/home/qidili2/dynamic-object-mask/Easi3R/results/MOCA/easi3r_textregion_refine"
# PRETRAINED="/home/qidili2/dynamic-object-mask/Easi3R/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
# TEXTREGION_ROOT="/home/qidili2/dynamic-object-mask/Easi3R/third_party/TextRegion/MOCA_out"
# DATA_DIR="/home/qidili2/dynamic-object-mask/Easi3R/FBMS_59"
# OUT_ROOT="/home/qidili2/dynamic-object-mask/Easi3R/results/FBMS/easi3r_textregion_refine"
# PRETRAINED="/home/qidili2/dynamic-object-mask/Easi3R/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
# TEXTREGION_ROOT="/home/qidili2/dynamic-object-mask/Easi3R/third_party/TextRegion/FBMS_out"
DATA_DIR="/home/qidili2/dynamic-object-mask/Easi3R/SegTrackv2/JPEGImages"
OUT_ROOT="/home/qidili2/dynamic-object-mask/Easi3R/results/SegTrackv2/easi3r_textregion_refine"
PRETRAINED="/home/qidili2/dynamic-object-mask/Easi3R/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
TEXTREGION_ROOT="/home/qidili2/dynamic-object-mask/Easi3R/third_party/TextRegion/Segtrackv2_out"
# 你要跑的序列列表（这里随便加/删）
SEQS=(
birdfall  bird_of_paradise  bmx  cheetah  drift  frog  girl  hummingbird  monkey  monkeydog  parachute  penguin  soldier  worm
)

# 可选：把每个seq的log分开存
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

FAILED=()

for seq in "${SEQS[@]}"; do
  echo "============================================================"
  echo "[EASi3R] Running seq=${seq}"
  echo "[EASi3R] out=${OUT_ROOT}/${seq}"
  echo "============================================================"

  mkdir -p "${OUT_ROOT}/${seq}"

  # 每个seq单独跑一次（--seq_list 只塞一个）
  python launch.py \
    --mode=eval_pose \
    --n_iter 0 \
    --pretrained="${PRETRAINED}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUT_ROOT}" \
    --use_atten_mask \
    --use_region_pooling \
    --textregion_annotations_dir "${TEXTREGION_ROOT}" \
    --seq_list "${seq}" \
    2>&1 | tee "${LOG_DIR}/${seq}.log" \
    || { echo "[EASi3R][WARN] seq failed: ${seq}"; FAILED+=("${seq}"); continue; }

  echo "[EASi3R] Done seq=${seq}"
done

if (( ${#FAILED[@]} > 0 )); then
  echo "[EASi3R] Some sequences failed:"
  printf '  - %s\n' "${FAILED[@]}"
  exit 1
else
  echo "[EASi3R] All sequences finished successfully."
fi
