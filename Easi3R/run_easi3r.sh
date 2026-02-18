#!/bin/bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate easi3r

cd /home/qidili2/dynamic-object-mask/Easi3R

echo "[EASi3R] host=$(hostname) date=$(date)"
echo "[EASi3R] python=$(which python)"
echo "[EASi3R] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true

DATA_DIR="/home/qidili2/dynamic-object-mask/SegAnyMo/MoCA/JPEGImages"
OUT_ROOT="/home/qidili2/dynamic-object-mask/Easi3R/results/MOCA/easi3r_origin"
PRETRAINED="/home/qidili2/dynamic-object-mask/Easi3R/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
TEXTREGION_ROOT="/home/qidili2/dynamic-object-mask/Easi3R/third_party/TextRegion/MOCA_out"

# 你要跑的序列列表（这里随便加/删）
SEQS=(
chameleon crab_2 flounder flounder_7 flower_crab_spider_1 goat_1 lion_cub_0 orchid_mantis polar_bear_0 pygmy_seahorse_1 scorpionfish_0 spider_tailed_horned_viper_1 flatfish_1 flounder_5 goat_2 grasshopper_0 scorpionfish_2 scorpionfish_3 stick_insect_0 peacock_flounder_0 flounder_8 bear egyptian_nightjar flatfish_4 peacock_flounder_2 snow_leopard_2 lion_cub_2 seal smallfish arctic_wolf_0 black_cat_1 lion_cub_1 copperhead_snake groundhog pallas_cat peacock_flounder_1 grasshopper_1 lion_cub_3 potoo goat_0 moth eastern_screech_owl_1 octopus snowshoe_hare eastern_screech_owl_0 octopus_1 sole flounder_6 snow_leopard_10 flatfish_0 plaice nile_monitor_0 snow_leopard_5 turtle starfish nile_monitor_1 lichen_katydid snow_leopard_4
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
    --sam2_mask_refine \
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
