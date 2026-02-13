#!/bin/bash
set -e

# 1. 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate seg

# 2. 路径设置
cd /home/qidili2/dynamic-object-mask/SegAnyMo
DATA_DIR="/home/qidili2/dynamic-object-mask/SegAnyMo/MoCA_short/JPEGImages"
RESULT_ROOT="/home/qidili2/dynamic-object-mask/SegAnyMo/result_moca_short"

mkdir -p "$RESULT_ROOT/moseg" "$RESULT_ROOT/sam2"
export PYTHONUNBUFFERED=1

# # 3. Step 1: depths / tracks / dinos
# python -u core/utils/run_inference.py \
#   --data_dir "$DATA_DIR" \
#   --depths \
#   --tracks \
#   --dinos \
#   --e

# 4. Step 2: motion segmentation
python -u core/utils/run_inference.py \
  --data_dir "$DATA_DIR" \
  --motin_seg_dir "$RESULT_ROOT/moseg" \
  --config_file "/home/qidili2/dynamic-object-mask/SegAnyMo/configs/example_train.yaml" \
  --motion_seg_infer \
  --e

# 5. Step 3: SAM2
python -u core/utils/run_inference.py \
  --data_dir "$DATA_DIR" \
  --sam2dir "$RESULT_ROOT/sam2" \
  --motin_seg_dir "$RESULT_ROOT/moseg" \
  --sam2 \
  --e
