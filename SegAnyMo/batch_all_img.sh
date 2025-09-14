#!/bin/bash

VIDEO_DIR="/mnt/data0/andy/Easi3R/DAVIS/davis_videos/images"
BASE_DATA_DIR="$VIDEO_DIR"
MOTIN_OUT_DIR="$BASE_DATA_DIR/result/moseg"
SAM2_OUT_DIR="$BASE_DATA_DIR/result/sam2"
CONFIG_FILE="./configs/example_train.yaml"
GPUS="0"

mkdir -p "$MOTIN_OUT_DIR"
mkdir -p "$SAM2_OUT_DIR"

for video_dir in "$VIDEO_DIR"/*/; do
  [ -d "$video_dir" ] || continue
  BASENAME=$(basename "$video_dir")
  NAME="${BASENAME%/}"

  # echo "▶️ Step 1: Preprocessing $NAME"
  # python core/utils/run_inference.py \
  #   --data_dir "$video_dir" \
  #   --gpus $GPUS \
  #   --depths --tracks --dinos --e

  echo "🧠 Step 2: Motion segmentation for $NAME"
  python core/utils/run_inference.py \
    --data_dir "$video_dir" \
    --motin_seg_dir "$MOTIN_OUT_DIR" \
    --config_file "$CONFIG_FILE" \
    --gpus $GPUS \
    --motion_seg_infer --e

  echo "🎨 Step 3: SAM2 final masks for $NAME"
  python core/utils/run_inference.py \
    --data_dir "$video_dir" \
    --sam2dir "$SAM2_OUT_DIR" \
    --motin_seg_dir "$MOTIN_OUT_DIR" \
    --gpus $GPUS \
    --sam2 --e

  echo "✅ All steps done for $NAME"
  echo "-------------------------------------------"
done
