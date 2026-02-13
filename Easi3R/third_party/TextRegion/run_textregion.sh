#!/bin/bash

source ~/.bashrc

conda activate TextRegion  

cd /home/qidili2/dynamic-object-mask/Easi3R/third_party/TextRegion

python batch_foreground_masks.py \
  --root /home/qidili2/dynamic-object-mask/Easi3R/SegTrackv2/JPEGImages \
  --yaml_out utils/SegTrackv2_final_labels.yaml \
  --out_dir /home/qidili2/dynamic-object-mask/Easi3R/third_party/TextRegion/Segtrackv2_out \
  --resize_method multi_resolution \
  --crop_size 336 \
  --points_per_side 16 \
  --dtype bf16 \
  --sam2_checkpoint /home/qidili2/dynamic-object-mask/Easi3R/third_party/sam2/checkpoints/sam2.1_hiera_large.pt 
