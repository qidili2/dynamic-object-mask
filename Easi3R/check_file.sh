GT=/mnt/data0/andy/Easi3R/DAVIS/JPEGImages/480p
RES=/mnt/data0/andy/Easi3R/results/davis/easi3r_sam1_region

# 列出“没有任何 dynamic_mask_*.png”的序列（待补跑）
for s in $(ls "$GT"); do
  [[ -d "$GT/$s" ]] || continue
  if ls "$RES/$s"/dynamic_mask_*.png >/dev/null 2>&1; then
    :  # 已有输出，跳过
  else
    echo "$s"
  fi
done
