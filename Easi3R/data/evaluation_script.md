# Dataset Preparation for Evaluation

We provide scripts to download and prepare the datasets for evaluation. The datasets include: **DAVIS**, **DyCheck**, **ADT**, and, **TUM-dynamics**.

> [!NOTE]
> The scripts provided here are for reference only. Please ensure you have obtained the necessary licenses from the original dataset providers before proceeding.


## Download and Prepare Datasets

### DAVIS
To download and prepare the **DAVIS** dataset, execute:
```bash
cd data
python download_prepare_davis.py
cd ..
```

### DyCheck
Download the [DyCheck dataset](https://drive.google.com/drive/folders/1xJaFS_3027crk7u36cue7BseAX80abRe) processed by [Shape of Motion](https://github.com/vye16/shape-of-motion/) in data, then execute:
```bash
cd data
python prepare_iphone.py
cd ..
```


### ADT
To download the **ADT** dataset, fowllow [TAPVid-3D](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid3d) to prepare TAPVid environment, then execute:
```bash
cd data
conda activate TAPVid
python download_adt.py
cd ..
```

To prepare the **ADT** dataset, execute:
```bash
cd data
conda activate easi3r
python prepare_adt.py
cd ..
```

### TUM-dynamics
To download the **TUM-dynamics** dataset, execute:
```bash
cd data
bash download_tum.sh
cd ..
```

To prepare the **TUM-dynamics** dataset, execute:
```bash
cd data
python prepare_tum.py
cd ..
```



# Evaluation Script


### DAVIS
To evaluate the **DAVIS** dataset, execute:
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29604 launch.py \
    --mode=eval_pose \
    --pretrained="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"   \
    --eval_dataset=davis --output_dir="results/davis/easi3r_dust3r" \
    --use_atten_mask
# To change backbone, --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"
# To use SAM2, add: --sam2_mask_refine
```
If you just need dynamic mask, execute:
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29604 launch.py \
    --mode=eval_pose --n_iter 0 \
    --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"   \
    --eval_dataset=davis --output_dir="results/davis/easi3r_monst3r_sam" \
    --use_atten_mask --sam2_mask_refine
```
The results will be saved in the `results/davis/easi3r_monst3r_sam` folder. You could then run `python mask_metric.py --results_path results/davis/easi3r_monst3r_sam` to evaluate the mask results, and run `python vis_attention.py --method_name easi3r_monst3r_sam --base_output_dir results/visualization` to see the visualization of attention as in the [webpage](https://easi3r.github.io/).


### DyCheck
To evaluate the **DyCheck** dataset, execute:
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29604 launch.py \
    --mode=eval_pose  --no_crop \
    --pretrained="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"   \
    --eval_dataset=iphone --output_dir="results/iphone/easi3r_dust3r" \
    --use_atten_mask
# To change backbone, --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"
# To use SAM2, add: --sam2_mask_refine
```
The results will be saved in the `results/iphone/easi3r_dust3r` folder. You could then run `CUDA_VISIBLE_DEVICES=4 python point_metric.py --result_path results/iphone` to evaluate the reconstruction results.

### ADT
To evaluate the **ADT** dataset, execute:
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29604 launch.py \
    --mode=eval_pose \
    --pretrained="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"   \
    --eval_dataset=adt --output_dir="results/adt/easi3r_dust3r" \
    --use_atten_mask
# To change backbone, --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"
# To use SAM2, add: --sam2_mask_refine
```
The results will be saved in the `results/adt/easi3r_dust3r` folder.

### TUM-dynamics
To evaluate the **TUM-dynamics** dataset, execute:
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29604 launch.py \
    --mode=eval_pose \
    --pretrained="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"   \
    --eval_dataset=tum --output_dir="results/tum/easi3r_dust3r" \
    --use_atten_mask
# To change backbone, --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"
# To use SAM2, add: --sam2_mask_refine
```
The results will be saved in the `results/tum/easi3r_dust3r` folder.