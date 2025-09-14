#!/usr/bin/env python3
"""
Standalone script to generate the dynamic attention mask (Aa=dyn) for the first frame of a specific video,
without interactive CLI. Video and model paths are hardcoded.
"""
import os
import time
import cv2
import numpy as np
import torch
import glob
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.inference import inference
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# --- Configuration (hardcoded) ---
video_path = "/mnt/data0/andy/data/videos/great_wall/great_wall.mp4"
model_path = "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
output_dir = os.path.join(os.path.dirname(video_path), "dyn_mask_out")
device_str = "cuda"
image_size = 512    # choose from [224, 512]
fps = 5            # 0: use original video FPS
num_frames = None      # number of frames to load for pairing

# Make output dir
os.makedirs(output_dir, exist_ok=True)

# 1. Read original frame size
cap = cv2.VideoCapture(video_path)
ret, frame0 = cap.read()
if not ret:
    raise RuntimeError("Cannot read first frame from video")
orig_h, orig_w = frame0.shape[:2]
cap.release()

# 2. Load model
device = torch.device(device_str if torch.cuda.is_available() else "cpu")
model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
model.eval()

# 3. Load and resize frames
start_time = time.time()
imgs, width, height, video_fps = load_images(
    [video_path], size=image_size, fps=fps, num_frames=num_frames, return_img_size=True
)
print(f"Loaded {len(imgs)} frames resized to {width}x{height}, FPS={video_fps}")

pairs = make_pairs(imgs, scene_graph="swin2stride", prefilter=None, symmetrize=True)
print(f"Built {len(pairs)} pairs")

# 5. Inference + align
out = inference(pairs, model, device=device, batch_size=1, verbose=False)
if isinstance(out, dict) and set(out.keys()) >= {'view1','view2','pred1','pred2'}:
    dust3r_output = out
elif isinstance(out, (list, tuple)) and len(out) == 4:
    dust3r_output = dict(zip(['view1','view2','pred1','pred2'], out))
else:
    raise RuntimeError(f"Inference returned unexpected type {type(out)} with length {len(out) if hasattr(out,'__len__') else 'N/A'}")
scene = global_aligner(
    dust3r_output, device=device,
    mode=GlobalAlignerMode.PointCloudOptimizer,
    verbose=False, shared_focal=True,
    temporal_smoothing_weight=0.0, translation_weight=1.0,
    flow_loss_weight=0.0, flow_loss_start_epoch=0.0,
    flow_loss_thre=25, use_self_mask=True,
    num_total_iter=0, empty_cache=False,
    batchify=True, use_atten_mask=True,
    sam2_mask_refine=False
)

imposes = scene.get_im_poses()
T_list = imposes.detach().cpu().numpy()
K_all = scene.get_intrinsics()
K = K_all[0].detach().cpu().numpy()

P_all = scene.get_im_poses().detach().cpu().numpy()     # (n_imgs,4,4)
K_all = scene.get_intrinsics().detach().cpu().numpy()    # (n_imgs,3,3)
P0    = P_all[0] 

# 9 Save dynamic attention masks
for idx, m in enumerate(scene.dynamic_masks):
    arr = m.cpu().numpy()
    cv2.imwrite(os.path.join(output_dir, f"group_token_dyn_mask{idx}.png"), (arr*255).astype(np.uint8))
print(f"Saved {len(scene.dynamic_masks)} dynamic masks")
# 读取所有 Aa_dyn_* 帧并排序
frame_paths = sorted(glob.glob(os.path.join(output_dir, "group_token_dyn_mask*.png")))
if len(frame_paths) == 0:
    raise RuntimeError("No dynamic mask frames found to build video.")

# 读取第一帧确定尺寸
frame0 = cv2.imread(frame_paths[0])
height, width = frame0.shape[:2]

# 定义输出视频路径
video_path_out = os.path.join(output_dir, "group_token_dyn_mask.mp4")

# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(video_path_out, fourcc, fps or video_fps, (width, height))

# 写入所有帧
for frame_file in frame_paths:
    frame = cv2.imread(frame_file)
    if frame is None:
        continue
    writer.write(frame)
writer.release()
print(f"Dynamic mask video saved to: {video_path_out}")

# Timing
print(f"Total time: {time.time()-start_time:.2f}s")