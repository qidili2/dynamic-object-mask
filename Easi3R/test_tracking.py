#!/usr/bin/env python3
"""
Standalone script: 生成第0帧动态注意力掩码，并追踪两个硬编码框（动态/静态）
在后续所有帧中的像素映射。使用 oneref-0 配对，启用 attention map。
"""
import os
import time
import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.inference import inference
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# --- Configuration ---
video_path     = "./demo_data/dog-gooses.mp4"
model_path     = "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
output_dir     = os.path.join(os.path.dirname(video_path), "dyn_mask_out")
dyn_track_dir  = os.path.join(os.path.dirname(video_path), "dyn_track")
stat_track_dir = os.path.join(os.path.dirname(video_path), "stat_track")
vis_dir        = os.path.join(os.path.dirname(video_path), "vis_tracks")

# 原图上手动标注的 bbox：（x0, y0, x1, y1）
dynamic_bbox = (280, 300, 400, 350)
static_bbox  = (400, 100, 650, 200)

device_str = "cuda"
image_size = 512     # 处理时的参考尺寸
fps        = 0
num_frames = None    # None 表示全文取帧

# --- Prepare directories ---
for d in (output_dir, dyn_track_dir, stat_track_dir, vis_dir):
    os.makedirs(d, exist_ok=True)

# --- 1. 读原始分辨率 ---
cap = cv2.VideoCapture(video_path)
ret, frame0 = cap.read()
if not ret:
    raise RuntimeError("Cannot read first frame from video")
orig_h, orig_w = frame0.shape[:2]
cap.release()

# --- 2. 加载模型 ---
device = torch.device(device_str if torch.cuda.is_available() else "cpu")
model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
model.eval()

start_time = time.time()

# --- 3. 加载并缩放视频帧 ---
imgs, W, H, video_fps = load_images(
    [video_path],
    size=image_size,
    fps=fps,
    num_frames=num_frames,
    return_img_size=True
)
print(f"Loaded {len(imgs)} frames resized to {W}×{H}, FPS={video_fps}")

# --- 4. 构建 oneref-0 配对 ---
pairs = make_pairs(
    imgs,
    scene_graph="oneref-0",
    prefilter=None,
    symmetrize=False
)
print(f"Built {len(pairs)} oneref-0 pairs (frame0 → frame i)")

# --- 5. Inference （拉 attention）---
out = inference(
    pairs,
    model,
    device=device,
    batch_size=1,
    verbose=False,
)
# 兼容 dict 或 list 返回
if isinstance(out, dict) and {'view1','view2','pred1','pred2'}.issubset(out):
    dust3r_output = out
elif isinstance(out, (list,tuple)) and len(out) == 4:
    dust3r_output = dict(zip(['view1','view2','pred1','pred2'], out))
else:
    raise RuntimeError(f"Inference returned unexpected format: {type(out)}")

# --- 6. 全局对齐 ---
scene = global_aligner(
    dust3r_output,
    device=device,
    mode=GlobalAlignerMode.PointCloudOptimizer,
    verbose=False,
    shared_focal=True,
    temporal_smoothing_weight=0.0,
    translation_weight=1.0,
    flow_loss_weight=0.0,
    flow_loss_start_epoch=0.0,
    flow_loss_thre=25,
    use_self_mask=True,
    num_total_iter=0,
    empty_cache=False,
    batchify=True,
    use_atten_mask=False,
    sam2_mask_refine=False
)
P_all = scene.get_im_poses().detach().cpu().numpy()

# --- 1. 提取 A→B 点云：（N_pairs, H_proc, W_proc, 3）---
pts3d_b_all = dust3r_output['pred2']['pts3d_in_other_view'] \
                   .detach().cpu().numpy()
# 如果它的 shape=(H,W,N_pairs,3)，挪轴到 (N_pairs,H,W,3)
if pts3d_b_all.ndim==4 and pts3d_b_all.shape[2]==len(pairs):
    pts3d_b_all = np.moveaxis(pts3d_b_all, 2, 0)
elif not (pts3d_b_all.ndim==4 and pts3d_b_all.shape[0]==len(pairs)):
    raise RuntimeError(f"Unexpected pts3d_in_other_view shape {pts3d_b_all.shape}")

# --- 2. 处理帧分辨率 & 缩放比例 ---
H_proc, W_proc = pts3d_b_all.shape[1:3]
sx = W_proc / orig_w
sy = H_proc / orig_h

# --- 3. 生成第0帧两个 bbox 区域的网格索引 ---
def make_grid(bbox):
    x0, y0, x1, y1 = bbox
    x0r = int(x0 * sx); x1r = int(x1 * sx)
    y0r = int(y0 * sy); y1r = int(y1 * sy)
    # clip 到处理帧大小
    x0r, x1r = np.clip([x0r, x1r], 0, W_proc)
    y0r, y1r = np.clip([y0r, y1r], 0, H_proc)
    xs = np.arange(x0r, x1r)
    ys = np.arange(y0r, y1r)
    return np.meshgrid(xs, ys)

xx_dyn, yy_dyn   = make_grid(dynamic_bbox)
xx_stat, yy_stat = make_grid(static_bbox)

# --- 4. 构建 frame0 → frame_i 对应索引映射 ---
map_0_to_i = {
    dst['idx']: idx_pair
    for idx_pair, (src, dst) in enumerate(pairs)
    if src['idx'] == 0
}

# --- 5. 对每一帧做映射并保存 ---
for i, pair_idx in map_0_to_i.items():
    atob = pts3d_b_all[pair_idx]  # shape (H_proc, W_proc, 3)
    # 第 i 帧内参
    Ki = scene.get_intrinsics().detach().cpu().numpy()[i]

    # dynamic 区域：取出 3D 点
    pts_dyn = atob[yy_dyn, xx_dyn, :].reshape(-1, 3).T  # (3, N_dyn)
    # 投影到像素
    proj_d = Ki @ pts_dyn                                # (3, N_dyn)
    xs_d   = (proj_d[0] / proj_d[2]).astype(int)
    ys_d   = (proj_d[1] / proj_d[2]).astype(int)
    tracked_dyn = np.vstack([xs_d, ys_d]).T              # (N_dyn,2)
    np.save(os.path.join(dyn_track_dir,  f"tracked_dyn_frame{i}.npy"), tracked_dyn)

    # static 区域同理
    pts_stat = atob[yy_stat, xx_stat, :].reshape(-1, 3).T
    proj_s   = Ki @ pts_stat
    xs_s     = (proj_s[0] / proj_s[2]).astype(int)
    ys_s     = (proj_s[1] / proj_s[2]).astype(int)
    tracked_stat = np.vstack([xs_s, ys_s]).T
    np.save(os.path.join(stat_track_dir, f"tracked_stat_frame{i}.npy"), tracked_stat)

print("Finished atob-based 3D cloud mapping for dynamic/static boxes.")

# --- 11. （可选）可视化 tracked 点 ---
cap = cv2.VideoCapture(video_path)
frame_idx = 1
while True:
    ret, frame = cap.read()
    if not ret or frame_idx >= len(P_all):
        break
    vis = cv2.resize(frame, (W*sx, H*sy))
    # 动态点
    pts_d = np.load(os.path.join(dyn_track_dir,  f"tracked_dyn_frame{frame_idx}.npy"))
    pts_s = np.load(os.path.join(stat_track_dir, f"tracked_stat_frame{frame_idx}.npy"))
    for x,y in pts_d:
        cv2.circle(vis, (int(x),int(y)), 1, (0,0,255), -1)
    for x,y in pts_s:
        cv2.circle(vis, (int(x),int(y)), 1, (255,0,0), -1)
    cv2.imwrite(os.path.join(vis_dir, f"vis_frame_{frame_idx:04d}.png"), vis)
    frame_idx += 1
cap.release()

print(f"Total time: {time.time()-start_time:.2f}s")
