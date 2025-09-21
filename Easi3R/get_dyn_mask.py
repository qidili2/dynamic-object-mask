# #!/usr/bin/env python3 
# """
# Generate dynamic attention masks (Aa=dyn) from either:
# - a video file (e.g., .mp4/.avi/.mov), or
# - a folder of frames (e.g., .png/.jpg)

# It auto-detects input type. Then runs DUSt3R to compute dynamic masks and
# writes them as images and a combined MP4 video.
# """

# import os
# import time
# import cv2
# import glob
# import numpy as np
# import torch
# from typing import Tuple

# from dust3r.model import AsymmetricCroCo3DStereo
# from dust3r.image_pairs import make_pairs
# from dust3r.utils.image import load_images
# from dust3r.inference import inference
# from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# # -----------------------------
# # --- Configuration (edit)  ---
# # -----------------------------
# input_path = "/mnt/data0/andy/Easi3R/DAVIS/davis_videos/bear.mp4"  # can be a video file or a directory of frames
# model_path = "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
# device_str = "cuda"
# image_size = 512        # {224, 512}
# fps = 5                 # only used for VIDEO input; 0 means use original video FPS
# num_frames = None       # optional limit of frames to load (both video & frames folder)
# mask_video_name = "Aa_dyn_group_video.mp4"

# # -----------------------------
# # --- Helpers               ---
# # -----------------------------
# VIDEO_EXTS = (".mp4", ".avi", ".mov", ".MP4", ".AVI", ".MOV")
# IMG_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")

# def is_video_file(path: str) -> bool:
#     return os.path.isfile(path) and path.lower().endswith(VIDEO_EXTS)

# def is_frames_dir(path: str) -> bool:
#     if not os.path.isdir(path):
#         return False
#     # Heuristics: directory that contains images
#     for fn in os.listdir(path):
#         if fn.lower().endswith(IMG_EXTS):
#             return True
#     return False

# def read_first_frame_size(path: str) -> Tuple[int, int]:
#     """Return (H, W) of first frame whether input is video or frames folder."""
#     if is_video_file(path):
#         cap = cv2.VideoCapture(path)
#         ret, frame0 = cap.read()
#         cap.release()
#         if not ret or frame0 is None:
#             raise RuntimeError(f"Cannot read first frame from video: {path}")
#         return frame0.shape[:2]
#     elif is_frames_dir(path):
#         # pick the first image by name order
#         frames = sorted([f for f in os.listdir(path) if f.lower().endswith(IMG_EXTS)])
#         if not frames:
#             raise RuntimeError(f"No image frames found in folder: {path}")
#         img_path = os.path.join(path, frames[0])
#         img = cv2.imread(img_path)
#         if img is None:
#             raise RuntimeError(f"Cannot read image: {img_path}")
#         return img.shape[:2]
#     else:
#         raise RuntimeError(f"Input path is neither a video nor a frames folder: {path}")

# def ensure_outdir(base_input: str) -> str:
#     """
#     Decide output directory:
#     - If video file: output dir is sibling to the video file
#     - If frames folder: output dir is inside that folder
#     """
#     if is_video_file(base_input):
#         # 视频文件 -> dyn_mask_out 放在视频文件同级
#         base_dir = os.path.dirname(base_input)
#         out_dir = os.path.join(base_dir, "dyn_mask_out")
#     elif is_frames_dir(base_input):
#         # 帧文件夹 -> dyn_mask_out 放在该文件夹内
#         out_dir = os.path.join(os.path.abspath(base_input), "dyn_mask_out")
#     else:
#         raise RuntimeError(f"Invalid input: {base_input}")
    
#     os.makedirs(out_dir, exist_ok=True)
#     return out_dir


# def write_mask_video(mask_dir: str, out_path: str, fps_val: float):
#     frame_paths = sorted(glob.glob(os.path.join(mask_dir, "Aa_dyn_group_*.png")))
#     if len(frame_paths) == 0:
#         raise RuntimeError("No dynamic-mask frames found to build video.")
#     first = cv2.imread(frame_paths[0])
#     h, w = first.shape[:2]
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     writer = cv2.VideoWriter(out_path, fourcc, fps_val, (w, h))
#     for fp in frame_paths:
#         im = cv2.imread(fp)
#         if im is not None:
#             writer.write(im)
#     writer.release()

# # -----------------------------
# # --- Main                   ---
# # -----------------------------
# def main():
#     if not (is_video_file(input_path) or is_frames_dir(input_path)):
#         raise RuntimeError("input_path must be a video file or a directory of frames.")

#     # Output dir
#     output_dir = ensure_outdir(input_path)

#     # Original size (for logging only)
#     orig_h, orig_w = read_first_frame_size(input_path)
#     print(f"[Info] First frame size (H x W): {orig_h} x {orig_w}")

#     # Device & model
#     device = torch.device(device_str if torch.cuda.is_available() else "cpu")
#     model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
#     model.eval()

#     # Load frames via DUSt3R loader:
#     # - if input is video: pass [video_file]
#     # - if input is frames folder: pass the directory path itself
#     start_time = time.time()
#     if is_video_file(input_path):
#         load_arg = [input_path]  # list to trigger video branch in load_images
#         fps_arg = fps            # use provided fps (0 means keep original)
#     else:
#         load_arg = input_path    # directory string -> image branch in load_images
#         fps_arg = 0              # fps is irrelevant for folder input; DUSt3R returns 24 by design

#     imgs, width, height, video_fps = load_images(
#         load_arg, size=image_size, fps=fps_arg, num_frames=num_frames, return_img_size=True
#     )
#     print(f"[Info] Loaded {len(imgs)} frames resized to {width}x{height}, FPS={video_fps}")

#     # Build pairs
#     pairs = make_pairs(imgs, scene_graph="swin2stride", prefilter=None, symmetrize=True)
#     print(f"[Info] Built {len(pairs)} pairs")

#     # Inference
#     out = inference(pairs, model, device=device, batch_size=1, verbose=False)
#     if isinstance(out, dict) and {'view1','view2','pred1','pred2'}.issubset(out.keys()):
#         dust3r_output = out
#     elif isinstance(out, (list, tuple)) and len(out) == 4:
#         dust3r_output = dict(zip(['view1','view2','pred1','pred2'], out))
#     else:
#         raise RuntimeError(f"Inference returned unexpected type {type(out)}")

#     # Align (no optimization, just to get dynamic masks)
#     scene = global_aligner(
#         dust3r_output, device=device,
#         mode=GlobalAlignerMode.PointCloudOptimizer,
#         verbose=False, shared_focal=True,
#         temporal_smoothing_weight=0.0, translation_weight=1.0,
#         flow_loss_weight=0.0, flow_loss_start_epoch=0.0,
#         flow_loss_thre=25, use_self_mask=True,
#         num_total_iter=0, empty_cache=False,
#         batchify=True, use_atten_mask=True,
#         sam2_mask_refine=False
#     )

#     # Save dynamic masks
#     count = 0
#     for idx, m in enumerate(scene.dynamic_masks):
#         arr = (m.detach().cpu().numpy() * 255).astype(np.uint8)
#         # ensure 3-channel for stable video writing (optional)
#         if arr.ndim == 2:
#             arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
#         out_png = os.path.join(output_dir, f"Aa_dyn_group_{idx:04d}.png")
#         cv2.imwrite(out_png, arr)
#         count += 1
#     print(f"[Info] Saved {count} dynamic mask frames to {output_dir}")

#     # Build a video from masks
#     out_video = os.path.join(output_dir, mask_video_name)
#     fps_for_video = fps if (is_video_file(input_path) and fps > 0) else (video_fps or 24)
#     write_mask_video(output_dir, out_video, fps_for_video)
#     print(f"[Info] Dynamic mask video saved to: {out_video}")

#     print(f"[Timing] Total time: {time.time() - start_time:.2f}s")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
Generate dynamic attention masks (Aa=dyn) and demo-style fused attentions from either:
- a video file (e.g., .mp4/.avi/.mov), or
- a folder of frames (e.g., .png/.jpg)

It auto-detects input type. Runs DUSt3R -> global_aligner, saves:
- Four fused attentions in demo-like folders
- (Optional) fused dynamic map in demo-like folder
- Aa_dyn_group_XXXX.png every N frames (default N=5)
and builds MP4s from the subsampled frames.
"""

import os
import time
import cv2
import glob
import shutil
import numpy as np
import torch
from typing import Tuple, List, Optional

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.inference import inference
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# -----------------------------
# --- Configuration (edit)  ---
# -----------------------------
input_path = "/mnt/data0/andy/Easi3R/DAVIS/davis_videos/dogs-jump.mp4"  # video file or a directory of frames
model_path = "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
device_str = "cuda"
image_size = 512        # {224, 512}
fps = 0                 # only used for VIDEO input; 0 means use original video FPS
num_frames = None       # optional limit of frames to load (both video & frames folder)
mask_video_name = "Aa_dyn_group_video.mp4"
every_n = 1             # <-- 只处理/导出每 N 帧（保留你的“每5帧”逻辑）

# 如果你也想把 fused dynamic 的 demo 图保留下来，设为 True
save_fused_dynamic_demo = True

# -----------------------------
# --- Helpers               ---
# -----------------------------
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".MP4", ".AVI", ".MOV")
IMG_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")

def is_video_file(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith(VIDEO_EXTS)

def is_frames_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for fn in os.listdir(path):
        if fn.lower().endswith(IMG_EXTS):
            return True
    return False

def input_stem(path: str) -> str:
    """For video: basename without ext; for frames folder: folder name."""
    if is_video_file(path):
        return os.path.splitext(os.path.basename(path))[0]
    elif is_frames_dir(path):
        return os.path.basename(os.path.normpath(path))
    else:
        raise RuntimeError(f"Invalid input: {path}")

def read_first_frame_size(path: str) -> Tuple[int, int]:
    """Return (H, W) of first frame whether input is video or frames folder."""
    if is_video_file(path):
        cap = cv2.VideoCapture(path)
        ret, frame0 = cap.read()
        cap.release()
        if not ret or frame0 is None:
            raise RuntimeError(f"Cannot read first frame from video: {path}")
        return frame0.shape[:2]
    elif is_frames_dir(path):
        frames = sorted([f for f in os.listdir(path) if f.lower().endswith(IMG_EXTS)])
        if not frames:
            raise RuntimeError(f"No image frames found in folder: {path}")
        img_path = os.path.join(path, frames[0])
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        return img.shape[:2]
    else:
        raise RuntimeError(f"Input path is neither a video nor a frames folder: {path}")

def ensure_outdir(base_input: str) -> str:
    """
    Decide output directory:
    - If video file: output dir is sibling to the video file
    - If frames folder: output dir is inside that folder
    """
    if is_video_file(base_input):
        base_dir = os.path.dirname(base_input)
        out_dir = os.path.join(base_dir, "dyn_mask_out")
    elif is_frames_dir(base_input):
        out_dir = os.path.join(os.path.abspath(base_input), "dyn_mask_out")
    else:
        raise RuntimeError(f"Invalid input: {base_input}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def write_video_from_frames(pattern: str, out_path: str, fps_val: float):
    frame_paths = sorted(glob.glob(pattern))
    if len(frame_paths) == 0:
        raise RuntimeError(f"No frames found: {pattern}")
    first = cv2.imread(frame_paths[0])
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps_val, (w, h))
    for fp in frame_paths:
        im = cv2.imread(fp)
        if im is not None:
            writer.write(im)
    writer.release()

def _save_resized_rgb_frames(imgs: List[np.ndarray], save_dir: str):
    """Save DUSt3R-resized RGB frames as BGR PNGs to {save_dir}/frame_XXXX.png"""
    os.makedirs(save_dir, exist_ok=True)
    for i, rgb in enumerate(imgs):
        if i % every_n != 0:
            continue
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, f"frame_{i:04d}.png"), bgr)
def to_rgb_uint8(img) -> np.ndarray:
    """
    将输入统一为 RGB uint8 的 HxWx3 numpy 数组。
    支持: numpy, torch.Tensor, PIL.Image
    也处理 [3,H,W]/[H,W,3]/灰度/float 等常见情况。
    """
    # 1) 转成 numpy
    if isinstance(img, np.ndarray):
        arr = img
    elif torch.is_tensor(img):
        arr = img.detach().cpu().numpy()
    elif hasattr(img, "convert"):  # PIL.Image
        arr = np.array(img.convert("RGB"))
    else:
        raise TypeError(f"Unsupported frame type: {type(img)}")

    # 2) 通道维度处理
    if arr.ndim == 2:                      # 灰度 -> 3通道
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):  # [C,H,W] -> [H,W,C]
            arr = np.transpose(arr, (1, 2, 0))
        # 若是单通道 [H,W,1] -> [H,W,3]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
    else:
        raise ValueError(f"Unexpected array shape: {arr.shape}")

    # 3) dtype/取值范围
    if arr.dtype != np.uint8:
        # 常见两类: float[0,1] 或 float[0,255]
        arr_min, arr_max = arr.min(), arr.max()
        if arr.dtype.kind in "fc":
            if arr_max <= 1.0:   # [0,1] -> [0,255]
                arr = (arr * 255.0).clip(0, 255)
            # 否则假定已在[0,255]区间，直接裁剪
            arr = arr.astype(np.float32).clip(0, 255)
        arr = arr.astype(np.uint8)

    # 4) 保证是 HxWx3
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expect HxWx3 after processing, got {arr.shape}")
    return arr

def _save_resized_rgb_frames(imgs, save_dir: str, every_n: int = 5):
    """Save DUSt3R-resized frames为BGR PNG，仅保存每 N 帧。"""
    os.makedirs(save_dir, exist_ok=True)
    for i, frame in enumerate(imgs):
        if i % every_n != 0:
            continue
        try:
            rgb = to_rgb_uint8(frame)           # 统一成 RGB uint8 HxWx3
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f"frame_{i:04d}.png"), bgr)
        except Exception as e:
            print(f"[Warn] skip frame {i}: {e}")


def _subsample_demo_frames(folder: str, pattern="frames_att/frame_*.png", n:int=5):
    """
    在 demo 风格的输出目录里（如 0_cross_att_k_i_mean_fused），只保留每 n 帧。
    会把原有 frames_att 下不满足条件的帧删掉（或移动到 _all 备份）。
    """
    frames_dir = os.path.join(folder, "frames_att")
    if not os.path.isdir(frames_dir):
        return
    # 可选：备份所有帧
    backup_dir = os.path.join(folder, "frames_att_all")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir, exist_ok=True)
        for p in glob.glob(os.path.join(frames_dir, "frame_*.png")):
            shutil.copy2(p, os.path.join(backup_dir, os.path.basename(p)))

    kept, removed = 0, 0
    for fp in sorted(glob.glob(os.path.join(frames_dir, "frame_*.png"))):
        # 解析编号
        base = os.path.basename(fp)
        try:
            idx = int(os.path.splitext(base)[0].split("_")[-1])
        except Exception:
            # 不规范的命名，跳过
            continue
        if idx % n == 0:
            kept += 1
            continue
        # 删除非每 n 帧
        os.remove(fp)
        removed += 1
    print(f"[Info] Subsample {folder}: kept={kept}, removed={removed} (every {n})")

# -----------------------------
# --- Main                   ---
# -----------------------------
def main():
    if not (is_video_file(input_path) or is_frames_dir(input_path)):
        raise RuntimeError("input_path must be a video file or a directory of frames.")

    # Output dir
    # Output dir: <name>_dyn_mask_out
    stem = input_stem(input_path)
    if is_video_file(input_path):
        base_dir = os.path.dirname(input_path)
        output_dir = os.path.join(base_dir, f"{stem}_dyn_mask_out")
    else:  # frames dir
        output_dir = os.path.join(os.path.abspath(input_path), f"{stem}_dyn_mask_out")
    os.makedirs(output_dir, exist_ok=True)


    # Original size (for logging only)
    orig_h, orig_w = read_first_frame_size(input_path)
    print(f"[Info] First frame size (H x W): {orig_h} x {orig_w}")

    # Device & model
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
    model.eval()

    # Load frames via DUSt3R loader
    start_time = time.time()
    if is_video_file(input_path):
        load_arg = [input_path]
        fps_arg = fps
    else:
        load_arg = input_path
        fps_arg = 0

    imgs, width, height, video_fps = load_images(
        load_arg, size=image_size, fps=fps_arg, num_frames=num_frames, return_img_size=True
    )
    print(f"[Info] Loaded {len(imgs)} frames resized to {width}x{height}, FPS={video_fps}")

    # 先把“原图帧（已resize）”每 N 帧输出一份，供后面拼图/检查
    frames_dir = os.path.join(output_dir, "frames")
    _save_resized_rgb_frames(imgs, frames_dir, every_n=every_n)

    # Build pairs & inference
    pairs = make_pairs(imgs, scene_graph="swin2stride", prefilter=None, symmetrize=True)
    print(f"[Info] Built {len(pairs)} pairs")
    out = inference(pairs, model, device=device, batch_size=1, verbose=False)
    if isinstance(out, dict) and {'view1','view2','pred1','pred2'}.issubset(out.keys()):
        dust3r_output = out
    elif isinstance(out, (list, tuple)) and len(out) == 4:
        dust3r_output = dict(zip(['view1','view2','pred1','pred2'], out))
    else:
        raise RuntimeError(f"Inference returned unexpected type {type(out)}")

    # Align (no optimization, just to get attention & dynamic masks)
    scene = global_aligner(
        dust3r_output, device=device,
        mode=GlobalAlignerMode.PointCloudOptimizer,
        verbose=False, shared_focal=True,
        temporal_smoothing_weight=0.0, translation_weight=1.0,
        flow_loss_weight=0.0, flow_loss_start_epoch=0.0,
        flow_loss_thre=25, use_self_mask=True,
        num_total_iter=0, empty_cache=False,
        batchify=True, use_atten_mask=True, use_region_pooling = True,
        sam2_group_output_dir = "/mnt/data0/andy/Easi3R/sam2_region_track/dogs-jump",
        sam2_mask_refine=False
    )

    # ===== Demo-like attention saving =====
    # 让 scene 自己把四个 fused attentions（以及可选的 fused dynamic）落盘到 demo 同款目录结构
    try:
        if hasattr(scene, "save_attention_maps"):
            scene.save_attention_maps(output_dir)
            # 只保留每 N 帧
            for sub in [
                "0_cross_att_k_i_mean_fused",
                "0_cross_att_k_j_mean_fused",
                "0_cross_att_k_i_var_fused",
                "0_cross_att_k_j_var_fused",
            ]:
                _subsample_demo_frames(os.path.join(output_dir, sub), n=every_n)

        if save_fused_dynamic_demo and hasattr(scene, "save_init_fused_dynamic_masks"):
            scene.save_init_fused_dynamic_masks(output_dir)
            _subsample_demo_frames(os.path.join(output_dir, "0_dynamic_map_fused"), n=every_n)

    except Exception as e:
        print(f"[Warn] demo-style attention saving failed: {e}")

    # ===== Save Aa_dyn (only every N frames) =====
    count = 0
    for idx, m in enumerate(scene.dynamic_masks):
        if idx % every_n != 0:
            continue
        arr = (m.detach().cpu().numpy() * 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        out_png = os.path.join(output_dir, f"Aa_dyn_group_{idx:04d}.png")
        cv2.imwrite(out_png, arr)
        count += 1
    print(f"[Info] Saved {count} dynamic mask frames (every {every_n}) to {output_dir}")

    # ===== Build videos from subsampled frames =====
    fps_for_video = fps if (is_video_file(input_path) and fps > 0) else (video_fps or 24)

    # Aa_dyn video
    try:
        write_video_from_frames(os.path.join(output_dir, "Aa_dyn_group_*.png"),
                                os.path.join(output_dir, mask_video_name),
                                fps_for_video)
        print(f"[Info] Dynamic mask video saved to: {os.path.join(output_dir, mask_video_name)}")
    except Exception as e:
        print(f"[Warn] writing Aa_dyn video failed: {e}")

    # Attention mosaics video（可选：如果你后续拼合一张大图的话再做；这里仅保留按 demo 输出的单幅帧图与目录）
    # 你可以继续复用之前给你的 compose/mosaic 函数来合成 attention_video.mp4

    print(f"[Timing] Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
