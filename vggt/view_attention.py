import os
import re
import glob
import numpy as np
from PIL import Image
import matplotlib.cm as cm

import torch
import torch.nn.functional as F

from vggt.models.aggregator import Aggregator


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

def save_inferno_png(map_hw_01: torch.Tensor, out_path: str):
    """
    map_hw_01: [H,W] float in [0,1]
    Save inferno-colored heatmap PNG (RGB).
    """
    m = map_hw_01.clamp(0, 1).cpu().numpy()          # [H,W]
    colored = cm.get_cmap("inferno")(m)[..., :3]     # [H,W,3] 0..1
    img = (colored * 255.0).astype(np.uint8)
    Image.fromarray(img, mode="RGB").save(out_path)


def save_inferno_overlay_png(map_hw_01: torch.Tensor, rgb_chw_01: torch.Tensor, out_path: str, alpha: float = 0.55):
    """
    map_hw_01: [H,W] 0..1
    rgb_chw_01: [3,H,W] 0..1  (use ORIGINAL frame, not padded)
    alpha: heatmap opacity
    """
    heat = map_hw_01.clamp(0, 1).cpu().numpy()       # [H,W]
    rgb = rgb_chw_01.clamp(0, 1).permute(1, 2, 0).cpu().numpy()  # [H,W,3]

    heat_rgb = cm.get_cmap("inferno")(heat)[..., :3] # [H,W,3]
    overlay = (1 - alpha) * rgb + alpha * heat_rgb
    overlay = np.clip(overlay, 0, 1)

    Image.fromarray((overlay * 255.0).astype(np.uint8), mode="RGB").save(out_path)


def extract_last_int(path: str) -> int:
    base = os.path.splitext(os.path.basename(path))[0]
    nums = re.findall(r"\d+", base)
    if not nums:
        return 10**18
    return int(nums[-1])


def list_images_sorted(seq_dir: str):
    paths = []
    for ext in IMG_EXTS:
        paths += glob.glob(os.path.join(seq_dir, f"*{ext}"))
        paths += glob.glob(os.path.join(seq_dir, f"*{ext.upper()}"))
    paths = sorted(set(paths))
    paths.sort(key=lambda p: (extract_last_int(p), os.path.basename(p)))
    return paths


def load_rgb_01(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # [3,H,W]


def pad_to_multiple(img_chw: torch.Tensor, multiple: int):
    C, H, W = img_chw.shape
    Hp = ((H + multiple - 1) // multiple) * multiple
    Wp = ((W + multiple - 1) // multiple) * multiple
    pad_h = Hp - H
    pad_w = Wp - W
    img_pad = F.pad(img_chw, (0, pad_w, 0, pad_h), mode="replicate")  # (L,R,T,B)
    return img_pad, (H, W, Hp, Wp)


def to_01(x: torch.Tensor) -> torch.Tensor:
    x = x - x.min()
    x = x / (x.max() + 1e-6)
    return x


def save_map01_png(map_hw_01: torch.Tensor, out_path: str):
    m = (map_hw_01.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(m, mode="L").save(out_path)


def infer_patch_grid(Hp: int, Wp: int, patch_size: int, P_patch: int):
    Hp0, Wp0 = Hp // patch_size, Wp // patch_size
    if Hp0 > 0 and Wp0 > 0 and Hp0 * Wp0 == P_patch:
        return Hp0, Wp0

    # fallback: factor pair near sqrt
    r = int(round(P_patch ** 0.5))
    best = None
    target_ratio = (Hp0 / (Wp0 + 1e-6)) if (Hp0 > 0 and Wp0 > 0) else 1.0
    for hp in range(max(1, r - 400), r + 401):
        if P_patch % hp == 0:
            wp = P_patch // hp
            score = abs((hp / (wp + 1e-6)) - target_ratio)
            cand = (score, hp, wp)
            if best is None or cand < best:
                best = cand
    if best is not None:
        return best[1], best[2]

    hp = int(round(P_patch ** 0.5))
    wp = max(1, P_patch // max(1, hp))
    return hp, wp


def main():
    # --------- config ---------
    seq_dir = "/mnt/data0/andy/Easi3R/DAVIS/JPEGImages/480p/boxing-fisheye"
    out_root = "attn_viz_boxing-fisheye"

    S = 4          # window size (cam-attn 版一般 S=8 也能跑，但先 4)
    stride = 1
    layer_idx = 0  # which block to visualize

    patch_size = 14
    # --------------------------

    print("=== view_attention (cam-attn) ===")
    print("seq_dir:", seq_dir)
    print("out_root:", os.path.abspath(out_root))

    paths = list_images_sorted(seq_dir)
    if len(paths) == 0:
        raise RuntimeError(f"No images found under {seq_dir} with exts={IMG_EXTS}")
    T = len(paths)
    print("num images:", T)
    print("first:", paths[0])
    print("last :", paths[-1])

    if T < S:
        S = T

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    img0 = load_rgb_01(paths[0])
    _, H0, W0 = img0.shape

    model = Aggregator(img_size=H0, patch_size=patch_size).to(device)
    model.eval()
    model.half()

    # enable cam-attn caching on selected layer
    model.frame_blocks[layer_idx].return_cam_attn = True
    model.global_blocks[layer_idx].return_cam_attn = True

    os.makedirs(out_root, exist_ok=True)
    out_frame = os.path.join(out_root, "frame_camattn_inferno")
    out_global = os.path.join(out_root, "global_camattn_inferno")
    out_frame_ov = os.path.join(out_root, "frame_camattn_overlay")
    out_global_ov = os.path.join(out_root, "global_camattn_overlay")

    os.makedirs(out_frame, exist_ok=True)
    os.makedirs(out_global, exist_ok=True)
    os.makedirs(out_frame_ov, exist_ok=True)
    os.makedirs(out_global_ov, exist_ok=True)
    # accumulators (original resolution)
    frame_sum = [torch.zeros((H0, W0), device=device) for _ in range(T)]
    global_sum = [torch.zeros((H0, W0), device=device) for _ in range(T)]
    frame_cnt = [0] * T
    global_cnt = [0] * T

    last_patch_start_idx = None

    for start in range(0, T - S + 1, stride):
        window_paths = paths[start:start + S]

        imgs_list = []
        sizes = None
        for p in window_paths:
            img = load_rgb_01(p)
            img_pad, (H, W, Hp, Wp) = pad_to_multiple(img, patch_size)
            imgs_list.append(img_pad)
            if sizes is None:
                sizes = (H, W, Hp, Wp)
            else:
                assert sizes == (H, W, Hp, Wp), "Frames in a window have different sizes!"

        H, W, Hp, Wp = sizes
        imgs = torch.stack(imgs_list, dim=0).unsqueeze(0).to(device)  # [1,S,3,Hp,Wp]

        # forward
        imgs = imgs.half()  # ★输入也 fp16

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _outputs, patch_start_idx = model(imgs)
        last_patch_start_idx = patch_start_idx

        # cached cam-attn
        cam_attn_frame = getattr(model.frame_blocks[layer_idx], "_last_cam_attn", None)
        cam_attn_global = getattr(model.global_blocks[layer_idx], "_last_cam_attn", None)
        if cam_attn_frame is None or cam_attn_global is None:
            raise RuntimeError("cam-attn not cached. Did you modify Attention/Block and set return_cam_attn=True?")

        # cam_attn_frame: [B*S, heads, 1, P] with B=1
        # cam_attn_global: [B, heads, 1, S*P] with B=1
        heads = cam_attn_frame.shape[1]
        P = cam_attn_frame.shape[-1]
        P_patch = P - patch_start_idx

        Hp_grid, Wp_grid = infer_patch_grid(Hp, Wp, patch_size, P_patch)

        # -------- frame cam-attn -> per-frame heatmap --------
        cam_frame_bs = cam_attn_frame.view(1, S, heads, 1, P)

        for t in range(S):
            row = cam_frame_bs[0, t]                 # [heads,1,P]
            row = row.mean(dim=0).squeeze(0)         # mean heads -> [P]
            key_patch = row[patch_start_idx:]        # [P_patch]

            map_patch = to_01(key_patch.view(Hp_grid, Wp_grid))
            map_img = F.interpolate(
                map_patch[None, None], size=(Hp, Wp), mode="bilinear", align_corners=False
            )[0, 0]
            map_img = map_img[:H, :W]                # crop back
            map_img = to_01(map_img)

            idx = start + t
            frame_sum[idx] += map_img
            frame_cnt[idx] += 1

        # -------- global cam-attn -> per-frame heatmap (slice by frame) --------
        rowg = cam_attn_global[0].mean(dim=0).squeeze(0)  # [S*P]

        for t in range(S):
            seg = rowg[t * P:(t + 1) * P]            # [P]
            key_patch = seg[patch_start_idx:]        # [P_patch]

            map_patch = to_01(key_patch.view(Hp_grid, Wp_grid))
            map_img = F.interpolate(
                map_patch[None, None], size=(Hp, Wp), mode="bilinear", align_corners=False
            )[0, 0]
            map_img = map_img[:H, :W]
            map_img = to_01(map_img)

            idx = start + t
            global_sum[idx] += map_img
            global_cnt[idx] += 1

        if start % 20 == 0:
            print(f"Processed window {start}/{max(0, T-S)} (S={S}, padded={Hp}x{Wp})")

    # save
    for i in range(T):
        if frame_cnt[i] > 0:
            m = to_01(frame_sum[i] / float(frame_cnt[i]))  # [H,W] 0..1
            # inferno heatmap
            save_inferno_png(m, os.path.join(out_frame, f"{i:05d}.png"))
            # overlay (读原图，注意这里用原始未pad的尺寸)
            rgb = load_rgb_01(paths[i])  # [3,H,W] 0..1  (如果你做了 resize，这里会自动一致)
            save_inferno_overlay_png(m, rgb, os.path.join(out_frame_ov, f"{i:05d}.png"), alpha=0.55)

        if global_cnt[i] > 0:
            m = to_01(global_sum[i] / float(global_cnt[i]))
            save_inferno_png(m, os.path.join(out_global, f"{i:05d}.png"))
            rgb = load_rgb_01(paths[i])
            save_inferno_overlay_png(m, rgb, os.path.join(out_global_ov, f"{i:05d}.png"), alpha=0.55)


    print("Done.")
    print("Saved frame cam-attn :", out_frame)
    print("Saved global cam-attn:", out_global)
    print("Mapping file         :", os.path.join(out_root, "index_mapping.txt"))
    print("patch_start_idx      :", last_patch_start_idx)


if __name__ == "__main__":
    main()
