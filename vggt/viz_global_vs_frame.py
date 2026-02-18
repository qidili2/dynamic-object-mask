import os, re, glob, argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import matplotlib.cm as cm

from vggt.models.aggregator import Aggregator

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

def extract_last_int(path: str) -> int:
    base = os.path.splitext(os.path.basename(path))[0]
    nums = re.findall(r"\d+", base)
    return int(nums[-1]) if nums else 10**18

def list_images_sorted(seq_dir: str):
    paths = []
    for ext in IMG_EXTS:
        paths += glob.glob(os.path.join(seq_dir, f"*{ext}"))
        paths += glob.glob(os.path.join(seq_dir, f"*{ext.upper()}"))
    paths = sorted(set(paths))
    paths.sort(key=lambda p: (extract_last_int(p), os.path.basename(p)))
    return paths

def load_rgb_01(path: str, short_side: int = 0) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if short_side and short_side > 0:
        W, H = img.size
        s = min(H, W)
        if s != short_side:
            if H < W:
                newH = short_side
                newW = int(round(W * (short_side / H)))
            else:
                newW = short_side
                newH = int(round(H * (short_side / W)))
            img = img.resize((newW, newH), resample=Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # [3,H,W]

def pad_to_multiple(img_chw: torch.Tensor, multiple: int):
    C, H, W = img_chw.shape
    Hp = ((H + multiple - 1) // multiple) * multiple
    Wp = ((W + multiple - 1) // multiple) * multiple
    img_pad = F.pad(img_chw, (0, Wp - W, 0, Hp - H), mode="replicate")
    return img_pad, (H, W, Hp, Wp)

def to_01(x: torch.Tensor) -> torch.Tensor:
    x = x - x.min()
    x = x / (x.max() + 1e-6)
    return x

def infer_patch_grid(Hp: int, Wp: int, patch_size: int, P_patch: int):
    Hp0, Wp0 = Hp // patch_size, Wp // patch_size
    if Hp0 * Wp0 == P_patch:
        return Hp0, Wp0
    # fallback factor search
    r = int(round(P_patch ** 0.5))
    best = None
    target = Hp0 / (Wp0 + 1e-6) if Hp0 > 0 and Wp0 > 0 else 1.0
    for hp in range(max(1, r - 400), r + 401):
        if P_patch % hp == 0:
            wp = P_patch // hp
            score = abs((hp / (wp + 1e-6)) - target)
            cand = (score, hp, wp)
            if best is None or cand < best:
                best = cand
    return (best[1], best[2]) if best else (r, max(1, P_patch // max(1, r)))

def save_inferno_png(map_hw_01: torch.Tensor, out_path: str):
    m = map_hw_01.clamp(0, 1).cpu().numpy()
    rgb = cm.get_cmap("inferno")(m)[..., :3]
    Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB").save(out_path)

def save_inferno_overlay(map_hw_01: torch.Tensor, rgb_chw_01: torch.Tensor, out_path: str, alpha: float = 0.55):
    heat = map_hw_01.clamp(0, 1).cpu().numpy()
    rgb = rgb_chw_01.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    heat_rgb = cm.get_cmap("inferno")(heat)[..., :3]
    ov = np.clip((1 - alpha) * rgb + alpha * heat_rgb, 0, 1)
    Image.fromarray((ov * 255).astype(np.uint8), mode="RGB").save(out_path)

def heatmap_from_keyvec(key_vec_patch: torch.Tensor, Hp_grid: int, Wp_grid: int, Hp: int, Wp: int, H: int, W: int):
    # key_vec_patch: [P_patch]
    m = to_01(key_vec_patch.view(Hp_grid, Wp_grid))
    m = F.interpolate(m[None, None], size=(Hp, Wp), mode="bilinear", align_corners=False)[0, 0]
    m = m[:H, :W]
    return to_01(m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--t0", type=int, default=0, help="query frame index")
    ap.add_argument("--K", type=int, default=256, help="num sampled patch queries from frame t0")
    ap.add_argument("--last_layers", type=int, default=12, help="average last N layers")
    ap.add_argument("--chunk_k", type=int, default=4096, help="key chunk size")
    ap.add_argument("--short_side", type=int, default=336, help="resize short side (0 to disable)")
    ap.add_argument("--patch_embed", type=str, default="dinov2_vitl14_reg",
                    choices=["dinov2_vitl14_reg","dinov2_vitb14_reg","dinov2_vits14_reg","dinov2_vitg2_reg"])
    ap.add_argument("--alpha", type=float, default=0.55)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = list_images_sorted(args.seq_dir)
    if not paths:
        raise RuntimeError(f"No images found under {args.seq_dir}")
    T = len(paths)
    t0 = max(0, min(args.t0, T - 1))

    print("seq_dir:", args.seq_dir)
    print("T:", T, "t0:", t0, "device:", device)
    print("patch_embed:", args.patch_embed, "short_side:", args.short_side)

    # load all frames (optionally resize), keep originals for overlay, and padded for model
    orig_list = []
    pad_list = []
    sizes = None
    patch_size = 14

    for p in paths:
        rgb = load_rgb_01(p, short_side=args.short_side)
        orig_list.append(rgb)  # [3,H,W]
        rgb_pad, (H, W, Hp, Wp) = pad_to_multiple(rgb, patch_size)
        pad_list.append(rgb_pad)
        if sizes is None:
            sizes = (H, W, Hp, Wp)
        else:
            assert sizes == (H, W, Hp, Wp), "All frames must have same size after resize."

    H, W, Hp, Wp = sizes
    imgs = torch.stack(pad_list, dim=0).unsqueeze(0).to(device)  # [1,T,3,Hp,Wp]

    # model
    model = Aggregator(img_size=H, patch_size=patch_size, patch_embed=args.patch_embed).to(device)
    model.eval().half()
    imgs = imgs.half()

    # compute token sizes
    P_patch = (Hp // patch_size) * (Wp // patch_size)
    Hp_grid, Wp_grid = infer_patch_grid(Hp, Wp, patch_size, P_patch)

    # run once to get real patch_start_idx
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _outs, patch_start_idx = model(imgs[:, :1])  # just 1 frame quick pass

    # tokens per frame
    P = patch_start_idx + P_patch
    print("patch_start_idx:", patch_start_idx, "P_patch:", P_patch, "P:", P)

    # sample K patch queries from frame token axis [patch_start_idx .. P-1]
    all_q = torch.arange(patch_start_idx, P, dtype=torch.long)
    K = min(args.K, all_q.numel())
    g = torch.Generator().manual_seed(0)
    sel = all_q[torch.randperm(all_q.numel(), generator=g)[:K]]

    # enable dumping on last layers only
    L = model.depth
    layers = list(range(max(0, L - args.last_layers), L))
    print("avg layers:", layers)

    for i in layers:
        # frame blocks dump queries on per-frame axis
        model.frame_blocks[i].attn.dump_q_idx = sel
        model.frame_blocks[i].attn.dump_chunk_k = args.chunk_k
        # global blocks dump queries on concatenated axis (frame offset)
        model.global_blocks[i].attn.dump_q_idx = sel + t0 * P
        model.global_blocks[i].attn.dump_chunk_k = args.chunk_k

    # full forward once (no window)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _outs, patch_start_idx2 = model(imgs)

    if patch_start_idx2 != patch_start_idx:
        patch_start_idx = patch_start_idx2
        P = patch_start_idx + P_patch
        print("NOTE: patch_start_idx updated to", patch_start_idx, "P updated to", P)

    # accumulate dumps across layers
    frame_acc = None  # will be [B*T, heads, K, P]
    global_acc = None # will be [B,   heads, K, T*P]
    for i in layers:
        a_f = getattr(model.frame_blocks[i].attn, "_last_dump_attn", None)
        a_g = getattr(model.global_blocks[i].attn, "_last_dump_attn", None)
        if a_f is None or a_g is None:
            raise RuntimeError(f"Missing dump at layer {i}. Check attention.py dump code.")
        frame_acc = a_f if frame_acc is None else (frame_acc + a_f)
        global_acc = a_g if global_acc is None else (global_acc + a_g)

    frame_avg = frame_acc / len(layers)
    global_avg = global_acc / len(layers)

    # ---- build FRAME self-attn heatmap for t0 ----
    # frame_avg: [T, heads, K, P] because B=1 -> B*T = T
    a = frame_avg[t0].mean(dim=0)     # mean heads -> [K, P]
    key_imp = a.mean(dim=0)           # mean queries -> [P]
    key_patch = key_imp[patch_start_idx:]  # [P_patch]
    hm_frame = heatmap_from_keyvec(key_patch, Hp_grid, Wp_grid, Hp, Wp, H, W)

    # ---- build GLOBAL attn heatmaps: t0 queries -> each frame j keys ----
    # global_avg: [1, heads, K, T*P]
    g0 = global_avg[0].mean(dim=0)    # mean heads -> [K, T*P]
    g_imp = g0.mean(dim=0)            # mean queries -> [T*P]

    # output dirs
    os.makedirs(args.out_root, exist_ok=True)
    frame_dir = os.path.join(args.out_root, "frame_self_inferno")
    frame_ov  = os.path.join(args.out_root, "frame_self_overlay")
    glob_dir  = os.path.join(args.out_root, "global_to_all_inferno")
    glob_ov   = os.path.join(args.out_root, "global_to_all_overlay")
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(frame_ov, exist_ok=True)
    os.makedirs(glob_dir, exist_ok=True)
    os.makedirs(glob_ov, exist_ok=True)

    # save frame self
    save_inferno_png(hm_frame, os.path.join(frame_dir, f"{t0:05d}.png"))
    save_inferno_overlay(hm_frame, orig_list[t0], os.path.join(frame_ov, f"{t0:05d}.png"), alpha=args.alpha)

    # save global-to-all per key frame
    # Also save a per-frame scalar importance file for quick sanity-check
    per_frame_scores = []
    for j in range(T):
        seg = g_imp[j * P:(j + 1) * P]               # [P]
        key_patch = seg[patch_start_idx:]            # [P_patch]
        hm = heatmap_from_keyvec(key_patch, Hp_grid, Wp_grid, Hp, Wp, H, W)
        save_inferno_png(hm, os.path.join(glob_dir, f"{j:05d}.png"))
        save_inferno_overlay(hm, orig_list[j], os.path.join(glob_ov, f"{j:05d}.png"), alpha=args.alpha)
        per_frame_scores.append(float(seg.sum().item()))

        if j % 20 == 0:
            print(f"saved global key-frame {j}/{T-1}")

    with open(os.path.join(args.out_root, "global_per_frame_score.txt"), "w") as f:
        for j, sc in enumerate(per_frame_scores):
            f.write(f"{j:05d}\t{os.path.basename(paths[j])}\t{sc:.6f}\n")

    print("Done.")
    print("Saved frame self-attn heatmap for t0 ->", os.path.join(frame_dir, f"{t0:05d}.png"))
    print("Saved global t0->all per-frame heatmaps ->", glob_dir)
    print("Scores ->", os.path.join(args.out_root, "global_per_frame_score.txt"))

if __name__ == "__main__":
    main()
