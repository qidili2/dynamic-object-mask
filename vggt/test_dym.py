import os
import cv2
import torch
import numpy as np
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# 设置设备与数据类型
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8) else torch.float16

# def select_points(frame,imgs):
# """
# 固定点追踪
# """
#     orig_h, orig_w = frame.shape[:2]
#     _, _, _, proc_h, proc_w = imgs.shape
#     default_pts = [(684,271),(681,306),(782,275),(753,330)]
#     scale_x = proc_w / orig_w
#     scale_y = proc_h / orig_h

#     scaled = []
#     for x, y in default_pts:
#         x_new = x * scale_x
#         y_new = y * scale_y
#         scaled.append((x_new, y_new))
#     # 长城 [(259,183),(383,183),(421,214),(306,68)]
#     # 斗兽场 [(320,75),(173,122),(255,181),(413,85)]
#     # 白车 [(622,143),(520,146),(446,121),(533,112)]
#     # 狗 [(300,310),(296,328),(381,313),(390,323)]
#     # 鹅 [(684,271),(681,306),(782,275),(753,330)]
#     print(f"default points: {default_pts}")
#     return scaled

def select_points(frame, imgs, stride=20):
    """
    以固定像素间距采样网格点。
    """
    # 原始 & 处理后 尺寸
    H_ori, W_ori = frame.shape[:2]
    _, _, _, H_proc, W_proc = imgs.shape

    # 按固定间距采样 processed 尺寸下的点
    pts_proc = [
        (x, y)
        for y in range(0, H_proc, stride)
        for x in range(0, W_proc, stride)
    ]
    print(f"以固定间距 {stride} 像素采样得到 {len(pts_proc)} 个点 (processed 分辨率)：{pts_proc}")
    return pts_proc

def track_video(input_path, output_dir=None):
    # 加载帧目录并预处理
    if os.path.isfile(input_path):
        video_path = input_path
        base, _     = os.path.splitext(os.path.basename(video_path))
        frames_dir  = os.path.join(os.path.dirname(video_path), base + "_frames")
        os.makedirs(frames_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS) 
        target_fps = 2   # 在这里每帧读取多少图片
        interval   = max(int(round(orig_fps / target_fps)), 1)
        idx = 0
        saved = 0
        print(f"Sampling video at {target_fps:.1f}fps (orig {orig_fps:.1f}fps), every {interval} frames, saving to {frames_dir}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                out_path = os.path.join(frames_dir, f"{saved:05d}.png")
                cv2.imwrite(out_path, frame)
                saved += 1
            idx += 1
        cap.release()
        print(f"  → Saved {saved} frames.")
        # 修改目录输入
        input_path = frames_dir
        image_names = [os.path.join(input_path, f)
                    for f in sorted(os.listdir(input_path))
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    elif os.path.isdir(input_path):
        image_names = [os.path.join(input_path, f)
                    for f in sorted(os.listdir(input_path))
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    image_names = image_names[:48]
    if not image_names:
        raise ValueError(f"目录 {input_path} 中未找到图片文件。")
    imgs = load_and_preprocess_images(image_names).to(device) 
    if imgs.ndim == 4:         # [S, C, H, W]
        imgs = imgs.unsqueeze(0)  # → [1, S, C, H, W]
    print("Input tensor shape:", imgs.shape)
    # 获取默认跟踪点
    first_frame = cv2.imread(image_names[0])
    if first_frame is None:
        raise FileNotFoundError(f"无法读取首帧: {image_names[0]}")
    init_pts = select_points(first_frame,imgs)
    pts_tensor = torch.tensor(init_pts, dtype=torch.float, device=device)
    print("points tensor shape:", pts_tensor.shape)

    # # 测试3帧是否爆显存 
    # image_names = ["./demo_data/dog-gooses/dog-gooses/00000.png", "./demo_data/dog-gooses/dog-gooses/00001.png", "./demo_data/dog-gooses/dog-gooses/00002.png"]  
    # imgs = load_and_preprocess_images(image_names).to(device)

    # 加载模型
    model = VGGT.from_pretrained("facebook/VGGT-1B").eval().to(device)

    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype if device=='cuda' else None):
                tokens_list, patch_start_idx = model.aggregator(imgs)
                query = pts_tensor.unsqueeze(0) 
                track_list, vis_score, conf_score = model.track_head(
                    tokens_list,
                    images=imgs,
                    patch_start_idx=patch_start_idx,
                    query_points=query
                )
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print("CUDA OOM")
        else:
            raise

    tracks = track_list[-1][0].cpu().numpy()  # [S, 4, 2]
    orig_h, orig_w = first_frame.shape[:2]
    _, _, _, proc_h, proc_w = imgs.shape

    scale_x = orig_w / proc_w
    scale_y = orig_h / proc_h
    
    # 输出路径
    base_dir = os.path.dirname(input_path) or '.'
    out_dir = output_dir or os.path.join(base_dir, 'tracking_result')
    os.makedirs(out_dir, exist_ok=True)
    out_video = os.path.join(out_dir, 'tracked_video.mp4')

    # 写入视频
    fps = 10
    h, w = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
    )
    S, M, _ = tracks.shape
    colors = []
    for idx in range(M):
        # HSV 色相均匀分布在 [0,180)
        hue = int(180 * idx / M)
        hsv_pix = np.uint8([[[hue, 255, 255]]])            # (1,1,3), HSV
        bgr_pix = cv2.cvtColor(hsv_pix, cv2.COLOR_HSV2BGR)
        bgr = tuple(int(x) for x in bgr_pix[0,0])          # (B,G,R)
        colors.append(bgr)
    for i, frame_path in enumerate(image_names):
        frame = cv2.imread(frame_path)
        j = 0
        for (x, y) in tracks[i]:
            x_orig = int(x * scale_x)
            y_orig = int(y * scale_y)
            color = colors[j] 
            j += 1
            cv2.circle(frame, (x_orig, y_orig), 3, color, -1)
        cv2.imwrite(os.path.join(out_dir, f'frame_{i:05d}.png'), frame)
        writer.write(frame)
    writer.release()

    print(f"Tracked video saved to: {out_video}")
    print(f"Annotated frames saved under: {out_dir}")
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='VGGT 4-point Hardcoded Tracking Demo')
    parser.add_argument('input', help='帧目录路径')
    parser.add_argument('--output_dir', help='输出 tracking_result 文件夹', default=None)
    args = parser.parse_args()
    track_video(args.input, args.output_dir)
