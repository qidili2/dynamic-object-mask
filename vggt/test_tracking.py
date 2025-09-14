import os
import cv2
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# 设置设备与数据类型
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8) else torch.float16

if device == 'cuda':
    num_gpus = torch.cuda.device_count()
    print(f"CUDA 可用 GPU 数量: {num_gpus}")
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        print(f"  GPU {i}: {name}")
    current = torch.cuda.current_device()
    print(f"当前使用 GPU 索引: {current}")
    print(f"当前使用 GPU 名称: {torch.cuda.get_device_name(current)}")
else:
    print("未检测到 CUDA，可在 CPU 上运行。")

def select_points(frame, n_points=4):
    """
    直接返回硬编码的默认跟踪点列表。
    """
    default_pts = [(622,143),(520,146),(446,121),(533,112)]
    # 白车 [(622,143),(520，146),(446，121),(533,112)]
    # 狗 [(300,310),(296，237),(381，313),(390,323)]
    # 鹅 [(684,271),(681，306),(782，275),(753,330)]
    print(f"default points: {default_pts}")
    return default_pts


def track_video(input_path, output_dir=None):
    # 加载帧目录并预处理
    if not os.path.isdir(input_path):
        raise NotImplementedError(
            "视频输入暂不支持，请先将视频导出为帧目录后再调用此函数。"
        )
    image_names = [os.path.join(input_path, f)
                   for f in sorted(os.listdir(input_path))
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_names = image_names[:32]
    if not image_names:
        raise ValueError(f"目录 {input_path} 中未找到图片文件。")
    imgs = load_and_preprocess_images(image_names).to(device)  # [1, S, 3, H, W]

    # 获取默认跟踪点
    first_frame = cv2.imread(image_names[0])
    if first_frame is None:
        raise FileNotFoundError(f"无法读取首帧: {image_names[0]}")
    init_pts = select_points(first_frame)
    pts_tensor = torch.tensor(init_pts, dtype=torch.float, device=device)

    # # 测试3帧是否爆显存 
    # image_names = ["./demo_data/dog-gooses/dog-gooses/00000.png", "./demo_data/dog-gooses/dog-gooses/00001.png", "./demo_data/dog-gooses/dog-gooses/00002.png"]  
    # imgs = load_and_preprocess_images(image_names).to(device)
    # # 测试完成

    # 加载模型
    model = VGGT.from_pretrained("facebook/VGGT-1B").eval().to(device)
    # 推理并处理可能的 OOM
    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype if device=='cuda' else None):
                outs = model(imgs, query_points=pts_tensor)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print("CUDA OOM")
        else:
            raise

    tracks = outs['track'][0].cpu().numpy()  # [S, 4, 2]
    # 准备输出路径
    base_dir = os.path.dirname(input_path) or '.'
    out_dir = output_dir or os.path.join(base_dir, 'tracking_result')
    os.makedirs(out_dir, exist_ok=True)
    out_video = os.path.join(out_dir, 'tracked_video.mp4')

    # 写入视频与保存帧
    fps = 10
    h, w = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
    )
    for i, frame_path in enumerate(image_names):
        frame = cv2.imread(frame_path)
        for (x, y) in tracks[i]:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), 2)
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
