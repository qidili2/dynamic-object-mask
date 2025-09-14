import torch
from sam2.build_sam import build_sam2_video_predictor
import os
import shutil
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from glob import glob
import json
import imageio
from imageio import get_writer
from collections import defaultdict
from sklearn.cluster import DBSCAN
import torchvision.transforms as T

DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"

class ObjectTracker:
    def __init__(self, predictor, check_interval=10, min_area_threshold=500):
        self.predictor = predictor
        self.check_interval = check_interval
        self.min_area_threshold = min_area_threshold
        self.next_obj_id = 1
        self.tracked_objects = {}  # obj_id -> last_seen_frame
        self.frame_count = 0
        
    def detect_new_objects(self, state, frame_idx, existing_masks=None, video_dir=None):
        """使用简化的方法检测新物体"""
        try:
            # 读取当前帧图像
            frame_names = sorted([
                os.path.splitext(p)[0]
                for p in os.listdir(video_dir)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
            ])
            
            img_ext = os.listdir(video_dir)[0]
            suffix = os.path.splitext(img_ext)[-1]
            current_frame_path = os.path.join(video_dir, f"{frame_names[frame_idx]}{suffix}")
            
            image = cv2.imread(current_frame_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            
            # 使用网格采样和颜色聚类来寻找新的候选点
            grid_points = grid_sample_points(height, width, grid_size=80)
            candidate_points = cluster_points_by_color(image_rgb, grid_points, eps=50, min_samples=3)
            
            if len(candidate_points) == 0:
                return []
            
            new_objects = []
            
            # 检查每个候选点是否在现有mask区域之外
            for point in candidate_points:
                x, y = int(point[0]), int(point[1])
                if x < 0 or x >= width or y < 0 or y >= height:
                    continue
                
                # 检查该点是否已经被现有物体覆盖
                is_covered = False
                if existing_masks is not None:
                    for existing_mask in existing_masks.values():
                        if isinstance(existing_mask, torch.Tensor):
                            mask_array = existing_mask.cpu().numpy()
                        else:
                            mask_array = existing_mask
                        
                        if mask_array.ndim > 2:
                            mask_array = mask_array.reshape(height, width)
                        
                        if y < mask_array.shape[0] and x < mask_array.shape[1]:
                            if mask_array[y, x] > 0:
                                is_covered = True
                                break
                
                if not is_covered:
                    center_point = np.array([[x, y]])
                    new_objects.append({
                        'center_point': center_point,
                        'area': self.min_area_threshold  # 估计面积
                    })
            
            # 限制新物体数量，避免过度检测
            return new_objects[:3]
            
        except Exception as e:
            print(f"Error in detect_new_objects: {e}")
            return []
    
    def find_mask_center(self, mask):
        """找到mask的中心点"""
        y_coords, x_coords = np.where(mask)
        if len(x_coords) == 0:
            return np.array([[0, 0]])
        
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        return np.array([[center_x, center_y]])
    
    def compute_iou(self, mask1, mask2):
        """计算两个mask的IoU"""
        if isinstance(mask1, torch.Tensor):
            mask1 = mask1.cpu().numpy()
        if isinstance(mask2, torch.Tensor):
            mask2 = mask2.cpu().numpy()
            
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0
    
    def should_check_new_objects(self):
        """判断是否需要检查新物体"""
        return self.frame_count % self.check_interval == 0
    
    def add_new_object(self, state, frame_idx, center_point):
        """添加新物体到追踪器"""
        obj_id = self.next_obj_id
        self.next_obj_id += 1
        
        labels = np.array([1], np.int32)
        
        try:
            _, _, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=center_point,
                labels=labels,
            )
            
            self.tracked_objects[obj_id] = frame_idx
            print(f"Added new object {obj_id} at frame {frame_idx}")
            return obj_id, out_mask_logits
            
        except Exception as e:
            print(f"Error adding new object: {e}")
            return None, None
    
    def update_frame_count(self):
        """更新帧计数"""
        self.frame_count += 1

def grid_sample_points(height, width, grid_size=50):
    """在图像上网格采样点"""
    points = []
    for y in range(grid_size//2, height, grid_size):
        for x in range(grid_size//2, width, grid_size):
            points.append([x, y])
    return np.array(points)

def cluster_points_by_color(image, points, eps=30, min_samples=5):
    """基于颜色聚类点"""
    if len(points) == 0:
        return []
    
    # 获取点的颜色特征
    colors = []
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            color = image[y, x]
            colors.append(color)
        else:
            colors.append([0, 0, 0])
    
    colors = np.array(colors)
    
    # DBSCAN聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(colors)
    labels = clustering.labels_
    
    # 提取每个聚类的代表点
    representative_points = []
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:  # 忽略噪声点
            continue
        cluster_points = points[labels == label]
        # 选择聚类中心作为代表点
        center = np.mean(cluster_points, axis=0)
        representative_points.append(center)
    
    return np.array(representative_points) if representative_points else np.array([])

def initialize_objects_from_first_frame(predictor, state, frame_idx, image_path, min_objects=3):
    """从第一帧初始化物体"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    
    # 网格采样点
    grid_points = grid_sample_points(height, width, grid_size=60)
    
    # 基于颜色聚类得到候选物体中心
    candidate_points = cluster_points_by_color(image_rgb, grid_points, eps=40, min_samples=3)
    
    if len(candidate_points) == 0:
        # 如果聚类失败，使用简单的网格点
        step = max(height, width) // 4
        candidate_points = np.array([
            [width//4, height//4],
            [3*width//4, height//4],
            [width//2, height//2],
            [width//4, 3*height//4],
            [3*width//4, 3*height//4]
        ])
    
    # 确保至少有最小数量的物体
    if len(candidate_points) < min_objects:
        # 添加更多点
        additional_points = []
        for i in range(min_objects - len(candidate_points)):
            x = np.random.randint(50, width-50)
            y = np.random.randint(50, height-50)
            additional_points.append([x, y])
        candidate_points = np.vstack([candidate_points, additional_points])
    
    initialized_objects = []
    obj_id = 1
    
    for point in candidate_points[:8]:  # 限制最多8个初始物体
        center_point = np.expand_dims(point, axis=0)
        labels = np.array([1], np.int32)
        
        try:
            _, _, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=center_point,
                labels=labels,
            )
            
            # 检查生成的mask是否有效
            mask = out_mask_logits[0] > 0.0
            if mask.sum() > 100:  # 确保mask不是太小
                initialized_objects.append(obj_id)
                print(f"Initialized object {obj_id} at point {point}")
                obj_id += 1
            else:
                # 如果mask太小，移除这个物体
                predictor.remove_object(state, obj_id)
                
        except Exception as e:
            print(f"Error initializing object at {point}: {e}")
            continue
    
    return initialized_objects, obj_id

def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        if isinstance(object_mask, torch.Tensor):
            object_mask = object_mask.cpu().numpy()
        object_mask = object_mask.reshape(height, width)
        mask[object_mask > 0] = object_id
    return mask

def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)

def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette

def save_multi_masks_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    output_palette,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
    output_mask_path = os.path.join(
        output_mask_dir, video_name, f"{frame_name}.png"
    )
    save_ann_png(output_mask_path, output_mask, output_palette)

def apply_mask_to_rgb(rgb_image, mask_image):
    """Apply mask to RGB image"""
    masked_rgb = np.zeros_like(rgb_image)
    masked_rgb[mask_image > 0] = rgb_image[mask_image > 0]
    return masked_rgb

def save_video_from_images(rgb_images, mask_images, video_dir, fps=30):
    """Save video from RGB and mask images"""
    assert len(rgb_images) > 0 and len(mask_images) > 0, "Image lists cannot be empty"
    
    height, width, _ = rgb_images[0].shape
    os.makedirs(video_dir, exist_ok=True)
    
    rgb_video_path = os.path.join(video_dir, "original_rgb.mp4")
    mask_video_path = os.path.join(video_dir, "mask.mp4")
    mask_rgb_video_path = os.path.join(video_dir, "mask_rgb.mp4")
    mask_rgb_color_video_path = os.path.join(video_dir, "mask_rgb_color.mp4")

    rgb_writer = get_writer(rgb_video_path, fps=fps)
    mask_writer = get_writer(mask_video_path, fps=fps)
    mask_rgb_writer = get_writer(mask_rgb_video_path, fps=fps)
    mask_rgb_color_writer = get_writer(mask_rgb_color_video_path, fps=fps)

    for rgb_img, mask_img in zip(rgb_images, mask_images):
        # Prepare mask image with white background
        mask_img_white_bg = np.ones_like(rgb_img) * 255
        mask_img_white_bg[mask_img > 0] = rgb_img[mask_img > 0]

        # Colored transparent overlay
        colored_mask = rgb_img.copy()
        overlay_color = np.array([0, 255, 0], dtype=np.uint8)
        alpha = 0.5

        colored_mask[mask_img > 0] = (
            alpha * overlay_color + (1 - alpha) * rgb_img[mask_img > 0]
        ).astype(np.uint8)

        rgb_writer.append_data(rgb_img)
        mask_writer.append_data(mask_img_white_bg)
        mask_rgb_writer.append_data(mask_img_white_bg)
        mask_rgb_color_writer.append_data(colored_mask)

    print(f'Videos saved to {video_dir}!')
    rgb_writer.close()
    mask_writer.close()
    mask_rgb_writer.close()
    mask_rgb_color_writer.close()

def main(args):
    video_name = os.path.basename(args.video_dir)
    output_mask_dir = args.output_mask_dir
    
    # 初始化SAM2
    checkpoint = "sam2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    
    # 获取帧名称
    frame_names = sorted([
        os.path.splitext(p)[0]
        for p in os.listdir(args.video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
    ])
    
    # 获取图像尺寸
    img_ext = os.listdir(args.video_dir)[0]
    suffix = os.path.splitext(img_ext)[-1]
    img_path = os.path.join(args.video_dir, img_ext)
    with Image.open(img_path) as img:
        width, height = img.size
    
    # 初始化物体追踪器
    tracker = ObjectTracker(predictor, check_interval=args.check_interval)
    
    video_segments = {}
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(args.video_dir)
        
        # 从第一帧初始化物体
        first_frame_path = os.path.join(args.video_dir, f"{frame_names[0]}{suffix}")
        initialized_objects, next_obj_id = initialize_objects_from_first_frame(
            predictor, state, 0, first_frame_path, min_objects=args.min_objects
        )
        
        tracker.next_obj_id = next_obj_id
        for obj_id in initialized_objects:
            tracker.tracked_objects[obj_id] = 0
        
        print(f"Initialized {len(initialized_objects)} objects: {initialized_objects}")
        
        # 传播分割到所有帧
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            if out_frame_idx not in video_segments:
                video_segments[out_frame_idx] = {}
            
            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                
                # 过滤掉太小的mask
                if mask.sum() > args.min_area_threshold:
                    video_segments[out_frame_idx][out_obj_id] = mask
                    tracker.tracked_objects[out_obj_id] = out_frame_idx
            
            tracker.update_frame_count()
            
            # 定期检查新物体
            if tracker.should_check_new_objects() and out_frame_idx > 0:
                print(f"Checking for new objects at frame {out_frame_idx}")
                
                existing_masks = video_segments.get(out_frame_idx, {})
                new_objects = tracker.detect_new_objects(state, out_frame_idx, existing_masks, args.video_dir)
                
                # 添加新发现的物体
                for new_obj in new_objects[:2]:  # 限制每次最多添加2个新物体
                    obj_id, mask_logits = tracker.add_new_object(
                        state, out_frame_idx, new_obj['center_point']
                    )
                    
                    if obj_id is not None and mask_logits is not None:
                        # 将新物体的mask添加到当前帧
                        mask = (mask_logits[0] > 0.0).cpu().numpy()
                        if mask.sum() > args.min_area_threshold:
                            video_segments[out_frame_idx][obj_id] = mask
                            
                        # 继续传播到后续帧
                        try:
                            for prop_frame_idx, prop_obj_ids, prop_mask_logits in predictor.propagate_in_video(
                                state, start_frame_idx=out_frame_idx
                            ):
                                if prop_frame_idx not in video_segments:
                                    video_segments[prop_frame_idx] = {}
                                
                                for j, prop_obj_id in enumerate(prop_obj_ids):
                                    if prop_obj_id == obj_id and prop_frame_idx > out_frame_idx:
                                        mask = (prop_mask_logits[j] > 0.0).cpu().numpy()
                                        if mask.sum() > args.min_area_threshold:
                                            video_segments[prop_frame_idx][prop_obj_id] = mask
                        except Exception as e:
                            print(f"Error propagating new object {obj_id}: {e}")
                            continue
    
    # 保存结果
    video_segments = dict(sorted(video_segments.items()))
    
    save_dirname = os.path.join(output_mask_dir, video_name)
    if os.path.exists(save_dirname):
        shutil.rmtree(save_dirname)
    
    os.makedirs(output_mask_dir, exist_ok=True)
    
    # 保存mask PNG文件
    for out_frame_idx, per_obj_output_mask in video_segments.items():
        if out_frame_idx < len(frame_names):
            save_multi_masks_to_dir(
                output_mask_dir=output_mask_dir,
                video_name=video_name,
                frame_name=frame_names[out_frame_idx],
                per_obj_output_mask=per_obj_output_mask,
                height=height,
                width=width,
                output_palette=DAVIS_PALETTE,
            )
    
    # 生成视频
    final_path = os.path.join(output_mask_dir, "final_res", video_name)
    os.makedirs(final_path, exist_ok=True)
    
    # 加载所有masks
    seq_mask_dir = os.path.join(output_mask_dir, video_name)
    all_masks = []
    rgbs = []
    
    for frame_name in frame_names:
        # 加载mask
        mask_path = os.path.join(seq_mask_dir, f'{frame_name}.png')
        if os.path.exists(mask_path):
            mask_img, _ = load_ann_png(mask_path)
        else:
            mask_img = np.zeros((height, width), dtype=np.uint8)
        
        mask_img = (mask_img > 0).astype(np.uint8)
        all_masks.append(mask_img)
        
        # 加载RGB
        rgb_path = os.path.join(args.video_dir, f'{frame_name}{suffix}')
        rgb_image = cv2.imread(rgb_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgbs.append(rgb_image)
    
    # 保存视频
    video_save_dir = os.path.join(final_path, "video")
    save_video_from_images(rgbs, all_masks, video_save_dir)
    
    print(f"Processing completed! Results saved to {output_mask_dir}")
    print(f"Total objects tracked: {len(tracker.tracked_objects)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Auto-track all objects in video using SAM2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--output_mask_dir', type=str, required=True,
                       help="Directory to save output masks")
    parser.add_argument('--video_dir', type=str, required=True,
                       help="Directory containing input video frames")
    parser.add_argument('--check_interval', type=int, default=10,
                       help="Interval (in frames) to check for new objects")
    parser.add_argument('--min_objects', type=int, default=3,
                       help="Minimum number of objects to initialize")
    parser.add_argument('--min_area_threshold', type=int, default=500,
                       help="Minimum area threshold for valid objects")
    
    args = parser.parse_args()
    main(args)