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
from sklearn.neighbors import NearestNeighbors

DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"

def generate_grid_points(height, width, grid_size=50):
    """Generate a grid of points across the image for initial segmentation"""
    points = []
    for y in range(grid_size//2, height, grid_size):
        for x in range(grid_size//2, width, grid_size):
            points.append([x, y])
    return np.array(points)

def generate_adaptive_points(image, num_points=20):
    """Generate points based on image features (corners, edges)"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect corners using goodFeaturesToTrack
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=num_points//2, qualityLevel=0.01, minDistance=30)
    
    # Detect edges and sample points
    edges = cv2.Canny(gray, 50, 150)
    edge_points = np.column_stack(np.where(edges > 0))
    
    # Sample edge points
    if len(edge_points) > num_points//2:
        indices = np.random.choice(len(edge_points), num_points//2, replace=False)
        edge_points = edge_points[indices]
        edge_points = edge_points[:, [1, 0]]  # Swap x, y coordinates
    else:
        edge_points = edge_points[:, [1, 0]] if len(edge_points) > 0 else np.array([])
    
    # Combine corner and edge points
    if corners is not None:
        corners = corners.reshape(-1, 2)
        if len(edge_points) > 0:
            points = np.vstack([corners, edge_points])
        else:
            points = corners
    else:
        points = edge_points if len(edge_points) > 0 else np.array([[100, 100]])
    
    return points.astype(int)

def cluster_points_by_mask(points, masks, min_points_per_cluster=3):
    """Cluster points based on which masks they belong to"""
    clusters = {}
    
    for i, mask in enumerate(masks):
        # Find points that fall within this mask
        points_in_mask = []
        for j, point in enumerate(points):
            x, y = int(point[0]), int(point[1])
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                if mask[y, x] > 0:  # Point is inside the mask
                    points_in_mask.append(j)
        
        if len(points_in_mask) >= min_points_per_cluster:
            clusters[i] = points_in_mask
    
    return clusters

def compute_mask_features(mask):
    """Compute features for a mask to help with object identification"""
    # Compute centroid
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0:
        return None
    
    centroid = np.array([np.mean(x_indices), np.mean(y_indices)])
    
    # Compute area
    area = np.sum(mask > 0)
    
    # Compute bounding box
    min_x, max_x = np.min(x_indices), np.max(x_indices)
    min_y, max_y = np.min(y_indices), np.max(y_indices)
    bbox = np.array([min_x, min_y, max_x, max_y])
    
    # Compute moments for shape description
    moments = cv2.moments(mask.astype(np.uint8))
    hu_moments = cv2.HuMoments(moments).flatten()
    
    return {
        'centroid': centroid,
        'area': area,
        'bbox': bbox,
        'hu_moments': hu_moments
    }

def match_objects_across_frames(prev_features, curr_features, max_distance=100):
    """Match objects across frames based on their features"""
    matches = {}
    used_curr = set()
    
    # Sort by area to prioritize larger objects
    prev_items = sorted(prev_features.items(), key=lambda x: x[1]['area'], reverse=True)
    
    for prev_id, prev_feat in prev_items:
        best_match = None
        best_score = float('inf')
        
        for curr_id, curr_feat in curr_features.items():
            if curr_id in used_curr:
                continue
                
            # Compute distance based on centroid and area
            centroid_dist = np.linalg.norm(prev_feat['centroid'] - curr_feat['centroid'])
            area_ratio = abs(prev_feat['area'] - curr_feat['area']) / max(prev_feat['area'], curr_feat['area'])
            
            # Compute shape similarity using Hu moments
            hu_dist = np.linalg.norm(prev_feat['hu_moments'] - curr_feat['hu_moments'])
            
            # Combined score
            score = centroid_dist + area_ratio * 100 + hu_dist * 10
            
            if score < best_score and centroid_dist < max_distance:
                best_score = score
                best_match = curr_id
        
        if best_match is not None:
            matches[prev_id] = best_match
            used_curr.add(best_match)
    
    return matches

def detect_new_objects_in_frame(predictor, state, frame_idx, image, existing_masks, method='adaptive', **kwargs):
    """Detect new objects in a frame, excluding existing tracked objects"""
    height, width = image.shape[:2]
    
    # Generate candidate points
    if method == 'grid':
        grid_size = kwargs.get('grid_size', 50)
        points = generate_grid_points(height, width, grid_size)
    elif method == 'adaptive':
        num_points = kwargs.get('num_points', 30)
        points = generate_adaptive_points(image, num_points)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
    
    # Filter out points that fall within existing masks
    if existing_masks:
        combined_existing_mask = np.zeros((height, width), dtype=bool)
        for mask in existing_masks.values():
            if mask.ndim == 3:
                mask = mask[0]  # Remove batch dimension if present
            combined_existing_mask |= (mask > 0)
        
        # Keep only points outside existing masks
        valid_points = []
        for point in points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < width and 0 <= y < height:
                if not combined_existing_mask[y, x]:
                    valid_points.append(point)
        
        if len(valid_points) == 0:
            return [], []
        
        points = np.array(valid_points)
    
    # Process points in smaller batches for new object detection
    batch_size = 5
    new_masks = []
    used_points = []
    temp_obj_id = 1000  # Use high ID to avoid conflicts
    
    for i in range(0, len(points), batch_size):
        batch_points = points[i:i+batch_size]
        labels = np.ones(len(batch_points), dtype=np.int32)
        
        try:
            predictor.reset_state(state)  # Reset state for new object detection
            
            _, _, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=temp_obj_id,
                points=batch_points,
                labels=labels,
            )
            
            # Convert logits to binary masks
            masks = (out_mask_logits > 0.0).cpu().numpy()
            
            # Filter masks that don't overlap significantly with existing objects
            for j, mask in enumerate(masks):
                if mask.ndim == 3:
                    mask = mask[0]
                
                # Check overlap with existing masks
                is_new_object = True
                if existing_masks:
                    for existing_mask in existing_masks.values():
                        if existing_mask.ndim == 3:
                            existing_mask = existing_mask[0]
                        
                        intersection = np.logical_and(mask > 0, existing_mask > 0).sum()
                        mask_area = (mask > 0).sum()
                        
                        if mask_area > 0:
                            overlap_ratio = intersection / mask_area
                            if overlap_ratio > 0.5:  # If more than 50% overlap, not a new object
                                is_new_object = False
                                break
                
                if is_new_object and mask.sum() > kwargs.get('min_mask_area', 100):
                    new_masks.append(mask)
                    used_points.append(batch_points[j])
            
            temp_obj_id += 1
            
        except Exception as e:
            print(f"Error detecting new objects in batch {i}: {e}")
            continue
    
    return new_masks, used_points

def save_mask_with_points(mask, points, save_path, point_color='red', point_size=20):
    """Visualize mask with points"""
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(mask_image)

    if len(points) > 0:
        x_coords, y_coords = points[:, 0], points[:, 1]
        ax.scatter(x_coords, y_coords, color=point_color, s=point_size)

    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

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

def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask.keys(), reverse=True)
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask

def save_video_from_images(rgb_images, mask_images, video_dir, fps=30):
    """Save RGB and mask images as videos"""
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

        # Colored transparent overlay for mask_rgb_color
        colored_mask = rgb_img.copy()
        overlay_color = np.array([0, 255, 0], dtype=np.uint8)  # Green color for mask
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

def segment_first_frame_sam2(predictor, state, frame_idx, image, method='adaptive', **kwargs):
    """Segment the first frame using various methods"""
    height, width = image.shape[:2]
    
    if method == 'grid':
        grid_size = kwargs.get('grid_size', 50)
        points = generate_grid_points(height, width, grid_size)
    elif method == 'adaptive':
        num_points = kwargs.get('num_points', 30)
        points = generate_adaptive_points(image, num_points)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
    
    # Process points in batches to avoid memory issues
    batch_size = 10
    all_masks = []
    obj_id = 1
    
    for i in range(0, len(points), batch_size):
        batch_points = points[i:i+batch_size]
        labels = np.ones(len(batch_points), dtype=np.int32)
        
        try:
            _, _, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=batch_points,
                labels=labels,
            )
            
            # Convert logits to binary masks
            masks = (out_mask_logits > 0.0).cpu().numpy()
            all_masks.extend(masks)
            obj_id += 1
            
        except Exception as e:
            print(f"Error processing point batch {i}: {e}")
            continue
    
    return all_masks, points

def process_video_segmentation(args):
    """Main function to process video segmentation"""
    
    # Initialize SAM2 predictor
    checkpoint = "sam2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    
    # Get frame information
    frame_names = sorted([
        os.path.splitext(p)[0]
        for p in os.listdir(args.video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
    ])
    
    if not frame_names:
        raise ValueError(f"No valid image files found in {args.video_dir}")
    
    # Get image dimensions
    img_ext = next(f for f in os.listdir(args.video_dir) 
                   if os.path.splitext(f)[-1].lower() in [".jpg", ".jpeg", ".png"])
    img_path = os.path.join(args.video_dir, img_ext)
    with Image.open(img_path) as img:
        width, height = img.size
    
    print(f"Processing {len(frame_names)} frames of size {width}x{height}")
    
    # Initialize video state
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(args.video_dir)
        
        # Process first frame
        first_frame_path = os.path.join(args.video_dir, f"{frame_names[0]}.{img_ext.split('.')[-1]}")
        first_frame = cv2.imread(first_frame_path)
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        
        print("Segmenting first frame...")
        masks, points = segment_first_frame_sam2(
            predictor, state, 0, first_frame, 
            method=args.method, 
            grid_size=args.grid_size,
            num_points=args.num_points
        )
        
        if args.visualize:
            vis_dir = os.path.join(args.output_dir, "visualization")
            os.makedirs(vis_dir, exist_ok=True)
            for i, mask in enumerate(masks[:5]):  # Visualize first 5 masks
                save_mask_with_points(
                    mask, points[i:i+1] if i < len(points) else np.array([]), 
                    os.path.join(vis_dir, f"first_frame_mask_{i}.png")
                )
        
        # Filter out small or invalid masks
        min_area = args.min_mask_area
        valid_masks = []
        valid_obj_ids = []
        
        for i, mask in enumerate(masks):
            if mask.sum() > min_area:
                valid_masks.append(mask)
                valid_obj_ids.append(i + 1)
        
        print(f"Found {len(valid_masks)} valid objects in first frame")
        
        if len(valid_masks) == 0:
            print("No valid objects found in first frame!")
            return
        
        # Propagate through video with periodic new object detection
        print("Propagating through video...")
        video_segments = {}
        prev_features = {}
        max_obj_id = 0
        detection_interval = args.detection_interval  # Every N frames
        
        # Initialize features for first frame
        for obj_id, mask in zip(valid_obj_ids, valid_masks):
            features = compute_mask_features(mask[0])  # Remove batch dimension
            if features is not None:
                prev_features[obj_id] = features
                max_obj_id = max(max_obj_id, obj_id)
        
        # Store initial state for new object detection
        detection_state = predictor.init_state(args.video_dir)
        
        # Propagate forward
        frame_count = 0
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            frame_masks = {}
            curr_features = {}
            
            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                if mask.sum() > min_area:
                    features = compute_mask_features(mask[0])
                    if features is not None:
                        curr_features[out_obj_id] = features
                        frame_masks[out_obj_id] = mask
            
            # Detect new objects every N frames
            if frame_count % detection_interval == 0 and frame_count > 0:
                print(f"Detecting new objects at frame {out_frame_idx}...")
                
                # Load current frame image
                frame_name = frame_names[out_frame_idx] if out_frame_idx < len(frame_names) else frame_names[-1]
                for ext in ['jpg', 'jpeg', 'png']:
                    frame_path = os.path.join(args.video_dir, f"{frame_name}.{ext}")
                    if os.path.exists(frame_path):
                        current_frame = cv2.imread(frame_path)
                        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                        break
                
                # Detect new objects
                new_masks, new_points = detect_new_objects_in_frame(
                    predictor, detection_state, out_frame_idx, current_frame, 
                    frame_masks, method=args.method,
                    grid_size=args.grid_size, num_points=args.num_points//2,  # Use fewer points for new detection
                    min_mask_area=min_area
                )
                
                # Add new objects to tracking
                for new_mask in new_masks:
                    max_obj_id += 1
                    new_features = compute_mask_features(new_mask)
                    if new_features is not None:
                        frame_masks[max_obj_id] = np.expand_dims(new_mask, axis=0)  # Add batch dimension
                        curr_features[max_obj_id] = new_features
                        
                        # Add new object to main tracking state
                        try:
                            # Find a good point for the new object
                            y_indices, x_indices = np.where(new_mask > 0)
                            if len(x_indices) > 0:
                                # Use centroid as the point
                                center_x = int(np.mean(x_indices))
                                center_y = int(np.mean(y_indices))
                                new_point = np.array([[center_x, center_y]])
                                
                                _, _, _ = predictor.add_new_points_or_box(
                                    inference_state=state,
                                    frame_idx=out_frame_idx,
                                    obj_id=max_obj_id,
                                    points=new_point,
                                    labels=np.array([1], np.int32),
                                )
                                print(f"Added new object {max_obj_id} at frame {out_frame_idx}")
                        except Exception as e:
                            print(f"Failed to add new object {max_obj_id}: {e}")
            
            # Match objects with previous frame
            if prev_features and curr_features:
                matches = match_objects_across_frames(prev_features, curr_features)
                
                # Create new mapping based on matches
                matched_masks = {}
                matched_features = {}
                
                for prev_id, curr_id in matches.items():
                    if curr_id in frame_masks:
                        matched_masks[prev_id] = frame_masks[curr_id]
                        matched_features[prev_id] = curr_features[curr_id]
                
                # Add unmatched existing objects with new IDs (these might be re-appeared objects)
                for curr_id, mask in frame_masks.items():
                    if curr_id not in matches.values() and curr_id not in matched_masks:
                        # Check if this might be a re-appeared object
                        best_match_id = None
                        best_score = float('inf')
                        
                        for prev_id, prev_feat in prev_features.items():
                            if prev_id not in matched_masks:
                                curr_feat = curr_features[curr_id]
                                centroid_dist = np.linalg.norm(prev_feat['centroid'] - curr_feat['centroid'])
                                area_ratio = abs(prev_feat['area'] - curr_feat['area']) / max(prev_feat['area'], curr_feat['area'])
                                score = centroid_dist + area_ratio * 100
                                
                                if score < best_score and centroid_dist < 200:  # More lenient for re-appearance
                                    best_score = score
                                    best_match_id = prev_id
                        
                        if best_match_id is not None:
                            matched_masks[best_match_id] = mask
                            matched_features[best_match_id] = curr_features[curr_id]
                        else:
                            # Truly new object, assign new ID
                            max_obj_id += 1
                            matched_masks[max_obj_id] = mask
                            matched_features[max_obj_id] = curr_features[curr_id]
                
                video_segments[out_frame_idx] = matched_masks
                prev_features = matched_features
            else:
                video_segments[out_frame_idx] = frame_masks
                prev_features = curr_features
            
            frame_count += 1
        
        # Save results
        output_dir = os.path.join(args.output_dir, "masks")
        os.makedirs(output_dir, exist_ok=True)
        
        video_name = os.path.basename(args.video_dir)
        save_dir = os.path.join(output_dir, video_name)
        
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save mask images
        for frame_idx, per_obj_masks in video_segments.items():
            if frame_idx < len(frame_names):
                combined_mask = put_per_obj_mask(per_obj_masks, height, width)
                mask_path = os.path.join(save_dir, f"{frame_names[frame_idx]}.png")
                save_ann_png(mask_path, combined_mask, DAVIS_PALETTE)
        
        print(f"Masks saved to {save_dir}")
        
        # Generate videos if requested
        if args.generate_video:
            print("Generating visualization videos...")
            
            rgb_images = []
            mask_images = []
            
            for frame_name in frame_names:
                # Load RGB image
                for ext in ['jpg', 'jpeg', 'png']:
                    rgb_path = os.path.join(args.video_dir, f"{frame_name}.{ext}")
                    if os.path.exists(rgb_path):
                        rgb_img = cv2.imread(rgb_path)
                        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                        rgb_images.append(rgb_img)
                        break
                
                # Load mask image
                mask_path = os.path.join(save_dir, f"{frame_name}.png")
                if os.path.exists(mask_path):
                    mask_img, _ = load_ann_png(mask_path)
                    mask_images.append((mask_img > 0).astype(np.uint8))
                else:
                    mask_images.append(np.zeros((height, width), dtype=np.uint8))
            
            video_output_dir = os.path.join(args.output_dir, "videos", video_name)
            save_video_from_images(rgb_images, mask_images, video_output_dir, fps=args.fps)

def main():
    parser = argparse.ArgumentParser(description='SAM2 Video Segmentation without Dynamic Trajectories')
    
    parser.add_argument('--video_dir', type=str, required=True,
                       help='Directory containing video frames')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory for results')
    parser.add_argument('--method', type=str, choices=['grid', 'adaptive'], default='adaptive',
                       help='Method for initial point generation')
    parser.add_argument('--grid_size', type=int, default=50,
                       help='Grid size for grid method')
    parser.add_argument('--num_points', type=int, default=30,
                       help='Number of points for adaptive method')
    parser.add_argument('--min_mask_area', type=int, default=100,
                       help='Minimum area for valid masks')
    parser.add_argument('--generate_video', action='store_true',
                       help='Generate visualization videos')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization images')
    parser.add_argument('--detection_interval', type=int, default=5,
                       help='Interval (in frames) for detecting new objects')
    parser.add_argument('--fps', type=int, default=30,
                       help='FPS for output videos')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_dir):
        raise ValueError(f"Video directory does not exist: {args.video_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    process_video_segmentation(args)
    print("Processing completed!")

if __name__ == "__main__":
    main()