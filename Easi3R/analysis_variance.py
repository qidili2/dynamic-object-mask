# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import defaultdict
# import cv2

# # Configuration
# base_dir = "/mnt/data0/andy/Easi3R/results/davis/easi3r_sam2_high_resolution"
# badcase_file = "/mnt/data0/andy/Easi3R/results/davis/davis17_badcase.txt"
# goodcase_file = "/mnt/data0/andy/Easi3R/results/davis/davis17_goodcase.txt"
# output_dir = "/mnt/data0/andy/Easi3R/results/davis/analysis_output"

# os.makedirs(output_dir, exist_ok=True)

# # Read sequence lists
# with open(badcase_file, 'r') as f:
#     bad_sequences = [line.strip() for line in f if line.strip()]

# with open(goodcase_file, 'r') as f:
#     good_sequences = [line.strip() for line in f if line.strip()]

# print(f"Found {len(bad_sequences)} bad cases and {len(good_sequences)} good cases")

# # Data collection
# bad_data = []
# good_data = []

# def load_sequence_data(sequence_name, case_type):
#     csv_path = os.path.join(base_dir, sequence_name, "object_statistics.csv")
    
#     if not os.path.exists(csv_path):
#         print(f"Warning: CSV not found for {sequence_name}")
#         return None
    
#     df = pd.read_csv(csv_path)
#     df['sequence'] = sequence_name
#     df['case_type'] = case_type
#     return df

# # Load all data
# for seq in bad_sequences:
#     df = load_sequence_data(seq, 'bad')
#     if df is not None:
#         bad_data.append(df)

# for seq in good_sequences:
#     df = load_sequence_data(seq, 'good')
#     if df is not None:
#         good_data.append(df)

# # Combine data
# bad_df = pd.concat(bad_data, ignore_index=True) if bad_data else pd.DataFrame()
# good_df = pd.concat(good_data, ignore_index=True) if good_data else pd.DataFrame()
# all_df = pd.concat([bad_df, good_df], ignore_index=True)

# print(f"\nTotal objects in bad cases: {len(bad_df)}")
# print(f"Total objects in good cases: {len(good_df)}")

# # ==================== Analysis 1 ====================
# # Bad case high mean attention vs good case variance
# print("\n" + "="*60)
# print("ANALYSIS 1: Bad case high mean attention objects")
# print("="*60)

# high_mean_threshold = 0.5
# bad_high_mean = bad_df[bad_df['attention_mean'] > high_mean_threshold]
# good_high_mean = good_df[good_df['attention_mean'] > high_mean_threshold]

# print(f"\nBad cases with attention_mean > {high_mean_threshold}:")
# print(f"  Count: {len(bad_high_mean)}")
# print(f"  Mean variance: {bad_high_mean['attention_variance'].mean():.6f}")
# print(f"  Median variance: {bad_high_mean['attention_variance'].median():.6f}")

# print(f"\nGood cases with attention_mean > {high_mean_threshold}:")
# print(f"  Count: {len(good_high_mean)}")
# print(f"  Mean variance: {good_high_mean['attention_variance'].mean():.6f}")
# print(f"  Median variance: {good_high_mean['attention_variance'].median():.6f}")

# # Statistical comparison
# if len(bad_high_mean) > 0 and len(good_high_mean) > 0:
#     from scipy import stats
#     t_stat, p_value = stats.ttest_ind(bad_high_mean['attention_variance'], 
#                                        good_high_mean['attention_variance'])
#     print(f"\nT-test result: t={t_stat:.4f}, p={p_value:.6f}")

# # ==================== Analysis 2 ====================
# # Good case high variance objects - are they small regions?
# print("\n" + "="*60)
# print("ANALYSIS 2: Good case high variance objects and region size")
# print("="*60)

# high_var_threshold = 0.05
# good_high_var = good_df[good_df['attention_variance'] > high_var_threshold]

# print(f"\nGood cases with attention_variance > {high_var_threshold}:")
# print(f"  Count: {len(good_high_var)}")
# print(f"  Mean pixel area: {good_high_var['avg_pixel_area'].mean():.2f}")
# print(f"  Median pixel area: {good_high_var['avg_pixel_area'].median():.2f}")

# # Compare with all good case objects
# print(f"\nAll good case objects:")
# print(f"  Mean pixel area: {good_df['avg_pixel_area'].mean():.2f}")
# print(f"  Median pixel area: {good_df['avg_pixel_area'].median():.2f}")

# # Correlation analysis
# corr = good_df[['attention_variance', 'avg_pixel_area']].corr()
# print(f"\nCorrelation between variance and pixel area in good cases: {corr.iloc[0,1]:.4f}")

# # ==================== Analysis 3 ====================
# # Bad case high mean + low variance objects - are these the real target objects?
# print("\n" + "="*60)
# print("ANALYSIS 3: Bad case high mean + low variance objects")
# print("="*60)

# low_var_threshold = 0.05  # Changed from 0.01 to 0.05
# bad_candidates = bad_df[(bad_df['attention_mean'] > high_mean_threshold) & 
#                         (bad_df['attention_variance'] < low_var_threshold)]

# print(f"\nBad cases with mean > {high_mean_threshold} AND variance < {low_var_threshold}:")
# print(f"  Count: {len(bad_candidates)}")

# if len(bad_candidates) > 0:
#     print(f"  Mean pixel area: {bad_candidates['avg_pixel_area'].mean():.2f}")
#     print(f"  Median pixel area: {bad_candidates['avg_pixel_area'].median():.2f}")
    
#     print("\nTop 10 candidate objects (likely real targets):")
#     top_candidates = bad_candidates.nlargest(10, 'avg_pixel_area')[
#         ['sequence', 'object_id', 'attention_mean', 'attention_variance', 'avg_pixel_area']
#     ]
#     print(top_candidates.to_string(index=False))

# # ==================== Visualization ====================
# print("\n" + "="*60)
# print("GENERATING VISUALIZATIONS")
# print("="*60)

# # Plot 1: Attention mean vs variance scatter plot
# fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# # Bad cases
# axes[0].scatter(bad_df['attention_mean'], bad_df['attention_variance'], 
#                 alpha=0.5, s=bad_df['avg_pixel_area']/100, c='red', label='Bad cases')
# axes[0].axvline(x=high_mean_threshold, color='blue', linestyle='--', alpha=0.7, label=f'Mean threshold={high_mean_threshold}')
# axes[0].axhline(y=low_var_threshold, color='green', linestyle='--', alpha=0.7, label=f'Var threshold={low_var_threshold}')
# axes[0].set_xlabel('Attention Mean')
# axes[0].set_ylabel('Attention Variance')
# axes[0].set_title('Bad Cases: Attention Mean vs Variance\n(size = pixel area)')
# axes[0].legend()
# axes[0].grid(True, alpha=0.3)

# # Good cases
# axes[1].scatter(good_df['attention_mean'], good_df['attention_variance'], 
#                 alpha=0.5, s=good_df['avg_pixel_area']/100, c='green', label='Good cases')
# axes[1].axvline(x=high_mean_threshold, color='blue', linestyle='--', alpha=0.7, label=f'Mean threshold={high_mean_threshold}')
# axes[1].axhline(y=high_var_threshold, color='orange', linestyle='--', alpha=0.7, label=f'Var threshold={high_var_threshold}')
# axes[1].set_xlabel('Attention Mean')
# axes[1].set_ylabel('Attention Variance')
# axes[1].set_title('Good Cases: Attention Mean vs Variance\n(size = pixel area)')
# axes[1].legend()
# axes[1].grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, '1_mean_vs_variance_comparison.png'), dpi=300, bbox_inches='tight')
# print(f"Saved: 1_mean_vs_variance_comparison.png")

# # Plot 2: Distribution comparison
# fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# # Attention mean distribution
# axes[0, 0].hist([bad_df['attention_mean'], good_df['attention_mean']], 
#                 bins=30, label=['Bad cases', 'Good cases'], alpha=0.6, color=['red', 'green'])
# axes[0, 0].axvline(x=high_mean_threshold, color='blue', linestyle='--', alpha=0.7)
# axes[0, 0].set_xlabel('Attention Mean')
# axes[0, 0].set_ylabel('Frequency')
# axes[0, 0].set_title('Attention Mean Distribution')
# axes[0, 0].legend()

# # Attention variance distribution
# axes[0, 1].hist([bad_df['attention_variance'], good_df['attention_variance']], 
#                 bins=30, label=['Bad cases', 'Good cases'], alpha=0.6, color=['red', 'green'])
# axes[0, 1].axvline(x=high_var_threshold, color='orange', linestyle='--', alpha=0.7)
# axes[0, 1].set_xlabel('Attention Variance')
# axes[0, 1].set_ylabel('Frequency')
# axes[0, 1].set_title('Attention Variance Distribution')
# axes[0, 1].legend()

# # Pixel area distribution
# axes[1, 0].hist([bad_df['avg_pixel_area'], good_df['avg_pixel_area']], 
#                 bins=30, label=['Bad cases', 'Good cases'], alpha=0.6, color=['red', 'green'])
# axes[1, 0].set_xlabel('Average Pixel Area')
# axes[1, 0].set_ylabel('Frequency')
# axes[1, 0].set_title('Pixel Area Distribution')
# axes[1, 0].legend()

# # Box plot comparison
# data_to_plot = [
#     bad_df['attention_variance'], good_df['attention_variance'],
#     bad_high_mean['attention_variance'] if len(bad_high_mean) > 0 else [],
#     good_high_mean['attention_variance'] if len(good_high_mean) > 0 else []
# ]
# labels = ['Bad\n(all)', 'Good\n(all)', f'Bad\n(mean>{high_mean_threshold})', f'Good\n(mean>{high_mean_threshold})']
# bp = axes[1, 1].boxplot([d for d in data_to_plot if len(d) > 0], 
#                         labels=[l for i, l in enumerate(labels) if len(data_to_plot[i]) > 0],
#                         patch_artist=True)
# for patch, color in zip(bp['boxes'], ['red', 'green', 'darkred', 'darkgreen'][:len(bp['boxes'])]):
#     patch.set_facecolor(color)
#     patch.set_alpha(0.6)
# axes[1, 1].set_ylabel('Attention Variance')
# axes[1, 1].set_title('Variance Comparison (Boxplot)')
# axes[1, 1].grid(True, alpha=0.3, axis='y')

# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, '2_distribution_comparison.png'), dpi=300, bbox_inches='tight')
# print(f"Saved: 2_distribution_comparison.png")

# # Plot 3: Variance vs Pixel Area (for good cases)
# plt.figure(figsize=(10, 8))
# plt.scatter(good_df['avg_pixel_area'], good_df['attention_variance'], alpha=0.5, c='green')
# plt.axhline(y=high_var_threshold, color='orange', linestyle='--', alpha=0.7, 
#             label=f'Var threshold={high_var_threshold}')
# plt.xlabel('Average Pixel Area')
# plt.ylabel('Attention Variance')
# plt.title('Good Cases: Variance vs Pixel Area')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, '3_variance_vs_pixel_area.png'), dpi=300, bbox_inches='tight')
# print(f"Saved: 3_variance_vs_pixel_area.png")

# # Save summary statistics to CSV
# summary_stats = pd.DataFrame({
#     'Category': [
#         'Bad - All', 'Bad - High Mean', 'Bad - High Mean & Low Var',
#         'Good - All', 'Good - High Mean', 'Good - High Var'
#     ],
#     'Count': [
#         len(bad_df), len(bad_high_mean), len(bad_candidates),
#         len(good_df), len(good_high_mean), len(good_high_var)
#     ],
#     'Mean_Attention_Mean': [
#         bad_df['attention_mean'].mean(), bad_high_mean['attention_mean'].mean() if len(bad_high_mean) > 0 else 0,
#         bad_candidates['attention_mean'].mean() if len(bad_candidates) > 0 else 0,
#         good_df['attention_mean'].mean(), good_high_mean['attention_mean'].mean() if len(good_high_mean) > 0 else 0,
#         good_high_var['attention_mean'].mean() if len(good_high_var) > 0 else 0
#     ],
#     'Mean_Attention_Variance': [
#         bad_df['attention_variance'].mean(), bad_high_mean['attention_variance'].mean() if len(bad_high_mean) > 0 else 0,
#         bad_candidates['attention_variance'].mean() if len(bad_candidates) > 0 else 0,
#         good_df['attention_variance'].mean(), good_high_mean['attention_variance'].mean() if len(good_high_mean) > 0 else 0,
#         good_high_var['attention_variance'].mean() if len(good_high_var) > 0 else 0
#     ],
#     'Mean_Pixel_Area': [
#         bad_df['avg_pixel_area'].mean(), bad_high_mean['avg_pixel_area'].mean() if len(bad_high_mean) > 0 else 0,
#         bad_candidates['avg_pixel_area'].mean() if len(bad_candidates) > 0 else 0,
#         good_df['avg_pixel_area'].mean(), good_high_mean['avg_pixel_area'].mean() if len(good_high_mean) > 0 else 0,
#         good_high_var['avg_pixel_area'].mean() if len(good_high_var) > 0 else 0
#     ]
# })

# summary_stats.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)
# print(f"Saved: summary_statistics.csv")

# # Save candidate objects for visualization
# if len(bad_candidates) > 0:
#     bad_candidates.to_csv(os.path.join(output_dir, 'bad_case_candidate_objects.csv'), index=False)
#     print(f"Saved: bad_case_candidate_objects.csv")

# print(f"\n" + "="*60)
# print(f"Analysis complete! Results saved to: {output_dir}")
# print("="*60)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import cv2
from pathlib import Path

# Configuration
base_dir = "/mnt/data0/andy/Easi3R/results/davis/easi3r_sam2_high_resolution"
badcase_file = "/mnt/data0/andy/Easi3R/results/davis/davis17_badcase.txt"
goodcase_file = "/mnt/data0/andy/Easi3R/results/davis/davis17_goodcase.txt"
output_dir_gt_inside = "/mnt/data0/andy/Easi3R/results/davis/analysis_output_gt_inside"
output_dir_gt_outside = "/mnt/data0/andy/Easi3R/results/davis/analysis_output_gt_outside"
gt_mask_dir = "/mnt/data0/andy/Easi3R/DAVIS/Annotations/480p"

os.makedirs(output_dir_gt_inside, exist_ok=True)
os.makedirs(output_dir_gt_outside, exist_ok=True)

# Read sequence lists
with open(badcase_file, 'r') as f:
    bad_sequences = [line.strip() for line in f if line.strip()]

with open(goodcase_file, 'r') as f:
    good_sequences = [line.strip() for line in f if line.strip()]

print(f"Found {len(bad_sequences)} bad cases and {len(good_sequences)} good cases")

def load_gt_masks(sequence_name):
    """Load GT masks for a sequence and return object IDs"""
    gt_seq_dir = os.path.join(gt_mask_dir, sequence_name)
    if not os.path.exists(gt_seq_dir):
        print(f"Warning: GT mask directory not found for {sequence_name}")
        return None, []
    
    mask_files = sorted(list(Path(gt_seq_dir).glob("*.png")))
    if not mask_files:
        print(f"Warning: No GT masks found for {sequence_name}")
        return None, []
    
    gt_masks = []
    for mask_file in mask_files:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not load {mask_file}")
            continue
        gt_masks.append(mask)
    
    if not gt_masks:
        return None, []
    
    gt_object_ids = set()
    for mask in gt_masks:
        unique_ids = np.unique(mask)
        gt_object_ids.update(unique_ids[unique_ids != 0].tolist())
    
    return gt_masks, sorted(list(gt_object_ids))

def compute_iou(mask1, mask2):
    """Compute IoU between two binary masks"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def resize_mask(mask, target_shape):
    """Resize mask to target shape using nearest neighbor interpolation"""
    if mask.shape == target_shape:
        return mask
    resized = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized

def find_matching_tracked_objects(sequence_name, gt_masks, gt_object_ids, iou_threshold=0.3):
    """
    Find tracked objects that match GT objects based on IoU
    Returns a dict mapping GT object IDs to lists of matching tracked object IDs
    """
    groups_npy_dir = os.path.join(base_dir, sequence_name, "groups_npy")
    if not os.path.exists(groups_npy_dir):
        print(f"Warning: groups_npy not found for {sequence_name}")
        return {}
    
    group_files = sorted(list(Path(groups_npy_dir).glob("group_*.npy")))
    if not group_files:
        print(f"Warning: No group files found for {sequence_name}")
        return {}
    
    tracked_regions = []
    for group_file in group_files:
        group = np.load(str(group_file))
        tracked_regions.append(group)
    
    if len(tracked_regions) != len(gt_masks):
        print(f"Warning: Mismatch in frame count for {sequence_name}: "
              f"GT={len(gt_masks)}, Tracked={len(tracked_regions)}")
        min_frames = min(len(gt_masks), len(tracked_regions))
        gt_masks = gt_masks[:min_frames]
        tracked_regions = tracked_regions[:min_frames]
    
    if len(tracked_regions) > 0:
        tracked_shape = tracked_regions[0].shape
        print(f"  {sequence_name}: GT shape={gt_masks[0].shape}, Tracked shape={tracked_shape}")
    
    gt_to_tracked = defaultdict(set)
    
    for gt_obj_id in gt_object_ids:
        tracked_obj_ious = defaultdict(list)
        
        for frame_idx in range(len(gt_masks)):
            gt_mask_original = gt_masks[frame_idx]
            tracked_mask = tracked_regions[frame_idx]
            
            gt_mask_resized = resize_mask(gt_mask_original, tracked_mask.shape)
            gt_binary_mask = (gt_mask_resized == gt_obj_id)
            
            tracked_ids = np.unique(tracked_mask)
            tracked_ids = tracked_ids[tracked_ids != 0]
            
            for tracked_id in tracked_ids:
                tracked_obj_mask = (tracked_mask == tracked_id)
                iou = compute_iou(gt_binary_mask, tracked_obj_mask)
                tracked_obj_ious[tracked_id].append(iou)
        
        for tracked_id, ious in tracked_obj_ious.items():
            mean_iou = np.mean(ious)
            max_iou = np.max(ious)
            if mean_iou >= iou_threshold:
                gt_to_tracked[gt_obj_id].add(int(tracked_id))
                if mean_iou > 0.5:
                    print(f"    GT obj {gt_obj_id} -> Tracked obj {tracked_id}: mean_iou={mean_iou:.3f}, max_iou={max_iou:.3f}")
    
    return gt_to_tracked

def load_sequence_data_split_by_gt(sequence_name, case_type, iou_threshold=0.3):
    """Load CSV data and split into GT-inside and GT-outside objects"""
    csv_path = os.path.join(base_dir, sequence_name, "object_statistics.csv")
    
    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found for {sequence_name}")
        return None, None
    
    # Load GT masks and find matching tracked objects
    gt_masks, gt_object_ids = load_gt_masks(sequence_name)
    if gt_masks is None or not gt_object_ids:
        print(f"Warning: No valid GT masks for {sequence_name}")
        return None, None
    
    print(f"\n{sequence_name}: Processing {len(gt_object_ids)} GT objects...")
    gt_to_tracked = find_matching_tracked_objects(sequence_name, gt_masks, gt_object_ids, iou_threshold)
    
    # Get all tracked object IDs that match any GT object
    gt_matched_tracked_ids = set()
    for gt_id, tracked_ids in gt_to_tracked.items():
        gt_matched_tracked_ids.update(tracked_ids)
        print(f"  GT object {gt_id} matched to {len(tracked_ids)} tracked objects: {sorted(tracked_ids)}")
    
    print(f"  Total: {len(gt_matched_tracked_ids)} GT-inside objects")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Check if global_variance column exists
    if 'global_variance' not in df.columns and 'attention_variance' in df.columns:
        df['global_variance'] = df['attention_variance']
    
    if 'mean_window_variance' not in df.columns:
        df['mean_window_variance'] = 0.0
    
    # Split into GT-inside and GT-outside
    df_gt_inside = df[df['object_id'].isin(gt_matched_tracked_ids)].copy()
    df_gt_outside = df[~df['object_id'].isin(gt_matched_tracked_ids)].copy()
    
    # Add metadata
    for df_subset in [df_gt_inside, df_gt_outside]:
        if len(df_subset) > 0:
            df_subset['sequence'] = sequence_name
            df_subset['case_type'] = case_type
            df_subset['gt_object_count'] = len(gt_object_ids)
    
    print(f"  GT-inside objects: {len(df_gt_inside)}, GT-outside objects: {len(df_gt_outside)}")
    
    return df_gt_inside if len(df_gt_inside) > 0 else None, df_gt_outside if len(df_gt_outside) > 0 else None

# Data collection with GT filtering
print("\n" + "="*60)
print("LOADING DATA WITH GT SPLIT")
print("="*60)

bad_data_gt_inside = []
bad_data_gt_outside = []
good_data_gt_inside = []
good_data_gt_outside = []

print("\nProcessing bad cases...")
for seq in bad_sequences:
    print(f"\n--- {seq} ---")
    df_inside, df_outside = load_sequence_data_split_by_gt(seq, 'bad', iou_threshold=0.3)
    if df_inside is not None:
        bad_data_gt_inside.append(df_inside)
    if df_outside is not None:
        bad_data_gt_outside.append(df_outside)

print("\n\nProcessing good cases...")
for seq in good_sequences:
    print(f"\n--- {seq} ---")
    df_inside, df_outside = load_sequence_data_split_by_gt(seq, 'good', iou_threshold=0.3)
    if df_inside is not None:
        good_data_gt_inside.append(df_inside)
    if df_outside is not None:
        good_data_gt_outside.append(df_outside)

# Combine data for GT-inside
bad_df_inside = pd.concat(bad_data_gt_inside, ignore_index=True) if bad_data_gt_inside else pd.DataFrame()
good_df_inside = pd.concat(good_data_gt_inside, ignore_index=True) if good_data_gt_inside else pd.DataFrame()
all_df_inside = pd.concat([bad_df_inside, good_df_inside], ignore_index=True)

# Combine data for GT-outside
bad_df_outside = pd.concat(bad_data_gt_outside, ignore_index=True) if bad_data_gt_outside else pd.DataFrame()
good_df_outside = pd.concat(good_data_gt_outside, ignore_index=True) if good_data_gt_outside else pd.DataFrame()
all_df_outside = pd.concat([bad_df_outside, good_df_outside], ignore_index=True)

print(f"\n" + "="*60)
print("GT-INSIDE OBJECTS:")
print(f"  Bad cases: {len(bad_df_inside)}")
print(f"  Good cases: {len(good_df_inside)}")
print(f"  Total: {len(all_df_inside)}")
print("\nGT-OUTSIDE OBJECTS:")
print(f"  Bad cases: {len(bad_df_outside)}")
print(f"  Good cases: {len(good_df_outside)}")
print(f"  Total: {len(all_df_outside)}")
print("="*60)

# ==================== FUNCTION TO GENERATE ANALYSIS ====================
def generate_analysis(bad_df, good_df, all_df, output_dir, label_suffix):
    """Generate complete analysis for a dataset"""
    
    if len(bad_df) == 0 and len(good_df) == 0:
        print(f"No data for {label_suffix}. Skipping.")
        return
    
    print(f"\n" + "="*60)
    print(f"GENERATING ANALYSIS FOR {label_suffix}")
    print("="*60)
    
    high_mean_threshold = 0.5
    high_var_threshold = 0.05
    low_var_threshold = 0.05
    
    bad_high_mean = bad_df[bad_df['attention_mean'] > high_mean_threshold] if len(bad_df) > 0 else pd.DataFrame()
    good_high_mean = good_df[good_df['attention_mean'] > high_mean_threshold] if len(good_df) > 0 else pd.DataFrame()
    good_high_var = good_df[good_df['global_variance'] > high_var_threshold] if len(good_df) > 0 else pd.DataFrame()
    bad_candidates = bad_df[(bad_df['attention_mean'] > high_mean_threshold) & 
                            (bad_df['global_variance'] < low_var_threshold)] if len(bad_df) > 0 else pd.DataFrame()
    
    # ==================== Analysis 1 ====================
    print("\n" + "="*60)
    print(f"ANALYSIS 1: Bad case high mean attention ({label_suffix})")
    print("="*60)
    
    print(f"\nBad cases with attention_mean > {high_mean_threshold}:")
    print(f"  Count: {len(bad_high_mean)}")
    if len(bad_high_mean) > 0:
        print(f"  Mean variance: {bad_high_mean['global_variance'].mean():.6f}")
        print(f"  Median variance: {bad_high_mean['global_variance'].median():.6f}")
    
    print(f"\nGood cases with attention_mean > {high_mean_threshold}:")
    print(f"  Count: {len(good_high_mean)}")
    if len(good_high_mean) > 0:
        print(f"  Mean variance: {good_high_mean['global_variance'].mean():.6f}")
        print(f"  Median variance: {good_high_mean['global_variance'].median():.6f}")
    
    if len(bad_high_mean) > 0 and len(good_high_mean) > 0:
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(bad_high_mean['global_variance'], 
                                           good_high_mean['global_variance'])
        print(f"\nT-test result: t={t_stat:.4f}, p={p_value:.6f}")
    
    # ==================== Analysis 2 ====================
    print("\n" + "="*60)
    print(f"ANALYSIS 2: Good case high variance ({label_suffix})")
    print("="*60)
    
    print(f"\nGood cases with global_variance > {high_var_threshold}:")
    print(f"  Count: {len(good_high_var)}")
    if len(good_high_var) > 0:
        print(f"  Mean pixel area: {good_high_var['avg_pixel_area'].mean():.2f}")
        print(f"  Median pixel area: {good_high_var['avg_pixel_area'].median():.2f}")
    
    if len(good_df) > 0:
        print(f"\nAll good case objects:")
        print(f"  Mean pixel area: {good_df['avg_pixel_area'].mean():.2f}")
        print(f"  Median pixel area: {good_df['avg_pixel_area'].median():.2f}")
        
        if len(good_df) > 1:
            corr = good_df[['global_variance', 'avg_pixel_area']].corr()
            print(f"\nCorrelation between variance and pixel area: {corr.iloc[0,1]:.4f}")
    
    # ==================== Analysis 3 ====================
    print("\n" + "="*60)
    print(f"ANALYSIS 3: Bad case candidates ({label_suffix})")
    print("="*60)
    
    print(f"\nBad cases with mean > {high_mean_threshold} AND variance < {low_var_threshold}:")
    print(f"  Count: {len(bad_candidates)}")
    
    if len(bad_candidates) > 0:
        print(f"  Mean pixel area: {bad_candidates['avg_pixel_area'].mean():.2f}")
        print(f"  Median pixel area: {bad_candidates['avg_pixel_area'].median():.2f}")
        
        print("\nTop 10 candidate objects:")
        top_candidates = bad_candidates.nlargest(10, 'avg_pixel_area')[
            ['sequence', 'object_id', 'attention_mean', 'global_variance', 'avg_pixel_area']
        ]
        print(top_candidates.to_string(index=False))
    
    # ==================== Visualizations ====================
    print(f"\nGenerating visualizations for {label_suffix}...")
    
    # Plot 1: Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    if len(bad_df) > 0:
        axes[0].scatter(bad_df['attention_mean'], bad_df['global_variance'], 
                        alpha=0.5, s=bad_df['avg_pixel_area']/100, c='red', label='Bad cases')
        axes[0].axvline(x=high_mean_threshold, color='blue', linestyle='--', alpha=0.7)
        axes[0].axhline(y=low_var_threshold, color='green', linestyle='--', alpha=0.7)
    axes[0].set_xlabel('Attention Mean')
    axes[0].set_ylabel('Global Variance')
    axes[0].set_title(f'Bad Cases ({label_suffix})\n(size = pixel area)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    if len(good_df) > 0:
        axes[1].scatter(good_df['attention_mean'], good_df['global_variance'], 
                        alpha=0.5, s=good_df['avg_pixel_area']/100, c='green', label='Good cases')
        axes[1].axvline(x=high_mean_threshold, color='blue', linestyle='--', alpha=0.7)
        axes[1].axhline(y=high_var_threshold, color='orange', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Attention Mean')
    axes[1].set_ylabel('Global Variance')
    axes[1].set_title(f'Good Cases ({label_suffix})\n(size = pixel area)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_mean_vs_variance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 1_mean_vs_variance_comparison.png")
    
    # Plot 2: Distributions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Mean distribution
    data_mean = []
    labels_mean = []
    colors_mean = []
    if len(bad_df) > 0:
        data_mean.append(bad_df['attention_mean'])
        labels_mean.append('Bad cases')
        colors_mean.append('red')
    if len(good_df) > 0:
        data_mean.append(good_df['attention_mean'])
        labels_mean.append('Good cases')
        colors_mean.append('green')
    
    if data_mean:
        axes[0, 0].hist(data_mean, bins=30, label=labels_mean, alpha=0.6, color=colors_mean)
        axes[0, 0].axvline(x=high_mean_threshold, color='blue', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('Attention Mean')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Attention Mean Distribution ({label_suffix})')
    axes[0, 0].legend()
    
    # Variance distribution
    data_var = []
    if len(bad_df) > 0:
        data_var.append(bad_df['global_variance'])
    if len(good_df) > 0:
        data_var.append(good_df['global_variance'])
    
    if data_var:
        axes[0, 1].hist(data_var, bins=30, label=labels_mean, alpha=0.6, color=colors_mean)
        axes[0, 1].axvline(x=high_var_threshold, color='orange', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('Global Variance')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Variance Distribution ({label_suffix})')
    axes[0, 1].legend()
    
    # Pixel area distribution
    data_pixel = []
    if len(bad_df) > 0:
        data_pixel.append(bad_df['avg_pixel_area'])
    if len(good_df) > 0:
        data_pixel.append(good_df['avg_pixel_area'])
    
    if data_pixel:
        axes[1, 0].hist(data_pixel, bins=30, label=labels_mean, alpha=0.6, color=colors_mean)
    axes[1, 0].set_xlabel('Average Pixel Area')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Pixel Area Distribution ({label_suffix})')
    axes[1, 0].legend()
    
    # Boxplot
    data_to_plot = []
    labels_box = []
    colors_box = []
    
    if len(bad_df) > 0:
        data_to_plot.append(bad_df['global_variance'])
        labels_box.append('Bad\n(all)')
        colors_box.append('red')
    if len(good_df) > 0:
        data_to_plot.append(good_df['global_variance'])
        labels_box.append('Good\n(all)')
        colors_box.append('green')
    if len(bad_high_mean) > 0:
        data_to_plot.append(bad_high_mean['global_variance'])
        labels_box.append(f'Bad\n(mean>{high_mean_threshold})')
        colors_box.append('darkred')
    if len(good_high_mean) > 0:
        data_to_plot.append(good_high_mean['global_variance'])
        labels_box.append(f'Good\n(mean>{high_mean_threshold})')
        colors_box.append('darkgreen')
    
    if data_to_plot:
        bp = axes[1, 1].boxplot(data_to_plot, labels=labels_box, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    axes[1, 1].set_ylabel('Global Variance')
    axes[1, 1].set_title(f'Variance Comparison ({label_suffix})')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_distribution_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 2_distribution_comparison.png")
    
    # Plot 3: Variance vs Pixel Area
    if len(good_df) > 0:
        plt.figure(figsize=(10, 8))
        plt.scatter(good_df['avg_pixel_area'], good_df['global_variance'], alpha=0.5, c='green')
        plt.axhline(y=high_var_threshold, color='orange', linestyle='--', alpha=0.7)
        plt.xlabel('Average Pixel Area')
        plt.ylabel('Global Variance')
        plt.title(f'Good Cases: Variance vs Pixel Area ({label_suffix})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '3_variance_vs_pixel_area.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: 3_variance_vs_pixel_area.png")
    
    # Save summary statistics
    summary_data = {
        'Category': [],
        'Count': [],
        'Mean_Attention_Mean': [],
        'Mean_Global_Variance': [],
        'Mean_Window_Variance': [],
        'Mean_Pixel_Area': []
    }
    
    categories = [
        ('Bad - All', bad_df),
        ('Bad - High Mean', bad_high_mean),
        ('Bad - High Mean & Low Var', bad_candidates),
        ('Good - All', good_df),
        ('Good - High Mean', good_high_mean),
        ('Good - High Var', good_high_var)
    ]
    
    for cat_name, cat_df in categories:
        if len(cat_df) > 0:
            summary_data['Category'].append(cat_name)
            summary_data['Count'].append(len(cat_df))
            summary_data['Mean_Attention_Mean'].append(cat_df['attention_mean'].mean())
            summary_data['Mean_Global_Variance'].append(cat_df['global_variance'].mean())
            summary_data['Mean_Window_Variance'].append(cat_df['mean_window_variance'].mean())
            summary_data['Mean_Pixel_Area'].append(cat_df['avg_pixel_area'].mean())
    
    summary_stats = pd.DataFrame(summary_data)
    summary_stats.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)
    print(f"  Saved: summary_statistics.csv")
    
    # Save candidate objects
    if len(bad_candidates) > 0:
        bad_candidates.to_csv(os.path.join(output_dir, 'bad_case_candidate_objects.csv'), index=False)
        print(f"  Saved: bad_case_candidate_objects.csv")
    
    # Save all data
    if len(all_df) > 0:
        all_df.to_csv(os.path.join(output_dir, 'all_objects.csv'), index=False)
        print(f"  Saved: all_objects.csv")
    
    print(f"\nAnalysis complete for {label_suffix}! Results saved to: {output_dir}")

# ==================== GENERATE BOTH ANALYSES ====================

# Generate analysis for GT-inside objects
generate_analysis(bad_df_inside, good_df_inside, all_df_inside, 
                 output_dir_gt_inside, "GT-INSIDE OBJECTS")

# Generate analysis for GT-outside objects
generate_analysis(bad_df_outside, good_df_outside, all_df_outside, 
                 output_dir_gt_outside, "GT-OUTSIDE OBJECTS")

print("\n" + "="*60)
print("ALL ANALYSES COMPLETE!")
print("="*60)
print(f"GT-inside results: {output_dir_gt_inside}")
print(f"GT-outside results: {output_dir_gt_outside}")
print("="*60)