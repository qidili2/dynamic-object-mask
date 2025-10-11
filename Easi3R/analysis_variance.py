import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import cv2

# Configuration
base_dir = "/mnt/data0/andy/Easi3R/results/davis/easi3r_sam2_high_resolution"
badcase_file = "/mnt/data0/andy/Easi3R/results/davis/davis17_badcase.txt"
goodcase_file = "/mnt/data0/andy/Easi3R/results/davis/davis17_goodcase.txt"
output_dir = "/mnt/data0/andy/Easi3R/results/davis/analysis_output"

os.makedirs(output_dir, exist_ok=True)

# Read sequence lists
with open(badcase_file, 'r') as f:
    bad_sequences = [line.strip() for line in f if line.strip()]

with open(goodcase_file, 'r') as f:
    good_sequences = [line.strip() for line in f if line.strip()]

print(f"Found {len(bad_sequences)} bad cases and {len(good_sequences)} good cases")

# Data collection
bad_data = []
good_data = []

def load_sequence_data(sequence_name, case_type):
    csv_path = os.path.join(base_dir, sequence_name, "object_statistics.csv")
    
    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found for {sequence_name}")
        return None
    
    df = pd.read_csv(csv_path)
    df['sequence'] = sequence_name
    df['case_type'] = case_type
    return df

# Load all data
for seq in bad_sequences:
    df = load_sequence_data(seq, 'bad')
    if df is not None:
        bad_data.append(df)

for seq in good_sequences:
    df = load_sequence_data(seq, 'good')
    if df is not None:
        good_data.append(df)

# Combine data
bad_df = pd.concat(bad_data, ignore_index=True) if bad_data else pd.DataFrame()
good_df = pd.concat(good_data, ignore_index=True) if good_data else pd.DataFrame()
all_df = pd.concat([bad_df, good_df], ignore_index=True)

print(f"\nTotal objects in bad cases: {len(bad_df)}")
print(f"Total objects in good cases: {len(good_df)}")

# ==================== Analysis 1 ====================
# Bad case high mean attention vs good case variance
print("\n" + "="*60)
print("ANALYSIS 1: Bad case high mean attention objects")
print("="*60)

high_mean_threshold = 0.5
bad_high_mean = bad_df[bad_df['attention_mean'] > high_mean_threshold]
good_high_mean = good_df[good_df['attention_mean'] > high_mean_threshold]

print(f"\nBad cases with attention_mean > {high_mean_threshold}:")
print(f"  Count: {len(bad_high_mean)}")
print(f"  Mean variance: {bad_high_mean['attention_variance'].mean():.6f}")
print(f"  Median variance: {bad_high_mean['attention_variance'].median():.6f}")

print(f"\nGood cases with attention_mean > {high_mean_threshold}:")
print(f"  Count: {len(good_high_mean)}")
print(f"  Mean variance: {good_high_mean['attention_variance'].mean():.6f}")
print(f"  Median variance: {good_high_mean['attention_variance'].median():.6f}")

# Statistical comparison
if len(bad_high_mean) > 0 and len(good_high_mean) > 0:
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(bad_high_mean['attention_variance'], 
                                       good_high_mean['attention_variance'])
    print(f"\nT-test result: t={t_stat:.4f}, p={p_value:.6f}")

# ==================== Analysis 2 ====================
# Good case high variance objects - are they small regions?
print("\n" + "="*60)
print("ANALYSIS 2: Good case high variance objects and region size")
print("="*60)

high_var_threshold = 0.05
good_high_var = good_df[good_df['attention_variance'] > high_var_threshold]

print(f"\nGood cases with attention_variance > {high_var_threshold}:")
print(f"  Count: {len(good_high_var)}")
print(f"  Mean pixel area: {good_high_var['avg_pixel_area'].mean():.2f}")
print(f"  Median pixel area: {good_high_var['avg_pixel_area'].median():.2f}")

# Compare with all good case objects
print(f"\nAll good case objects:")
print(f"  Mean pixel area: {good_df['avg_pixel_area'].mean():.2f}")
print(f"  Median pixel area: {good_df['avg_pixel_area'].median():.2f}")

# Correlation analysis
corr = good_df[['attention_variance', 'avg_pixel_area']].corr()
print(f"\nCorrelation between variance and pixel area in good cases: {corr.iloc[0,1]:.4f}")

# ==================== Analysis 3 ====================
# Bad case high mean + low variance objects - are these the real target objects?
print("\n" + "="*60)
print("ANALYSIS 3: Bad case high mean + low variance objects")
print("="*60)

low_var_threshold = 0.05  # Changed from 0.01 to 0.05
bad_candidates = bad_df[(bad_df['attention_mean'] > high_mean_threshold) & 
                        (bad_df['attention_variance'] < low_var_threshold)]

print(f"\nBad cases with mean > {high_mean_threshold} AND variance < {low_var_threshold}:")
print(f"  Count: {len(bad_candidates)}")

if len(bad_candidates) > 0:
    print(f"  Mean pixel area: {bad_candidates['avg_pixel_area'].mean():.2f}")
    print(f"  Median pixel area: {bad_candidates['avg_pixel_area'].median():.2f}")
    
    print("\nTop 10 candidate objects (likely real targets):")
    top_candidates = bad_candidates.nlargest(10, 'avg_pixel_area')[
        ['sequence', 'object_id', 'attention_mean', 'attention_variance', 'avg_pixel_area']
    ]
    print(top_candidates.to_string(index=False))

# ==================== Visualization ====================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Plot 1: Attention mean vs variance scatter plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bad cases
axes[0].scatter(bad_df['attention_mean'], bad_df['attention_variance'], 
                alpha=0.5, s=bad_df['avg_pixel_area']/100, c='red', label='Bad cases')
axes[0].axvline(x=high_mean_threshold, color='blue', linestyle='--', alpha=0.7, label=f'Mean threshold={high_mean_threshold}')
axes[0].axhline(y=low_var_threshold, color='green', linestyle='--', alpha=0.7, label=f'Var threshold={low_var_threshold}')
axes[0].set_xlabel('Attention Mean')
axes[0].set_ylabel('Attention Variance')
axes[0].set_title('Bad Cases: Attention Mean vs Variance\n(size = pixel area)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Good cases
axes[1].scatter(good_df['attention_mean'], good_df['attention_variance'], 
                alpha=0.5, s=good_df['avg_pixel_area']/100, c='green', label='Good cases')
axes[1].axvline(x=high_mean_threshold, color='blue', linestyle='--', alpha=0.7, label=f'Mean threshold={high_mean_threshold}')
axes[1].axhline(y=high_var_threshold, color='orange', linestyle='--', alpha=0.7, label=f'Var threshold={high_var_threshold}')
axes[1].set_xlabel('Attention Mean')
axes[1].set_ylabel('Attention Variance')
axes[1].set_title('Good Cases: Attention Mean vs Variance\n(size = pixel area)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '1_mean_vs_variance_comparison.png'), dpi=300, bbox_inches='tight')
print(f"Saved: 1_mean_vs_variance_comparison.png")

# Plot 2: Distribution comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Attention mean distribution
axes[0, 0].hist([bad_df['attention_mean'], good_df['attention_mean']], 
                bins=30, label=['Bad cases', 'Good cases'], alpha=0.6, color=['red', 'green'])
axes[0, 0].axvline(x=high_mean_threshold, color='blue', linestyle='--', alpha=0.7)
axes[0, 0].set_xlabel('Attention Mean')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Attention Mean Distribution')
axes[0, 0].legend()

# Attention variance distribution
axes[0, 1].hist([bad_df['attention_variance'], good_df['attention_variance']], 
                bins=30, label=['Bad cases', 'Good cases'], alpha=0.6, color=['red', 'green'])
axes[0, 1].axvline(x=high_var_threshold, color='orange', linestyle='--', alpha=0.7)
axes[0, 1].set_xlabel('Attention Variance')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Attention Variance Distribution')
axes[0, 1].legend()

# Pixel area distribution
axes[1, 0].hist([bad_df['avg_pixel_area'], good_df['avg_pixel_area']], 
                bins=30, label=['Bad cases', 'Good cases'], alpha=0.6, color=['red', 'green'])
axes[1, 0].set_xlabel('Average Pixel Area')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Pixel Area Distribution')
axes[1, 0].legend()

# Box plot comparison
data_to_plot = [
    bad_df['attention_variance'], good_df['attention_variance'],
    bad_high_mean['attention_variance'] if len(bad_high_mean) > 0 else [],
    good_high_mean['attention_variance'] if len(good_high_mean) > 0 else []
]
labels = ['Bad\n(all)', 'Good\n(all)', f'Bad\n(mean>{high_mean_threshold})', f'Good\n(mean>{high_mean_threshold})']
bp = axes[1, 1].boxplot([d for d in data_to_plot if len(d) > 0], 
                        labels=[l for i, l in enumerate(labels) if len(data_to_plot[i]) > 0],
                        patch_artist=True)
for patch, color in zip(bp['boxes'], ['red', 'green', 'darkred', 'darkgreen'][:len(bp['boxes'])]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1, 1].set_ylabel('Attention Variance')
axes[1, 1].set_title('Variance Comparison (Boxplot)')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '2_distribution_comparison.png'), dpi=300, bbox_inches='tight')
print(f"Saved: 2_distribution_comparison.png")

# Plot 3: Variance vs Pixel Area (for good cases)
plt.figure(figsize=(10, 8))
plt.scatter(good_df['avg_pixel_area'], good_df['attention_variance'], alpha=0.5, c='green')
plt.axhline(y=high_var_threshold, color='orange', linestyle='--', alpha=0.7, 
            label=f'Var threshold={high_var_threshold}')
plt.xlabel('Average Pixel Area')
plt.ylabel('Attention Variance')
plt.title('Good Cases: Variance vs Pixel Area')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3_variance_vs_pixel_area.png'), dpi=300, bbox_inches='tight')
print(f"Saved: 3_variance_vs_pixel_area.png")

# Save summary statistics to CSV
summary_stats = pd.DataFrame({
    'Category': [
        'Bad - All', 'Bad - High Mean', 'Bad - High Mean & Low Var',
        'Good - All', 'Good - High Mean', 'Good - High Var'
    ],
    'Count': [
        len(bad_df), len(bad_high_mean), len(bad_candidates),
        len(good_df), len(good_high_mean), len(good_high_var)
    ],
    'Mean_Attention_Mean': [
        bad_df['attention_mean'].mean(), bad_high_mean['attention_mean'].mean() if len(bad_high_mean) > 0 else 0,
        bad_candidates['attention_mean'].mean() if len(bad_candidates) > 0 else 0,
        good_df['attention_mean'].mean(), good_high_mean['attention_mean'].mean() if len(good_high_mean) > 0 else 0,
        good_high_var['attention_mean'].mean() if len(good_high_var) > 0 else 0
    ],
    'Mean_Attention_Variance': [
        bad_df['attention_variance'].mean(), bad_high_mean['attention_variance'].mean() if len(bad_high_mean) > 0 else 0,
        bad_candidates['attention_variance'].mean() if len(bad_candidates) > 0 else 0,
        good_df['attention_variance'].mean(), good_high_mean['attention_variance'].mean() if len(good_high_mean) > 0 else 0,
        good_high_var['attention_variance'].mean() if len(good_high_var) > 0 else 0
    ],
    'Mean_Pixel_Area': [
        bad_df['avg_pixel_area'].mean(), bad_high_mean['avg_pixel_area'].mean() if len(bad_high_mean) > 0 else 0,
        bad_candidates['avg_pixel_area'].mean() if len(bad_candidates) > 0 else 0,
        good_df['avg_pixel_area'].mean(), good_high_mean['avg_pixel_area'].mean() if len(good_high_mean) > 0 else 0,
        good_high_var['avg_pixel_area'].mean() if len(good_high_var) > 0 else 0
    ]
})

summary_stats.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)
print(f"Saved: summary_statistics.csv")

# Save candidate objects for visualization
if len(bad_candidates) > 0:
    bad_candidates.to_csv(os.path.join(output_dir, 'bad_case_candidate_objects.csv'), index=False)
    print(f"Saved: bad_case_candidate_objects.csv")

print(f"\n" + "="*60)
print(f"Analysis complete! Results saved to: {output_dir}")
print("="*60)