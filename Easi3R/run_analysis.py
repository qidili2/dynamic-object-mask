import pandas as pd
import numpy as np
import re

def analyze_sequence_data(csv_file_path):
    """
    分析序列数据，计算每个sequence的J-Mean平均分，找出最高和最低帧
    
    Args:
        csv_file_path (str): CSV文件路径
    """
    
    # 读取CSV文件
    print("正在读取CSV文件...")
    df = pd.read_csv(csv_file_path)
    print(f"总共读取了 {len(df)} 行数据")
    
    # 提取基础序列名称和帧号
    print("正在解析序列名称和帧号...")
    sequence_data = []
    
    for index, row in df.iterrows():
        sequence_full = row['Sequence']
        j_mean = row['J-Mean']
        f_mean = row['F-Mean']
        
        # 使用正则表达式提取基础序列名称和帧号
        match = re.match(r'^(.+)_(\d+)$', sequence_full)
        if match:
            base_name = match.group(1)
            frame_num = int(match.group(2))
            sequence_data.append({
                'base_sequence': base_name,
                'frame': frame_num,
                'j_mean': j_mean,
                'f_mean': f_mean,
                'full_sequence': sequence_full
            })
        else:
            print(f"警告: 无法解析的序列名称: {sequence_full}")
    
    # 转换为DataFrame进行分析
    analysis_df = pd.DataFrame(sequence_data)
    
    # 按base_sequence分组计算统计信息
    print("正在计算每个序列的统计信息...")
    sequence_stats = []
    
    for base_seq in analysis_df['base_sequence'].unique():
        seq_data = analysis_df[analysis_df['base_sequence'] == base_seq]
        
        # 计算J-Mean的平均值
        j_mean_avg = seq_data['j_mean'].mean()
        
        # 找出J-Mean最高和最低的帧
        max_j_idx = seq_data['j_mean'].idxmax()
        min_j_idx = seq_data['j_mean'].idxmin()
        
        max_j_frame = seq_data.loc[max_j_idx]
        min_j_frame = seq_data.loc[min_j_idx]
        
        sequence_stats.append({
            'base_sequence': base_seq,
            'frame_count': len(seq_data),
            'j_mean_average': j_mean_avg,
            'max_j_mean': max_j_frame['j_mean'],
            'max_j_frame': max_j_frame['frame'],
            'max_j_full_name': max_j_frame['full_sequence'],
            'min_j_mean': min_j_frame['j_mean'],
            'min_j_frame': min_j_frame['frame'],
            'min_j_full_name': min_j_frame['full_sequence']
        })
    
    # 转换为DataFrame并按J-Mean平均值排序
    stats_df = pd.DataFrame(sequence_stats)
    stats_df = stats_df.sort_values('j_mean_average', ascending=False)
    
    # 输出结果
    print("\n" + "="*80)
    print("序列分析结果 (按J-Mean平均值排序)")
    print("="*80)
    print(f"总共发现 {len(stats_df)} 个不同的序列")
    print()
    
    # 输出详细结果
    for idx, row in stats_df.iterrows():
        print(f"序列: {row['base_sequence']}")
        print(f"  帧数: {row['frame_count']}")
        print(f"  J-Mean平均值: {row['j_mean_average']:.4f}")
        print(f"  最高J-Mean: {row['max_j_mean']:.4f} (帧{row['max_j_frame']}: {row['max_j_full_name']})")
        print(f"  最低J-Mean: {row['min_j_mean']:.4f} (帧{row['min_j_frame']}: {row['min_j_full_name']})")
        print()
    
    # 输出排名前10和后10的序列
    print("\n" + "="*80)
    print("排名前10的序列 (J-Mean平均值最高)")
    print("="*80)
    top_10 = stats_df.head(10)
    for i, (idx, row) in enumerate(top_10.iterrows(), 1):
        print(f"{i:2d}. {row['base_sequence']:<20} | 平均J-Mean: {row['j_mean_average']:.4f} | 帧数: {row['frame_count']}")
    
    print("\n" + "="*80)
    print("排名后10的序列 (J-Mean平均值最低)")
    print("="*80)
    bottom_10 = stats_df.tail(10)
    for i, (idx, row) in enumerate(bottom_10.iterrows(), 1):
        rank = len(stats_df) - 10 + i
        print(f"{rank:2d}. {row['base_sequence']:<20} | 平均J-Mean: {row['j_mean_average']:.4f} | 帧数: {row['frame_count']}")
    
    # 保存结果到CSV文件
    output_file = 'sequence_analysis_results.csv'
    stats_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n详细结果已保存到: {output_file}")
    
    return stats_df

def main():
    """
    主函数
    """
    # 设置CSV文件路径
    csv_file = '/mnt/data0/andy/Easi3R/results/davis/easi3r_monst3r_sam_noregion/per-sequence_results.csv'  # 请根据实际文件路径修改
    
    try:
        # 执行分析
        results = analyze_sequence_data(csv_file)
        
        # 输出总体统计信息
        print("\n" + "="*80)
        print("总体统计信息")
        print("="*80)
        print(f"序列总数: {len(results)}")
        print(f"总帧数: {results['frame_count'].sum()}")
        print(f"J-Mean平均值范围: {results['j_mean_average'].min():.4f} - {results['j_mean_average'].max():.4f}")
        print(f"J-Mean平均值的平均值: {results['j_mean_average'].mean():.4f}")
        print(f"J-Mean平均值的标准差: {results['j_mean_average'].std():.4f}")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{csv_file}'")
        print("请确保文件存在并且路径正确")
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()