import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from permetrics.regression import RegressionMetric



def calculate_metrics_for_flow_category(real_vals, predicted_vals, percentile=None):
    """
    Calculate regression metrics for different flow categories.
    
    Args:
        real_vals: Numpy array of ground truth values
        predicted_vals: Numpy array of predicted values
        percentile: Tuple (low, high) defining flow category, e.g., (None, 90) for high flows
                   or (10, None) for low flows. None means all flows.
    
    Returns:
        Tuple of (KGE, NSE, MSE) metrics
    """
    # Filter for valid values
    valid_indices = (real_vals >= 0) & (~np.isnan(real_vals)) & (~np.isnan(predicted_vals))
    real_vals = real_vals[valid_indices]
    predicted_vals = predicted_vals[valid_indices]
    
    # Apply percentile filtering if specified
    if percentile:
        low_pct, high_pct = percentile
        if low_pct is not None and high_pct is not None:
            # Middle range
            low_threshold = np.percentile(real_vals, low_pct)
            high_threshold = np.percentile(real_vals, high_pct)
            indices = (real_vals >= low_threshold) & (real_vals <= high_threshold)
        elif low_pct is not None:
            # Low flows
            threshold = np.percentile(real_vals, low_pct)
            indices = real_vals <= threshold
            predicted_vals = np.maximum(predicted_vals, 0)  # Ensure non-negative predictions
        elif high_pct is not None:
            # High flows
            threshold = np.percentile(real_vals, high_pct)
            indices = real_vals >= threshold
        
        real_vals = real_vals[indices]
        predicted_vals = predicted_vals[indices]
    
    # Calculate metrics
    evaluator = RegressionMetric(real_vals, predicted_vals)
    mse = evaluator.mean_squared_error()
    rmse = np.sqrt(mse)  # Calculate RMSE from MSE

    return (evaluator.kling_gupta_efficiency(), 
            evaluator.nash_sutcliffe_efficiency(), 
            mse,
            rmse)




def plot_prediction_comparison(real_vals, predicted_vals, basin_id, modelname, start_index=0, num_days=100, save_path='plots/comparison'):
    """
    绘制预测值与真实值的比较图
    
    Args:
        real_vals: 真实值数组
        predicted_vals: 预测值数组
        basin_id: 流域ID，用于图表标题
        start_index: 开始绘制的索引位置
        num_days: 要显示的天数
        save_path: 保存图表的路径
    """
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # 准备数据
    end_index = min(start_index + num_days, len(real_vals))
    real_segment = real_vals[start_index:end_index]
    pred_segment = predicted_vals[start_index:end_index]
    
    # 计算这段时间的指标
    segment_metrics = calculate_metrics_for_flow_category(real_segment, pred_segment)
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 使用Seaborn样式
    sns.set_style("whitegrid")
    
    # 绘制真实值和预测值 - 使用数据长度范围替代days
    x_values = np.arange(len(real_segment))
    plt.plot(x_values, real_segment, 'b-', linewidth=2, label='Observed')
    plt.plot(x_values, pred_segment, 'r--', linewidth=2, label='Predicted')
    
    # 添加指标文本
    metrics_text = f'NSE: {segment_metrics[1]:.4f}\nKGE: {segment_metrics[0]:.4f}\nRMSE: {segment_metrics[3]:.4f}'
    plt.annotate(metrics_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # 添加标题和标签
    plt.title(f'Basin {basin_id}: Observed vs Predicted Flow', fontsize=16)
    plt.xlabel('Time Steps', fontsize=12)  # 修改标签以反映我们使用的是时间步长
    plt.ylabel('Flow', fontsize=12)
    plt.legend(fontsize=12)
    
    # 定制X轴
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(f'{save_path}/prediction_comparison_{modelname}_basin_{basin_id}.png', dpi=300)
    plt.close()
    
    print(f"Plot saved to {save_path}/prediction_comparison_basin_{basin_id}.png")





def plot_high_low_flow_comparison(real_vals, predicted_vals, basin_id, modelname, save_path='plots/highlowcomparision'):
    """
    分别绘制高流量和低流量区间的预测比较图
    
    Args:
        real_vals: 真实值数组
        predicted_vals: 预测值数组
        basin_id: 流域ID，用于图表标题
        save_path: 保存图表的路径
    """
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # 计算高流量和低流量的阈值
    valid_indices = (real_vals >= 0) & (~np.isnan(real_vals)) & (~np.isnan(predicted_vals))

    valid_real_vals = real_vals[valid_indices]
    valid_pred_vals = predicted_vals[valid_indices]
    
    high_threshold = np.percentile(valid_real_vals, 90)
    low_threshold = np.percentile(valid_real_vals, 10)
    
    # 筛选高流量和低流量数据点
    high_flow_indices = valid_real_vals >= high_threshold
    low_flow_indices = valid_real_vals <= low_threshold
    
    high_real = valid_real_vals[high_flow_indices]
    high_pred = valid_pred_vals[high_flow_indices]
    
    low_real = valid_real_vals[low_flow_indices]
    low_pred = valid_pred_vals[low_flow_indices]
    
    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 设置Seaborn样式
    sns.set_style("whitegrid")
    
    # 绘制高流量比较
    high_indices = np.arange(len(high_real))
    axes[0].plot(high_indices, high_real, 'b-', linewidth=2, label='Observed')
    axes[0].plot(high_indices, high_pred, 'r--', linewidth=2, label='Predicted')
    axes[0].set_title(f'High Flow Comparison (>90th percentile)', fontsize=14)
    axes[0].set_xlabel('Index', fontsize=12)
    axes[0].set_ylabel('Flow', fontsize=12)
    
    # 计算并显示高流量指标
    high_metrics = calculate_metrics_for_flow_category(high_real, high_pred)
    high_metrics_text = f'NSE: {high_metrics[1]:.4f}\nKGE: {high_metrics[0]:.4f}\nRMSE: {high_metrics[3]:.4f}'
    axes[0].annotate(high_metrics_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    axes[0].legend()
    
    # 绘制低流量比较
    low_indices = np.arange(len(low_real))
    axes[1].plot(low_indices, low_real, 'b-', linewidth=2, label='Observed')
    axes[1].plot(low_indices, low_pred, 'r--', linewidth=2, label='Predicted')
    axes[1].set_title(f'Low Flow Comparison (<10th percentile)', fontsize=14)
    axes[1].set_xlabel('Index', fontsize=12)
    axes[1].set_ylabel('Flow', fontsize=12)
    
    # 计算并显示低流量指标
    low_metrics = calculate_metrics_for_flow_category(low_real, low_pred)
    low_metrics_text = f'NSE: {low_metrics[1]:.4f}\nKGE: {low_metrics[0]:.4f}\nRMSE: {low_metrics[3]:.4f}'
    axes[1].annotate(low_metrics_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    axes[1].legend()
    
    # 调整布局并保存
    plt.suptitle(f'Basin {basin_id}: Flow Regime Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_path}/high_low_flow_comparison_{modelname}_basin_{basin_id}.png', dpi=300)
    plt.close()
    
    print(f"High/Low flow comparison plot saved to {save_path}/high_low_flow_comparison_basin_{basin_id}.png")





def plot_detailed_prediction_results(real_vals, predicted_vals, basin_id, modelname, save_dir='plots/detailed'):
    """
    创建一个更详细、更专业的预测结果可视化函数
    
    Args:
        real_vals: 真实观测值数组
        predicted_vals: 模型预测值数组
        basin_id: 流域ID
        save_dir: 保存图表的目录
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置图表风格
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 创建多子图的画布 - 包含:
    # 1. 时间序列对比
    # 2. 散点图
    # 3. 残差分析
    # 4. 分位数-分位数图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 过滤有效值
    valid_indices = (real_vals >= 0) & (~np.isnan(real_vals)) & (~np.isnan(predicted_vals))
    valid_real = real_vals[valid_indices]
    valid_pred = predicted_vals[valid_indices]
    
    # === 1. 时间序列对比图 (最多显示300个数据点，避免过度拥挤) ===
    display_length = min(300, len(valid_real))
    time_indices = np.arange(display_length)
    display_real = valid_real[:display_length]
    display_pred = valid_pred[:display_length]
    
    axes[0, 0].plot(time_indices, display_real, 'b-', linewidth=2, alpha=0.8, label='Observed')
    axes[0, 0].plot(time_indices, display_pred, 'r--', linewidth=2, alpha=0.8, label='Predicted')
    axes[0, 0].set_title(f'Time Series Comparison', fontweight='bold')
    axes[0, 0].set_xlabel('Time Index')
    axes[0, 0].set_ylabel('Flow')
    axes[0, 0].legend(loc='upper right')
    
    # 在图上添加指标信息
    metrics = calculate_metrics_for_flow_category(valid_real, valid_pred)
    metrics_text = f'NSE: {metrics[1]:.3f}\nKGE: {metrics[0]:.3f}\nRMSE: {metrics[3]:.3f}'
    axes[0, 0].text(0.03, 0.97, metrics_text, transform=axes[0, 0].transAxes, 
                   fontsize=12, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # === 2. 散点图与1:1线 ===
    max_val = max(np.max(valid_real), np.max(valid_pred)) * 1.1
    min_val = min(np.min(valid_real), np.min(valid_pred)) * 0.9
    min_val = max(0, min_val)  # 确保非负
    
    axes[0, 1].scatter(valid_real, valid_pred, c='blue', alpha=0.6, s=20)
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    # 添加回归线
    z = np.polyfit(valid_real, valid_pred, 1)
    p = np.poly1d(z)
    axes[0, 1].plot(np.sort(valid_real), p(np.sort(valid_real)), 'g-', linewidth=2)
    
    axes[0, 1].set_xlim(min_val, max_val)
    axes[0, 1].set_ylim(min_val, max_val)
    axes[0, 1].set_title('Observed vs Predicted', fontweight='bold')
    axes[0, 1].set_xlabel('Observed Flow')
    axes[0, 1].set_ylabel('Predicted Flow')
    axes[0, 1].text(0.03, 0.97, f'y = {z[0]:.3f}x + {z[1]:.3f}', transform=axes[0, 1].transAxes, 
                   fontsize=12, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # === 3. 残差分析图 ===
    residuals = valid_pred - valid_real
    axes[1, 0].scatter(valid_real, residuals, c='red', alpha=0.6, s=20)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 0].set_title('Residuals Analysis', fontweight='bold')
    axes[1, 0].set_xlabel('Observed Flow')
    axes[1, 0].set_ylabel('Residuals (Predicted - Observed)')
    
    # 添加残差均值和标准差
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    axes[1, 0].axhline(y=residual_mean, color='blue', linestyle='--', linewidth=1)
    axes[1, 0].axhline(y=residual_mean + 2*residual_std, color='green', linestyle='--', linewidth=1)
    axes[1, 0].axhline(y=residual_mean - 2*residual_std, color='green', linestyle='--', linewidth=1)
    axes[1, 0].text(0.03, 0.97, f'Mean: {residual_mean:.3f}\nStd: {residual_std:.3f}', 
                   transform=axes[1, 0].transAxes, fontsize=12, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # === 4. 分位数-分位数图 (QQ图) ===
    sorted_real = np.sort(valid_real)
    sorted_pred = np.sort(valid_pred)
    
    # 如果长度不同，进行插值以匹配长度
    if len(sorted_real) != len(sorted_pred):
        indices = np.linspace(0, len(sorted_pred)-1, len(sorted_real)).astype(int)
        sorted_pred = sorted_pred[indices]
    
    axes[1, 1].scatter(sorted_real, sorted_pred, c='purple', alpha=0.6, s=20)
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1, 1].set_title('Quantile-Quantile Plot', fontweight='bold')
    axes[1, 1].set_xlabel('Observed Flow Quantiles')
    axes[1, 1].set_ylabel('Predicted Flow Quantiles')
    
    # === 总体标题和布局调整 ===
    plt.suptitle(f'Basin {basin_id}: Detailed Prediction Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为总标题留出空间
    
    # 保存高分辨率图像
    plt.savefig(f'{save_dir}/detailed_analysis_{modelname}_basin_{basin_id}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed analysis saved to {save_dir}/detailed_analysis_basin_{basin_id}.png")



def plot_flow_duration_curve(real_vals, predicted_vals, basin_id, modelname, save_dir='plots/fdc'):
    """
    绘制流量持续曲线（FDC），常用于水文分析
    
    Args:
        real_vals: 真实观测值数组
        predicted_vals: 模型预测值数组
        basin_id: 流域ID
        save_dir: 保存图表的目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置风格
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    # 过滤有效值
    valid_indices = (real_vals >= 0) & (~np.isnan(real_vals)) & (~np.isnan(predicted_vals))
    valid_real = real_vals[valid_indices]
    valid_pred = predicted_vals[valid_indices]
    
    # 计算超越概率
    sorted_real = np.sort(valid_real)[::-1]  # 降序排列
    sorted_pred = np.sort(valid_pred)[::-1]
    
    exceedance_prob = np.arange(1, len(sorted_real) + 1) / (len(sorted_real) + 1) * 100
    
    # 绘制流量持续曲线
    plt.plot(exceedance_prob, sorted_real, 'b-', linewidth=2, label='Observed')
    plt.plot(exceedance_prob, sorted_pred, 'r--', linewidth=2, label='Predicted')
    
    # 使用对数刻度以更好地观察低流量区域
    plt.yscale('log')
    
    # 添加网格线和标签
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Exceedance Probability (%)')
    plt.ylabel('Flow (log scale)')
    plt.title(f'Basin {basin_id}: Flow Duration Curve', fontsize=14, fontweight='bold')
    plt.legend()
    
    # 标记Q10, Q50, Q90的位置
    q10_idx = int(0.1 * len(sorted_real))
    q50_idx = int(0.5 * len(sorted_real))
    q90_idx = int(0.9 * len(sorted_real))
    
    plt.axvline(x=10, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=50, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=90, color='gray', linestyle='--', alpha=0.7)
    
    plt.axhline(y=sorted_real[q10_idx], color='gray', linestyle='--', alpha=0.7)
    plt.axhline(y=sorted_real[q50_idx], color='gray', linestyle='--', alpha=0.7)
    plt.axhline(y=sorted_real[q90_idx], color='gray', linestyle='--', alpha=0.7)
    
    plt.text(12, sorted_real[q10_idx]*1.1, 'Q10', fontsize=10)
    plt.text(52, sorted_real[q50_idx]*1.1, 'Q50', fontsize=10)
    plt.text(92, sorted_real[q90_idx]*1.1, 'Q90', fontsize=10)
    
    # 添加指标信息
    metrics = calculate_metrics_for_flow_category(valid_real, valid_pred)
    metrics_text = f'NSE: {metrics[1]:.3f}\nKGE: {metrics[0]:.3f}\nRMSE: {metrics[3]:.3f}'
    plt.text(0.05, 0.05, metrics_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='bottom', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 设置x轴范围
    plt.xlim(0, 100)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(f'{save_dir}/flow_duration_curve_{modelname}_basin_{basin_id}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Flow duration curve saved to {save_dir}/flow_duration_curve_basin_{basin_id}.png")


def calculate_metrics_for_flow_category(real_vals, predicted_vals, percentile=None):
    """
    为了在可视化函数中使用的计算指标的函数副本
    """
    # Filter for valid values
    valid_indices = (real_vals >= 0) & (~np.isnan(real_vals)) & (~np.isnan(predicted_vals))
    real_vals = real_vals[valid_indices]
    predicted_vals = predicted_vals[valid_indices]
    
    # Apply percentile filtering if specified
    if percentile:
        low_pct, high_pct = percentile
        if low_pct is not None and high_pct is not None:
            # Middle range
            low_threshold = np.percentile(real_vals, low_pct)
            high_threshold = np.percentile(real_vals, high_pct)
            indices = (real_vals >= low_threshold) & (real_vals <= high_threshold)
        elif low_pct is not None:
            # Low flows
            threshold = np.percentile(real_vals, low_pct)
            indices = real_vals <= threshold
            predicted_vals = np.maximum(predicted_vals, 0)  # Ensure non-negative predictions
        elif high_pct is not None:
            # High flows
            threshold = np.percentile(real_vals, high_pct)
            indices = real_vals >= threshold
        
        real_vals = real_vals[indices]
        predicted_vals = predicted_vals[indices]
    
    # 计算NSE
    nse_val = 1 - (np.sum((real_vals - predicted_vals)**2) / 
                   np.sum((real_vals - np.mean(real_vals))**2))
    
    # 计算KGE
    r = np.corrcoef(real_vals, predicted_vals)[0, 1]
    alpha = np.std(predicted_vals) / np.std(real_vals)
    beta = np.mean(predicted_vals) / np.mean(real_vals)
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    # 计算MSE和RMSE
    mse = np.mean((real_vals - predicted_vals)**2)
    rmse = np.sqrt(mse)
    
    return (kge, nse_val, mse, rmse)
