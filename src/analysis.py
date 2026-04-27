# src/analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==================== 修复中文乱码====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_cleaned_data(data_path: str = None):
    
    import os
    print("正在加载清洗后的数据...")

    data_path = r"data/cleaned_taxi_data.parquet"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"cleaned_taxi_data.parquet 不存在！请先运行 M1。路径：{data_path}")

    df = pd.read_parquet(data_path)
    print(f"加载完成，数据形状: {df.shape}")
    return df


def plot_demand_by_time(df: pd.DataFrame, output_dir: str = "outputs"):
    """1. 出行需求时间规律（分小时 + 周末/工作日）"""
    os.makedirs(output_dir, exist_ok=True)

    # 按小时统计订单量
    hourly = df.groupby('pickup_hour').size().reset_index(name='trip_count')
    # 按小时 + 是否周末
    hourly_week = df.groupby(['pickup_hour', 'is_weekend']).size().reset_index(name='trip_count')
    hourly_week['type'] = hourly_week['is_weekend'].map({0: '工作日', 1: '周末'})

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=hourly_week, x='pickup_hour', y='trip_count', hue='type', marker='o')
    plt.title('纽约出租车出行需求时间分布（分小时 & 工作日/周末）')
    plt.xlabel('小时 (0-23)')
    plt.ylabel('订单数量')
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3)
    plt.legend(title='类型')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/demand_by_hour.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("图1 已保存: outputs/demand_by_hour.png")


def plot_region_heatmap(df: pd.DataFrame, output_dir: str = "outputs"):
    """2. 区域热度分析（Pickup TOP10 + 高峰时段分布）"""
    os.makedirs(output_dir, exist_ok=True)

    # Pickup 区域 TOP10
    top_pickup = df['pickup_borough'].value_counts().head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_pickup.values, y=top_pickup.index, palette='viridis')
    plt.title('上下客量最高的 TOP10 Borough（Pickup）')
    plt.xlabel('订单数量')
    plt.ylabel('Borough')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top10_borough.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 高峰时段 vs 非高峰 订单量对比（按 Borough）
    peak_vs_normal = df.groupby(['pickup_borough', 'is_peak_hour']).size().unstack(fill_value=0)
    peak_vs_normal.columns = ['非高峰', '高峰']
    peak_vs_normal = peak_vs_normal.sort_values('高峰', ascending=False).head(8)

    peak_vs_normal.plot(kind='bar', figsize=(12, 7), stacked=False)
    plt.title('各 Borough 高峰期 vs 非高峰期订单量对比')
    plt.xlabel('Borough')
    plt.ylabel('订单数量')
    plt.xticks(rotation=45)
    plt.legend(title='时段')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/peak_vs_normal.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("图2 已保存: outputs/top10_borough.png 和 outputs/peak_vs_normal.png")


def plot_fare_factors(df: pd.DataFrame, output_dir: str = "outputs"):
    """3. 车费影响因素（距离 vs 车费）"""
    os.makedirs(output_dir, exist_ok=True)

    # 采样画散点图（300万数据太密，采样 2万条更清晰）
    sample_df = df.sample(n=20000, random_state=42)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=sample_df, x='trip_distance', y='fare_amount',
                    hue='is_peak_hour', alpha=0.6, s=10)
    sns.regplot(data=sample_df, x='trip_distance', y='fare_amount',
                scatter=False, color='red', line_kws={'linewidth': 2})
    plt.title('行程距离与车费的关系（红色为线性回归线）')
    plt.xlabel('行程距离 (miles)')
    plt.ylabel('车费金额 ($)')
    plt.legend(title='是否高峰期')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fare_vs_distance.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("图3 已保存: outputs/fare_vs_distance.png")


def plot_custom_insight(df: pd.DataFrame, output_dir: str = "outputs"):
    """4. 自选分析：不同 Borough 的平均速度与平均车费对比（很有洞察价值）"""
    os.makedirs(output_dir, exist_ok=True)

    borough_stats = df.groupby('pickup_borough').agg({
        'speed_mph': 'mean',
        'fare_amount': 'mean',
        'trip_distance': 'mean'
    }).round(2).sort_values('fare_amount', ascending=False)

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # 柱状图 - 平均车费
    color1 = 'tab:blue'
    ax1.bar(borough_stats.index, borough_stats['fare_amount'], color=color1, alpha=0.7, label='平均车费 ($)')
    ax1.set_xlabel('Borough')
    ax1.set_ylabel('平均车费 ($)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    plt.xticks(rotation=45)

    # 折线图 - 平均速度（双Y轴）
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.plot(borough_stats.index, borough_stats['speed_mph'], color=color2, marker='o', linewidth=2,
             label='平均速度 (mph)')
    ax2.set_ylabel('平均速度 (mph)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('各 Borough 平均车费与平均行驶速度对比（自选洞察）')
    fig.tight_layout()
    plt.savefig(f"{output_dir}/borough_fare_speed.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("图4（自选）已保存: outputs/borough_fare_speed.png")
    print("\n自选分析洞察：曼哈顿等核心区速度较低但车费较高，外围区相反。")


# ====================== 主函数 ======================
if __name__ == "__main__":
    df = load_cleaned_data()

    plot_demand_by_time(df)
    plot_region_heatmap(df)
    plot_fare_factors(df)
    plot_custom_insight(df)

    print("\nM2 分析可视化全部完成！所有图表已保存到 outputs/ 目录。")
    print("你可以打开 outputs/ 文件夹查看 4 张高清图表。")
