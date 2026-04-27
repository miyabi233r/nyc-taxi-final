# src/data_processing.py
import pandas as pd
import numpy as np
import os
from datetime import datetime


# ====================== M1 数据处理模块 ======================
# 作业要求：加载数据 → 生成质量报告 → 清洗数据（注释说明理由）→ 提取特征 + 至少2个衍生特征

def load_data(data_path: str = None, zone_path: str = None):
    """
    使用绝对路径加载数据（按你的要求修改）
    """
    import os

    print("正在加载数据（约300万条记录，请耐心等待）...")

    # === 这里改成你的绝对路径（最重要！）===
    if data_path is None:
        data_path = r"data/yellow_tripdata_2023-01.parquet"
        zone_path = r"data/taxi_zone_lookup.csv"

    # 检查文件是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在！请检查路径：\n{data_path}")

    df = pd.read_parquet(data_path, engine='pyarrow')
    zones = pd.read_csv(zone_path)

    print(f"原始数据形状: {df.shape}")
    print(f"区域映射表形状: {zones.shape}")
    return df, zones


def generate_data_quality_report(df: pd.DataFrame, output_dir: str = "outputs"):
    """
    生成数据质量报告：缺失率、异常值统计
    报告会保存为 outputs/data_quality_report.txt
    """
    os.makedirs(output_dir, exist_ok=True)

    report = []
    report.append("=== 纽约出租车数据质量报告 (2023年1月) ===\n")
    report.append(f"总记录数: {len(df):,}")
    report.append(f"总列数: {len(df.columns)}")
    report.append(f"数据占用内存: {df.memory_usage(deep=True).sum() / (1024 ** 3):.2f} GB\n")

    # 缺失率
    report.append("=== 缺失值统计 ===")
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    for col, rate in missing.items():
        report.append(f"{col}: {rate:.4f}%")
    if len(missing) == 0:
        report.append("无缺失值")

    # 异常值统计（关键业务字段）
    report.append("\n=== 异常值统计 ===")
    report.append(
        f"行程距离 <= 0: {len(df[df['trip_distance'] <= 0]):,} 条 ({len(df[df['trip_distance'] <= 0]) / len(df) * 100:.4f}%)")
    report.append(f"车费金额 <= 0: {len(df[df['fare_amount'] <= 0]):,} 条")
    report.append(f"乘客人数异常 (>6 或 0): {len(df[(df['passenger_count'] == 0) | (df['passenger_count'] > 6)]):,} 条")
    report.append(f"行程时间 <= 0 秒: {len(df[df['tpep_dropoff_datetime'] <= df['tpep_pickup_datetime']]):,} 条")

    # 保存报告
    report_path = os.path.join(output_dir, "data_quality_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"数据质量报告已保存至: {report_path}")
    print(df.describe().round(2))  # 控制台快速查看数值统计
    return report_path


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗策略（每步都写明理由，符合作业要求）
    """
    print("开始数据清洗...")
    original_shape = df.shape

    # 1. 删除完全重复的记录（理由：同一行程不应重复记录，减少噪声）
    df = df.drop_duplicates()

    # 2. 处理缺失值（理由：passenger_count 缺失可能是系统问题，填众数1更合理；其他关键字段缺失直接删除）
    df['passenger_count'] = df['passenger_count'].fillna(1)
    df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime',
                           'trip_distance', 'fare_amount'])

    # 3. 过滤明显异常值（理由：这些记录不符合实际业务逻辑，会严重影响后续分析和模型）
    df = df[df['trip_distance'] > 0]  # 距离必须 > 0
    df = df[df['fare_amount'] > 0]  # 车费必须 > 0
    df = df[df['tpep_dropoff_datetime'] > df['tpep_pickup_datetime']]  # 结束时间 > 开始时间

    # 乘客数限制在合理范围（1-6人，理由：出租车最多载6人，0人无意义）
    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]

    # 4. 行程时间限制（理由：过滤超长或极短行程，避免异常影响统计）
    df['trip_duration_minutes'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    df = df[(df['trip_duration_minutes'] > 1) & (df['trip_duration_minutes'] < 180)]  # 1分钟 ~ 3小时

    print(f"清洗前: {original_shape[0]:,} 条 → 清洗后: {len(df):,} 条")
    print(
        f"共删除 {original_shape[0] - len(df):,} 条异常/无效记录 ({(original_shape[0] - len(df)) / original_shape[0] * 100:.2f}%)")

    return df


def feature_engineering(df: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    """
    特征工程：
    1. 从行程时间提取：小时、星期、是否周末、是否高峰期
    2. 至少2个衍生特征（作业要求）：trip_duration_minutes、speed_mph（平均速度）
    """
    print("开始特征工程...")

    # 时间特征
    df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday  # 0=周一 ... 6=周日
    df['is_weekend'] = df['pickup_weekday'].isin([5, 6]).astype(int)

    # 高峰期定义（作业常见做法）：早高峰 7-9点，晚高峰 17-19点
    df['is_peak_hour'] = df['pickup_hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # 衍生特征1：行程时长（分钟）—— 已在上一步清洗中计算，这里确保存在
    if 'trip_duration_minutes' not in df.columns:
        df['trip_duration_minutes'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

    # 衍生特征2：平均速度（mph）—— 非常有意义的衍生特征，能反映交通状况
    # 公式：速度(mph) = 距离(miles) / (时长(hours))
    df['trip_duration_hours'] = df['trip_duration_minutes'] / 60
    df['speed_mph'] = df['trip_distance'] / df['trip_duration_hours']
    # 限制极端速度（理由：实际车速不可能超过100mph或低于1mph）
    df = df[(df['speed_mph'] > 1) & (df['speed_mph'] < 100)]

    # 衍生特征3：每英里费用（修复 SettingWithCopyWarning）
    df = df.copy()  # 显式创建副本，避免警告
    df['fare_per_mile'] = df['fare_amount'] / df['trip_distance']

    # 合并区域信息（非常重要！把LocationID转为可读的 Borough 和 Zone）
    zones = zones[['LocationID', 'Borough', 'Zone']]
    df = df.merge(zones, left_on='PULocationID', right_on='LocationID', how='left')
    df = df.rename(columns={'Borough': 'pickup_borough', 'Zone': 'pickup_zone'})
    df = df.drop(columns=['LocationID'])

    df = df.merge(zones, left_on='DOLocationID', right_on='LocationID', how='left')
    df = df.rename(columns={'Borough': 'dropoff_borough', 'Zone': 'dropoff_zone'})
    df = df.drop(columns=['LocationID'])

    print("特征工程完成！新增特征包括：pickup_hour, is_weekend, is_peak_hour, trip_duration_minutes, speed_mph 等")
    return df


def save_cleaned_data(df: pd.DataFrame, output_path: str = "data/cleaned_taxi_data.parquet"):
    """保存清洗后的数据（后续模块可复用）"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"清洗后的数据已保存至: {output_path} （形状: {df.shape}）")


# ====================== 主函数（便于测试） ======================
if __name__ == "__main__":
    df, zones = load_data()
    generate_data_quality_report(df)

    df_clean = clean_data(df)
    df_clean = feature_engineering(df_clean, zones)

    save_cleaned_data(df_clean)

    print("\nM1 数据处理模块执行完成！")
    print("下一步可以进行 M2 可视化分析。")
