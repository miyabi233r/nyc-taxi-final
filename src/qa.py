# src/qa.py
import pandas as pd
import re
import os


def load_qa_resources():

    import os
    import pandas as pd

    print("正在加载问答系统资源...")

    cleaned_path = r"data/cleaned_taxi_data.parquet"
    demand_path = r"data/demand_data.parquet"

    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"cleaned_taxi_data.parquet 不存在！路径：{cleaned_path}")

    df = pd.read_parquet(cleaned_path)

    # 生成或加载聚合需求数据
    if os.path.exists(demand_path):
        demand_df = pd.read_parquet(demand_path)
    else:
        demand_df = df.groupby(['pickup_hour', 'pickup_borough', 'is_weekend', 'is_peak_hour']).size().reset_index(
            name='demand')
        demand_df.to_parquet(demand_path)

    return df, demand_df


def process_question(question: str, df: pd.DataFrame, demand_df: pd.DataFrame):
    """最终版问题处理 - 调整匹配优先级"""
    q = question.lower().strip()

    # ==================== 1. 需求预测（最高优先级：包含“预测/预估/需求量”） ====================
    if any(k in q for k in ['预测', '预估', '需求量', '会多少', '有多少订单', '需求']):
        hour_match = re.search(r'(\d{1,2})[点时]?', q)
        borough_match = re.search(r'(曼哈顿|皇后|布鲁克林|布朗克斯|斯塔滕岛|manhattan|queens|brooklyn|bronx|staten)', q,
                                  re.I)

        hour = int(hour_match.group(1)) if hour_match else None
        borough = None
        if borough_match:
            b_map = {
                "曼哈顿": "Manhattan", "皇后": "Queens", "布鲁克林": "Brooklyn",
                "布朗克斯": "Bronx", "斯塔滕岛": "Staten Island",
                "manhattan": "Manhattan", "queens": "Queens", "brooklyn": "Brooklyn",
                "bronx": "Bronx", "staten": "Staten Island"
            }
            raw = borough_match.group(1).lower()
            borough = b_map.get(raw, raw.capitalize())

        if hour is not None and borough:
            pred = demand_df[(demand_df['pickup_hour'] == hour) & (demand_df['pickup_borough'] == borough)]
            if not pred.empty:
                demand = int(pred['demand'].iloc[0])
                is_peak = "高峰期" if hour in [7, 8, 9, 17, 18, 19] else "非高峰"
                return f"预测 **{borough}** 区 **{hour}点** 出行需求量约为 **{demand:,}** 单（{is_peak}）。", "outputs/nn_loss_curve.png"
            else:
                return f"暂无 **{borough}** 区 {hour}点的历史数据。", None

        # 只提到小时的预测
        elif hour is not None:
            count = len(df[df['pickup_hour'] == hour])
            return f"{hour}点全纽约出租车预计订单总量约 {count:,} 单。", "outputs/demand_by_hour.png"

    # ==================== 2. 纯时段订单查询 ====================
    if any(k in q for k in ['小时', '点', '时段', '晚上', '早上', '上午', '下午']) and any(
            k in q for k in ['订单', '多少', '多吗']):
        hour_match = re.search(r'(\d{1,2})[点时]?', q)
        if hour_match:
            hour = int(hour_match.group(1))
            if 0 <= hour <= 23:
                count = len(df[df['pickup_hour'] == hour])
                is_peak = "高峰期" if hour in [7, 8, 9, 17, 18, 19] else "非高峰"
                return f"{hour}点共有约 {count:,} 单出租车订单，属于{is_peak}。", "outputs/demand_by_hour.png"

    # ==================== 3. 区域热度排名 ====================
    if any(k in q for k in ['哪个区域', '哪里', '最热门', 'top', '最多', '热度', '排名']):
        top_borough = df['pickup_borough'].value_counts().head(5)
        result = "上下客量最高的 Borough（前5名）：\n"
        for b, c in top_borough.items():
            if pd.notna(b) and b != "Unknown":
                result += f"  - {b}: {c:,} 单\n"
        return result.strip(), "outputs/top10_borough.png"

    # ==================== 4. 车费估算 ====================
    if any(k in q for k in ['多少钱', '车费', '费用', '大概', '大约', '要花']):
        dist_match = re.search(r'(\d+\.?\d*)', q)
        dist = float(dist_match.group(1)) if dist_match else 5.0
        sample = df[df['trip_distance'].between(dist - 3, dist + 3)]
        avg_fare = sample['fare_amount'].mean() if not sample.empty else df['fare_amount'].mean()
        return f"行程距离约 {dist} 英里时，平均车费约为 **${avg_fare:.2f}**（高峰期或拥堵时会更高）。", "outputs/fare_vs_distance.png"

    # ==================== 5. 时间规律对比 ====================
    if any(k in q for k in ['周末', '工作日', '高峰', '非高峰', '差异', '对比']):
        weekend = df[df['is_weekend'] == 1].shape[0]
        weekday = df[df['is_weekend'] == 0].shape[0]
        peak = df[df['is_peak_hour'] == 1].shape[0]
        return (f"工作日订单总量: {weekday:,} 单\n"
                f"周末订单总量: {weekend:,} 单\n"
                f"高峰时段订单总量: {peak:,} 单\n"
                f"结论：工作日出行需求显著高于周末，高峰期集中明显。"), "outputs/peak_vs_normal.png"

    # ==================== 默认回复 ====================
    return ("我目前支持以下查询：\n"
            "• 时段订单（如“晚上8点订单多少”）\n"
            "• 区域热度（如“哪个区域最热门”）\n"
            "• 需求预测（如“预测皇后区9点需求”）\n"
            "• 车费估算（如“10英里大概多少钱”）\n"
            "• 时间对比（如“高峰期和周末差异”）\n"
            "请尝试更具体的自然语言提问~"), None


def simple_qa_system():
    print("\n" + "=" * 70)
    print("   纽约出租车出行数据智能问答系统（M4 - 最终版）")
    print("支持自然语言查询 • 已整合 M1-M3 分析结果")
    print("=" * 70)
    print("推荐测试问题：")
    print("  - 预测皇后区9点需求")
    print("  - 预测曼哈顿晚上8点需求量")
    print("  - 哪个区域最热门？")
    print("  - 从曼哈顿到机场大概多少钱？")
    print("  - 高峰期和周末差异？")
    print("输入 '退出' 或 'exit' 结束\n")

    df, demand_df = load_qa_resources()

    while True:
        question = input("请输入你的问题: ").strip()
        if question.lower() in ['退出', 'exit', 'quit', 'bye']:
            print("感谢使用，再见！")
            break

        answer, img_path = process_question(question, df, demand_df)
        print(f"\n回答: {answer}")
        if img_path:
            print(f"相关图表路径: {img_path}")
        print("-" * 60)


if __name__ == "__main__":
    simple_qa_system()
