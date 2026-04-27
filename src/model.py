# src/model.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 设置中文字体（与 M2 保持一致）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_cleaned_data(data_path: str = None):
    """使用绝对路径加载清洗后的数据（M3）"""
    import os
    print("正在加载清洗后的数据用于建模...")

    data_path = r"C:\Users\26779\Videos\桌面\学习\人工智能编程语言\AI_Prog_HW\data\cleaned_taxi_data.parquet"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"cleaned_taxi_data.parquet 不存在！路径：{data_path}")

    df = pd.read_parquet(data_path)
    print(f"加载完成，原始形状: {df.shape}")
    return df


def prepare_prediction_data(df: pd.DataFrame):
    """
    增强版特征工程：增加更多特征，让模型有足够信息学习
    """
    print("正在准备增强版预测数据集...")

    # 1. 基础聚合：小时 + Borough 的需求量
    demand_df = df.groupby(['pickup_hour', 'pickup_borough', 'is_weekend', 'is_peak_hour']).size().reset_index(
        name='demand')

    # 2. 增加丰富特征
    # - 该Borough的整体平均需求
    borough_avg = demand_df.groupby('pickup_borough')['demand'].mean().reset_index(name='borough_avg_demand')
    demand_df = demand_df.merge(borough_avg, on='pickup_borough', how='left')

    # - 该小时的整体平均需求（已存在）
    hourly_avg = demand_df.groupby('pickup_hour')['demand'].mean().reset_index(name='hourly_avg_demand')
    demand_df = demand_df.merge(hourly_avg, on='pickup_hour', how='left')

    # - Borough + 是否周末的交互特征（平均需求）
    borough_weekend_avg = demand_df.groupby(['pickup_borough', 'is_weekend'])['demand'].mean().reset_index(
        name='borough_weekend_avg')
    demand_df = demand_df.merge(borough_weekend_avg, on=['pickup_borough', 'is_weekend'], how='left')

    # - 是否机场相关区域（简单规则，JFK, EWR 等通常需求特殊）
    airport_boroughs = ['Queens', 'Brooklyn']  # 简化，实际可更精确
    demand_df['is_airport_area'] = demand_df['pickup_borough'].isin(airport_boroughs).astype(int)

    print(f"增强后数据集形状: {demand_df.shape} （共 {len(demand_df)} 个样本）")
    print("新增特征：borough_avg_demand, borough_weekend_avg, is_airport_area 等")
    return demand_df


def build_and_train_models(demand_df: pd.DataFrame, output_dir: str = "outputs"):
    """
    构建并训练神经网络 + 随机森林，对比实验
    """
    os.makedirs(output_dir, exist_ok=True)

    # 特征和目标
    features = ['pickup_hour', 'is_weekend', 'is_peak_hour', 'hourly_avg_demand']
    categorical_features = ['pickup_borough']  # 需要 One-Hot

    X = demand_df[features + categorical_features]
    y = demand_df['demand']

    # 数据集划分 8:2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"训练集大小: {X_train.shape[0]}，测试集大小: {X_test.shape[0]}")

    # 数据预处理（OneHot + 标准化）
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', StandardScaler(), [f for f in features if f != 'pickup_borough'])
        ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # ====================== 1. 随机森林模型 ======================
    print("\n=== 训练 Random Forest 模型 ===")
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_processed, y_train)

    y_pred_rf = rf_model.predict(X_test_processed)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    print(f"Random Forest - MAE: {mae_rf:.2f}，RMSE: {rmse_rf:.2f}")

    # ====================== 2. 神经网络模型 (优化版) ======================
    print("\n=== 训练 神经网络 (MLP) 模型（优化后） ===")

    input_dim = X_train_processed.shape[1]

    nn_model = keras.Sequential([
        layers.Input(shape=(input_dim,)),  # 修复 UserWarning
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    nn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])

    # 早停 + 更合理的训练参数（适合小样本）
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = nn_model.fit(
        X_train_processed, y_train,
        validation_split=0.2,
        epochs=100,  # 增加 epochs，但有早停
        batch_size=32,  # 小样本用更小 batch
        callbacks=[early_stopping],
        verbose=1
    )

    # 测试集评估
    y_pred_nn = nn_model.predict(X_test_processed).flatten()
    mae_nn = mean_absolute_error(y_test, y_pred_nn)
    rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))

    print(f"Neural Network - MAE: {mae_nn:.2f}，RMSE: {rmse_nn:.2f}")

    # 保存模型
    nn_model.save("models/taxi_demand_nn_model.keras")
    print("神经网络模型已保存至: models/taxi_demand_nn_model.keras")

    # ====================== 3. 绘制 Loss 曲线 ======================
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='训练 Loss')
    plt.plot(history.history['val_loss'], label='验证 Loss')
    plt.title('神经网络训练 Loss 曲线')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/nn_loss_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss 曲线已保存: outputs/nn_loss_curve.png")

    # ====================== 4. 对比分析 ======================
    print("\n=== 模型对比结果 ===")
    print(f"{'模型':<20} {'MAE':<10} {'RMSE':<10}")
    print(f"{'Random Forest':<20} {mae_rf:<10.2f} {rmse_rf:<10.2f}")
    print(f"{'Neural Network':<20} {mae_nn:<10.2f} {rmse_nn:<10.2f}")

    if mae_nn < mae_rf:
        print("结论：神经网络在本任务上表现更好（MAE 更低）。")
    else:
        print("结论：随机森林在本任务上表现更好（MAE 更低）。")

    print("\n分析：随机森林通常训练更快、可解释性强；神经网络在数据量大、特征复杂时可能捕捉更深层非线性关系。")

    return {
        'rf_mae': mae_rf, 'rf_rmse': rmse_rf,
        'nn_mae': mae_nn, 'nn_rmse': rmse_nn
    }


# ====================== 主函数 ======================
if __name__ == "__main__":
    # 创建 models 目录
    os.makedirs("models", exist_ok=True)

    df = load_cleaned_data()
    demand_df = prepare_prediction_data(df)
    metrics = build_and_train_models(demand_df)

    print("\nM3 预测模型模块执行完成！")
    print("已生成：Loss 曲线图 + 两个模型的 MAE/RMSE 对比")
    print("下一步可以进行 M4 问答接口。")