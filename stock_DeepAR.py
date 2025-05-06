import warnings
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE

# 忽略 pandas 的 SettingWithCopyWarning
warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)

def main():
    # 设置随机种子，确保结果可复现
    pl.seed_everything(42, workers=True)

    # 1. 加载数据
    data = pd.read_csv("AAPL_stock_data_with_indicators.csv")

    # 2. 数据预处理
    # 将日期转换为 datetime 格式
    data["Date"] = pd.to_datetime(data["Date"], format="%Y/%m/%d")
    data = data.sort_values("Date").reset_index(drop=True)
    data["time_idx"] = data.index + 1  # 时间索引从1开始
    # 假设数据只包含单一股票，将 stock_id 固定为 "AAPL"
    data["stock_id"] = "AAPL"
    data["static"] = data["stock_id"]

    # 如果存在列名中包含 "#NAME?" 的列，则剔除
    drop_cols = [col for col in data.columns if "#NAME?" in col]
    if drop_cols:
        print(f"剔除异常列: {drop_cols}")
        data = data.drop(columns=drop_cols)

    # 填充缺失值，可以根据需要选择其他填充策略
    data = data.fillna(method='ffill')

    # 3. 定义 encoder 和 decoder 的步长
    max_encoder_length = 60
    max_prediction_length = 20

    training_cutoff = data["time_idx"].max() - (max_encoder_length + max_prediction_length)
    print(f"数据总长度: {data['time_idx'].max()}, 训练截止点: {training_cutoff}")

    # 4. 定义额外的技术指标（注意部分指标名称包含特殊字符，如 “%K”、“Williams %R”，需保证和 CSV 列名一致）
    additional_features = [
        "Open", "High", "Low", "Volume", "SMA_20", "EMA_20", "RSI_14",
        "BB_Middle", "BB_Upper", "BB_Lower", "EMA_12", "EMA_26",
        "MACD", "MACD_Signal", "ATR_14", "%K", "%D", "Momentum", "WMA_20",
        "TRIX", "VWAP", "CCI", "Bias", "RVI", "Williams %R", "MFI_14",
        "TR", "DM+", "DM-", "DX", "ADX_14", "SMA_Smooth_20", "Volatility",
        "Upper_Price_Channel", "Lower_Price_Channel", "Keltner_Middle",
        "Keltner_Upper", "Keltner_Lower"
    ]

    # 将时间索引用作基础的已知变量，再加上技术指标：
    known_reals = ["time_idx"] + additional_features

    # 5. 划分数据集
    training_data = data[data.time_idx <= training_cutoff]
    validation_data = data[data.time_idx > training_cutoff]
    print(f"验证集数量: {len(validation_data)}")

    # 6. 构造 TimeSeriesDataSet 数据集
    training_dataset = TimeSeriesDataSet(
        data=training_data,
        time_idx="time_idx",
        target="Close",
        group_ids=["stock_id"],
        categorical_encoders={"stock_id": NaNLabelEncoder().fit(data["stock_id"])},
        static_categoricals=["static"],
        # 将额外技术指标作为“已知的时变实数变量”
        time_varying_known_reals=known_reals,
        # 目标 "Close" 被视为未知变量（注意：这样设定时，历史值会作为参考，但未来目标需要模型预测）
        time_varying_unknown_reals=["Close"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        target_normalizer=GroupNormalizer(groups=["stock_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        randomize_length=None,
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        validation_data,
        stop_randomization=True,
    )

    # 7. 构造 DataLoader
    batch_size = 64
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=2)

    # 8. 定义回调函数和 Trainer
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min"
    )
    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu",  # 如果没有 GPU，可设为 "cpu"
        devices="auto",
        gradient_clip_val=0.1,
        limit_train_batches=30,  # 调试时限制批次数，可根据实际情况取消
        limit_val_batches=3,
        callbacks=[lr_logger, early_stop_callback],
    )

    # 9. 使用 TemporalFusionTransformer 模型
    # TFT 利用多重协变量及注意力机制往往能更好地融合丰富的历史信息
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,  # 输出多个分位数，默认包含 10%, 50%, 90%
        loss=MAE(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    print(f"模型参数数量: {tft.size() / 1e3:.1f}k")

    # 根据需要设置 CPU 线程数
    torch.set_num_threads(10)

    # 10. 模型训练
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # 11. 预测和评估
    # 使用默认中位数（50% 分位数）作为预测结果
    predictions = tft.predict(val_dataloader, mode="prediction")
    # 收集实际目标值（这里注意集成方式可能因数据不同而不同）
    actuals = torch.cat([y for x, (y, _) in iter(val_dataloader)])
    mae = (actuals - predictions).abs().mean()
    print(f"模型平均绝对误差 (MAE): {mae.item():.4f}")

if __name__ == '__main__':
    main()