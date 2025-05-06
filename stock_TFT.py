import warnings
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer, TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.data import NaNLabelEncoder

# 忽略 pandas 的 SettingWithCopyWarning
warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)

def main():
    # 设置随机种子，确保结果可复现
    pl.seed_everything(42, workers=True)

    # 1. 加载数据（使用 Kaggle 上的文件路径）
    file_path = "/kaggle/input/vhzmdd/AAPL_stock_data_with_indicators.csv"
    data = pd.read_csv(file_path)

    # 2. 数据预处理
    # 转换日期格式，注意日期格式需和 CSV 文件一致
    data["Date"] = pd.to_datetime(data["Date"], format="%Y/%m/%d")
    data = data.sort_values("Date").reset_index(drop=True)
    data["time_idx"] = data.index + 1  # 时间索引从 1 开始
    # 假设数据只包含单只股票，将 stock_id 固定为 "AAPL"
    data["stock_id"] = "AAPL"
    data["static"] = data["stock_id"]

    # 剔除列名中包含异常字符串（如 "#NAME?"）的列
    drop_cols = [col for col in data.columns if "#NAME?" in col]
    if drop_cols:
        print(f"剔除异常列: {drop_cols}")
        data = data.drop(columns=drop_cols)

    # 前向填充缺失值
    data = data.fillna(method='ffill')

    # 3. 定义采样参数
    max_encoder_length = 60
    max_prediction_length = 20
    training_cutoff = data["time_idx"].max() - (max_encoder_length + max_prediction_length)
    print(f"数据总长度: {data['time_idx'].max()}, 训练截止点: {training_cutoff}")

    # 4. 定义额外技术指标（确保与 CSV 列名完全一致）
    additional_features = [
        "Open", "High", "Low", "Volume", "SMA_20", "EMA_20", "RSI_14",
        "BB_Middle", "BB_Upper", "BB_Lower", "EMA_12", "EMA_26",
        "MACD", "MACD_Signal", "ATR_14", "%K", "%D", "Momentum", "WMA_20",
        "TRIX", "VWAP", "CCI", "Bias", "RVI", "Williams %R", "MFI_14",
        "TR", "DM+", "DM-", "DX", "ADX_14", "SMA_Smooth_20", "Volatility",
        "Upper_Price_Channel", "Lower_Price_Channel", "Keltner_Middle",
        "Keltner_Upper", "Keltner_Lower"
    ]
    known_reals = ["time_idx"] + additional_features

    # 5. 构造训练集与验证集
    training_data = data[data.time_idx <= training_cutoff]
    validation_data = data[data.time_idx > training_cutoff]
    print(f"验证集数量: {len(validation_data)}")

    training_dataset = TimeSeriesDataSet(
        data=training_data,
        time_idx="time_idx",
        target="Close",
        group_ids=["stock_id"],
        categorical_encoders={"stock_id": NaNLabelEncoder().fit(data["stock_id"])},
        static_categoricals=["static"],
        time_varying_known_reals=known_reals,         # 使用额外的技术指标
        time_varying_unknown_reals=["Close"],           # 目标为 Close
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

    # 6. 构造 DataLoader
    batch_size = 64
    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=2, persistent_workers=True
    )
    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=batch_size, num_workers=2, persistent_workers=True
    )

    # 7. 定义回调函数和日志记录
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=1, verbose=True, mode="min"
    )
    lr_logger = LearningRateMonitor()
    tensorboard_logger = TensorBoardLogger("lightning_logs")

    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu",   # 使用 GPU，如无 GPU 则设置为 "cpu"
        devices=1,
        gradient_clip_val=0.1,
        limit_train_batches=30,  # 调试时每个 epoch 限制 30 个批次
        callbacks=[lr_logger, early_stop_callback],
        logger=tensorboard_logger
    )

    # 8. 构造 TemporalFusionTransformer 模型
    # 采用 QuantileLoss（这里默认设置量化分位数为 [0.1, 0.5, 0.9]）
    # 可以根据需要调整 hidden_size、dropout、hidden_continuous_size 等参数以提高预测精度
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        hidden_size=32,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=QuantileLoss(),
        log_interval=2,
        learning_rate=0.03,
        reduce_on_plateau_patience=4
    )
    print(f"模型参数数量: {tft.size() / 1e3:.1f}k")

    torch.set_num_threads(10)

    # 9. 开始训练
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # 10. 验证集预测与误差计算（采用中位数，即 quantile=0.5）
    predictions = tft.predict(val_dataloader, mode="quantiles", quantiles=[0.5])
    # 拼接批次中的实际目标值
    actuals = torch.cat([y for x, (y, _) in iter(val_dataloader)])
    actuals = actuals.to(predictions.device)  # 确保设备一致
    mae = (actuals - predictions).abs().mean()
    print(f"模型平均绝对误差 (MAE): {mae.item():.4f}")

if __name__ == '__main__':
    main()