import warnings
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer, NHiTS, MQF2DistributionLoss
from pytorch_forecasting.data import NaNLabelEncoder

# 忽略 pandas 的 SettingWithCopyWarning
warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)


def main():
    # 固定随机种子，保证结果可复现
    pl.seed_everything(42, workers=True)

    # 1. 加载数据
    # 如果你使用 Kaggle 数据，请修改为相应路径
    file_path = "/kaggle/input/vhzmdd/AAPL_stock_data_with_indicators.csv"
    data = pd.read_csv(file_path)

    # 2. 数据预处理
    data["Date"] = pd.to_datetime(data["Date"], format="%Y/%m/%d")
    data = data.sort_values("Date").reset_index(drop=True)
    data["time_idx"] = data.index + 1  # 时间索引从 1 开始
    # 假设数据仅包含单只股票，将 stock_id 固定
    data["stock_id"] = "AAPL"
    data["static"] = data["stock_id"]

    # 剔除含异常字符串（如 "#NAME?"）的列
    drop_cols = [col for col in data.columns if "#NAME?" in col]
    if drop_cols:
        print(f"剔除异常列: {drop_cols}")
        data = data.drop(columns=drop_cols)

    # 前向填充缺失值
    data = data.fillna(method='ffill')

    # 3. 设置采样参数
    max_encoder_length = 30
    max_prediction_length = 10
    training_cutoff = data["time_idx"].max() - (max_encoder_length + max_prediction_length)
    print(f"数据总长度: {data['time_idx'].max()}, 训练截止点: {training_cutoff}")

    # 4. 定义额外技术指标（请确保数据中包含对应的列，如 Open, High, Low 等）
    additional_features = [
        "Open", "High", "Low", "Volume", "SMA_20", "EMA_20", "RSI_14",
        "BB_Middle", "BB_Upper", "BB_Lower", "EMA_12", "EMA_26",
        "MACD", "MACD_Signal", "ATR_14", "%K", "%D", "Momentum", "WMA_20"
    ]
    known_reals = ["time_idx"] + additional_features

    # 5. 构造训练集
    training_data = data[data.time_idx <= training_cutoff]
    training_dataset = TimeSeriesDataSet(
        data=training_data,
        time_idx="time_idx",
        target="Close",
        group_ids=["stock_id"],
        categorical_encoders={"stock_id": NaNLabelEncoder().fit(data["stock_id"])},
        static_categoricals=["static"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        # 目标仅为 Close
        time_varying_unknown_reals=["Close"],
        # 已知变量包含时间索引和额外技术指标
        time_varying_known_reals=known_reals,
        target_normalizer=GroupNormalizer(groups=["stock_id"], transformation="softplus"),
        # 对 NHiTS 此处设置为 False，若需要可考虑设置为 True
        add_relative_time_idx=False,
        add_target_scales=True,
        randomize_length=None,
    )

    # 构造验证集，起始位置设置为训练截止点之后
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, data, min_prediction_idx=training_cutoff + 1
    )

    batch_size = 64
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    # 6. 设置回调：EarlyStopping与LearningRateMonitor
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu",  # 如无 GPU，可设置为 "cpu"
        enable_model_summary=True,
        gradient_clip_val=1.0,
        callbacks=[lr_logger, early_stop_callback],
        limit_train_batches=30,
        enable_checkpointing=True,
    )

    # 7. 构造 NHiTS 模型
    net = NHiTS.from_dataset(
        training_dataset,
        learning_rate=5e-3,
        log_interval=10,
        log_val_interval=1,
        weight_decay=1e-2,
        backcast_loss_ratio=0.0,
        hidden_size=64,
        optimizer="AdamW",
        loss=MQF2DistributionLoss(prediction_length=max_prediction_length),
    )

    print(f"模型参数数量: {net.size() / 1e3:.1f}k")
    torch.set_num_threads(10)

    # 8. 开始训练
    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # 9. 验证集预测与误差计算（此处以平均绝对误差 MAE 为例，注意数据为归一化后的结果）
    actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
    predictions = net.predict(val_dataloader)
    device = predictions.device
    actuals = actuals.to(device)
    mae = (actuals - predictions).abs().mean()
    print(f"Mean absolute error of model: {mae.item():.4f}")


if __name__ == '__main__':
    main()