import warnings
import pandas as pd
import torch
from pytorch_forecasting import GroupNormalizer,  MQF2DistributionLoss
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import lightning.pytorch as pl
from pytorch_forecasting import NHiTS, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder

# 不再提示pd的SettingWithCopyWarning为错误
warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)

def main():
    # 设置随机种子，确保结果可复现
    pl.seed_everything(42, workers=True)

    # 1. 加载股票数据
    data = pd.read_csv("stock_data.csv")

    # 2. 数据预处理
    data["Date"] = pd.to_datetime(data["Date"], format="%Y/%m/%d")
    data = data.sort_values("Date").reset_index(drop=True)
    data["time_idx"] = data.index + 1  # 从1开始构造顺序的时间索引
    data["static"] = data["stock_id"]

    # 3. 定义采样参数
    max_encoder_length = 30
    max_prediction_length = 10

    # 调整训练截止点，确保验证数据中至少有 max_encoder_length + max_prediction_length 个时间步
    training_cutoff = data["time_idx"].max() - (max_encoder_length + max_prediction_length)
    print(f"数据总长度: {data['time_idx'].max()}, 训练截止点: {training_cutoff}")

    # 4. 构造训练数据集
    training = TimeSeriesDataSet(
        data=data[data.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="Close",
        group_ids=["stock_id"],
        categorical_encoders={"stock_id": NaNLabelEncoder().fit(data["stock_id"])},
        static_categoricals=["static"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=["Close"],
        time_varying_known_reals=["time_idx"],
        target_normalizer=GroupNormalizer(groups=["stock_id"], transformation="softplus"),
        add_relative_time_idx=False,
        add_target_scales=True,
        randomize_length=None,
    )

    validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training_cutoff + 1)
    batch_size = 64
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    pl.seed_everything(42)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu",
        enable_model_summary=True,
        gradient_clip_val=1.0,
        callbacks=[lr_logger, early_stop_callback],
        limit_train_batches=30,
        enable_checkpointing=True,
    )

    net = NHiTS.from_dataset(
        training,
        learning_rate=5e-3,
        log_interval=10,
        log_val_interval=1,
        weight_decay=1e-2,
        backcast_loss_ratio=0.0,
        hidden_size=64,
        optimizer="AdamW",
        loss=MQF2DistributionLoss(prediction_length=max_prediction_length),
    )

    # 9. 开始训练
    torch.set_num_threads(10)
    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # 10. 验证集预测和误差计算（以平均绝对误差MAE为例）
    actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
    predictions = net.predict(val_dataloader)
    device = predictions.device
    actuals = actuals.to(device)
    mae = (actuals - predictions).abs().mean()
    print(f"Mean absolute error of model: {mae.item():.4f}")

if __name__ == '__main__':
    main()
