import warnings
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
from pytorch_forecasting.data import NaNLabelEncoder
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from lightning.pytorch.tuner import Tuner

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
    max_encoder_length = 60
    max_prediction_length = 20

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
        add_relative_time_idx=True,
        add_target_scales=True,
        randomize_length=None,
    )

    # 5. 构造验证数据集
    validation_data = data[data.time_idx > training_cutoff]
    print(f"验证集数量: {len(validation_data)}")

    validation = TimeSeriesDataSet.from_dataset(
        training,
        validation_data,
        stop_randomization=True,
    )

    # 6. 创建 DataLoader（降低 num_workers 防止内存不足问题）
    batch_size = 64
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=2, persistent_workers=True
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=2, persistent_workers=True
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu",  # 使用 GPU
        devices=1,
        gradient_clip_val=0.1,
        limit_train_batches=30,  # 30 batches per epoch
        callbacks=[lr_logger, early_stop_callback],
        logger=TensorBoardLogger("lightning_logs")
    )

    # define network to train - the architecture is mostly inferred from the dataset, so that only a few hyperparameters have to be set by the user
    tft = TemporalFusionTransformer.from_dataset(
        # dataset
        training,
        # architecture hyperparameters
        hidden_size=32,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=16,
        # loss metric to optimize
        loss=QuantileLoss(),
        # logging frequency
        log_interval=2,
        # optimizer parameters
        learning_rate=0.03,
        reduce_on_plateau_patience=4
    )
    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

    # 9. 开始训练
    torch.set_num_threads(10)
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # 10. 验证集预测和误差计算（以平均绝对误差MAE为例）
    actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
    predictions = tft.predict(val_dataloader)
    device = predictions.device
    actuals = actuals.to(device)
    mae = (actuals - predictions).abs().mean()
    print(f"Mean absolute error of model: {mae.item():.4f}")


if __name__ == '__main__':
    main()
