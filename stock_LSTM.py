import warnings
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.models.nn.rnn import LSTM as ForecastingLSTM

# 忽略 pandas 的 SettingWithCopyWarning
warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)


class LSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, prediction_length, num_layers=1, lr=1e-3):
        """
        :param input_size: 输入特征数
        :param hidden_size: LSTM 隐藏层维度
        :param prediction_length: 预测区间长度
        :param num_layers: LSTM 层数
        :param lr: 学习率
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        # 使用 pytorch_forecasting 中的 LSTM 层（设置 batch_first=True）
        self.lstm = ForecastingLSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True
        )
        # 使用全连接层，将最后的隐藏状态映射到预测区间
        self.fc = torch.nn.Linear(hidden_size, prediction_length)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        """
        前向传播：使用 x["encoder_cont"] (形状为 (batch, encoder_length, num_features))
        """
        encoder_input = x["encoder_cont"]
        # lstm_out: (batch, encoder_length, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        _, (h_n, _) = self.lstm(encoder_input)
        # 使用最后一层隐藏状态 (batch, hidden_size)
        last_hidden = h_n[-1]
        prediction = self.fc(last_hidden)  # (batch, prediction_length)
        prediction = prediction.unsqueeze(-1)  # (batch, prediction_length, 1)
        return prediction

    def training_step(self, batch, batch_idx):
        x, (y, weight) = batch
        y_hat = self(x)
        # 调整目标维度为 (batch, prediction_length, 1)
        # 若 y 的形状不符合要求，则进行变换
        if y.dim() == 2:
            y = y.unsqueeze(-1)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, (y, weight) = batch
        y_hat = self(x)
        if y.dim() == 2:
            y = y.unsqueeze(-1)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1,
            }
        }


def main():
    pl.seed_everything(42, workers=True)

    # 1. 加载数据（使用 Kaggle 上的文件路径，新数据需包含丰富的技术指标）
    file_path = "/kaggle/input/vhzmdd/AAPL_stock_data_with_indicators.csv"
    data = pd.read_csv(file_path)

    # 2. 数据预处理
    # 将日期转换为 datetime 类型，并按照日期排序
    data["Date"] = pd.to_datetime(data["Date"], format="%Y/%m/%d")
    data = data.sort_values("Date").reset_index(drop=True)
    data["time_idx"] = data.index + 1  # 构造时间索引，从 1 开始
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

    # 3. 设置采样参数
    max_encoder_length = 30
    max_prediction_length = 10
    training_cutoff = data["time_idx"].max() - (max_encoder_length + max_prediction_length)
    print(f"数据总长度: {data['time_idx'].max()}, 训练截止点: {training_cutoff}")

    # 4. 定义额外技术指标（请确保 CSV 文件中包含对应的列）
    additional_features = [
        "Open", "High", "Low", "Volume", "SMA_20", "EMA_20", "RSI_14",
        "BB_Middle", "BB_Upper", "BB_Lower", "EMA_12", "EMA_26",
        "MACD", "MACD_Signal", "ATR_14", "%K", "%D", "Momentum", "WMA_20",
    ]
    # 将时间索引与额外指标合并
    known_reals = ["time_idx"] + additional_features

    # 5. 构造训练集与验证集数据集
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
        time_varying_known_reals=known_reals,  # 使用额外技术指标作为已知特征
        time_varying_unknown_reals=["Close"],  # 目标为 Close
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

    # 7. 定义回调函数与日志记录
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min")
    lr_logger = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu",  # 如无 GPU，请设置为 "cpu"
        devices="auto",
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback],
    )

    # 打印输入特征数
    sample_batch = next(iter(train_dataloader))
    input_size = sample_batch[0]["encoder_cont"].shape[-1]
    print(f"Input feature size for LSTM: {input_size}")

    # 8. 构造 LSTM 模型
    lstm_model = LSTMModel(
        input_size=input_size,
        hidden_size=128,
        prediction_length=max_prediction_length,
        num_layers=1,
        lr=1e-3
    )
    total_params = sum(p.numel() for p in lstm_model.parameters())
    print(f"Number of parameters in network: {total_params / 1e3:.1f}k")

    torch.set_num_threads(10)
    trainer.fit(lstm_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # 9. 评估阶段：注意预测和实际值均为归一化后的结果
    lstm_model.eval()
    predictions_list = []
    actuals_list = []
    with torch.no_grad():
        for batch in val_dataloader:
            x, (y, weight) = batch
            y_hat = lstm_model(x)
            predictions_list.append(y_hat)
            actuals_list.append(y)
    predictions = torch.cat(predictions_list, dim=0)
    actuals = torch.cat(actuals_list, dim=0)
    mae = (actuals - predictions).abs().mean()
    print(f"Mean absolute error: {mae.item():.4f}")


if __name__ == '__main__':
    main()