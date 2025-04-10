import os
import torch

import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


@dataclass
class TimeSplit:
    """
    Dataset splits based on time.
    """

    train: tuple[str, str]
    val: tuple[str, str]
    test: tuple[str, str]


ETT_SPLIT = TimeSplit(
    val=("2017-10-01", "2018-02-01"),
    test=("2018-02-01", "2018-06-01"),
    train=("2016-10-01", "2017-10-01"),
)

DATA_PATH = Path(__file__).parent / "datasets"


def get_datasets(
    dataset_name: str, lookback_size: int, horizon_size: int
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Utility function to obtain Train/Val/Test datasets.
    :param dataset_name: dataset name
    :param lookback_size: history window length used for model input
    :param horizon_size: horizon window length to forecast
    :returns: corresponding Train/Val/Test datasets
    """
    if "ETT" in dataset_name:
        data = pd.read_csv(DATA_PATH / dataset_name)

        train_df = data[data.date.between(*ETT_SPLIT.train)].copy()
        train_ds = ETTDataset(
            dataframe=train_df,
            lookback_size=lookback_size,
            horizon_size=horizon_size,
            scaler=None,
        )

        val_df = data[data.date.between(*ETT_SPLIT.val)].copy()
        val_ds = ETTDataset(
            dataframe=val_df,
            lookback_size=lookback_size,
            horizon_size=horizon_size,
            scaler=train_ds.scaler,
        )

        test_df = data[data.date.between(*ETT_SPLIT.test)].copy()
        test_ds = ETTDataset(
            dataframe=test_df,
            lookback_size=lookback_size,
            horizon_size=horizon_size,
            scaler=train_ds.scaler,
        )

        return train_ds, val_ds, test_ds

    raise ValueError(f"Unknown dataset name: {dataset_name=}")


class ETTDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        lookback_size: int,
        horizon_size: int,
        scaler: StandardScaler | None,
    ):
        """
        Electricity Transformer Temperature dataset.
        :param dataframe: dataframe containing time series data
        :param lookback_size: history window length used for model input
        :param horizon_size: horizon window length to forecast
        """
        super().__init__()
        self.horizon_size = horizon_size
        self.lookback_size = lookback_size
        self.window_size = horizon_size + lookback_size

        dataframe["month"] = dataframe["date"].apply(
            lambda date: datetime.fromisoformat(date).month
        )
        dataframe["weekday"] = dataframe["date"].apply(
            lambda date: datetime.fromisoformat(date).weekday()
        )
        dataframe["day"] = dataframe["date"].apply(
            lambda date: datetime.fromisoformat(date).day
        )
        dataframe["hour"] = dataframe["date"].apply(
            lambda date: datetime.fromisoformat(date).hour
        )
        dataframe["minute"] = (
            dataframe["date"].apply(lambda date: datetime.fromisoformat(date).minute)
            // 15
        )  # 15 minutes intervals only

        time_values = dataframe[["month", "weekday", "day", "hour", "minute"]].values
        data_values = dataframe[
            ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        ].values

        if scaler is None:
            self.scaler = StandardScaler()
            data_values = self.scaler.fit_transform(data_values)
        else:
            self.scaler = scaler
            data_values = self.scaler.transform(data_values)

        self.time_values = torch.tensor(data=time_values, dtype=torch.long)
        self.data_values = torch.tensor(data=data_values, dtype=torch.float32)

    def __len__(self) -> int:
        return max(0, self.time_values.shape[0] - self.window_size + 1)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        start_idx = idx
        x_slice = slice(start_idx, start_idx + self.lookback_size)
        x_time, x_data = self.time_values[x_slice], self.data_values[x_slice]
        y_slice = slice(start_idx + self.lookback_size, start_idx + self.window_size)
        _, y_data = self.time_values[y_slice], self.data_values[y_slice]
        return x_time, x_data, y_data
