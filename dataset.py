import torch
import pandas as pd

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATA_PATH = Path(__file__).parent / "datasets"

@dataclass
class Split:
    """
    Dataset split.
    """
    train_size: int
    val_size: int
    test_size: int

DATASET_SPLITS = {
    "ETTh": Split(train_size=8640, val_size=2880, test_size=2880),
    "ETTm": Split(train_size=34560, val_size=11520, test_size=11520)
}

def get_datasets(
    dataset_name: str,
    lookback_size: int,
    horizon_size: int
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Utility function to obtain Train/Val/Test datasets.
    :param dataset_name: dataset name
    :param lookback_size: history window length used for model input
    :param horizon_size: horizon window length to forecast
    :returns: corresponding Train/Val/Test datasets
    """
    if "ETT" in dataset_name:
        start_idx = 0
        data = pd.read_csv(DATA_PATH / dataset_name)
        split = DATASET_SPLITS["ETTh"] if "h" in dataset_name else DATASET_SPLITS["ETTm"]

        train_slice = slice(start_idx, start_idx + split.train_size)
        train_df = data.loc[train_slice].copy()
        train_ds = ETTDataset(
            dataframe=train_df,
            lookback_size=lookback_size,
            horizon_size=horizon_size,
            scaler=None,
        )

        start_idx += split.train_size
        val_slice = slice(start_idx, start_idx + split.val_size)
        val_df = data.loc[val_slice].copy()
        val_ds = ETTDataset(
            dataframe=val_df,
            lookback_size=lookback_size,
            horizon_size=horizon_size,
            scaler=train_ds.scaler,
        )

        start_idx += split.val_size
        test_slice = slice(start_idx, start_idx + split.test_size)
        test_df = data.loc[test_slice].copy()
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
        scaler: StandardScaler | None
    ):
        """
        Electricity Transformer Temperature dataset.
        :param dataframe: dataframe containing time series data
        :param lookback_size: history window length used for model input
        :param horizon_size: horizon window length to forecast
        :param scaler: data scaler
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
            dataframe["date"].apply(lambda date: datetime.fromisoformat(date).minute) // 15
        )  # 15 minutes intervals only
        
        ohe = OneHotEncoder()
        time_values = dataframe[["month", "weekday", "day", "hour", "minute"]]
        self.time_values = ohe.fit_transform(time_values).toarray()
        
        self.x_values = dataframe[["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]].values
        self.y_values = dataframe[["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]].values
        self.target_idx = -1

        if scaler is None:
            self.scaler = StandardScaler()
            self.x_values = self.scaler.fit_transform(self.x_values)
        else:
            self.scaler = scaler
            self.x_values = self.scaler.transform(self.x_values)

        self.x_values = torch.tensor(self.x_values, dtype=torch.float32)
        self.y_values = torch.tensor(self.y_values, dtype=torch.float32)
        self.time_values = torch.tensor(self.time_values, dtype=torch.float32)

    def __len__(self) -> int:
        return max(0, self.x_values.shape[0] - self.window_size + 1)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        x_slice = slice(idx, idx + self.lookback_size)
        x_sample = self.x_values[x_slice, :]
        
        time_values = self.time_values[idx + self.lookback_size - 1, :]

        y_slice = slice(idx + self.lookback_size, idx + self.window_size)
        y_sample = self.y_values[y_slice, self.target_idx]
        
        return x_sample, time_values, y_sample
