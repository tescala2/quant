import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from quant.utils.useful_funcs import peaks_valleys


def load_data(data_dir, ma_method='close', target_method='VWAP'):
    '''
    loads csv data into pandas dataframe with timestamp as the index
    creates new columns:
        average - average of the open, close, high, and low
        smooth average - a smoothed curve of the average of the open, close, high, and low
        open/VWAP - ratio of open to VWAP
        high/VWAP - ratio of high to VWAP
        low/VWAP - ratio of low to VWAP
        close/VWAP - ratio of close to VWAP
        MA_12 - 12 point moving average
    '''

    df = pd.read_csv(data_dir, index_col='time').drop(
        columns=['Plot.1', 'Upper Bollinger Band', 'Lower Bollinger Band'])
    df.index = pd.to_datetime(df.index, unit='s')

    df['average'] = (df.open + df.high + df.low + df.close) / 4

    filter_length = 9
    df['smooth average'] = np.convolve(df['average'], np.ones((filter_length)), mode='same')
    df['smooth average'] /= filter_length

    df['open/VWAP'] = df.open / df.VWAP
    df['high/VWAP'] = df.high / df.VWAP
    df['low/VWAP'] = df.low / df.VWAP
    df['close/VWAP'] = df.close / df.VWAP

    targets = np.zeros_like(df[target_method])
    peaks, valleys = peaks_valleys(df, method=target_method)
    targets[peaks] = 1
    targets[valleys] = 2
    df['target'] = targets

    return df


class SequenceDataset(Dataset):
    def __init__(self, X: np.array, Y: np.array, sequence_length: int, feature_first: bool):
        self.sequence_length = sequence_length

        self.X = torch.tensor(X).float()
        self.y = torch.tensor(Y).float()
        self.feature_first = feature_first

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1)]

        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)

            x = self.X[0:(i + 1)]
            x = torch.cat((padding, x.view(-1, 1)))

        if self.feature_first:
            return x.view(1, -1), self.y[i]

        return x.view(-1, 1), self.y[i]


def get_loaders(df: pd.DataFrame, forecast_lead: int = 1, target_column: str = None, sequence_length: int = 300,
                bs: int = 16, feature_first: bool = False):
    val_start = int(len(df) * 0.6)
    test_start = int(len(df) * 0.8)

    if not target_column:
        target_column = df.columns

    features = df.columns

    mean, std = df[:val_start].mean(), df[:val_start].std()
    print(mean, std)

    df = (df - df.mean()) / df.std()

    df['target'] = df[target_column].shift(-forecast_lead)
    df = df.iloc[:-forecast_lead]

    df_train = df.iloc[:val_start].copy()
    df_val = df.iloc[val_start:test_start].copy()
    df_test = df.iloc[test_start:].copy()

    train_dataset = SequenceDataset(
        X=df_train[features].to_numpy(),
        Y=df_train["target"].to_numpy(),
        sequence_length=sequence_length,
        feature_first=feature_first
    )

    val_dataset = SequenceDataset(
        X=df_val[features].to_numpy(),
        Y=df_val["target"].to_numpy(),
        sequence_length=sequence_length,
        feature_first=feature_first
    )

    test_dataset = SequenceDataset(
        X=df_test[features].to_numpy(),
        Y=df_test["target"].to_numpy(),
        sequence_length=sequence_length,
        feature_first=feature_first
    )

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    return train_loader, val_loader, test_loader