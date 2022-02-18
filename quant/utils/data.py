import pandas as pd
import numpy as np

from utils.useful_funcs import peaks_valleys


def load_data(data_dir):
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

    df['target'] = 0
    peaks, valleys = peaks_valleys(df, method='VWAP')
    df['target'][peaks] = 1
    df['target'][valleys] = 2

    return df
