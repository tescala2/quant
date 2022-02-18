import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt

from quant.utils.data import load_data
from quant.utils.useful_funcs import peaks_valleys

from pathlib import Path


def three_plot(subset, width=.025, width2=.005, col1='green', col2='red'):
    '''
    makes a plot with 3 subplots
    first plot contains the bar chart with vwap overlaid
    second plot shows ratios of open, close, high, and low to vwap
    third plot shows rsi and the moving average of rsi
    '''

    up = subset[subset.close >= subset.open]
    down = subset[subset.close < subset.open]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    # plot 1
    ax1.bar(up.index, up.close - up.open, width, bottom=up.open, color=col1)
    ax1.bar(up.index, up.high - up.close, width2, bottom=up.close, color=col1)
    ax1.bar(up.index, up.low - up.open, width2, bottom=up.open, color=col1)

    ax1.bar(down.index, down.close - down.open, width, bottom=down.open, color=col2)
    ax1.bar(down.index, down.high - down.open, width2, bottom=down.open, color=col2)
    ax1.bar(down.index, down.low - down.close, width2, bottom=down.close, color=col2)

    ax1.plot(subset.index, subset.VWAP, linewidth=0.5, label='VWAP')
    ax1.fill_between(subset.index, subset['Lower Band'], subset['Upper Band'], color='purple', linewidth=0.5,
                     alpha=0.25)
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Price (USD)')

    # plot 2
    ax2.plot(subset.index, subset['open/VWAP'], linewidth=1, label='open')
    ax2.plot(subset.index, subset['high/VWAP'], linewidth=1, label='high')
    ax2.plot(subset.index, subset['low/VWAP'], linewidth=1, label='low')
    ax2.plot(subset.index, subset['close/VWAP'], linewidth=1, label='close')
    ax2.hlines(1, xmin=subset.index[0], xmax=subset.index[-1], colors='k', linestyles='dashed', linewidth=0.5)
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Price / VWAP')

    # plot 3
    ax3.plot(subset.index, subset['RSI'], label='RSI', linewidth=1)
    ax3.plot(subset.index, subset['RSI-based MA'], label='RSI MA', linewidth=1)
    ax3.hlines(70, xmin=subset.index[0], xmax=subset.index[-1], colors='k', linestyles='dashed', linewidth=0.5)
    ax3.hlines(50, xmin=subset.index[0], xmax=subset.index[-1], colors='k', linestyles='dashed', linewidth=0.5)
    ax3.hlines(30, xmin=subset.index[0], xmax=subset.index[-1], colors='k', linestyles='dashed', linewidth=0.5)
    ax3.legend(loc='upper right')
    ax3.set_ylabel('RSI')

    plt.xticks(rotation=45, ha='right')
    plt.show()


def buy_sell_plot(data, show_bars=True, method='VWAP', buy_col='orange', sell_col='navy'):
    '''
    makes a plot showing the bar chart (optional) with buy and sell targets overlaid
    on a given method (can be 'VWAP', 'average', or 'smooth average')
    '''

    width = .025
    width2 = .005

    up = data[data.close >= data.open]
    down = data[data.close < data.open]

    col1 = 'green'
    col2 = 'red'

    peaks, valleys = peaks_valleys(data, method)

    plt.figure()

    if show_bars:
        plt.bar(up.index, up.close - up.open, width, bottom=up.open, color=col1)
        plt.bar(up.index, up.high - up.close, width2, bottom=up.close, color=col1)
        plt.bar(up.index, up.low - up.open, width2, bottom=up.open, color=col1)

        plt.bar(down.index, down.close - down.open, width, bottom=down.open, color=col2)
        plt.bar(down.index, down.high - down.open, width2, bottom=down.open, color=col2)
        plt.bar(down.index, down.low - down.close, width2, bottom=down.close, color=col2)

    plt.scatter(data.index[peaks], data[method][peaks], c=sell_col, linewidths=0.5, label='sell')
    plt.scatter(data.index[valleys], data[method][valleys], c=buy_col, linewidths=0.5, label='buy')

    plt.plot(data.index, data[method], linewidth=0.5, label=method)
    plt.legend(loc='upper right')
    plt.ylabel('Price (USD)')

    plt.show()


if __name__ == '__main__':
    data_dir = Path('/Users/tomasescalante/Downloads/FTX_SOLUSD, 60.csv')

    solusd = load_data(data_dir)

    subset = solusd[solusd.index >= dt.datetime(2022, 1, 20)]

    buy_sell_plot(subset)
