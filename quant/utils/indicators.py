import pandas as pd

from quant.utils.data import load_data

from pathlib import Path

class Columns(object):
    OPEN='open'
    HIGH='high'
    LOW='low'
    CLOSE='close'
    VOLUME='volume'



class Settings(object):
    join=True
    col=Columns()

SETTINGS=Settings()


def out(settings, df, result):
    if not settings.join:
        return result
    else:
        df=df.join(result)
        return df


def MA(df, n, price='close'):
    """
    Moving Average
    """

    name='MA_{n}'.format(n=n)
    result = pd.Series(df[price].rolling(n).mean(), name=name)
    return out(SETTINGS, df, result)


def BBANDS(df, n, price='close'):
    """
    Bollinger Bands
    """
    MA = pd.Series(df[price].rolling(n).mean())
    STD = pd.Series(df[price].rolling(n).std())
    b1 = MA + 2*STD
    B1 = pd.Series(b1, name='BollingerB_' + str(n))
    b2 = MA - 2*STD
    B2 = pd.Series(b2, name='Bollinger%b_' + str(n))
    result = pd.DataFrame([B1, B2]).transpose()
    return out(SETTINGS, df, result)


def KELCH(df, n):
    """
    Keltner Channel
    """
    temp = (df['high'] + df['low'] + df['close']) / 3
    KelChM = pd.Series(temp.rolling(n).mean(), name='KelChM_' + str(n))
    temp = (4 * df['high'] - 2 * df['low'] + df['close']) / 3
    KelChU = pd.Series(temp.rolling(n).mean(), name='KelChU_' + str(n))
    temp = (-2 * df['high'] + 4 * df['low'] + df['close']) / 3
    KelChD = pd.Series(temp.rolling(n).mean(), name='KelChD_' + str(n))
    result = pd.DataFrame([KelChM, KelChU, KelChD]).transpose()
    return out(SETTINGS, df, result)



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data_dir = Path('../FTX_SOLUSD, 60.csv')

    solusd = load_data(data_dir)

    n = 12

    ma1 = 'MA_' + str(n)
    b1 = 'BollingerB_' + str(n)
    b2 = 'Bollinger%b_' + str(n)
    kchu = 'KelChU_' + str(n)
    kchm = 'KelChM_' + str(n)
    kchd = 'KelChD_' + str(n)

    solusd = MA(solusd, n)
    solusd = BBANDS(solusd, n)
    solusd = KELCH(solusd, n)


    width, width2, col1, col2 = .025, .005, 'green', 'red'

    up = solusd[solusd.close >= solusd.open]
    down = solusd[solusd.close < solusd.open]

    plt.figure()

    plt.bar(up.index, up.close - up.open, width, bottom=up.open, color=col1)
    plt.bar(up.index, up.high - up.close, width2, bottom=up.close, color=col1)
    plt.bar(up.index, up.low - up.open, width2, bottom=up.open, color=col1)

    plt.bar(down.index, down.close - down.open, width, bottom=down.open, color=col2)
    plt.bar(down.index, down.high - down.open, width2, bottom=down.open, color=col2)
    plt.bar(down.index, down.low - down.close, width2, bottom=down.close, color=col2)

    plt.plot(solusd.index, solusd[ma1], linewidth=0.5, c='blue', label='Moving Average')
    plt.fill_between(solusd.index, solusd[b1], solusd[b2], color='red', linewidth=0.5,
                     alpha=0.25, label='Bollinger Bands')

    plt.plot(solusd.index, solusd[kchm], linewidth=0.5, c='red', label='Keltner Middle')
    plt.fill_between(solusd.index, solusd[kchu], solusd[kchd], color='blue', linewidth=0.5,
                     alpha=0.25, label='Keltner Channel')

    plt.legend(loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.show()
