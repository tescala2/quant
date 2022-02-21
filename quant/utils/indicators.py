import pandas as pd

from pathlib import Path

class Columns(object):
    OPEN = 'open'
    HIGH = 'high'
    LOW = 'low'
    CLOSE = 'close'
    VOLUME = 'volume'


class Settings(object):
    join = True
    col = Columns()

SETTINGS = Settings()


def out(settings, df, result):
    if not settings.join:
        return result
    else:
        df = df.join(result)
        return df


def MA(df, n, method='close'):
    """
    Moving Average
    """
    name = 'MA_{n}'.format(n=n)
    result = pd.Series(df[method].rolling(n).mean(), name=name)
    return out(SETTINGS, df, result)


def BBANDS(df, n, method='close'):
    """
    Bollinger Bands
    """
    MA = pd.Series(df[method].rolling(n).mean())
    STD = pd.Series(df[method].rolling(n).std())
    b1 = MA + 2*STD
    B1 = pd.Series(b1, name=f'BBandU_{n}')
    b2 = MA - 2*STD
    B2 = pd.Series(b2, name=f'BBandD_{n}')
    result = pd.DataFrame([B1, B2]).transpose()
    return out(SETTINGS, df, result)


def KELCH(df, n):
    """
    Keltner Channel
    """
    temp = (df['high'] + df['low'] + df['close']) / 3
    KelChM = pd.Series(temp.rolling(n).mean(), name=f'KelChM_{n}')
    temp = (4 * df['high'] - 2 * df['low'] + df['close']) / 3
    KelChU = pd.Series(temp.rolling(n).mean(), name=f'KelChU_{n}')
    temp = (-2 * df['high'] + 4 * df['low'] + df['close']) / 3
    KelChD = pd.Series(temp.rolling(n).mean(), name=f'KelChD_{n}')
    result = pd.DataFrame([KelChM, KelChU, KelChD]).transpose()
    return out(SETTINGS, df, result)
