import numpy as np
from scipy.signal import argrelextrema


def peaks_valleys(data, method='VWAP'):
    peaks = argrelextrema(np.array(data[method]), np.greater)
    peaks = peaks[0]
    valleys = argrelextrema(np.array(data[method]), np.less)
    valleys = valleys[0]

    return peaks, valleys
