import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from quant.utils.data import load_data
from quant.utils import indicators

from pathlib import Path

data_dir = Path('data/FTX_SOLUSD, 60.csv')

solusd = load_data(data_dir)

print(solusd.columns)

