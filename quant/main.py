import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from utils.data import load_data

from pathlib import Path

data_dir = Path('FTX_SOLUSD, 60.csv')

solusd = load_data(data_dir)

