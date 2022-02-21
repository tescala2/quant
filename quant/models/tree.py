import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from quant.utils.data import load_data
from pathlib import Path
import matplotlib.pyplot as plt

data_dir = Path('../data/FTX_SOLUSD, 60.csv')

cols = ['open/VWAP', 'high/VWAP', 'low/VWAP',
       'close/VWAP', 'open/MA_12', 'high/MA_12', 'low/MA_12', 'close/MA_12',
       'open/BBandU_12', 'high/BBandU_12', 'low/BBandU_12', 'close/BBandU_12',
       'open/BBandD_12', 'high/BBandD_12', 'low/BBandD_12', 'close/BBandD_12',
       'open/KelChU_12', 'high/KelChU_12', 'low/KelChU_12', 'close/KelChU_12',
       'open/KelChD_12', 'high/KelChD_12', 'low/KelChD_12', 'close/KelChD_12',
       'open/KelChM_12', 'high/KelChM_12', 'low/KelChM_12', 'close/KelChM_12',
       'labels']

df = load_data(data_dir)
X = df[cols[:-1]]
y = df[cols[-1]].shift(-1)
y[-1] = y[-2]

clf = DecisionTreeClassifier(max_depth=5).fit(X, y)
preds = clf.predict(X)

df['preds'] = preds

sell_labels = np.where(df['labels'] == 1, df['VWAP'], np.nan)
buy_labels = np.where(df['labels'] == 2, df['VWAP'], np.nan)

sell_preds = np.where(df['preds'] == 1, df['VWAP'], np.nan)
buy_preds = np.where(df['preds'] == 2, df['VWAP'], np.nan)

plt.figure()
plt.plot(df.index, df['VWAP'], label='VWAP')
# plt.scatter(df.index, sell_labels, c='maroon', marker='*', label='true sell')
# plt.scatter(df.index, buy_labels, c='darkgreen', marker='*', label='true sell')
plt.scatter(df.index, sell_preds, c='red', marker='.', label='pred sell')
plt.scatter(df.index, buy_preds, c='lime', marker='.', label='pred buy')
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Price')
plt.show()


fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (8,2), dpi=300)
plot_tree(clf,
          feature_names = cols[:-1],
          class_names=['nothing', 'sell', 'buy'],
          filled = True)
plt.show()
