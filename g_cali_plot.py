import os, sys
import numpy as np
import matplotlib.pyplot as plt

for item in os.listdir('./'):
    if 'batch' in item and '.npy' in item:
        loss = item.replace('.npy', '')
        if 'batch_cal' not in loss: continue
        arr = np.load(item)
        x_arr = np.linspace(0.01, 1.0, arr.shape[0])
        if 'orig' in loss:
            pattern = '^-'
        elif 'no_sq' in loss:
            pattern = '--'
        else:
            pattern = 'o-'
        plt.plot(x_arr, np.log10(arr), pattern, label=loss, )

plt.legend()

plt.show()
