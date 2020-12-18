import math
import sys, os
import pickle as pkl
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import erf

sys.path.append("../")


in_file = sys.argv[1]
data_list = [
    "aminor",
    "dssdenest",
    "efsbetan",
    "efsli",
    "efsvolume",
    "ip",
    "kappa",
    "R0",
    "tribot",
    "tritop",
]
data_list = [
    "concrete",
    "power",
    "wine",
    "yacht",
    "naval",
    "energy",
    "boston",
    "kin8nm",
]


# items_ordered = []
# for d in data_list:
#     for item in os.listdir(file_dir):
#         if d in item:
#             items_ordered.append(item)
#             break

# test_data = pkl.load(open('uci_test_data.pkl', 'rb'))

data_name = in_file.split("_")[0]
result = pkl.load(open("{}".format(in_file), "rb"))
calis = result["per_seed_cali"]
sharps = result["per_seed_sharp"]
gcalis = result["per_seed_gcali"]
# models = result['models_per_seed']
num_seeds = len(calis)

print(
    "cali: {:.3f} ({:.3f})".format(
        np.mean(calis), np.std(calis, ddof=1) / np.sqrt(num_seeds)
    )
)
print(
    "sharp: {:.3f} ({:.3f})".format(
        np.mean(sharps), np.std(sharps, ddof=1) / np.sqrt(num_seeds)
    )
)
# print('nlls: {:.4f} ({:.4f})'.format(np.mean(nlls), np.std(nlls, ddof=1)/np.sqrt(num_seeds)))
# mean_gcali = np.mean(np.stack(gcalis, axis=1), axis=1)
# plt.plot(np.linspace(0.01, 1.0, 10), mean_gcali, '-o')
# plt.title(item)
# plt.show()
g_cali_mat = np.stack(gcalis, axis=0)
np.save("int_{}_gcm.npy".format(data_name), g_cali_mat)
