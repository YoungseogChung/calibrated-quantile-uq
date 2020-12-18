import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

item_list = []
loss_list = []
num_list = []
for item in os.listdir("./"):
    if "batch" in item and ".npy" in item:
        loss = item.replace(".npy", "")

        # if 'scaled_batch_cal_' not in loss: continue
        # if 'concrete_batch_cal_' not in loss: continue
        # if 'concrete_scaled_batch_cal_' not in loss: continue
        # if 'concrete_batch_qr_' not in loss: continue
        # if 'boston_scaled_batch_cal_' not in loss: continue
        # if 'boston_batch_qr_' not in loss: continue
        if sys.argv[1] not in loss:
            continue

        if not loss.split("_")[-1].isnumeric():
            continue
        num_list.append(float(loss.split("_")[-1]))
        item_list.append(item)
        loss_list.append(loss)

order = np.argsort(num_list)
colors = cm.rainbow(np.linspace(0, 1, len(order)))

i = len(order) - 1
for item_idx in order:
    item = item_list[item_idx]
    loss = loss_list[item_idx]
    arr = np.load(item)
    x_arr = np.linspace(0.1, 1.0, arr.shape[0])
    if "orig" in loss:
        pattern = "^-"
    elif "no_sq" in loss:
        pattern = "--"
    else:
        pattern = "o-"
    # plt.plot(x_arr, np.log10(arr), pattern, label=loss, color=colors[i])
    plt.plot(x_arr, arr, pattern, label=loss, color=colors[i])
    i -= 1

plt.xlabel("Group Size")
plt.ylabel("Avg Miscalibration")
plt.title("_".join(loss.split("_")[:-1]))
plt.legend()

plt.show()
