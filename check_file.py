import sys
import pickle as pkl
import numpy as np

data = pkl.load(open(sys.argv[1], "rb"))

print(
    "Cali: {:.3f}, ({:.3f})".format(
        np.mean(data["per_seed_cali"]),
        (np.std(data["per_seed_cali"], ddof=1) / np.sqrt(5)),
    )
)
print(
    "Sharp: {:.3f}, ({:.3f})".format(
        np.mean(data["per_seed_sharp"]),
        (np.std(data["per_seed_sharp"], ddof=1) / np.sqrt(5)),
    )
)
print(
    "NLL: {:.3f}, ({:.3f})".format(
        np.mean(data["per_seed_nll"]),
        (np.std(data["per_seed_nll"], ddof=1) / np.sqrt(5)),
    )
)
print(
    "CRPS: {:.3f}, ({:.3f})".format(
        np.mean(data["per_seed_crps"]),
        (np.std(data["per_seed_crps"], ddof=1) / np.sqrt(5)),
    )
)
print(
    "Check: {:.3f}, ({:.3f})".format(
        np.mean(data["per_seed_check"]),
        (np.std(data["per_seed_check"], ddof=1) / np.sqrt(5)),
    )
)
print(
    "Int: {:.3f}, ({:.3f})".format(
        np.mean(data["per_seed_int"]),
        (np.std(data["per_seed_int"], ddof=1) / np.sqrt(5)),
    )
)
print(
    "Int-Cali: {:.3f}, ({:.3f})".format(
        np.mean(data["per_seed_int_cali"]),
        (np.std(data["per_seed_int_cali"], ddof=1) / np.sqrt(5)),
    )
)
mean_gcali = np.mean(np.stack(data["per_seed_gcali"], axis=0), axis=0)
print(mean_gcali[:5])
print(mean_gcali[5:])
