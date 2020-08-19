import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from argparse import Namespace


def get_uci_data(args):
    data = np.loadtxt('{}/{}.txt'.format(args.data_dir, args.data))
    x_al = data[:, :-1]
    y_al = data[:, -1].reshape(-1, 1)

    x_tr, x_te, y_tr, y_te = train_test_split(
        x_al, y_al, test_size=0.1, random_state=args.seed)
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_tr, y_tr, test_size=0.2, random_state=args.seed)

    s_tr_x = StandardScaler().fit(x_tr)
    s_tr_y = StandardScaler().fit(y_tr)

    x_tr = torch.Tensor(s_tr_x.transform(x_tr))
    x_va = torch.Tensor(s_tr_x.transform(x_va))
    x_te = torch.Tensor(s_tr_x.transform(x_te))

    y_tr = torch.Tensor(s_tr_y.transform(y_tr))
    y_va = torch.Tensor(s_tr_y.transform(y_va))
    y_te = torch.Tensor(s_tr_y.transform(y_te))
    y_al = torch.Tensor(s_tr_y.transform(y_al))

    out_namespace = Namespace(x_tr=x_tr, x_va=x_va, x_te=x_te,
                              y_tr=y_tr, y_va=y_va, y_te=y_te, y_al=y_al)

    return out_namespace

