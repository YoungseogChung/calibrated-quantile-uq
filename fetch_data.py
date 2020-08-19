import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from argparse import Namespace

PR_DIR = '/zfsauton/project/public/ysc/'

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

def get_fusion_data(args):

    cur_dir = '{}/{}'.format(args.data_dir, args.data) 

    print('loading fusion data {}'.format(args.data))
    x_tr = np.load('{}/x_tr_{}.npy'.format(cur_dir, args.data))
    y_tr = np.load('{}/y_tr_{}.npy'.format(cur_dir, args.data))
    x_va = np.load('{}/x_va_{}.npy'.format(cur_dir, args.data))
    y_va = np.load('{}/y_va_{}.npy'.format(cur_dir, args.data))
    x_te = np.load('{}/x_te_{}.npy'.format(cur_dir, args.data))
    y_te = np.load('{}/y_te_{}.npy'.format(cur_dir, args.data))
    y_al = np.concatenate([y_tr, y_va, y_te], axis=0)

    x_tr = torch.Tensor(x_tr)
    y_tr = torch.Tensor(y_tr)
    x_va = torch.Tensor(x_va)
    y_va = torch.Tensor(y_va)
    x_te = torch.Tensor(x_te)
    y_te = torch.Tensor(y_te)
    y_al = torch.Tensor(y_al)

    out_namespace = Namespace(x_tr=x_tr, x_va=x_va, x_te=x_te,
                              y_tr=y_tr, y_va=y_va, y_te=y_te, y_al=y_al)

    return out_namespace

    
