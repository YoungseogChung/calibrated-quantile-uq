import sys
import argparse
from argparse import Namespace
import numpy as np
import tqdm
import torch
from fetch_uci import get_uci_data
from misc_utils import test_uq, set_seeds
from recal import iso_recal
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

sys.path.append("NNKit/")
from NNKit.models.model import vanilla_nn
from losses import (
    cali_loss,
    batch_cali_loss,
    qr_loss,
    batch_qr_loss,
    mod_cali_loss,
    batch_mod_cali_loss,
)
from losses import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Synth_Datasets",
        help="parent directory of datasets",
    )
    parser.add_argument("--data", type=str, help="dataset to use")
    parser.add_argument("--gpu", type=int, default=0, help="gpu num to use")

    parser.add_argument(
        "--num_ep", type=int, default=100, help="number of epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="weight decay")
    parser.add_argument("--bs", type=int, default=64, help="batch size")

    parser.add_argument(
        "--loss",
        type=str,
        default="ours",
        help="'ours' for our loss, 'qr' for pinball loss",
    )
    parser.add_argument(
        "--recal", type=int, default=0, help="1 to recalibrate afterwards"
    )

    parser.add_argument("--debug", type=int, default=0, help="1 to debug")

    args = parser.parse_args()

    args.recal = bool(args.recal)
    args.debug = bool(args.debug)

    device_name = (
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_name)
    args.device = device

    return args


if __name__ == "__main__":
    args = parse_args()
    set_seeds(args.seed)
    print("DEVICE: {}".format(args.device))

    if args.debug:
        import pudb

        pudb.set_trace()

    data_arr = np.load("{}/{}.npy".format(args.data_dir, args.data))
    # data_arr = np.asarray(list(np.arange(10)) + list(np.arange(30, 40)))
    print(data_arr.shape)
    plt.hist(data_arr, bins=50)
    plt.xlabel("datapoint")
    plt.ylabel("count")
    plt.title(args.data)
    plt.show()
    data_tensor = torch.from_numpy(data_arr).reshape(-1, 1)
    y = data_tensor
    y_range = (y.max() - y.min()).item()
    print("y range: {:.3f}".format(y_range))

    num_pts = y.size(0)
    dim_y = y.size(1)
    # there is no 'x', model only takes in q
    model = vanilla_nn(
        input_size=1, output_size=dim_y, hidden_size=64, num_layers=2
    )
    model = model.to(args.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    loader = DataLoader(TensorDataset(y), shuffle=False, batch_size=num_pts)
    # batch_size=args.bs)

    if args.loss == "cal":
        loss_fn = cali_loss
    elif args.loss == "qr":
        loss_fn = qr_loss
    elif args.loss == "batch_cal":
        loss_fn = batch_cali_loss
    elif args.loss == "batch_qr":
        loss_fn = batch_qr_loss
    elif args.loss == "mod_cal":
        loss_fn = mod_cali_loss
    elif args.loss == "batch_mod_cal":
        loss_fn = batch_mod_cali_loss
    elif args.loss == "cov":
        loss_fn = cov_loss
    else:
        raise ValueError("loss arg not valid")

    # train loop
    model = model.to(args.device)
    va_loss = []
    # for ep in tqdm.tqdm(range(args.num_ep)):
    for ep in range(args.num_ep):
        for yi in loader:
            xi = None
            yi = yi[0].to(args.device)
            optimizer.zero_grad()
            # q_list = torch.linspace(0.00, 1.00, 100)
            # q_list = torch.Tensor(list(np.random.uniform(size=30)) + [0.0, 1.0])
            # q_list = torch.Tensor([0.1, 0.5, 0.9])
            q_list = torch.Tensor([0.7])
            # xi = x_tr[:2].to(args.device)
            # yi = y_tr[:2].to(args.device)
            if "batch" in args.loss:
                loss = loss_fn(model, yi, xi, q_list, args.device)
            else:
                loss_list = []
                for q in q_list:
                    q_loss = loss_fn(model, yi, xi, q, args.device)
                    loss_list.append(q_loss)
                loss = torch.sum(torch.stack(loss_list))
            if torch.isnan(loss):
                import pudb

                pudb.set_trace()
            loss.backward()
            optimizer.step()
        if (ep % 20 == 0) or (ep == args.num_ep - 1):
            print("EP:{}: loss {:.3f}".format(ep, loss.item()))

    print("Last loss: {:.4f}".format(loss.item()))

    for q in q_list:
        single_q_tensor = torch.Tensor([q]).reshape(1, 1).to(args.device)
        with torch.no_grad():
            q_pred = model(single_q_tensor).cpu()
        q_idx_1 = data_arr[np.argsort(data_arr)[int(num_pts * q) - 1]]
        q_idx_2 = data_arr[np.argsort(data_arr)[int(num_pts * q)]]
        q_np = np.quantile(data_arr, q=q)
        print("{:.2f}: prediction {:.4f}".format(q, q_pred.item()))
        print(
            "Np quantile: {:.4f}, between {:.4f}, {:.4f}\n".format(
                q_np, q_idx_1, q_idx_2
            )
        )
