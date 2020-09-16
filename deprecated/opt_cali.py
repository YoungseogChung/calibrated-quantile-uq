import sys
import argparse
from argparse import Namespace
from copy import deepcopy
import numpy as np
import tqdm
import torch
from fetch_uci import get_uci_data
from misc_utils import test_uq, set_seeds, gather_loss_per_q
from recal import iso_recal
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

sys.path.append('NNKit/')
from NNKit.models.model import vanilla_nn
from losses import cali_loss, batch_cali_loss, qr_loss, batch_qr_loss, \
    mod_cali_loss, batch_mod_cali_loss, crps_loss, cov_loss, batch_interval_loss


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ens', type=int, default=0,
                        help='1 to make an ensemble')
    parser.add_argument('--num_ens', type=int, default=5,
                        help='number of members in ensemble')

    parser.add_argument('--seed', type=int,
                        help='random seed')
    parser.add_argument('--data_dir', type=str, default='UCI_Datasets',
                        help='parent directory of datasets')
    parser.add_argument('--data', type=str,
                        help='dataset to use')
    parser.add_argument('--num_q', type=int, default=30,
                        help='number of quantiles you want to sample each step')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu num to use')

    parser.add_argument('--num_ep', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--bs', type=int, default=64,
                        help='batch size')
    parser.add_argument('--wait', type=int, default=200,
                        help='how long to wait for lower validation loss')

    parser.add_argument('--loss', type=str, default='ours',
                        help="'ours' for our loss, 'qr' for pinball loss")
    parser.add_argument('--recal', type=int, default=0,
                        help='1 to recalibrate afterwards')

    parser.add_argument('--debug', type=int, default=0,
                        help='1 to debug')

    args = parser.parse_args()

    args.recal = bool(args.recal)
    args.debug = bool(args.debug)

    device_name = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    args.device = device

    return args


class q_model_ens(object):
    def __init__(self, input_size, output_size, hidden_size, num_layers, lr, wd,
                 num_ens, device):

        self.num_ens = num_ens
        self.device = device
        self.model = [vanilla_nn(input_size=input_size, output_size=output_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers).to(device)
                      for _ in range(num_ens)]
        self.optimizers = [torch.optim.Adam(x.parameters()) for x in self.model]
        self.keep_training = [True for _ in range(num_ens)]
        self.best_va_loss = [np.inf for _ in range(num_ens)]
        self.best_va_model = [None for _ in range(num_ens)]
        self.best_va_ep = [0 for _ in range(num_ens)]



    def loss(self, loss_fn, x, y, q_list, batch_q, take_step):
        ens_loss = []
        for idx in range(self.num_ens):
            self.optimizers[idx].zero_grad()
            if self.keep_training[idx]:
                if batch_q:
                    loss = loss_fn(self.model[idx], y, x, q_list, self.device)
                else:
                    loss = gather_loss_per_q(loss_fn, self.model[idx], y, x,
                                             q_list, self.device)
                ens_loss.append(loss.item())

                if take_step:
                    loss.backward()
                    self.optimizers[idx].step()
            else:
                ens_loss.append(None)

        return np.asarray(ens_loss)






if __name__ == '__main__':
    args = parse_args()

    if args.debug:
        import pudb; pudb.set_trace()

    set_seeds(args.seed)
    print('DEVICE: {}'.format(args.device))

    if 'uci' in args.data_dir.lower():
        data_args = Namespace(data_dir=args.data_dir, dataset=args.data,
                              seed=args.seed)
        data_out = get_uci_data(args)
        x_tr, x_va, x_te, y_tr, y_va, y_te, y_al = \
            data_out.x_tr, data_out.x_va, data_out.x_te, data_out.y_tr, \
            data_out.y_va, data_out.y_te, data_out.y_al
        y_range = (y_al.max() - y_al.min()).item()

    print('y range: {:.3f}'.format(y_range))

    dim_x = x_tr.size(1)
    dim_y = y_tr.size(1)
    model = vanilla_nn(input_size=dim_x + 1, output_size=dim_y,
                       hidden_size=64, num_layers=2)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.wd)
    loader = DataLoader(TensorDataset(x_tr, y_tr),
                        shuffle=True,
                        batch_size=args.bs)

    if args.loss == 'cal':
        loss_fn = cali_loss
    elif args.loss == 'qr':
        loss_fn = qr_loss
    elif args.loss == 'batch_cal':
        loss_fn = batch_cali_loss
    elif args.loss == 'batch_qr':
        loss_fn = batch_qr_loss
    elif args.loss == 'mod_cal':
        loss_fn = mod_cali_loss
    elif args.loss == 'batch_mod_cal':
        loss_fn = batch_mod_cali_loss
    elif args.loss == 'cov':
        loss_fn = cov_loss
    elif args.loss == 'batch_crps':
        loss_fn = crps_loss
    elif args.loss == 'int_score':
        args.loss = 'batch_interval_loss'
        loss_fn = batch_interval_loss
    else:
        raise ValueError('loss arg not valid')

    """ train loop """
    model = model.to(args.device)
    tr_loss_list = []
    va_loss_list = []
    te_loss_list = []

    ### tracking best model
    best_va_loss = np.inf
    best_va_model = None
    best_va_ep = 0

    # for ep in tqdm.tqdm(range(args.num_ep)):
    for ep in (range(args.num_ep)):
        ep_train_loss = []
        for (xi, yi) in loader:
            xi, yi = xi.to(args.device), yi.to(args.device)
            optimizer.zero_grad()
            # q_list = torch.linspace(0.00, 1.00, 100)
            # q_list = torch.Tensor(list(np.random.uniform(size=30)) + [0.0, 1.0])
            q_list = torch.rand(args.num_q)
            # xi = x_tr[:2].to(args.device)
            # yi = y_tr[:2].to(args.device)
            # q_list = torch.Tensor([0.1, 0.5, 0.9])
            if 'batch' in args.loss:
                loss = loss_fn(model, yi, xi, q_list, args.device)
            else:
                loss = gather_loss_per_q(loss_fn, model, yi, xi, q_list,
                                         args.device)
                # loss_list = []
                # for q in q_list:
                #     q_loss = loss_fn(model, yi, xi, q, args.device)
                #     loss_list.append(q_loss)
                # loss = torch.sum(torch.stack(loss_list))
            ep_train_loss.append(loss.item())
            if not torch.isfinite(loss):
                import pudb; pudb.set_trace()
            loss.backward()
            optimizer.step()

        if (ep % 20 == 0) or (ep == args.num_ep-1):
            tr_loss_list.append(np.mean(ep_train_loss))
            print('EP:{}'.format(ep))
            print('Train loss {:.3f}'.format(loss.item()))

            """ get val loss """
            x_va, y_va = x_va.to(args.device), y_va.to(args.device)
            va_te_q_list = torch.linspace(0.01, 0.99, 99)
            with torch.no_grad():
                if 'batch' in args.loss:
                    temp_va_loss = loss_fn(model, y_va, x_va, va_te_q_list,
                                           args.device)
                else:
                    temp_va_loss = gather_loss_per_q(loss_fn, model, y_va, x_va,
                                                     va_te_q_list, args.device)
                print('va_loss {:.3f}'.format(temp_va_loss.item()))
                va_loss_list.append(temp_va_loss.item())

            """ get test loss """
            x_te, y_te = x_te.to(args.device), y_te.to(args.device)
            with torch.no_grad():
                if 'batch' in args.loss:
                    temp_te_loss = loss_fn(model, y_te, x_te, va_te_q_list,
                                           args.device)
                else:
                    temp_te_loss = gather_loss_per_q(loss_fn, model, y_te, x_te,
                                                     va_te_q_list, args.device)
                print('te_loss {:.3f}'.format(temp_te_loss.item()))
                te_loss_list.append(temp_te_loss.item())

            """ continue training? """
            if temp_va_loss.item() < best_va_loss:
                best_va_loss = temp_va_loss.item()
                best_va_ep = ep
                best_va_model = deepcopy(model)
            else:
                if ep - best_va_ep > args.wait:
                    print('Validation loss stagnate, cutting training')
                    break

    x_va, y_va, x_te, y_te = x_va.cpu(), y_va.cpu(), x_te.cpu(), y_te.cpu()
    model = best_va_model

    plt.plot(np.arange(len(tr_loss_list)) * 20, np.log(tr_loss_list), label='train')
    plt.plot(np.arange(len(va_loss_list)) * 20, np.log(va_loss_list), label='val')
    plt.plot(np.arange(len(te_loss_list)) * 20, np.log(te_loss_list), label='test')
    plt.legend()
    plt.show()

    # Test UQ on train
    tr_exp_props = torch.linspace(-3.0, 3.0, 600)
    tr_cali_score, tr_sharp_score, tr_obs_props = \
        test_uq(model, x_tr, y_tr, tr_exp_props, y_range,
                recal_model=None, recal_type=None)

    # Test UQ on val
    va_exp_props = torch.linspace(-10.0, 10.0, 10000)
    va_cali_score, va_sharp_score, va_obs_props = \
        test_uq(model, x_va, y_va, va_exp_props, y_range,
                recal_model=None, recal_type=None)

    # Test UQ on test
    te_exp_props = torch.linspace(-3.0, 3.0, 600)
    te_cali_score, te_sharp_score, te_obs_props = \
        test_uq(model, x_te, y_te, te_exp_props, y_range,
                recal_model=None, recal_type=None)

    if args.recal:
        recal_model = iso_recal(va_exp_props, va_obs_props)

        exp_props = torch.linspace(0.00, 1.01, 102)
        recal_va_cali_score, recal_va_sharp_score, recal_va_obs_props = \
            test_uq(model, x_va, y_va, exp_props, y_range,
                    recal_model=recal_model, recal_type='sklearn')

        recal_te_cali_score, recal_te_sharp_score, recal_te_obs_props = \
            test_uq(model, x_te, y_te, exp_props, y_range,
                    recal_model=recal_model, recal_type='sklearn')

        recal_tr_cali_score, recal_tr_sharp_score, recal_tr_obs_props = \
            test_uq(model, x_tr, y_tr, exp_props, y_range,
                    recal_model=recal_model, recal_type='sklearn')
