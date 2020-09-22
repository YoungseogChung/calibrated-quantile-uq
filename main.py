import os, sys
import argparse
from argparse import Namespace
from copy import deepcopy
import numpy as np
import pickle as pkl
import tqdm
import torch
from data.fetch_data import get_uci_data, get_fusion_data
from utils.misc_utils import test_uq, set_seeds, gather_loss_per_q
from recal import iso_recal
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# sys.path.append('NNKit/')
# from NNKit.models.model import vanilla_nn
from utils.q_model_ens import QModelEns
from losses import cali_loss, batch_cali_loss, qr_loss, batch_qr_loss, \
    mod_cali_loss, batch_mod_cali_loss, crps_loss, cov_loss, \
    interval_loss, batch_interval_loss


def get_loss_fn(loss_name):
    if loss_name == 'qr':
        fn = qr_loss
    elif loss_name == 'batch_qr':
        fn = batch_qr_loss
    elif loss_name in ['cal', 'scaled_cal',
                       'cal_penalty', 'scaled_cal_penalty']:
        fn = cali_loss
    elif loss_name in ['batch_cal', 'scaled_batch_cal',
                       'batch_cal_penalty', 'scaled_batch_cal_penalty']:
        fn = batch_cali_loss
    elif loss_name in ['mod_cal', 'scaled_mod_cal']:
        fn = mod_cali_loss
    elif loss_name in ['batch_mod_cal', 'scaled_batch_mod_cal']:
        fn = batch_mod_cali_loss
    elif loss_name == 'cov':
        fn = cov_loss
    elif loss_name == 'batch_crps':
        fn = crps_loss
    elif loss_name == 'int':
        fn = interval_loss
    elif loss_name == 'batch_int':
        fn = batch_interval_loss
    else:
        raise ValueError('loss arg not valid')

    return fn


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_ens', type=int, default=1,
                        help='number of members in ensemble')
    parser.add_argument('--boot', type=int, default=0,
                        help='1 to bootstrap samples')

    parser.add_argument('--seed', type=int,
                        help='random seed')
    parser.add_argument('--data_dir', type=str, 
                        default='data/UCI_Datasets',
                        help='parent directory of datasets')
    parser.add_argument('--data', type=str,
                        help='dataset to use')
    parser.add_argument('--epist', type=int, default=0,
                        help='1 to switch train and test splits for epistemic uncertainty')
    parser.add_argument('--num_q', type=int, default=30,
                        help='number of quantiles you want to sample each step')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu num to use')

    parser.add_argument('--num_ep', type=int, default=10000,
                        help='number of epochs')
    parser.add_argument('--nl', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--hs', type=int, default=64,
                        help='hidden size')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--bs', type=int, default=64,
                        help='batch size')
    parser.add_argument('--wait', type=int, default=200,
                        help='how long to wait for lower validation loss')

    parser.add_argument('--loss', type=str,
                        help='specify type of loss')

    # only for cali losses
    parser.add_argument('--penalty', dest='sharp_penalty', type=float,
                        help='coefficient for sharpness penalty; 0 for non')
    parser.add_argument('--rand_ref', type=int,
                        help='1 to use rand reference idxs for cali loss')
    parser.add_argument('--sharp_wide', type=int,
                        help='1 to penalize only widths that are over covered')

    parser.add_argument('--recal', type=int, default=1,
                        help='1 to recalibrate afterwards')
    parser.add_argument('--save_dir', type=str, 
                        default='/zfsauton/project/public/ysc/uq_uci/all_default',
                        help='dir to save results')
    parser.add_argument('--debug', type=int, default=0,
                        help='1 to debug')
    args = parser.parse_args()

    if 'penalty' in args.loss:
        assert isinstance(args.sharp_penalty, float)
        assert 0.0 <= args.sharp_penalty <= 1.0

        if args.sharp_wide is not None:
            args.sharp_wide = bool(args.sharp_wide)
    else:
        args.sharp_penalty = None
        args.sharp_wide = None
    # else:
    #     if hasattr(args, 'sharp_penalty'):
    #         delattr(args, 'sharp_penalty')
    #     assert not hasattr(args, 'sharp_penalty')
    #
    #     if hasattr(args, 'sharp_wide'):
    #         delattr(args, 'sharp_wide')
    #     assert not hasattr(args, 'sharp_wide')

    if args.rand_ref is not None:
        args.rand_ref = bool(args.rand_ref)
    # else:
    #     delattr(args, 'rand_ref')

    args.boot = bool(args.boot)
    args.epist = bool(args.epist)
    args.recal = bool(args.recal)
    args.debug = bool(args.debug)

    if args.boot:
        if not args.num_ens > 1:
            raise RuntimeError('num_ens must be above > 1 for bootstrap')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    args.device = device

    return args


if __name__ == '__main__':
    DATA_NAMES = \
        ['wine', 'naval', 'kin8nm', 'energy', 'yacht', 'concrete', 'power', 'boston']
    SEEDS = [0,1,2,3,4]

    args = parse_args()

    print('DEVICE: {}'.format(args.device))

    if args.debug:
        import pudb; pudb.set_trace()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for d in DATA_NAMES:
        for s in SEEDS:
            args.data = d
            args.seed = s

            # Save file name
            if 'penalty' not in args.loss:
                save_file_name = '{}/{}_loss{}_epist{}_ens{}_boot{}_seed{}.pkl'.format(
                    args.save_dir,
                    args.data, args.loss, args.epist, args.num_ens, args.boot, args.seed)
            else:
                # penalizing sharpness
                if args.sharp_wide is not None and args.sharp_wide:
                    save_file_name = '{}/{}_loss{}_pen{}_wideonly_epist{}_ens{}_boot{}_seed{}.pkl'.format(
                        args.save_dir,
                        args.data, args.loss, args.sharp_penalty,
                        args.epist, args.num_ens, args.boot, args.seed)
                else:
                    save_file_name = '{}/{}_loss{}_pen{}_epist{}_ens{}_boot{}_seed{}.pkl'.format(
                        args.save_dir,
                        args.data, args.loss, args.sharp_penalty, args.epist,
                        args.num_ens, args.boot, args.seed)
            if os.path.exists(save_file_name):
                print('skipping {}'.format(save_file_name))
                continue
                sys.exit()

            # Set seeds
            set_seeds(args.seed)

            # Fetching data 
            data_args = Namespace(data_dir=args.data_dir, dataset=args.data,
                                  seed=args.seed)
            if 'uci' in args.data_dir.lower():
                data_out = get_uci_data(args)
            else:
                data_out = get_fusion_data(args)
            x_tr, x_va, x_te, y_tr, y_va, y_te, y_al = \
                data_out.x_tr, data_out.x_va, data_out.x_te, data_out.y_tr, \
                data_out.y_va, data_out.y_te, data_out.y_al
            y_range = (y_al.max() - y_al.min()).item()
            print('y range: {:.3f}'.format(y_range))

            if args.epist:
                temp_x_tr, temp_y_tr = deepcopy(x_tr), deepcopy(y_tr)
                x_tr, y_tr = x_te, y_te
                x_te, y_te = temp_x_tr, temp_y_tr
                assert ((x_tr.shape[0]) < x_te.shape[0]) and ((y_tr.shape[0]) < y_te.shape[0])

            # Making models
            num_tr = x_tr.shape[0]
            dim_x = x_tr.shape[1]
            dim_y = y_tr.shape[1]

            model_ens = QModelEns(input_size=dim_x+1, output_size=dim_y,
                                  hidden_size=args.hs, num_layers=args.nl,
                                  lr=args.lr, wd=args.wd,
                                  num_ens=args.num_ens, device=args.device)
            # model = vanilla_nn(input_size=dim_x + 1, output_size=dim_y,
            #                    hidden_size=64, num_layers=2)
            # model = model.to(args.device)
            #
            # optimizer = torch.optim.Adam(model.parameters(),
            #                              lr=args.lr, weight_decay=args.wd)

            # Data loader 
            if not args.boot:
                loader = DataLoader(TensorDataset(x_tr, y_tr),
                                    shuffle=True,
                                    batch_size=args.bs)
            else:
                rand_idx_list = [
                    np.random.choice(num_tr, size=num_tr, replace=True)
                    for _ in range(args.num_ens)]
                loader_list = [DataLoader(TensorDataset(x_tr[idxs], y_tr[idxs]),
                                          shuffle=True, batch_size=args.bs)
                               for idxs in rand_idx_list]

            # Loss function
            loss_fn = get_loss_fn(args.loss)
            args.scale = True if 'scaled' in args.loss else False
            batch_loss = True if 'batch' in args.loss else False

            """ train loop """
            tr_loss_list = []
            va_loss_list = []
            te_loss_list = []

            # for ep in tqdm.tqdm(range(args.num_ep)):
            for ep in (range(args.num_ep)):

                if model_ens.done_training:
                    print('Done training ens at EP {}'.format(ep))
                    break

                # Take train step
                ep_train_loss = []  # list of losses from each batch, for one epoch
                if not args.boot:
                    for (xi, yi) in loader:
                        xi, yi = xi.to(args.device), yi.to(args.device)
                        q_list = torch.rand(args.num_q)
                        loss = model_ens.loss(loss_fn, xi, yi, q_list, batch_q=batch_loss,
                                              take_step=True, args=args)
                        ep_train_loss.append(loss)
                else:
                    for xi_yi_samp in zip(*loader_list):
                        xi_list = [item[0].to(args.device) for item in xi_yi_samp]
                        yi_list = [item[1].to(args.device) for item in xi_yi_samp]
                        assert len(xi_list) == len(yi_list) == args.num_ens
                        q_list = torch.rand(args.num_q)
                        loss = model_ens.loss_boot(
                            loss_fn, xi_list, yi_list, q_list,
                            batch_q=batch_loss, take_step=True, args=args
                        )
                        ep_train_loss.append(loss)
                ep_tr_loss = np.nanmean(np.stack(ep_train_loss, axis=0), axis=0)
                tr_loss_list.append(ep_tr_loss)

                # Validation loss
                x_va, y_va = x_va.to(args.device), y_va.to(args.device)
                va_te_q_list = torch.linspace(0.01, 0.99, 99)

                ep_va_loss = model_ens.update_va_loss(
                    loss_fn, x_va, y_va, va_te_q_list,
                    batch_q=batch_loss, curr_ep=ep, num_wait=args.wait,
                    args=args
                )
                va_loss_list.append(ep_va_loss)

                # Test loss
                x_te, y_te = x_te.to(args.device), y_te.to(args.device)
                with torch.no_grad():
                    ep_te_loss = model_ens.loss(
                        loss_fn, x_te, y_te, va_te_q_list,
                        batch_q=batch_loss, take_step=False,
                        args=args
                    )
                te_loss_list.append(ep_te_loss)

                # Printing some losses
                if (ep % 100 == 0) or (ep == args.num_ep-1):
                    print('EP:{}'.format(ep))
                    print('Train loss {}'.format(ep_tr_loss))
                    print('Val loss {}'.format(ep_va_loss))
                    print('Test loss {}'.format(ep_te_loss))

            # Move everything to cpu
            x_tr, y_tr, x_va, y_va, x_te, y_te = \
                x_tr.cpu(), y_tr.cpu(), x_va.cpu(), y_va.cpu(), x_te.cpu(), y_te.cpu()
            model_ens.use_device(torch.device('cpu'))

            # plt.plot(np.arange(len(tr_loss_list)) * 20, np.log(tr_loss_list), label='train')
            # plt.plot(np.arange(len(va_loss_list)) * 20, np.log(va_loss_list), label='val')
            # plt.plot(np.arange(len(te_loss_list)) * 20, np.log(te_loss_list), label='test')
            # plt.legend()
            # plt.show()

            """ Test UQ on train """
            print('Testing UQ on train')
            tr_exp_props = torch.linspace(0.01, 0.99, 99)
            tr_cali_score, tr_sharp_score, tr_obs_props, tr_q_preds = \
                test_uq(model_ens, x_tr, y_tr, tr_exp_props, y_range,
                        recal_model=None, recal_type=None)

            """ Test UQ on val """
            print('Testing UQ on val')
            va_exp_props = torch.linspace(-2.0, 3.0, 501)
            va_cali_score, va_sharp_score, va_obs_props, va_q_preds = \
                test_uq(model_ens, x_va, y_va, va_exp_props, y_range,
                        recal_model=None, recal_type=None)

            """Test UQ on test """
            print('Testing UQ on test')
            te_exp_props = torch.linspace(0.01, 0.99, 99)
            te_cali_score, te_sharp_score, te_obs_props, te_q_preds = \
                test_uq(model_ens, x_te, y_te, te_exp_props, y_range,
                        recal_model=None, recal_type=None)

            if args.recal:
                recal_model = iso_recal(va_exp_props, va_obs_props)
                recal_exp_props = torch.linspace(0.01, 0.99, 99)

                recal_va_cali_score, recal_va_sharp_score, recal_va_obs_props, \
                recal_va_q_preds = \
                    test_uq(model_ens, x_va, y_va, recal_exp_props, y_range,
                            recal_model=recal_model, recal_type='sklearn')

                recal_te_cali_score, recal_te_sharp_score, recal_te_obs_props, \
                recal_te_q_preds = \
                    test_uq(model_ens, x_te, y_te, recal_exp_props, y_range,
                            recal_model=recal_model, recal_type='sklearn')

                recal_tr_cali_score, recal_tr_sharp_score, recal_tr_obs_props, \
                recal_tr_q_preds = \
                    test_uq(model_ens, x_tr, y_tr, recal_exp_props, y_range,
                            recal_model=recal_model, recal_type='sklearn')

            save_dic = {
                'tr_loss_list': tr_loss_list,    # loss lists
                'va_loss_list': va_loss_list,
                'te_loss_list': te_loss_list,

                'tr_cali_score': tr_cali_score,   # test on tr
                'tr_sharp_score': tr_sharp_score,
                'tr_exp_props': tr_exp_props,
                'tr_obs_props': tr_obs_props,
                'tr_q_preds': tr_q_preds,

                'va_cali_score': va_cali_score,   # test on va
                'va_sharp_score': va_sharp_score,
                'va_exp_props': va_exp_props,
                'va_obs_props': va_obs_props,
                'va_q_preds': va_q_preds,

                'te_cali_score': te_cali_score,   # test on te
                'te_sharp_score': te_sharp_score,
                'te_exp_props': te_exp_props,
                'te_obs_props': te_obs_props,
                'te_q_preds': te_q_preds,

                'recal_model': recal_model,
                'recal_exp_props': recal_exp_props,

                'recal_tr_cali_score': recal_tr_cali_score,
                'recal_tr_sharp_score': recal_tr_sharp_score,
                'recal_tr_obs_props': recal_tr_obs_props,
                'recal_tr_q_preds': recal_tr_q_preds,

                'recal_va_cali_score': recal_va_cali_score,
                'recal_va_sharp_score': recal_va_sharp_score,
                'recal_va_obs_props': recal_va_obs_props,
                'recal_va_q_preds': recal_va_q_preds,

                'recal_te_cali_score': recal_te_cali_score,
                'recal_te_sharp_score': recal_te_sharp_score,
                'recal_te_obs_props': recal_te_obs_props,
                'recal_te_q_preds': recal_te_q_preds,

                'x_va': x_va,
                'y_va': y_va,
                'x_te': x_te,
                'y_te': y_te,

                'args': args,

                'model': model_ens
            }

            with open(save_file_name, 'wb') as pf:
                pkl.dump(save_dic, pf)


