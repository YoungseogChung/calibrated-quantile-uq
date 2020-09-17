import random
import numpy as np
from scipy.interpolate import interp1d
import torch
import matplotlib.pyplot as plt
from cali_plot import get_props, plot_calibration_curve, ens_get_props


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_q_idx(exp_props, q):
    target_idx = None
    for idx, x in enumerate(exp_props):
        if x <= q < exp_props[idx + 1]:
            target_idx = idx+1
            break
    if target_idx is None:
        import pdb; pdb.set_trace()
        raise ValueError('q must be within exp_props')
    return target_idx


def test_uq(model, x, y, exp_props, y_range, recal_model=None, recal_type=None,
            make_plots=False):

    # obs_props, quantile_preds, quantile_preds_mat = \
    #     ens_get_props(model, x, y, exp_props=exp_props, recal_model=recal_model,
    #                   recal_type=recal_type)

    num_pts = x.shape[0]
    y = y.detach().cpu().reshape(num_pts, -1)

    # of shape (num_pts, num_q)
    quantile_preds = model.predict_q(
        x, exp_props, ens_pred_type='conf',
        recal_model=recal_model, recal_type=recal_type
    )
    obs_props = torch.sum((quantile_preds >= y).float(), dim=1).flatten()

    assert exp_props.shape == obs_props.shape

    idx_01 = get_q_idx(exp_props, 0.01)
    idx_99 = get_q_idx(exp_props, 0.99)
    cali_score = plot_calibration_curve(exp_props[idx_01:idx_99 + 1],
                                        obs_props[idx_01:idx_99 + 1],
                                        make_plots=make_plots)

    order = torch.argsort(y.flatten())
    q_025 = quantile_preds[:, get_q_idx(exp_props, 0.025)][order]
    q_975 = quantile_preds[:, get_q_idx(exp_props, 0.975)][order]
    sharp_score = torch.mean(q_975 - q_025).item() / y_range

    if make_plots:
        plt.figure(figsize=(8, 5))
        plt.plot(torch.arange(y.size(0)), q_025)
        plt.plot(torch.arange(y.size(0)), q_975)
        plt.scatter(torch.arange(y.size(0)), y[order], c='r')
        plt.title('Mean Width: {:.3f}'.format(sharp_score))
        plt.show()

    return cali_score, sharp_score, obs_props, quantile_preds


def gather_loss_per_q(loss_fn, model, y, x, q_list, device, args):
    loss_list = []
    for q in q_list:
        q_loss = loss_fn(model, y, x, q, device, args)
        loss_list.append(q_loss)
    loss = torch.mean(torch.stack(loss_list))

    return loss

