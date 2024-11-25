"""
Functions to output evaluation metrics for quantile models
Quantile models map (X, alpha) --> (q),
  where q is the quantile level in the target space for alpha
i.e. P(y<q|X) = alpha

Denote dimension of X as dim_x
The model must have a __call__ function implemented that
    only takes in an (N, dim_x + 1), and possibly other optional parameters
    (e.g. device, options) returns an (N, 1) output

Input and Output will be treated as numpy arrays (utilities will take over
  conversions to Tensors, etc)
If the quantile model is a marginal model (i.e. doesn't take in X), then X
  should be set to None

Model can a PyTorch model, CatBoost model, etc
"""

import os, sys
import tqdm
import torch
import numpy as np
from copy import deepcopy
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.interpolate import interp1d

from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils.q_model_ens import QModelEns


DATA_TYPES = ["numpy", "torch"]


### Utils ###
def convert_input_type(in_arr, out_type):
    if out_type not in DATA_TYPES:
        raise RuntimeError("type is invalid, must be one of {}".format(DATA_TYPES))
    if out_type == "numpy":
        output = in_arr
    elif out_type == "torch":
        output = torch.from_numpy(in_arr)

    return output


def use_device(model, X, y, device):
    if isinstance(model, QModelEns):
        model.use_device(device)
        X, y = X.to(device), y.to(device)
    else:  # TODO take care of more cases
        import pudb

        pudb.set_trace()
    return model, X, y


def get_score_fn(metric):
    if metric == "check":
        return check_loss
    elif metric == "crps":
        return crps_score
    elif metric == "int":
        return interval_score
    elif metric in ["cal_q", "cal_i"]:
        return average_calibration
    elif metric == "mpiw":
        return mpiw
    elif metric == "nll":
        return bag_nll
    # TODO: more cases


def recal_quantiles(recal_model, q_arr):
    orig_shape = q_arr.shape
    if isinstance(q_arr, torch.Tensor):
        orig_device = q_arr.device
        in_q_arr = q_arr.to("cpu").detach().numpy().flatten()
        out_q_arr = recal_model.predict(in_q_arr)
        out_q_arr = (
            torch.from_numpy(out_q_arr).float().reshape(orig_shape).to(orig_device)
        )
    elif isinstance(q_arr, np.ndarray):
        in_q_arr = q_arr.flatten()
        out_q_arr = recal_model.predict(in_q_arr).reshape(orig_shape)

    return out_q_arr


""" Proper Scoring Rules """


def check_loss(model, X, y, args):  # all done
    """
    Check loss scoring rule (pinball loss)
    :param model:
    :param X: assuming tensor
    :param y: assuming tensor
    :param args: required - q_list (assuming tensor)
    :return:
    """
    if not hasattr(args, "q_list"):
        raise RuntimeError("args must have q_list for check_loss")
    q_list = args.q_list

    num_pts = y.size(0)
    num_q = q_list.size(0)

    q_rep = q_list.view(-1, 1).repeat(1, num_pts).view(-1, 1).to(args.device)
    y_stacked = y.repeat(num_q, 1)

    # recalibrate if needed
    if hasattr(args, "recal_model") and args.recal_model is not None:
        q_rep = recal_quantiles(args.recal_model, q_rep)

    if X is None:
        model_in = q_rep
    else:
        x_stacked = X.repeat(num_q, 1)
        model_in = torch.cat([x_stacked, q_rep], dim=1)

    pred_y = model.predict(model_in)

    diff = pred_y - y_stacked
    mask = (diff.ge(0).float() - q_rep).detach()
    # TODO: mean or sum?
    check_loss_score = torch.mean((mask * diff))

    return check_loss_score


def crps_score(model, X, y, args):  # all done
    """
    CRPS score integrated along range of CDF, not domain of CDF
    :param model:
    :param X: assuming tensor
    :param y: assuming tensor
    :param args: optional - num_p (assuming tensor)
    :return:
    """

    # if not hasattr(args, 'num_p'):
    #     # print('args does not have num_p for crps_score, replacing with default')
    args.num_p = 1000
    resolution = 1000

    num_pts = y.size(0)
    p_list = torch.linspace(0.01, 0.99, args.num_p)
    num_q = p_list.size(0)
    assert num_q == args.num_p
    p_rep = p_list.view(-1, 1).repeat(1, num_pts).view(-1, 1).to(args.device)
    y_stacked = y.repeat(num_q, 1)

    # recalibrate if needed
    if hasattr(args, "recal_model") and args.recal_model is not None:
        p_rep = recal_quantiles(args.recal_model, p_rep)

    if X is None:
        model_in = p_rep
    else:
        x_stacked = X.repeat(num_q, 1)
        model_in = torch.cat([x_stacked, p_rep], dim=1)

    with torch.no_grad():
        pred_y = model.predict(model_in)
    pred_y_mat = pred_y.reshape(num_q, num_pts).T

    p_list = p_list.numpy().flatten()
    scores_list = []
    for pt_idx in range(num_pts):
        curr_y = float(y[pt_idx].item())
        curr_q_preds = pred_y_mat[pt_idx].cpu().numpy().flatten()
        min_q_pred, max_q_pred = np.min(curr_q_preds), np.max(curr_q_preds)

        # #####
        # curr_cdf = interp1d(curr_q_preds, p_list, kind='linear')
        # sample_xs = np.linspace(min_q_pred, max_q_pred, resolution)
        # sample_ys = curr_cdf(sample_xs)
        #
        # if curr_y < min_q_pred:
        #     curr_crps = np.nanmean(np.square(1 - sample_ys))
        # elif max_q_pred < curr_y:
        #     curr_crps = np.nanmean(np.square(sample_ys))
        # else:
        #     assert (min_q_pred <= curr_y) and (curr_y <= max_q_pred)
        #     below_idx = (sample_xs <= curr_y).astype(float)
        #     above_idx = 1.0 - below_idx
        #     crps_arr = ((np.square(sample_ys) * below_idx) +
        #                 (np.square(1 - sample_ys) * above_idx))
        #     curr_crps = np.nanmean(crps_arr)
        # #####

        ##### Non-inverted
        if curr_y < min_q_pred:
            curr_crps = np.mean(np.square(1 - p_list))
        elif max_q_pred < curr_y:
            curr_crps = np.mean(np.square(p_list))
        else:
            assert (np.min(curr_q_preds) <= curr_y) and (curr_y <= np.max(curr_q_preds))
            below_idx = (curr_q_preds <= curr_y).astype(float)
            above_idx = 1.0 - below_idx
            curr_crps = np.mean(
                (np.square(p_list) * below_idx) + (np.square(1 - p_list) * above_idx)
            )
        #####
        if not np.isfinite(curr_crps):
            import pdb

            pdb.set_trace()

        scores_list.append(curr_crps)

    loss = np.mean(scores_list)
    return loss


def orig_crps_score(model, X, y, args):  # all done
    """
    CRPS score integrated along range of CDF, not domain of CDF
    :param model:
    :param X: assuming tensor
    :param y: assuming tensor
    :param args: optional - num_p (assuming tensor)
    :return:
    """

    if not hasattr(args, "num_p"):
        print("args does not have num_p for crps_score, replacing with default")
        args.num_p = 100

    num_pts = y.size(0)
    p_list = torch.linspace(0.0, 1.0, args.num_p)
    num_q = p_list.size(0)
    p_rep = p_list.view(-1, 1).repeat(1, num_pts).view(-1, 1).to(args.device)
    y_stacked = y.repeat(num_q, 1)

    # recalibrate if needed
    if hasattr(args, "recal_model") and args.recal_model is not None:
        p_rep = recal_quantiles(args.recal_model, p_rep)

    if X is None:
        model_in = p_rep
    else:
        x_stacked = X.repeat(num_q, 1)
        model_in = torch.cat([x_stacked, p_rep], dim=1)

    pred_y = model.predict(model_in)

    abs_diff = torch.abs(pred_y - y_stacked)
    # TODO: mean or sum?
    loss = torch.mean(abs_diff)
    # abs_diff = torch.abs(pred_y - y_stacked)
    # crps_per_pt = torch.mean(abs_diff, dism=1)
    # mean_crps = torch.mean(crps_per_pt)

    return loss


def interval_score(model, X, y, args):  # all done
    """
    Interval score
    :param model:
    :param X: assuming tensor
    :param y: assuming tensor
    :param args: optional - alpha_list (assuming tensor)
    :return:
    """

    # if not hasattr(args, 'alpha_list'):
    # print('args does not have alpha_list for interval_score, '
    #       'replacing with default')
    # alpha_list = torch.arange(50, dtype=float)/100.0
    # alpha_list = torch.arange(0.01, 0.50, 0.01)

    # alpha_list = args.alpha_list
    alpha_list = torch.linspace(0.01, 0.99, 99)
    num_pts = y.size(0)
    num_alpha = alpha_list.size(0)
    assert num_alpha == 99

    with torch.no_grad():
        l_list = torch.min(
            torch.stack([(alpha_list / 2.0), 1 - (alpha_list / 2.0)], dim=1), dim=1
        )[0]
        u_list = 1.0 - l_list

    # recalibrate if needed
    if hasattr(args, "recal_model") and args.recal_model is not None:
        l_list = recal_quantiles(args.recal_model, l_list)
        u_list = recal_quantiles(args.recal_model, u_list)

    l_rep = l_list.view(-1, 1).repeat(1, num_pts).view(-1, 1).to(args.device).float()
    u_rep = u_list.view(-1, 1).repeat(1, num_pts).view(-1, 1).to(args.device).float()
    num_l = l_rep.size(0)
    num_u = u_rep.size(0)

    if X is None:
        model_in = torch.cat([l_list, u_list], dim=0)
    else:
        x_stacked = X.repeat(num_alpha, 1)
        l_in = torch.cat([x_stacked, l_rep], dim=1)
        u_in = torch.cat([x_stacked, u_rep], dim=1)
        model_in = torch.cat([l_in, u_in], dim=0)

    pred_y = model.predict(model_in)
    pred_l = pred_y[:num_l].view(num_alpha, num_pts)
    pred_u = pred_y[num_l:].view(num_alpha, num_pts)

    below_l = (pred_l - y.view(-1)).gt(0)
    above_u = (y.view(-1) - pred_u).gt(0)

    score_per_alpha = (
        (pred_u - pred_l)
        + (1.0 / l_list).view(-1, 1).to(args.device) * (pred_l - y.view(-1)) * below_l
        + (1.0 / l_list).view(-1, 1).to(args.device) * (y.view(-1) - pred_u) * above_u
    )
    int_score = torch.mean(score_per_alpha)

    return int_score


""" Average Calibration """


def get_obs_props(
    model,
    X,
    y,
    exp_props,
    device,
    type,  # all done
    recal_model=None,
    recal_type="sklearn",
):
    """
    Outputs observed proportions by model per expected proportions
    :param model: assumes a torch model (for now)
    :param X:
    :param y:
    :param exp_props:
    :param device:
    :param recal_model:
    :param recal_type:
    :return:
    """

    if exp_props is None:
        exp_props = torch.linspace(0.01, 0.99, 99)
    else:
        exp_props = exp_props.flatten()

    if type not in ["quantile", "interval"]:
        raise ValueError("type must be one of quantile or interval")

    num_pts = X.size(0)
    obs_props = []
    cdf_preds = []

    for p in exp_props:
        if recal_model is not None:
            if recal_type == "torch":
                recal_model.cpu()
                with torch.no_grad():
                    p = recal_model(p.reshape(1, -1)).item()
            elif recal_type == "sklearn":
                p = float(recal_model.predict(p.flatten()))
            else:
                raise ValueError("recal_type incorrect")

        p_tensor = (p * torch.ones(num_pts)).reshape(-1, 1).to(device)
        cdf_in = torch.cat([X, p_tensor], dim=1)

        with torch.no_grad():
            cdf_pred = model.predict(cdf_in).reshape(num_pts, -1)

        # store cdf prediction at each quantile, regardless of type
        cdf_preds.append(cdf_pred)

        if type == "quantile":
            prop = torch.mean((y <= cdf_pred).float())
            obs_props.append(prop.item())
        elif type == "interval":
            lower_p = 0.5 - (p / 2.0)
            upper_p = 0.5 + (p / 2.0)
            if not (float(upper_p - lower_p) - float(p) < 1e-5):
                import pudb

                pudb.set_trace()

            lower_p_tensor = (lower_p * torch.ones(num_pts)).reshape(-1, 1).to(device)
            upper_p_tensor = (upper_p * torch.ones(num_pts)).reshape(-1, 1).to(device)

            with torch.no_grad():
                lower_cdf_pred = model.predict(
                    torch.cat([X, lower_p_tensor], dim=1)
                ).reshape(num_pts, -1)
                upper_cdf_pred = model.predict(
                    torch.cat([X, upper_p_tensor], dim=1)
                ).reshape(num_pts, -1)

            above_lower = (lower_cdf_pred <= y).float()
            below_upper = (y <= upper_cdf_pred).float()
            prop = torch.mean(above_lower * below_upper)
            obs_props.append(prop.item())

    cdf_preds = torch.cat(
        cdf_preds, dim=1
    ).T  # shape (num_quantiles, num_pts)...most likely (99, num_pts)
    obs_props = torch.Tensor(obs_props)  # flat tensor of props

    return exp_props, obs_props, cdf_preds


def average_calibration(model, X, y, args):  # all done
    """
    Calculates average calibration of model
    :param model:
    :param X:
    :param y:
    :param args: required - cali_type: one of 'quantile' or 'interval'
                 optional - exp_props (assume flat tensor),
                            recal_model, recal_type
    :return:
    """

    if not hasattr(args, "exp_props"):
        args.exp_props = None
    if not hasattr(args, "recal_model"):
        args.recal_model, args.recal_type = None, None

    if args.metric == "cal_q":
        cali_type = "quantile"
    elif args.metric == "cal_i":
        cali_type = "interval"

    exp_props, obs_props, cdf_preds = get_obs_props(
        model,
        X,
        y,
        args.exp_props,
        device=args.device,
        type=cali_type,
        recal_model=args.recal_model,
        recal_type=args.recal_type,
    )

    # # Compute miscalibration area
    # polygon_points = []
    # for point in zip(exp_props, obs_props):
    #     polygon_points.append(point)
    # for point in zip(reversed(exp_props), reversed(exp_props)):
    #     polygon_points.append(point)
    # polygon_points.append((exp_props[0], obs_props[0]))
    # polygon = Polygon(polygon_points)
    # x, y = polygon.exterior.xy  # original data
    # ls = LineString(np.c_[x, y])  # closed, non-simple
    # lr = LineString(ls.coords[:] + ls.coords[0:1])
    # mls = unary_union(lr)
    # polygon_area_list = [poly.area for poly in polygonize(mls)]
    # miscal_area = np.asarray(polygon_area_list).sum()

    miscal_area = torch.mean(torch.abs(exp_props - obs_props)).item()

    return miscal_area


""" Sharpness """


def mpiw(model, X, y, args):  # all done
    """
    Calculates MPIW of a single alpha
    :param model:
    :param X:
    :param y:
    :param args: optional - alpha_list (assume flat tensor)
    :return:
    """

    if not hasattr(args, "alpha_list"):
        args.alpha_list = torch.Tensor([0.05])

    num_pts = y.size(0)
    num_alpha = args.alpha_list.size(0)

    with torch.no_grad():
        l_list = torch.min(
            torch.stack([(args.alpha_list / 2.0), 1 - (args.alpha_list / 2.0)], dim=1),
            dim=1,
        )[0]
        u_list = 1.0 - l_list

    # recalibrate if needed
    if hasattr(args, "recal_model") and args.recal_model is not None:
        l_list = recal_quantiles(args.recal_model, l_list)
        u_list = recal_quantiles(args.recal_model, u_list)

    l_rep = l_list.view(-1, 1).repeat(1, num_pts).view(-1, 1).to(args.device)
    u_rep = u_list.view(-1, 1).repeat(1, num_pts).view(-1, 1).to(args.device)
    num_l = l_rep.size(0)

    if X is None:
        model_in = torch.cat([l_list, u_list], dim=0)
    else:
        x_stacked = X.repeat(num_alpha, 1)
        l_in = torch.cat([x_stacked, l_rep], dim=1)
        u_in = torch.cat([x_stacked, u_rep], dim=1)
        model_in = torch.cat([l_in, u_in], dim=0)

    with torch.no_grad():
        pred_y = model.predict(model_in)
    pred_l = pred_y[:num_l].view(num_alpha, num_pts)
    pred_u = pred_y[num_l:].view(num_alpha, num_pts)

    mpiw_per_alpha = torch.mean(pred_u - pred_l, dim=1)

    return mpiw_per_alpha


def bag_nll(model, X, y, args):  # working
    """

    :param model:
    :param X:
    :param y:
    :param args:
    :return:
    """

    q_list = torch.linspace(0.01, 0.99, 99)
    num_pts = y.size(0)
    num_q = q_list.size(0)

    q_rep = q_list.view(-1, 1).repeat(1, num_pts).view(-1, 1).to(args.device)
    y_stacked = y.repeat(num_q, 1)

    # recalibrate if needed
    if hasattr(args, "recal_model") and args.recal_model is not None:
        q_rep = recal_quantiles(args.recal_model, q_rep)

    if X is None:
        model_in = q_rep
    else:
        x_stacked = X.repeat(num_q, 1)
        model_in = torch.cat([x_stacked, q_rep], dim=1)

    pred_y = model.predict(model_in)

    pred_y_mat = pred_y.reshape(num_q, num_pts).T
    nll_list = []
    for pt_idx in range(num_pts):
        curr_quantiles = pred_y_mat[pt_idx].detach().cpu().numpy().flatten()
        init_mean = float(np.mean(curr_quantiles))
        init_std = float(np.std(curr_quantiles))

        def obj(x):
            trial_qs = norm.ppf(
                q_list.cpu().numpy(), loc=float(x[0]), scale=float(x[1])
            ).flatten()
            sum_squared_diff = np.sum(np.square(trial_qs - curr_quantiles))
            return sum_squared_diff

        bounds = [
            (init_mean - (2 * init_std), init_mean + (2 * init_std)),
            (1e-10, 3 * init_std),
        ]

        result = minimize(obj, x0=[init_mean, init_std], bounds=bounds)
        if not result["success"]:
            print("pt {} bag not optimized well".format(pt_idx))
        opt_mean, opt_std = result["x"][0], result["x"][1]

        pt_nll = norm.logpdf(float(y[pt_idx]), loc=opt_mean, scale=opt_std)
        nll_list.append(-1 * pt_nll)

    return np.nanmean(nll_list)


### main procedure ###
def eval_quantile_uq(model, X, y, args):
    """
    Evaluate this model with data (X, y) on device

    X, y must be correct datatypes s.t. a naive pass y-model(X) works
    :param model:
    :param X: size (N, dim_x), should be provided in correct datatype
    :param y: size (N, 1), should be provided in correct datatype
    :param args: required - "metric", "device"
                 optional - "display_plot", "save_plot", "save_plot_dir",
                            "save_plot_name"
                 required_per_metric - (refer to each metric definition)
    :return: a scalar score
    """
    model, X, y = use_device(model, X, y, args.device)
    if isinstance(args.metric, list):
        score = {}
        metric_list = deepcopy(args.metric)
        for curr_metric in metric_list:
            curr_score_fn = get_score_fn(curr_metric)
            args.metric = curr_metric
            curr_score = curr_score_fn(model, X, y, args).item()
            score[curr_metric] = curr_score
        return score
    elif isinstance(args.metric, str):
        import pdb

        pdb.set_trace()
        score_fn = get_score_fn(args.metric)
        score = score_fn(model, X, y, args).item()
        return score
