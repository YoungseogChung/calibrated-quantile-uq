import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union
from scipy.interpolate import interp1d
import torch
import tqdm


def plot_calibration_curve(
    exp_proportions,
    obs_proportions,
    title=None,
    curve_label=None,
    make_plots=False,
):
    # Set figure defaults
    if make_plots:
        width = 5
        fontsize = 12
        rc = {
            "figure.figsize": (width, width),
            "font.size": fontsize,
            "axes.labelsize": fontsize,
            "axes.titlesize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
        }
        sns.set(rc=rc)
        sns.set_style("ticks")

        # Set label
        if curve_label is None:
            curve_label = "Predictor"

        # Plot
        plt.figure()
        if title is not None:
            plt.title(title)
        plt.plot([0, 1], [0, 1], "--", label="Ideal")
        plt.plot(exp_proportions, obs_proportions, label=curve_label)
        plt.fill_between(
            exp_proportions, exp_proportions, obs_proportions, alpha=0.2
        )
        plt.xlabel("Expected proportion in interval")
        plt.ylabel("Observed proportion in interval")
        plt.axis("square")
        buff = 0.01
        plt.xlim([0 - buff, 1 + buff])
        plt.ylim([0 - buff, 1 + buff])

        # Compute miscalibration area
        polygon_points = []
        for point in zip(exp_proportions, obs_proportions):
            polygon_points.append(point)
        for point in zip(reversed(exp_proportions), reversed(exp_proportions)):
            polygon_points.append(point)
        polygon_points.append((exp_proportions[0], obs_proportions[0]))
        polygon = Polygon(polygon_points)
        x, y = polygon.exterior.xy  # original data
        ls = LineString(np.c_[x, y])  # closed, non-simple
        lr = LineString(ls.coords[:] + ls.coords[0:1])
        mls = unary_union(lr)
        polygon_area_list = [poly.area for poly in polygonize(mls)]
        miscalibration_area = np.asarray(polygon_area_list).sum()

        # Annotate plot with the miscalibration area
        plt.text(
            x=0.95,
            y=0.05,
            s="Miscalibration area = %.2f" % miscalibration_area,
            verticalalignment="bottom",
            horizontalalignment="right",
            fontsize=fontsize,
        )
        plt.show()
    else:
        # not making plots, just computing ECE
        miscalibration_area = torch.mean(
            torch.abs(exp_proportions - obs_proportions)
        ).item()

    return miscalibration_area


def get_props(
    cdf_model,
    x_tensor,
    y_tensor,
    exp_props=None,
    recal_model=None,
    recal_type=None,
):
    x_tensor = x_tensor.cpu()
    y_tensor = y_tensor.cpu()
    cdf_model.cpu()

    if exp_props is None:
        exp_props = torch.linspace(0.00, 1.00, 101)

    num_pts = x_tensor.size(0)
    props = []
    cdf_preds = []
    for p in exp_props:

        # getting recalibrated prop
        if recal_model is not None:
            if recal_type == "torch":
                recal_model.cpu()
                with torch.no_grad():
                    p = recal_model(p.reshape(1, -1)).item()
            elif recal_type == "sklearn":
                p = float(recal_model.predict(p.flatten()))
            else:
                raise ValueError("recal_type incorrect")

        p_tensor = (p * torch.ones(num_pts)).reshape(-1, 1)
        cdf_in = torch.Tensor(torch.cat([x_tensor, p_tensor], dim=1))

        with torch.no_grad():
            cdf_pred = cdf_model(cdf_in).reshape(num_pts, -1)
        cdf_preds.append(cdf_pred)

        prop = torch.mean(cdf_pred > y_tensor, dtype=float)

        props.append(prop.item())

    return torch.Tensor(props), cdf_preds


def get_ens_pred(unc_preds, taus):
    """unc_preds 3D ndarray  (ens_size, 99, num_x) where each row
    corresonds to tau 0.01, 0.02... and the columns
    are for the set of x being predicted over.
    """
    # taus = np.arange(0.01, 1, 0.01)
    y_min, y_max = np.min(unc_preds), np.max(unc_preds)
    y_grid = np.linspace(y_min, y_max, 10000)
    new_quants = []
    avg_cdfs = []
    for x_idx in tqdm.tqdm(range(unc_preds.shape[-1])):
        x_cdf = []
        for ens_idx in range(unc_preds.shape[0]):
            xs, ys = [], []
            targets = unc_preds[ens_idx, :, x_idx]
            for idx in np.argsort(targets):
                if len(xs) != 0 and targets[idx] <= xs[-1]:
                    continue
                xs.append(targets[idx])
                ys.append(taus[idx])
            intr = interp1d(
                xs, ys, kind="linear", fill_value=([0], [1]), bounds_error=False
            )
            x_cdf.append(intr(y_grid))
        x_cdf = np.asarray(x_cdf)
        avg_cdf = np.mean(x_cdf, axis=0)
        avg_cdfs.append(avg_cdf)
        t_idx = 0
        x_quants = []
        for idx in range(len(avg_cdf)):
            if t_idx >= len(taus):
                break
            if taus[t_idx] <= avg_cdf[idx]:
                x_quants.append(y_grid[idx])
                t_idx += 1
        while t_idx < len(taus):
            x_quants.append(y_grid[-1])
            t_idx += 1
        new_quants.append(x_quants)
    return np.asarray(new_quants).T


def ens_get_props(
    cdf_model,
    x_tensor,
    y_tensor,
    exp_props=None,
    recal_model=None,
    recal_type=None,
):

    x_tensor = x_tensor.cpu()
    y_tensor = y_tensor.cpu()
    num_ens = cdf_model.num_ens
    for m in cdf_model.best_va_model:
        m.cpu()

    if exp_props is None:
        exp_props = torch.arange(0.01, 1.00, 0.01)
    num_q = exp_props.size(0)

    num_pts = x_tensor.size(0)
    props = []
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

        p_tensor = (p * torch.ones(num_pts)).reshape(-1, 1)
        cdf_in = torch.Tensor(torch.cat([x_tensor, p_tensor], dim=1))

        ens_preds_p = []
        with torch.no_grad():
            for m in cdf_model.best_va_model:
                cdf_pred = m(cdf_in).reshape(num_pts, -1)
                ens_preds_p.append(cdf_pred.flatten())

        cdf_preds.append(torch.stack(ens_preds_p, dim=0).unsqueeze(1))
    ens_pred_mat = torch.cat(cdf_preds, dim=1).numpy()

    if num_ens > 1:
        assert ens_pred_mat.shape == (num_ens, num_q, num_pts)
        ens_pred = get_ens_pred(ens_pred_mat, taus=exp_props)
    else:
        ens_pred = ens_pred_mat.reshape(num_q, num_pts)

    ens_pred = torch.from_numpy(ens_pred)
    props = torch.mean((ens_pred - y_tensor.flatten()).ge(0).float(), dim=1)

    # props.append(prop.item())

    return torch.Tensor(props), ens_pred, ens_pred_mat
