import torch
import numpy as np
import matplotlib.pyplot as plt


def cali_loss(model, y, x, q, device, args):
    """
    original proposed loss function:
        when coverage is low, pulls from above
        when coverage is high, pulls from below

    q: scalar
    """
    num_pts = y.size(0)
    q = float(q)

    if x is None:
        q_tensor = torch.Tensor([q]).to(device)
        pred_y = model(q_tensor)
    else:
        q_tensor = q * torch.ones(num_pts).view(-1, 1).to(device)
        pred_y = model(torch.cat([x, q_tensor], dim=1))

    idx_under = y <= pred_y
    idx_over = ~idx_under
    coverage = torch.mean(idx_under.float()).item()

    if coverage < q:
        loss = torch.mean((y - pred_y)[idx_over])
    else:
        loss = torch.mean((pred_y - y)[idx_under])

    if hasattr(args, 'scale') and args.scale:
        loss = torch.abs(q - coverage) * loss

    if hasattr(args, 'sharp_penalty'):
        import pudb; pudb.set_trace()
        assert isinstance(args.sharp_penalty, float)

        if x is None:
            opp_q_tensor = torch.Tensor([1-q]).to(device)
            opp_pred_y = model(opp_q_tensor)
        else:
            opp_q_tensor = (1-q) * torch.ones(num_pts).view(-1, 1).to(device)
            opp_pred_y = model(torch.cat([x, opp_q_tensor], dim=1))

        with torch.no_grad():
            below_med = (q <= 0.5)
            above_med = ~below_med

        sharp_penalty = (below_med * (opp_pred_y - pred_y) +
                   above_med * (pred_y - opp_pred_y))

        if sharp_penalty <= 0.0:
            sharp_penalty = 0.0

        loss = ((1 - args.sharp_penalty) * loss +
                (args.sharp_penalty * sharp_penalty))

    return loss


def batch_cali_loss(model, y, x, q_list, device, args):
    """
    batched version of original proposed loss function for batch of quantiles

    q_list: torch tensor of size [N]
    """

    num_pts = y.size(0)
    num_q = q_list.size(0)

    q_rep = q_list.view(-1, 1).repeat(1, num_pts).view(-1, 1).to(device)
    y_stacked = y.repeat(num_q, 1)
    y_mat = y_stacked.reshape(num_q, num_pts)

    if x is None:
        model_in = q_rep
    else:
        x_stacked = x.repeat(num_q, 1)
        model_in = torch.cat([x_stacked, q_rep], dim=1)
    pred_y = model(model_in)

    idx_under = (y_stacked <= pred_y).reshape(num_q, num_pts)
    idx_over = ~idx_under
    coverage = torch.mean(idx_under.float(), dim=1)  # shape (num_q,)

    pred_y_mat = pred_y.reshape(num_q, num_pts)
    diff_mat = y_mat - pred_y_mat

    mean_diff_under = torch.mean(-1 * diff_mat * idx_under, dim=1)
    mean_diff_over = torch.mean(diff_mat * idx_over, dim=1)

    cov_under = coverage < q_list.to(device)
    cov_over = ~cov_under

    if hasattr(args, 'rand_ref') and args.rand_ref:
        import pudb; pudb.set_trace()
    else:
        loss_list = (cov_under * mean_diff_over) + (cov_over * mean_diff_under)

    # handle scaling
    if hasattr(args, 'scale') and args.scale:
        cov_diff = torch.abs(coverage - q_list.to(device)) 
        import pdb; pdb.set_trace()
        loss_list = cov_diff * loss_list
        loss = torch.mean(loss_list)
    else:
        loss = torch.mean(loss_list)



    # handle sharpness penalty
    if hasattr(args, 'sharp_penalty'):
        assert isinstance(args.sharp_penalty, float)

        if x is None:
            opp_q_model_in = 1.0 - q_rep
        else:
            opp_q_model_in = torch.cat([x_stacked, (1.0 - q_rep)], dim=1)
        opp_pred_y = model(opp_q_model_in)

        with torch.no_grad():
            below_med = (q_rep <= 0.5)
            above_med = ~below_med

        sharp_penalty = (below_med * (opp_pred_y - pred_y) +
                         above_med * (pred_y - opp_pred_y))
        with torch.no_grad():
            width_positive = (sharp_penalty > 0.0)
            assert tuple(width_positive.shape) == tuple(sharp_penalty.shape)

        # penalize sharpness only if centered interval obs props is too high
        if hasattr(args, 'sharp_wide_only') and args.sharp_wide_only:
            import pudb; pudb.set_trace()
            with torch.no_grad():
                opp_pred_y_mat = opp_pred_y.reshape(num_q, num_pts)
                below_med_mat = below_med.reshape(num_q, num_pts)
                exp_interval_props = torch.abs((2 * q_list) - 1)

                interval_lower_mat = ((below_med_mat * pred_y_mat) +
                                      (~below_med_mat * opp_pred_y_mat))
                interval_upper_mat = ((~below_med_mat * pred_y_mat) +
                                      (below_med_mat * opp_pred_y_mat))

                within_interval_mat = ((interval_lower_mat <= y_mat) *
                                       (y_mat <= interval_upper_mat))
                obs_interval_props = torch.mean(within_interval_mat.float(), dim=1)
                obs_over_exp = (obs_interval_props > exp_interval_props)
            sharp_penalty = obs_over_exp * width_positive * sharp_penalty
        else:
            sharp_penalty = width_positive * sharp_penalty

        loss = ((1 - args.sharp_penalty) * loss +
                (args.sharp_penalty * torch.mean(sharp_penalty)))

    return loss


def mod_cali_loss(model, y, x, q, device, args):
    """
    modified proposed loss function that uses all points:
        when coverage is low, pull from above and push from below
        when coverage is high, pull from below and push from above

    """
    num_pts = y.size(0)

    if x is None:
        q_tensor = torch.Tensor([q]).to(device)
        pred_y = model(q_tensor)
    else:
        q_tensor = q * torch.ones(num_pts).reshape(-1, 1).to(device)
        pred_y = model(torch.cat([x, q_tensor], dim=1))

    idx_under = y <= pred_y
    coverage = torch.mean(idx_under.float()).item()

    if coverage < q:
        loss = torch.mean(y - pred_y)
    else:
        loss = torch.mean(pred_y - y)

    if hasattr(args, 'scale') and args.scale:
        loss = torch.abs(q - coverage) * loss

    return loss


def batch_mod_cali_loss(model, y, x, q_list, device, args):
    """
    batched version of modified proposed loss function, for batch of quantiles

    q_list: torch tensor of size [N]
    """

    num_pts = y.size(0)
    num_q = q_list.size(0)

    q_rep = q_list.view(-1, 1).repeat(1, num_pts).view(-1, 1).to(device)
    y_stacked = y.repeat(num_q, 1)
    y_mat = y_stacked.reshape(num_q, num_pts)

    if x is None:
        model_in = q_rep
    else:
        x_stacked = x.repeat(num_q, 1)
        model_in = torch.cat([x_stacked, q_rep], dim=1)
    pred_y = model(model_in)

    idx_under = (y_stacked <= pred_y).reshape(num_q, num_pts)
    coverage = torch.mean(idx_under.float(), dim=1)

    pred_y_mat = pred_y.reshape(num_q, num_pts)
    diff_mat = y_mat - pred_y_mat

    # mean_diff_under = torch.sum(-1 * diff_mat * idx_under, dim=1) / \
    #                   torch.sum(idx_under, dtype=float, dim=1)
    # mean_diff_over = torch.sum(diff_mat * idx_over, dim=1) / \
    #                  torch.sum(idx_over, dtype=float, dim=1)

    mean_diff_under = torch.mean(-1 * diff_mat, dim=1)
    mean_diff_over = torch.mean(diff_mat, dim=1)

    cov_under = coverage < q_list.to(device)
    cov_over = ~cov_under

    if hasattr(args, 'scale') and args.scale:
        cov_diff = torch.abs(coverage - q_list.to(device)) 
        loss_list = (cov_diff * cov_under * mean_diff_over) + \
                    (cov_diff * cov_over * mean_diff_under)
    else:
        loss_list = (cov_under * mean_diff_over) + (cov_over * mean_diff_under)

    loss = torch.mean(loss_list)

    return loss


def qr_loss(model, y, x, q, device, args):
    """
    original quantile regression loss

    q: a scalar
    """
    num_pts = y.size(0)

    if x is None:
        q_tensor = torch.Tensor([q]).to(device)
        pred_y = model(q_tensor)
    else:
        q_tensor = q * torch.ones(num_pts).view(-1, 1).to(device)
        try:
            pred_y = model(torch.cat([x, q_tensor], dim=1))
        except:
            import pdb; pdb.set_trace()

    diff = pred_y - y
    mask = (diff.ge(0).float() - q).detach()

    loss = (mask * diff).mean()

    return loss


def batch_qr_loss(model, y, x, q_list, device, args):
    """
    batched version of original quantile regression loss, for batch of quantiles

    q_list: torch tensor of size [N]
    """

    num_pts = y.size(0)
    num_q = q_list.size(0)

    q_rep = q_list.view(-1, 1).repeat(1, num_pts).view(-1, 1).to(device)
    y_stacked = y.repeat(num_q, 1)

    if x is None:
        model_in = q_rep
    else:
        x_stacked = x.repeat(num_q, 1)
        model_in = torch.cat([x_stacked, q_rep], dim=1)
    pred_y = model(model_in)

    diff = pred_y - y_stacked
    mask = (diff.ge(0).float() - q_rep).detach()

    loss = (mask * diff).mean()

    return loss


def crps_loss(model, y, x, q_list, device, args):
    """
    implementation of a modified version of the CRPS score

    NOTE: optimizes to a degenerate case
    """

    num_pts = y.size(0)
    q_list = torch.arange(101)/100.0
    num_q = q_list.size(0)
    q_rep = q_list.view(-1, 1).repeat(1, num_pts).view(-1, 1).to(device)
    y_stacked = y.repeat(num_q, 1)
    y_mat = y_stacked.reshape(num_q, num_pts)

    if x is None:
        model_in = q_rep
    else:
        x_stacked = x.repeat(num_q, 1)
        model_in = torch.cat([x_stacked, q_rep], dim=1)

    pred_y = model(model_in)
    sq_diff = (pred_y - y_stacked)**2
    loss = torch.mean(sq_diff)
    # abs_diff = torch.abs(pred_y - y_stacked)
    # crps_per_pt = torch.mean(abs_diff, dism=1)
    # mean_crps = torch.mean(crps_per_pt)

    return loss


def cov_loss(model, y, x, q, device, args):
    """
    setting the loss to just be the difference in coverage

    NOTE: no traceable gradient in loss
    """

    num_pts = y.size(0)
    if x is None:
        q_tensor = torch.Tensor([q]).to(device)
        pred_y = model(q_tensor)
    else:
        q_tensor = q * torch.ones(num_pts).view(-1, 1).to(device)
        pred_y = model(torch.cat([x, q_tensor], dim=1))

    idx_under = y <= pred_y
    idx_over = ~idx_under
    coverage = torch.mean(idx_under.float())

    loss = (coverage - q)**2
    # if coverage < q:
    #     loss = torch.mean((y - pred_y)[idx_over])
    # else:
    #     loss = torch.mean((pred_y - y)[idx_under])


    return loss


def interval_loss(model, y, x, q, device, args):
    """
    implementation of interval score
    
    q: scalar
    """

    num_pts = y.size(0)

    with torch.no_grad():
        lower = torch.min(torch.stack([q, 1-q], dim=0), dim=0)[0]
        upper = 1.0 - lower
        #l_list = torch.min(torch.stack([q_list, 1-q_list], dim=1), dim=1)[0]
        #u_list = 1.0 - l_list

    l_rep = lower.view(-1, 1).repeat(1, num_pts).view(-1, 1).to(device)
    u_rep = upper.view(-1, 1).repeat(1, num_pts).view(-1, 1).to(device)

    if x is None:
        model_in = torch.cat([lower, upper], dim=0)
    else:
        l_in = torch.cat([x, l_rep], dim=1)
        u_in = torch.cat([x, u_rep], dim=1)
        model_in = torch.cat([l_in, u_in], dim=0)

    pred_y = model(model_in)
    pred_l = pred_y[:num_pts].view(-1)
    pred_u = pred_y[num_pts:].view(-1)

    below_l = (pred_l - y.view(-1)).gt(0)
    above_u = (y.view(-1) - pred_u).gt(0)

    loss = (pred_u - pred_l) + \
           (1.0/lower) * (pred_l - y.view(-1)) * below_l + \
           (1.0/lower) * (y.view(-1) - pred_u) * above_u

    return torch.mean(loss)


def batch_interval_loss(model, y, x, q_list, device, args):
    """
    implementation of interval score, for batch of quantiles
    """

    num_pts = y.size(0)
    num_q = q_list.size(0)

    with torch.no_grad():
        l_list = torch.min(torch.stack([q_list, 1-q_list], dim=1), dim=1)[0]
        u_list = 1.0 - l_list

    l_rep = l_list.view(-1, 1).repeat(1, num_pts).view(-1, 1).to(device)
    u_rep = u_list.view(-1, 1).repeat(1, num_pts).view(-1, 1).to(device)
    num_l = l_rep.size(0)
    num_u = u_rep.size(0)

    if x is None:
        model_in = torch.cat([l_list, u_list], dim=0)
    else:
        x_stacked = x.repeat(num_q, 1)
        l_in = torch.cat([x_stacked, l_rep], dim=1)
        u_in = torch.cat([x_stacked, u_rep], dim=1)
        model_in = torch.cat([l_in, u_in], dim=0)

    pred_y = model(model_in)
    pred_l = pred_y[:num_l].view(num_q, num_pts)
    pred_u = pred_y[num_l:].view(num_q, num_pts)

    below_l = (pred_l - y.view(-1)).gt(0)
    above_u = (y.view(-1) - pred_u).gt(0)

    loss = (pred_u - pred_l) + \
           (1.0/l_list).view(-1, 1).to(device) * (pred_l - y.view(-1)) * below_l + \
           (1.0/ l_list).view(-1, 1).to(device) * (y.view(-1) - pred_u) * above_u

    return torch.mean(loss)

