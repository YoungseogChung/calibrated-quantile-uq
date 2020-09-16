import torch
from sklearn.isotonic import IsotonicRegression
from utils.misc_utils import get_q_idx


def iso_recal(exp_props, obs_props):
    exp_props = exp_props.flatten()
    obs_props = obs_props.flatten()
    min_obs = torch.min(obs_props)
    max_obs = torch.max(obs_props)

    iso_model = IsotonicRegression(increasing=True, out_of_bounds='clip')
    try:
        assert torch.min(obs_props) == 0.0
        assert torch.max(obs_props) == 1.0
    except:
        print('Obs props not ideal: from {} to {}'.format(min_obs, max_obs))
    # just need observed prop values between 0 and 1
    # problematic if min_obs_p > 0 and max_obs_p < 1

    exp_0_idx = get_q_idx(exp_props, 0.0)
    exp_1_idx = get_q_idx(exp_props, 1.0)
    within_01 = obs_props[exp_0_idx : exp_1_idx+1]

    beg_idx, end_idx = None, None
    # handle beg_idx
    min_obs_below = torch.min(obs_props[:exp_0_idx])
    min_obs_within = torch.min(within_01)
    if min_obs_below < min_obs_within:
        i = exp_0_idx-1
        while obs_props[i] > min_obs_below:
            i -= 1
        beg_idx = i
    elif torch.sum((within_01 == min_obs_within).float()) > 1:
        # multiple minima in within_01 ==> get last min idx
        i = exp_1_idx - 1
        while obs_props[i] > min_obs_within:
            i -= 1
        beg_idx = i
    elif torch.sum((within_01 == min_obs_within).float()) == 1:
        beg_idx = torch.argmin(within_01) + exp_0_idx
    else:
        import pudb; pudb.set_trace()

    # handle end_idx
    max_obs_above = torch.max(obs_props[exp_1_idx+1:])
    max_obs_within = torch.max(within_01)
    if max_obs_above > max_obs_within:
        i = exp_1_idx + 1
        while obs_props[i] < max_obs_above:
            i += 1
        end_idx = i+1
    elif torch.sum((within_01 == max_obs_within).float()) > 1:
        # multiple minima in within_01 ==> get last min idx
        i = beg_idx
        while obs_props[i] < max_obs_within:
            i += 1
        end_idx = i+1
    elif torch.sum((within_01 == max_obs_within).float()) == 1:
        end_idx = exp_0_idx + torch.argmax(within_01) + 1
    else:
        import pudb; pudb.set_trace()

    assert end_idx > beg_idx

    # min_idx = torch.argmin(obs_props)
    # last_idx = obs_props.size(0)
    # beg_idx, end_idx = None, None
    #
    # for idx in range(min_idx, last_idx):
    #     if obs_props[idx] >= 0.0:
    #         if beg_idx is None:
    #             beg_idx = idx
    #     if obs_props[idx] >= 1.0:
    #         if end_idx is None:
    #             end_idx = idx+1
    # # if beg_idx is None:
    # #     beg_idx = 0
    # # if end_idx is None:
    # #     end_idx = obs_props.size(0)
    # print(beg_idx, end_idx)

    filtered_obs_props = obs_props[beg_idx:end_idx]
    filtered_exp_props = exp_props[beg_idx:end_idx]

    try:
        iso_model = iso_model.fit(filtered_obs_props, filtered_exp_props)
    except:
        import pudb; pudb.set_trace()

    return iso_model

if __name__ == '__main__':
    exp = torch.linspace(-0.5, 1.5, 200)
    from copy import deepcopy
    obs = deepcopy(exp)
    obs[:80] = 0
    obs[-80:] = 1

    iso_recal(exp, obs)

