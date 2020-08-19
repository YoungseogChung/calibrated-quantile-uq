import torch
from sklearn.isotonic import IsotonicRegression


def iso_recal(exp_props, obs_props):
    iso_model = IsotonicRegression()
    try:
        assert torch.min(obs_props) == 0.0
        assert torch.max(obs_props) == 1.0
    except:
        print(torch.min(obs_props), torch.max(obs_props))

    min_idx = torch.argmin(obs_props)
    last_idx = obs_props.size(0)
    beg_idx, end_idx = None, None
    
    for idx in range(min_idx, last_idx):
        if obs_props[idx] >= 0.0:
            if beg_idx is None:
                beg_idx = idx
        if obs_props[idx] >= 1.0:
            if end_idx is None:
                end_idx = idx+1
    # if beg_idx is None:
    #     beg_idx = 0
    # if end_idx is None:
    #     end_idx = obs_props.size(0)
    print(beg_idx, end_idx)
    filtered_obs_props = obs_props[beg_idx:end_idx]
    filtered_exp_props = exp_props[beg_idx:end_idx]
    try:
        iso_model = iso_model.fit(filtered_obs_props, filtered_exp_props)
    except:
        import pdb; pdb.set_trace()

    return iso_model
