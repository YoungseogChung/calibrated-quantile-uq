import numpy as np
import torch

def qr_loss(y, pred_y, q):
    assert pred_y.requires_grad
    assert not(y.requires_grad)
    assert isinstance(q, float)
    
    diff = pred_y - y
    mask = (diff.ge(0).float() - q)
    loss = (mask * diff).mean()
    
    return loss


def qr_loss_terms(y, pred_y, q):
    assert pred_y.requires_grad
    assert not(y.requires_grad)
    assert isinstance(q, float)

    num_pts = y.size(0)
    diff = pred_y - y
    mask = diff.ge(0).float()
    num_under = torch.sum(diff.ge(0))
    cov_pts_diff = (num_under - (q*num_pts))
    
    term_1 = cov_pts_diff * pred_y
    term_2 = q * torch.sum(y)
    term_3 = torch.sum(mask * y)
    
    loss = (term_1 + term_2 - term_3)/num_pts
    
    return loss


def qr_loss_simple(y, pred_y, q):
    assert pred_y.requires_grad
    assert not(y.requires_grad)
    assert isinstance(q, float)

    num_pts = y.size(0)
    diff = pred_y - y
    num_under = torch.sum(diff.ge(0))
    cov_pts_diff = (num_under - (q*num_pts))
    
    loss = (cov_pts_diff * pred_y)/num_pts
    
    return loss

"""
Now, our proposed losses
"""

def cov_loss_1(y, pred_y, q):
    assert pred_y.requires_grad
    assert not(y.requires_grad)
    assert isinstance(q, float)

    idx_under = (y <= pred_y)
    cov = torch.mean(idx_under, dtype=float)
    cov_sign = 1 if (q < cov) else -1
#     loss = cov_sign * torch.sum(pred_y - y[idx_under])
    loss = cov_sign * torch.mean(pred_y - y[idx_under])
        
    return loss


def cov_loss_2(y, pred_y, q):
    assert pred_y.requires_grad
    assert not(y.requires_grad)
    assert isinstance(q, float)

    idx_under = (y <= pred_y)
    cov = torch.mean(idx_under, dtype=float)
    cov_sign = 1 if (q < cov) else -1
#     loss = cov_sign * torch.sum((pred_y - y[idx_under])**2)
    loss = cov_sign * torch.mean((pred_y - y[idx_under])**2)
        
    return loss


def cov_loss_3(y, pred_y, q):
    assert pred_y.requires_grad
    assert not(y.requires_grad)
    assert isinstance(q, float)

    idx_under = (y <= pred_y)
    idx_over = ~idx_under
    cov = torch.mean(idx_under, dtype=float)
    cov_under = q > cov
    
    if cov_under:
#         loss = torch.sum(y[idx_under] - pred_y)
        loss = torch.mean(y[idx_under] - pred_y)
    else:
#         loss = torch.sum(pred_y - y[idx_over])
        loss = torch.mean(pred_y - y[idx_over])
        
    return loss


def cov_loss_4(y, pred_y, q):
    assert pred_y.requires_grad
    assert not(y.requires_grad)
    assert isinstance(q, float)

    idx_under = (y <= pred_y)
    idx_over = ~idx_under
    cov = torch.mean(idx_under, dtype=float)
    cov_under = q > cov
    
    if cov_under:
#         loss = torch.sum(y[idx_over] - pred_y)
        loss = torch.mean(y[idx_over] - pred_y)
    else:
#         loss = torch.sum(pred_y - y[idx_under])
        loss = torch.mean(pred_y - y[idx_under])
        
    return loss


def cov_loss_5(y, pred_y, q):
    assert pred_y.requires_grad
    assert not(y.requires_grad)
    assert isinstance(q, float)

    idx_under = (y <= pred_y)
    idx_over = ~idx_under
    cov = torch.mean(idx_under, dtype=float)
    cov_under = q > cov
    
    if cov_under:
        loss = torch.mean(y - pred_y)
    else:
        loss = torch.mean(pred_y - y)
        
    return loss


def cov_loss_6(y, pred_y, q): # undifferentiable
    assert pred_y.requires_grad
    assert not(y.requires_grad)
    assert isinstance(q, float)

    idx_under = (y <= pred_y)
    idx_over = ~idx_under
    cov = torch.mean(idx_under, dtype=float)
    cov_under = q > cov
    
    if cov_under:
        loss = q - cov
    else:
        loss = cov - q
        
    return loss


def cov_loss_01(y, pred_y, q): 
    assert pred_y.requires_grad
    assert not(y.requires_grad)
    assert isinstance(q, float)

    idx_under = (y <= pred_y)
    idx_over = ~idx_under
    cov = torch.mean(idx_under, dtype=float)
    loss = (q - cov) * torch.mean(y - pred_y)
    
    return loss


def cov_loss_02(y, pred_y, q): 
    assert pred_y.requires_grad
    assert not(y.requires_grad)
    assert isinstance(q, float)

    idx_under = (y <= pred_y)
    idx_over = ~idx_under
    cov = torch.mean(idx_under, dtype=float)
    cov_under = q > cov
    
    if cov_under:
        loss = (q - cov) * torch.mean(y[idx_over] - pred_y)
    else:
        loss = (cov - q) * torch.mean(pred_y - y[idx_under])
        
    return loss


def cov_loss_02_2(y, pred_y, q): 
    assert pred_y.requires_grad
    assert not(y.requires_grad)
    assert isinstance(q, float)

    idx_under = (y <= pred_y)
    idx_over = ~idx_under
    cov = torch.mean(idx_under, dtype=float)
    cov_under = q > cov
    
    if cov_under:
        loss = (q - cov) * torch.sum(y[idx_over] - pred_y)
    else:
        loss = (cov - q) * torch.sum(pred_y - y[idx_under])
        
    return loss


def cov_loss_03(y, pred_y, q): 
    assert pred_y.requires_grad
    assert not(y.requires_grad)
    assert isinstance(q, float)

    idx_under = (y <= pred_y)
    idx_over = ~idx_under
    cov = torch.mean(idx_under, dtype=float)
    cov_under = q > cov
    
    if cov_under:
        loss = (q - cov) * (torch.sum(y[idx_over] - pred_y) + torch.sum(pred_y - y[idx_under]))
    else:
        loss = (cov - q) * (torch.sum(y[idx_over] - pred_y) + torch.sum(pred_y - y[idx_under]))
        
    return loss


def cov_loss_04(y, pred_y, q): 
    assert pred_y.requires_grad
    assert not(y.requires_grad)
    assert isinstance(q, float)

    idx_under = (y <= pred_y)
    idx_over = ~idx_under
    cov = torch.mean(idx_under, dtype=float)
    
    loss = torch.mean(((q - cov) * (y - pred_y))**2)
        
    return loss

