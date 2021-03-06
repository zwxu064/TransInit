import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .train_util import upfeat, poolfeat

'''
Loss function
author:Fengting Yang 
Mar.1st 2019

We only use "compute_semantic_pos_loss" func. in our final version, best result achieved with weight = 3e-3
'''

def compute_semantic_pos_loss(mode, prob_in, labxy_feat, pos_weight=0.003,
                              kernel_size=16, valid_area=None):
    # this wrt the slic paper who used sqrt of (mse)

    # rgbxy1_feat: B*50+2*H*W
    # output : B*9*H*w
    # NOTE: this loss is only designed for one level structure

    # todo: currently we assume the downsize scale in x,y direction are always same
    S = kernel_size
    m = pos_weight
    prob = prob_in.clone()

    b, c, h, w = labxy_feat.shape
    pooled_labxy = poolfeat(labxy_feat, prob, kernel_size, kernel_size)
    reconstr_feat = upfeat(pooled_labxy, prob, kernel_size, kernel_size)

    loss_map = reconstr_feat[:,-2:,:,:] - labxy_feat[:,-2:,:,:]

    # self def cross entropy  -- the official one combined softmax
    logit = torch.log(reconstr_feat[:, :-2, :, :] + 1e-8)

    if valid_area is not None:
        valid_area = valid_area.bool()
        valid_area_exp = valid_area.unsqueeze(1).repeat(1, c - 2, 1, 1)

        if mode == 'mean':
            loss_sem = -torch.mean(logit[valid_area_exp] * labxy_feat[:, :-2, :, :][valid_area_exp])
            loss_pos = torch.mean((torch.norm(loss_map, p=2, dim=1)[valid_area])) * m / S
            loss_sem_denom = logit.new_ones(1)
            loss_pos_denom = logit.new_ones(1)
        else:
            loss_sem = logit[valid_area_exp] * labxy_feat[:, :-2, :, :][valid_area_exp]
            loss_sem_denom = logit.new_ones(1) * loss_sem.numel()
            loss_sem = -torch.sum(loss_sem)
            loss_pos = torch.norm(loss_map, p=2, dim=1)[valid_area]
            loss_pos_denom = logit.new_ones(1) * loss_pos.numel()
            loss_pos = torch.sum(loss_pos) * m / S
    else:
        loss_sem = -torch.sum(logit * labxy_feat[:, :-2, :, :]) / b
        loss_pos = torch.sum(torch.norm(loss_map, p=2, dim=1)) / b * m / S
        loss_sem_denom = logit.new_ones(1)  # this is useless
        loss_pos_denom = logit.new_ones(1)

    # empirically we find timing 0.005 tend to better performance
    # loss_sem_sum = 0.005 * loss_sem
    # loss_pos_sum = 0.005 * loss_pos
    # loss_sum = loss_sem + loss_pos
    results = {'sem_loss': loss_sem, 'sem_denom': loss_sem_denom,
               'pos_loss': loss_pos, 'pos_denom': loss_pos_denom}

    return results