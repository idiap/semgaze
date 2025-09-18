#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_info_nce_loss(emb_pred, emb_gt, io_gt, logit_scale):
    mask = io_gt.bool()
    emb_gt, emb_pred = emb_gt[mask], emb_pred[mask]
    emb_gt, labels = torch.unique(emb_gt, dim=0, return_inverse=True) # get unique labels
    logits = torch.matmul(emb_pred, emb_gt.t()) * logit_scale.exp()
    loss = F.cross_entropy(logits, labels)
    return loss


def compute_dist_loss(gp_pred, gp_gt, io_gt):
    dist_loss = (gp_pred - gp_gt).pow(2).sum(dim=1)
    dist_loss = torch.mul(dist_loss, io_gt)
    dist_loss = torch.sum(dist_loss) / torch.sum(io_gt)
    return dist_loss


def compute_heatmap_loss(hm_pred, hm_gt, io_gt, loss_fn="mse"):
    if loss_fn == "mse":
        heatmap_loss = F.mse_loss(hm_pred, hm_gt, reduce=False).mean([1, 2])
    elif loss_fn == "bce":
        heatmap_loss = F.binary_cross_entropy_with_logits(hm_pred, hm_gt, reduction="none").mean([1, 2])
    else:
        raise Exception("loss_fn should be either 'mse' or 'bce'.")
    heatmap_loss = torch.mul(heatmap_loss, io_gt)
    heatmap_loss = torch.sum(heatmap_loss) / torch.sum(io_gt)
    return heatmap_loss


def compute_angular_loss(gv_pred, gv_gt, io_gt):
    angular_loss = (1 - torch.einsum("ij,ij->i", gv_pred, gv_gt)) / 2
    angular_loss = torch.mul(angular_loss, io_gt)
    angular_loss = torch.sum(angular_loss) / torch.sum(io_gt)
    return angular_loss
