# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F


def get_mask(tscale):
    mask = np.zeros([tscale, tscale], np.float32)
    for i in range(tscale):
        for j in range(i, tscale):
            mask[i, j] = 1
    return torch.Tensor(mask)


def bmn_loss_func(pred_bm, pred_start, pred_end, gt_iou_map, gt_start, gt_end, bm_mask):
    gt_iou_map = gt_iou_map * bm_mask
    pred_bm = pred_bm*bm_mask

    pred_bm_reg = pred_bm[:, 0].contiguous()
    pred_bm_cls = pred_bm[:, 1].contiguous()

    pem_reg_loss = pem_reg_loss_func(pred_bm_reg, gt_iou_map, bm_mask)
    pem_cls_loss = pem_cls_loss_func(pred_bm_cls, gt_iou_map, bm_mask)
    tem_loss = tem_loss_func(pred_start, pred_end, gt_start, gt_end)

    loss = tem_loss + pem_reg_loss + pem_cls_loss
    return loss, tem_loss, pem_reg_loss, pem_cls_loss


def tem_loss_func(pred_start, pred_end, gt_start, gt_end):
    def bi_loss(pred_score, gt_label):
        max_value = gt_label.max(dim=1, keepdim=True)[0]
        min_value = gt_label.min(dim=1, keepdim=True)[0]
        threshold_value = max_value - min_value
        pmask = (gt_label > (0.5*threshold_value+min_value)).float()
        pred_score = pred_score.view(-1)
        pmask = pmask.view(-1)
        num_entries = len(pmask)
        num_positive = torch.sum(pmask)
        ratio = num_entries / num_positive
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        epsilon = 0.000001
        loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
        loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * (1.0 - pmask)
        loss = -1 * torch.mean(loss_pos + loss_neg)
        return loss

    loss_start = bi_loss(pred_start, gt_start)
    loss_end = bi_loss(pred_end, gt_end)
    loss = loss_start + loss_end
    return loss


def pem_reg_loss_func(pred_score, gt_iou_map, mask):
    max_value = gt_iou_map.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
    min_value = (gt_iou_map+100-mask*100).min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
    threshold_value = max_value-min_value
    u_hmask = (gt_iou_map > (0.7*threshold_value+min_value)).float()
    u_mmask = ((gt_iou_map <= (0.7*threshold_value+min_value)) & (gt_iou_map > (0.3*threshold_value+min_value))).float()
    u_lmask = ((gt_iou_map <= (0.3*threshold_value+min_value)) & (gt_iou_map > min_value)).float()
    u_lmask = u_lmask * mask

    num_h = torch.sum(u_hmask)
    num_m = torch.sum(u_mmask)
    num_l = torch.sum(u_lmask)

    r_m = num_h / num_m
    u_smmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
    u_smmask = u_mmask * u_smmask
    u_smmask = (u_smmask > (1. - r_m)).float()

    r_l = num_h / num_l
    u_slmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
    u_slmask = u_lmask * u_slmask
    u_slmask = (u_slmask > (1. - r_l)).float()

    weights = u_hmask + u_smmask + u_slmask

    loss = F.mse_loss(pred_score * weights, gt_iou_map * weights)
    loss = 0.5 * torch.sum(loss * torch.ones(*weights.shape).cuda()) / torch.sum(weights)

    return loss * 10


def pem_cls_loss_func(pred_score, gt_iou_map, mask):
    max_value = gt_iou_map.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
    min_value = (gt_iou_map+100-mask*100).min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
    threshold_value = max_value-min_value
    pmask = (gt_iou_map > (0.9*threshold_value+min_value)).float()
    nmask = (gt_iou_map <= (0.9*threshold_value+min_value)).float()
    nmask = nmask * mask

    num_positive = torch.sum(pmask)
    num_entries = num_positive + torch.sum(nmask)
    ratio = num_entries / num_positive
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    epsilon = 0.000001
    loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
    loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * nmask
    loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries
    return loss
