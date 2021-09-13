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

def teacher_bmn_loss_func(pred_bm, pred_start, pred_end, gt_iou_map, gt_start, gt_end, bm_mask):
    gt_bm_reg = gt_iou_map[:, 0].contiguous()
    gt_bm_cls = gt_iou_map[:, 1].contiguous()
    gt_bm_reg = gt_bm_reg * bm_mask
    gt_bm_cls = gt_bm_cls * bm_mask

    pred_bm_reg = pred_bm[:, 0].contiguous()
    pred_bm_cls = pred_bm[:, 1].contiguous()
    pred_bm_reg = pred_bm_reg * bm_mask
    pred_bm_cls = pred_bm_cls * bm_mask

    tem_loss = teacher_tem_loss_func(pred_start, pred_end, gt_start, gt_end)
    pem_reg_loss = teacher_pem_reg_loss_func(pred_bm_reg, gt_bm_reg, bm_mask)
    pem_cls_loss = teacher_pem_cls_loss_func(pred_bm_cls, gt_bm_cls, bm_mask)

    loss = tem_loss + pem_reg_loss + pem_cls_loss
    return loss, tem_loss, pem_reg_loss, pem_cls_loss

def teacher_tem_loss_func(pred_start, pred_end, gt_start, gt_end):
    '''
    pred_start: bs, 100
    pred_end: bs, 100
    gt_start: [teacher_num x (bs, 100)]
    gt_end: [teacher_num x (bs, 100)]
    weight: bs, teacher_num
    '''
    def bi_loss(pred_score, gt_label):
        # max_value = gt_label.max(dim=1, keepdim=True)[0]
        # min_value = gt_label.min(dim=1, keepdim=True)[0]
        # threshold_value = max_value - min_value
        # pmask = (gt_label > (threshold_value*0.5+min_value)).float()
        # num_entries = pmask.view(-1).shape[0]
        pred_score = pred_score.view(-1)
        gt_label = gt_label.view(-1)
        pmask = (gt_label>0.6).float()
        num_entries = gt_label.shape[0]
        num_positive = torch.sum(pmask)
        ratio = num_entries / num_positive
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        epsilon = 0.000001
        loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
        loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * (1.0 - pmask)
        loss = -1 * torch.mean(loss_pos + loss_neg) # (bs)
        return loss

    loss_start = bi_loss(pred_start, gt_start)
    if torch.isnan(loss_start):
        print('start tem loss is nan\n', gt_start)
        exit()
    loss_end = bi_loss(pred_end, gt_end)
    if torch.isnan(loss_end):
        print('end tem loss is nan\n', gt_end)
        exit()
    loss = loss_start + loss_end
    return loss

def teacher_pem_reg_loss_func(pred_score, gt_iou_map, mask):
    def reg_loss(pred_score, gt_iou_map, mask):
        # max_value = gt_iou_map.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        # min_value = (gt_iou_map+100-mask*100).min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        # threshold_value = max_value-min_value
        # u_hmask = (gt_iou_map > (threshold_value*0.7+min_value)).float()
        # u_mmask = ((gt_iou_map <= (threshold_value*0.7+min_value)) & (gt_iou_map > (threshold_value*0.3+min_value))).float()
        # u_lmask = ((gt_iou_map <= (threshold_value*0.3+min_value)) & (gt_iou_map >= min_value)).float()

        u_hmask = (gt_iou_map > 0.6).float()
        # u_mmask = ((gt_iou_map <= 0.7) & (gt_iou_map > 0.4)).float()
        u_lmask = ((gt_iou_map <= 0.6) & (gt_iou_map > 0.)).float()
        u_lmask = u_lmask * mask
        num_h = torch.sum(u_hmask)#,dim=(1,2),keepdim=True)
        # num_m = torch.sum(u_mmask)#,dim=(1,2),keepdim=True)
        num_l = torch.sum(u_lmask)#,dim=(1,2),keepdim=True)

        # r_m = num_h / num_m
        # u_smmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
        # u_smmask = u_mmask * u_smmask
        # u_smmask = (u_smmask > (1. - r_m)).float()


        r_l = num_h / num_l
        u_slmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
        u_slmask = u_lmask * u_slmask
        u_slmask = (u_slmask > (1. - r_l)).float()

        weights = u_hmask + u_slmask
        loss = F.mse_loss(pred_score * weights, gt_iou_map * weights)
        loss = 0.5 * torch.sum(loss * torch.ones(*weights.shape).cuda()) / torch.sum(weights)

        # loss = torch.sum((pred_score*weights - gt_iou_map*weights).pow(2), dim=(1,2), keepdim=True)/torch.mean(num_h+num_m+num_l) # (bs, 100, 100)
        # loss = 0.5 * torch.mean(loss) / torch.sum(weights * mask)
        return loss * 10
    loss = reg_loss(pred_score, gt_iou_map, mask)
    if torch.isnan(loss):
        print('pem reg loss is nan')
        exit()
    return loss

def teacher_pem_cls_loss_func(pred_score, gt_iou_map, mask):
    def cls_loss(pred_score, gt_iou_map, mask):
        # max_value = gt_iou_map.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        # min_value = (gt_iou_map + 100 - mask * 100).min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        # threshold_value = max_value - min_value
        # pmask = (gt_iou_map > (threshold_value*0.9+min_value)).float() # (bs, 100, 100)
        # nmask = (gt_iou_map <= (threshold_value*0.9+min_value)).float()
        pmask = (gt_iou_map>0.6).float()
        nmask = (gt_iou_map<=0.6).float()
        nmask = nmask * mask # (bs, 100, 100)

        num_positive = torch.sum(pmask)#, dim=(1, 2), keepdim=True) # (bs, 100, 100)
        num_entries = num_positive + torch.sum(nmask)#, dim=(1, 2), keepdim=True) # (bs, 100, 100)
        ratio = num_entries / num_positive
        coef_0 = 0.5 * ratio / (ratio - 1) * mask
        coef_1 = 0.5 * ratio * mask
        epsilon = 0.000001
        loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
        loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * nmask
        # loss = -1 * torch.sum((loss_pos + loss_neg)/(num_entries*mask))
        loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries

        return loss
    loss = cls_loss(pred_score, gt_iou_map, mask)
    if torch.isnan(loss):
        print('pem cls loss is nan')
        exit()
    return loss

def bmn_loss_func(pred_bm, pred_start, pred_end, gt_iou_map, gt_start, gt_end, bm_mask):
    gt_iou_map = gt_iou_map * bm_mask
    pred_bm = pred_bm * bm_mask

    pred_bm_reg = pred_bm[:, 0].contiguous()
    pred_bm_cls = pred_bm[:, 1].contiguous()

    pem_reg_loss = pem_reg_loss_func(pred_bm_reg, gt_iou_map, bm_mask)
    pem_cls_loss = pem_cls_loss_func(pred_bm_cls, gt_iou_map, bm_mask)
    tem_loss = tem_loss_func(pred_start, pred_end, gt_start, gt_end)

    loss = tem_loss + 10 * pem_reg_loss + pem_cls_loss
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

    return loss

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