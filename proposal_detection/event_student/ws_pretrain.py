import sys
from dataset import *
from loss_function import *
import os
import json
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import opts
from models import *
import pandas as pd
from ws_processing import BMN_post_processing
from ws_eval import evaluation_proposal

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "8,9,10,11,12,13,14,15"

# distillation version
def train_BMN_distillation(data_loader, student_model, teacher_models, optimizer, epoch, bm_mask):
    student_model.train()
    teacher_models = [teacher_model.eval() for teacher_model in teacher_models]
    epoch_tem_loss = 0
    epoch_pem_reg_loss = 0
    epoch_pem_cls_loss = 0
    epoch_loss = 0
    epoch_base_loss = 0
    for n_iter, (input_data, _, _, _) in enumerate(data_loader):
        input_data = input_data.cuda()
        with torch.no_grad():
            t_maps = torch.zeros(input_data.shape[0], 2, 100, 100).cuda()
            t_starts = torch.zeros(input_data.shape[0], 100).cuda()
            t_ends = torch.zeros(input_data.shape[0], 100).cuda()
            for teacher_model in teacher_models:
                t_map, t_start, t_end, _ = teacher_model(input_data)
                map_max_reg = t_map[:, 0].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
                map_min_reg = t_map[:, 0].min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
                threshold_reg = map_max_reg - map_min_reg
                t_maps[:, 0] += (t_map[:, 0] > (0.5 * threshold_reg + map_min_reg)).float()
                map_max_cls = t_map[:, 1].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
                map_min_cls = t_map[:, 1].min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
                threshold_cls = map_max_cls - map_min_cls
                t_maps[:, 1] += (t_map[:, 1] > (0.5 * threshold_cls + map_min_cls)).float()
                start_max = t_start.max(dim=1, keepdim=True)[0]
                start_min = t_start.min(dim=1, keepdim=True)[0]
                threshold_start = start_max - start_min
                t_starts += (t_start > (0.8 * threshold_start + start_min)).float()
                end_max = t_end.max(dim=1, keepdim=True)[0]
                end_min = t_end.min(dim=1, keepdim=True)[0]
                threshold_end = end_max - end_min
                t_ends += (t_end > (0.8 * threshold_end + end_min)).float()
            t_maps /= 3
            t_starts /= 3
            t_ends /= 3
        confidence_map, start, end = student_model.forward(input_data)
        loss = teacher_bmn_loss_func(confidence_map, start, end, t_maps, t_starts, t_ends, bm_mask.cuda())
        total_loss = loss[0]
        epoch_pem_reg_loss += loss[2].cpu().detach().numpy()
        epoch_pem_cls_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += total_loss.cpu().detach().numpy()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if (n_iter+1)%10 == 0:
            print(
                "BMN training loss(epoch %d, iter %d): tem_loss: %.03f, pem_reg_loss: %.03f, pem_cls_loss: %.03f, total_loss: %.03f" % (
                    epoch, n_iter+1, epoch_tem_loss / (n_iter + 1),
                    epoch_pem_reg_loss / (n_iter + 1),
                    epoch_pem_cls_loss / (n_iter + 1),
                    epoch_loss / (n_iter + 1)))
    print(
        "BMN training loss(epoch %d): tem_loss: %.03f, pem_reg_loss: %.03f, pem_cls_loss: %.03f, total_loss: %.03f" % (
            epoch, epoch_tem_loss / (n_iter + 1),
            epoch_pem_reg_loss / (n_iter + 1),
            epoch_pem_cls_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1)))
    state = {'epoch': epoch+1,
               'state_dict': student_model.state_dict(),
               'optimizer': optimizer.state_dict()}
    torch.save(state, opt['checkpoint_path'] + '/ws_pretrain_checkpoint.pth.tar')

def test_BMN_distillation(data_loader, student_model, epoch, bm_mask, best_loss):
    student_model.eval()
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):
        input_data = input_data.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()
        confidence_map, start, end = student_model.forward(input_data)
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())
        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()
    print(
        "BMN test loss(epoch %d): tem_loss: %.03f, pem reg_loss: %.03f, pem class_loss: %.03f, total_loss: %.03f" % (
            epoch, epoch_tem_loss / (n_iter + 1),
            epoch_pemreg_loss / (n_iter + 1),
            epoch_pemclr_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1)))

    state = {'epoch': epoch + 1,
             'state_dict': student_model.state_dict()}
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(state, opt["checkpoint_path"] + "/ws_pretrain_event.pth.tar")
    return best_loss

def BMN_Train_Distillation(opt):
    best_loss = 1e10
    # teacher model
    teacher_models = []
    action_teacher = BMN(opt)
    activity_teacher = BMN(opt)
    highlight_teacher = BMN(opt)
    action_model = torch.nn.DataParallel(action_teacher, device_ids=[0,1,3,4,5,6,7,8]).cuda()
    activity_model = torch.nn.DataParallel(activity_teacher, device_ids=[0,1,3,4,5,6,7,8]).cuda()
    highlight_model = torch.nn.DataParallel(highlight_teacher, device_ids=[0,1,3,4,5,6,7,8]).cuda()
    action_checkpoint = torch.load(opt["checkpoint_path"] + "/BMN_action.pth.tar")
    activity_checkpoint = torch.load(opt["checkpoint_path"] + "/BMN_activity.pth.tar")
    highlight_checkpoint = torch.load(opt["checkpoint_path"] + "/BMN_highlight.pth.tar")
    action_model.load_state_dict(action_checkpoint['state_dict'])
    activity_model.load_state_dict(activity_checkpoint['state_dict'])
    highlight_model.load_state_dict(highlight_checkpoint['state_dict'])
    teacher_models.append(action_model)
    teacher_models.append(activity_model)
    teacher_models.append(highlight_model)

    # student model
    event_student = BMN_Student(opt)
    student_model = torch.nn.DataParallel(event_student, device_ids=[0,1,3,4,5,6,7,8]).cuda()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, student_model.parameters()), lr=opt["training_lr"], weight_decay=opt["weight_decay"], momentum=0.9)
    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset='train'),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset='validation'),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["step_gamma"])
    bm_mask = get_mask(opt["temporal_scale"])
    for epoch in range(opt["train_epochs"]):
        scheduler.step()
        train_BMN_distillation(train_loader, student_model, teacher_models, optimizer, epoch, bm_mask)
        best_loss = test_BMN_distillation(test_loader, student_model, epoch, bm_mask, best_loss)

def BMN_inference_Distillation(opt):
    model = BMN_Student(opt)
    model = torch.nn.DataParallel(model, device_ids=[0,1,3,4,5,6,7,8]).cuda()
    checkpoint = torch.load(opt["checkpoint_path"] + "/ws_pretrain_event.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    inference_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset='validation'),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False)
    tscale = opt["temporal_scale"]
    with torch.no_grad():
        for idx, input_data in inference_loader:
            video_name = inference_loader.dataset.video_list[idx[0]]
            input_data = input_data.cuda()
            confidence_map, start, end = model(input_data)
            start_scores = start[0].detach().cpu().numpy()
            end_scores = end[0].detach().cpu().numpy()
            clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
            reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()
            # 遍历起始分界点与结束分界点的组合
            new_props = []
            for idx in range(tscale):
                for jdx in range(tscale):
                    start_index = idx
                    end_index = jdx + 1
                    if start_index < end_index and end_index < tscale:
                        xmin = start_index / tscale
                        xmax = end_index / tscale
                        xmin_score = start_scores[start_index]
                        xmax_score = end_scores[end_index]
                        clr_score = clr_confidence[idx, jdx]
                        reg_score = reg_confidence[idx, jdx]
                        score = xmin_score * xmax_score * clr_score * reg_score
                        new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score])
            new_props = np.stack(new_props)
            col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_socre", "score"]
            new_df = pd.DataFrame(new_props, columns=col_name)
            new_df.to_csv("./output/ws_pretrain_results/" + video_name + ".csv", index=False)

def main(opt):
    if opt["mode"] == "train":
        BMN_Train_Distillation(opt)
    elif opt["mode"] == "inference":
        if not os.path.exists("output/ws_pretrain_results"):
            os.makedirs("output/ws_pretrain_results")
        BMN_inference_Distillation(opt)
        print("Post processing start")
        BMN_post_processing(opt)
        print("Post processing finished")
        evaluation_proposal(opt)

if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    opt_file = open(opt["checkpoint_path"] + "/opts.json", "w")
    json.dump(opt, opt_file)
    opt_file.close()
    main(opt)
