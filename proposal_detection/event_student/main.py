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
from post_processing import BMN_post_processing
from eval import evaluation_proposal

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "8,9,10,11,12,13,14,15"

# normal version
def train_BMN(data_loader, model, optimizer, epoch, bm_mask):
    model.train()
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):
        input_data = input_data.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()
        confidence_map, start, end, _ = model(input_data)
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())
        optimizer.zero_grad()
        loss[0].backward()
        optimizer.step()

        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()
    print(
        "BMN training loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
            epoch, epoch_tem_loss / (n_iter + 1),
            epoch_pemclr_loss / (n_iter + 1),
            epoch_pemreg_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1)))
    state = {'epoch': epoch + 1,
               'state_dict': model.state_dict(),
               'optimizer': optimizer.state_dict()}
    torch.save(state, opt['checkpoint_path'] + '/BMN_checkpoint.pth.tar')

def test_BMN(data_loader, model, epoch, bm_mask, best_loss):
    model.eval()
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):
        input_data = input_data.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()

        confidence_map, start, end, _ = model(input_data)
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())

        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()
    print(
        "BMN test loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
            epoch, epoch_tem_loss / (n_iter + 1),
            epoch_pemclr_loss / (n_iter + 1),
            epoch_pemreg_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1)))
    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(state, opt["checkpoint_path"] + "/BMN_event.pth.tar")
    return best_loss

def BMN_Train(opt):
    gpu_num = opt['gpu_num']
    best_loss = 1e10
    model = BMN(opt)
    model = torch.nn.DataParallel(model, device_ids=[0,1,3,4,5,6,7]).cuda()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt["training_lr"],
                           weight_decay=opt["weight_decay"], momentum=0.9)

    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=opt["batch_size"], shuffle=False,
                                              num_workers=8, pin_memory=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["step_gamma"])
    bm_mask = get_mask(opt["temporal_scale"])
    for epoch in range(opt["train_epochs"]):
        scheduler.step()
        train_BMN(train_loader, model, optimizer, epoch, bm_mask)
        best_loss = test_BMN(test_loader, model, epoch, bm_mask, best_loss)

def BMN_inference(opt):
    gpu_num = opt['gpu_num']
    # event_model = BMN(opt)
    # activity_model = BMN(opt)
    # action_model = BMN(opt)
    highlight_model = BMN(opt)
    # event_model = torch.nn.DataParallel(event_model, device_ids=[0,1,3,4,5,6,7]).cuda()
    # activity_model = torch.nn.DataParallel(activity_model, device_ids=[0,1,3,4,5,6,7]).cuda()
    # action_model = torch.nn.DataParallel(action_model, device_ids=[0,1,3,4,5,6,7]).cuda()
    highlight_model = torch.nn.DataParallel(highlight_model, device_ids=[0,1,3,4,5,6,7]).cuda()
    # event_checkpoint = torch.load(opt["checkpoint_path"] + "/BMN_event.pth.tar")
    # activity_checkpoint = torch.load(opt["checkpoint_path"] + "/BMN_activity.pth.tar")
    # action_checkpoint = torch.load(opt["checkpoint_path"] + "/BMN_action.pth.tar")
    highlight_checkpoint = torch.load(opt["checkpoint_path"] + "/BMN_highlight.pth.tar")

    # event_model.load_state_dict(event_checkpoint['state_dict'])
    # activity_model.load_state_dict(activity_checkpoint['state_dict'])
    # action_model.load_state_dict(action_checkpoint['state_dict'])
    highlight_model.load_state_dict(highlight_checkpoint['state_dict'])
    # event_model.eval()
    # activity_model.eval()
    # action_model.eval()
    highlight_model.eval()
    inference_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=1, shuffle=False,
                                              num_workers=32, pin_memory=True, drop_last=False)
    tscale = opt["temporal_scale"]
    with torch.no_grad():
        for idx, input_data in inference_loader:
            video_name = inference_loader.dataset.video_list[idx[0]]
            input_data = input_data.cuda()
            # event_confidence_map, event_start, event_end, _ = event_model(input_data)
            # activity_confidence_map, activity_start, activity_end, _ = activity_model(input_data)
            # action_confidence_map, action_start, action_end, _ = action_model(input_data)
            highlight_confidence_map, highlight_start, highlight_end, _ = highlight_model(input_data)

            # confidence_map = (activity_confidence_map + action_confidence_map + highlight_confidence_map) / 3
            # start = (activity_start + action_start + highlight_start) / 3
            # end = (activity_end + action_end + highlight_end) / 3
            start_scores = highlight_start[0].detach().cpu().numpy()
            end_scores = highlight_end[0].detach().cpu().numpy()
            clr_confidence = (highlight_confidence_map[0][1]).detach().cpu().numpy()
            reg_confidence = (highlight_confidence_map[0][0]).detach().cpu().numpy()

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
            new_df.to_csv("./output/highlight_results/" + video_name + ".csv", index=False)

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
                t_map, t_start, t_end = teacher_model(input_data)
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
        map, start, end = student_model.forward(input_data)
        loss = teacher_bmn_loss_func(map, start, end, t_maps, t_starts, t_ends, bm_mask.cuda())
        total_loss = loss[0]
        epoch_pem_reg_loss += loss[2].cpu().detach().numpy()
        epoch_pem_cls_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += total_loss.cpu().detach().numpy()
        # TODO:start/end score
        # a_map = torch.flip(a_map, [1])
        # e_map = torch.flip(e_map, [1])
        # a_start_score = torch.sum(torch.triu(a_map), dim=2)
        # e_start_score = torch.sum(torch.triu(e_map), dim=2)
        # a_end_score = torch.sum(torch.triu(a_map), dim=1)
        # e_end_score = torch.sum(torch.triu(e_map), dim=1)
        # a_end_score = torch.zeros(end.shape[0], end.shape[1]).cuda()
        # for bs in range(input_data.shape[0]):
        #     for idx in range(input_data.shape[2]):
        #         a_end_score[bs, idx] = torch.sum(torch.diag(a_map[bs], (input_data.shape[2]-idx-1)))
        # e_end_score = torch.zeros(end.shape[0], end.shape[1]).cuda()
        # for bs in range(input_data.shape[0]):
        #     for idx in range(input_data.shape[2]):
        #         e_end_score[bs, idx] = torch.sum(torch.diag(e_map[bs], (input_data.shape[2]-idx-1)))
        # score_map = torch.zeros(score.shape[0], score.shape[1], score.shape[1]).cuda()
        # for i in range(score.shape[1]):
        #     for j in range(i, score.shape[1]-2):
        #         score_map[:, i, j] = torch.sum(score[:, i:j+2], dim=1)/(j+2-i)
        #
        # tem_loss = 0.1*(tem_loss_func(start, end, a_start, a_end) + tem_loss_func(start, end, e_start, e_end))
        # score_loss = 10*(score_mse_loss(score_map, a_map, bm_mask.cuda()) + score_mse_loss(score_map, e_map, bm_mask.cuda()))
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    print(
        "BMN training loss(epoch %d): tem_loss: %.03f, pem_reg_loss: %.03f, pem_cls_loss: %.03f, total_loss: %.03f" % (
            epoch, epoch_tem_loss / (n_iter + 1),
            epoch_pem_reg_loss / (n_iter + 1),
            epoch_pem_cls_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1)))
    state = {'epoch': epoch+1,
               'state_dict': student_model.state_dict(),
               'optimizer': optimizer.state_dict()}
    torch.save(state, opt['checkpoint_path'] + '/student_checkpoint.pth.tar')

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
        torch.save(state, opt["checkpoint_path"] + "/student_event.pth.tar")
    return best_loss

def BMN_Train_Distillation(opt):
    gpu_num = opt['gpu_num']
    best_loss = 1e10
    # teacher model
    teacher_models = []
    action_teacher = BMN(opt)
    activity_teacher = BMN(opt)
    highlight_teacher = BMN(opt)
    action_model = torch.nn.DataParallel(action_teacher, device_ids=range(gpu_num)).cuda()
    activity_model = torch.nn.DataParallel(activity_teacher, device_ids=range(gpu_num)).cuda()
    highlight_model = torch.nn.DataParallel(highlight_teacher, device_ids=range(gpu_num)).cuda()
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
    student_model = torch.nn.DataParallel(event_student, device_ids=range(gpu_num)).cuda()
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
    gpu_num = opt['gpu_num']
    model = BMN(opt)
    model = torch.nn.DataParallel(model, device_ids=range(gpu_num)).cuda()
    checkpoint = torch.load(opt["checkpoint_path"] + "/ws_event.pth.tar")
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
            confidence_map, start, end, _ = model(input_data)
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
            new_df.to_csv("./output/ws_results/" + video_name + ".csv", index=False)

def main(opt):
    if opt["mode"] == "train":
        # BMN_Train(opt)
        BMN_Train_Distillation(opt)
    elif opt["mode"] == "inference":
        if not os.path.exists("output/student_results"):
            os.makedirs("output/student_results")
        if not os.path.exists("output/event_results"):
            os.makedirs("output/event_results")
        if not os.path.exists("output/action_results"):
            os.makedirs("output/action_results")
        if not os.path.exists("output/activity_results"):
            os.makedirs("output/activity_results")
        if not os.path.exists("output/highlight_results"):
            os.makedirs("output/highlight_results")
        if not os.path.exists("output/full_results"):
            os.makedirs("output/full_results")
        if not os.path.exists("output/ws_results"):
            os.makedirs("output/ws_results")
        # BMN_inference_Distillation(opt)
        # BMN_inference(opt)
        # print("Post processing start")
        # BMN_post_processing(opt)
        # print("Post processing finished")
        evaluation_proposal(opt)


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    opt_file = open(opt["checkpoint_path"] + "/opts.json", "w")
    json.dump(opt, opt_file)
    opt_file.close()

    # model = BMN(opt)
    # a = torch.randn(1, 400, 100)
    # b, c = model(a)
    # print(b.shape, c.shape)
    # print(b)
    # print(c)
    main(opt)
