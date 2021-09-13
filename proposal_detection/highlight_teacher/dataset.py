# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
import torch.utils.data as data
import torch
from utils import ioa_with_anchors, iou_with_anchors
import h5py


def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data


class VideoDataSet(data.Dataset):
    def __init__(self, opt, subset="train"):
        self.temporal_scale = opt["temporal_scale"]  # 100
        self.temporal_gap = 1. / self.temporal_scale
        self.video_features = None
        self.subset = subset
        self.mode = opt["mode"]
        self.feature_path = opt["feature_path"]
        self.anno_path = opt["anno_path"]
        self._getDatasetDict()
        self.anchor_xmin = [self.temporal_gap * (i - 0.5) for i in range(self.temporal_scale)]
        self.anchor_xmax = [self.temporal_gap * (i + 0.5) for i in range(self.temporal_scale)]

    def _getDatasetDict(self):
        self.video_dict = {}
        video_database = load_json(self.anno_path + 'new_BH.json')
        video_list1 = load_json(self.anno_path + self.subset + '_list.json')
        video_list2 = list(video_database.keys())
        self.video_list = list(set(video_list1).intersection(set(video_list2)))
        for video_name in self.video_list:
            self.video_dict[video_name] = video_database[video_name]
        print("%s subset video numbers: %d" % (self.subset, len(self.video_list)))

    def __getitem__(self, index):
        if self.video_features == None:
            self.video_features = h5py.File(self.feature_path, 'r')
        video_data = self._load_file(index)
        if self.mode == "train":
            match_score_start, match_score_end, confidence_score = self._get_train_label(index, self.anchor_xmin,
                                                                                         self.anchor_xmax)
            return video_data, confidence_score, match_score_start, match_score_end
        else:
            return index, video_data

    def _load_file(self, index):
        video_name = self.video_list[index]
        video_data = self.video_features[video_name]['c3d_features']
        video_data = torch.Tensor(video_data)
        video_data = torch.transpose(video_data, 0, 1)
        video_data.float()
        return video_data

    def _get_train_label(self, index, anchor_xmin, anchor_xmax):
        video_name = self.video_list[index]
        video_info = self.video_dict[video_name]
        duration_second = video_info['duration_second']
        corrected_second = float(duration_second)  # there are some frames not used
        video_labels = video_info['annotations']  # the measurement is second, not frame
        # change the measurement from second to percentage
        gt_bbox = []
        gt_iou_map = []
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info['segment'][0] / corrected_second), 0)
            tmp_end = max(min(1, tmp_info['segment'][1] / corrected_second), 0)
            gt_bbox.append([tmp_start, tmp_end])
        # generate R_s and R_e
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 * self.temporal_gap  # np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
        gt_iou_map = np.zeros([self.temporal_scale, self.temporal_scale])
        for i in range(self.temporal_scale):
            for j in range(i, self.temporal_scale):
                gt_iou_map[i, j] = np.max(
                    iou_with_anchors(i * self.temporal_gap, (j + 1) * self.temporal_gap, gt_xmins, gt_xmaxs))
        gt_iou_map = torch.Tensor(gt_iou_map)
        # calculate the ioa for all timestamp
        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        return match_score_start, match_score_end, gt_iou_map

    def __len__(self):
        return len(self.video_list)


if __name__ == '__main__':
    import opts

    opt = opts.parse_opt()
    opt = vars(opt)
    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=True)
    # print(len(train_loader))
    for a, b, c, d in train_loader:
        # print('video_data', a)
        # print('confidence_score', b)
        # print('match_score_start', c)
        # print('match_score_end', d)
        print(a.shape, b.shape, c.shape, d.shape)
        for i in range(100):
            print(b[0][i])
        break
    # box_min = [0.1, 0.2, 0.3]
    # box_max = [0.3, 0.5, 0.9]
    # int_xmin = np.maximum(0.15, box_min)
    # int_xmax = np.minimum(0.45, box_max)
    # print(int_xmin, int_xmax)
    # inter_len = np.maximum(int_xmax - int_xmin, 0.)
    # print(inter_len)
    # union_len = 0.3 - inter_len + box_max - box_min
    # print(union_len)
    # jaccard = np.divide(inter_len, union_len)
    # print(jaccard)
    # print(np.max(jaccard))

