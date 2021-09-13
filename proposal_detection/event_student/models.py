# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# TCN
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=4))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=4))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class BMN_Distillation(nn.Module):
    def __init__(self, opt):
        super(BMN_Distillation, self).__init__()
        self.feat_dim=opt["feat_dim"]

        self.hidden_dim_1d = 256

        # Base Module
        self.x_1d_b = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True)
        )

        # Temporal Evaluation Module
        self.x_1d_s = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.x_1d_e = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.score = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        base_feature = self.x_1d_b(x)
        start = self.x_1d_s(base_feature)
        end = self.x_1d_e(base_feature)
        score = self.score(base_feature)
        return score.squeeze(1), start.squeeze(1), end.squeeze(1)

class BMN(nn.Module):
    def __init__(self, opt):
        super(BMN, self).__init__()
        self.tscale = opt["temporal_scale"]
        self.prop_boundary_ratio = opt["prop_boundary_ratio"]
        self.num_sample = opt["num_sample"]
        self.num_sample_perbin = opt["num_sample_perbin"]
        self.feat_dim=opt["feat_dim"]

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        self._get_interp1d_mask()

        # Base Module
        self.base = TemporalConvNet(500, [512, 512, 512, 512, 512, 512, 256])

        # Proposal Evaluation Module
        self.start = nn.Sequential(
            TemporalConvNet(256, [256, 256, 256, 256, 256, 256, 256]),
            nn.Conv1d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.end = nn.Sequential(
            TemporalConvNet(256, [256, 256, 256, 256, 256, 256, 256]),
            nn.Conv1d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(self.hidden_dim_1d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1),stride=(self.num_sample, 1, 1)),
            nn.ReLU(inplace=True)
        )
        self.x_2d_p = nn.Sequential(
            nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1),
            nn.Sigmoid()
        )
        self.p = TemporalConvNet(256, [256, 256, 256, 256, 256, 256, 256])
    def forward(self, x):
        base_feature = self.base(x)
        start = self.start(base_feature).squeeze(1)
        end = self.end(base_feature).squeeze(1)
        confidence_map = self.p(base_feature)
        confidence_map = self._boundary_matching_layer(confidence_map)
        confidence_map = self.x_3d_p(confidence_map).squeeze(2)
        confidence_map = self.x_2d_p(confidence_map)
        return confidence_map, start, end, base_feature

    def _boundary_matching_layer(self, x):
        input_size = x.size()
        out = torch.matmul(x, self.sample_mask).reshape(input_size[0],input_size[1],self.num_sample,self.tscale,self.tscale)
        return out

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for end_index in range(self.tscale):
            mask_mat_vector = []
            for start_index in range(self.tscale):
                if start_index <= end_index:
                    p_xmin = start_index
                    p_xmax = end_index + 1
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin, sample_xmax, self.tscale, self.num_sample,
                        self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        self.sample_mask = nn.Parameter(torch.Tensor(mask_mat).view(self.tscale, -1), requires_grad=False)
        # print(torch.sum(self.sample_mask[0]))
        # exit()

class BMN_Student(BMN):
    def __init__(self, opt):
        super(BMN_Student, self).__init__(opt)

        self.temporal_scale = opt['temporal_scale']
        self.base_dim = 256
        # Weight Module
        self.shared_weight = nn.Sequential(
            nn.Linear(self.temporal_scale*self.base_dim, self.temporal_scale*self.base_dim//4),
            nn.ReLU(True),
            # nn.Linear(self.temporal_scale*self.feat_dim//2, self.temporal_scale*self.feat_dim//4),
            # nn.ReLU(True),
            nn.MaxPool1d(kernel_size=self.temporal_scale, stride=self.temporal_scale),
            nn.Linear(self.base_dim//4, self.base_dim//8),
            nn.ReLU(True),
            nn.Linear(self.base_dim//8, 1)
            )

    def forward(self, *x):
        if len(x) == 1:
            base_feature = self.base(x[0]) # (bs, 256, 100)
            start = self.start(base_feature).squeeze(1) # (bs, 100)
            end = self.end(base_feature).squeeze(1) # (bs, 100)
            confidence_map = self.p(base_feature)
            confidence_map = self._boundary_matching_layer(confidence_map)
            confidence_map = self.x_3d_p(confidence_map).squeeze(2)
            confidence_map = self.x_2d_p(confidence_map) # (bs, 2, 100, 100)
            return confidence_map, start, end
        if len(x) == 2:
            base_feature = self.base(x[0])
            batch_size = base_feature.shape[0]
            student_feature = base_feature.view(batch_size, -1) # (bs, 256, 100)
            teacher_features = [teacher_feature.view(batch_size, -1) for teacher_feature in x[1]]
            cross_features = [self.shared_weight((teacher_feature*student_feature).unsqueeze(1)) for teacher_feature in teacher_features]
            weight = torch.softmax(torch.cat(cross_features, dim=2).squeeze(1), dim=1)
            start = self.start(base_feature).squeeze(1)
            end = self.end(base_feature).squeeze(1)
            confidence_map = self.p(base_feature)
            confidence_map = self._boundary_matching_layer(confidence_map)
            confidence_map = self.x_3d_p(confidence_map).squeeze(2)
            confidence_map = self.x_2d_p(confidence_map)
            return weight, confidence_map, start, end, base_feature

if __name__ == '__main__':
    import opts
    opt = opts.parse_opt()
    opt = vars(opt)
    model = BMN(opt)
    score = BMN_Student(opt)
    score = score.eval()
    s = torch.randn(2, 500, 100)
    t1 = torch.randn(2, 500, 100)
    t2 = torch.randn(2, 500, 100)
    t3 = torch.randn(2, 500, 100)
    t1_m, t1_s, t1_e, t1_b = model(t1)
    t2_m, t2_s, t2_e, t2_b = model(t2)
    t3_m, t3_s, t3_e, t3_b = model(t3)
    output, map, start, end = score.forward_train(s, [t1_b, t2_b, t3_b])
    map_, start_, end_ = score(s)
    t_s = [t1_s, t2_s, t3_s]
    t_s = torch.sum(torch.cat([t_s_.unsqueeze(2) for t_s_ in t_s], dim=2) * output.unsqueeze(1), dim=2)
    t_m = [t1_m, t2_m, t3_m]
    t_m = torch.sum(torch.cat([t_m_.unsqueeze(4) for t_m_ in t_m], dim=4) * output.unsqueeze(1).unsqueeze(2).unsqueeze(3), dim=4)
    print(output.shape, map.shape, start.shape, end.shape)

