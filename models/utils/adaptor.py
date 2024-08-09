# -*- coding: utf-8 -*-
import torch.nn as nn
from models.pose.hourglass.base.layers import Conv2


class MultiAdaptor(object):
    def __init__(self, device, stream_num, feature_dim, adaptor_type='0'):
        self.unit = 10
        self.stream_num = stream_num
        self.adaptor_type = adaptor_type

        if self.adaptor_type == '0':  # Classification
            self.adaptors = nn.ModuleList([
                nn.Sequential(
                    Conv2(feature_dim, 512+self.unit*stIdx, 1, bn=False, relu_idx=stIdx % 5),
                    Conv2(512+self.unit*stIdx, 1024+self.unit*stIdx, 1, bn=False, relu_idx=stIdx % 5),
                    Conv2(1024+self.unit*stIdx, 1024+self.unit*stIdx, 1, bn=False, relu_idx=stIdx % 5),
                    Conv2(1024+self.unit*stIdx, 512+self.unit*stIdx, 1, bn=False, relu_idx=stIdx % 5),
                    Conv2(512+self.unit*stIdx, feature_dim, 1, bn=False, relu_idx=stIdx % 5)
                ) for stIdx in range(self.stream_num)]).to(device)
        elif self.adaptor_type == '1':  # Pose Estimation
            self.adaptors = nn.ModuleList([
                nn.Sequential(
                    Conv2(feature_dim, feature_dim+self.unit*stIdx, 1, bn=False, relu_idx=stIdx % 5),
                    Conv2(feature_dim+self.unit*stIdx, feature_dim, 1, bn=False, relu_idx=stIdx % 5)
                ) for stIdx in range(self.stream_num)]).to(device)

    def forward(self, idx, x):
        return self.adaptors[idx](x)
