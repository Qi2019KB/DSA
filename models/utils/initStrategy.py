# -*- coding: utf-8 -*-
import math
import torch.nn as nn


class InitializeStrategy:
    def __init__(self):
        pass
    @classmethod
    def parameters_initialize(cls, net, mode):
        for layer in net.modules():
            if mode == "constant":
                if isinstance(layer, nn.Conv2d):
                    nn.init.constant_(layer.weight, 0.)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=1e-3)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
            elif mode == "uniform":
                if isinstance(layer, nn.Conv2d):
                    n = layer.in_channels
                    for k in layer.kernel_size: n *= k
                    stdv = 1. / math.sqrt(n)
                    nn.init.uniform_(layer.weight, -stdv, stdv)
                    if layer.bias is not None:
                        nn.init.uniform_(layer.bias, -stdv, stdv)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.uniform_(layer.weight)
                    nn.init.uniform_(layer.bias)
                elif isinstance(layer, nn.Linear):
                    n = layer.in_channels
                    stdv = 1. / math.sqrt(n)
                    nn.init.uniform_(layer.weight, -stdv, stdv)
                    if layer.bias is not None:
                        nn.init.uniform_(layer.bias, -stdv, stdv)
            elif mode == "normal":
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.normal_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.normal_(layer.weight)
                    nn.init.normal_(layer.bias)
                elif isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.normal_(layer.bias)
            elif mode == "xavier":
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=1e-3)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
            elif mode == "kaiming_normal":
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, a=math.sqrt(5))
                    if layer.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(layer.bias, -bound, bound)
                elif isinstance(layer, nn.BatchNorm2d):
                    layer.reset_running_stats()
                    if layer.affine:
                        nn.init.ones_(layer.weight)
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, a=math.sqrt(5))
                    if layer.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(layer.bias, -bound, bound)
            else:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                    if layer.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(layer.bias, -bound, bound)
                elif isinstance(layer, nn.BatchNorm2d):
                    layer.reset_running_stats()
                    if layer.affine:
                        nn.init.ones_(layer.weight)
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                    if layer.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(layer.bias, -bound, bound)

    @classmethod
    def ms_fc_initialize(cls, net):
        for iIdx, item in enumerate(net.ms_fc):
            mIdx = iIdx % 5
            layer = item[0]
            if mIdx == 0:  # "constant"
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=1e-3)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
            elif mIdx == 1:  # "uniform":
                if isinstance(layer, nn.Linear):
                    n = layer.in_features
                    stdv = 1. / math.sqrt(n)
                    nn.init.uniform_(layer.weight, -stdv, stdv)
                    if layer.bias is not None:
                        nn.init.uniform_(layer.bias, -stdv, stdv)
            elif mIdx == 2:  # "xavier"
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=1e-3)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
            elif mIdx == 3:  # "kaiming_normal"
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, a=math.sqrt(5))
                    if layer.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(layer.bias, -bound, bound)
            elif mIdx == 4:  # "kaiming_uniform"
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                    if layer.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(layer.bias, -bound, bound)
            else:  # "normal"
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.normal_(layer.bias)

    @classmethod
    def ms_adapter_initialize(cls, net):
        for iIdx, adaptor in enumerate(net.ms.ma.adaptors):
            mIdx = iIdx % 5
            for layers in adaptor:
                layer = layers.conv
                if mIdx == 0:  # "constant"
                    if isinstance(layer, nn.Conv2d):
                        nn.init.normal_(layer.weight, std=1e-3)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
                elif mIdx == 1:  # "xavier"
                    if isinstance(layer, nn.Conv2d):
                        nn.init.normal_(layer.weight, std=1e-3)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
                elif mIdx == 2:  # "kaiming_normal"
                    if isinstance(layer, nn.Conv2d):
                        nn.init.kaiming_normal_(layer.weight, a=math.sqrt(5))
                        if layer.bias is not None:
                            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                            bound = 1 / math.sqrt(fan_in)
                            nn.init.uniform_(layer.bias, -bound, bound)
                elif mIdx == 3:  # "kaiming_uniform"
                    if isinstance(layer, nn.Conv2d):
                        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                        if layer.bias is not None:
                            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                            bound = 1 / math.sqrt(fan_in)
                            nn.init.uniform_(layer.bias, -bound, bound)
                else:  # "normal"
                    if isinstance(layer, nn.Conv2d):
                        nn.init.normal_(layer.weight)
                        if layer.bias is not None:
                            nn.init.normal_(layer.bias)



    @classmethod
    def hg_ms_adapter_initialize(cls, net):
        for msIdx, ms in enumerate(net.ms):
            for iIdx, adaptor in enumerate(ms.ma.adaptors):
                mIdx = iIdx % 5
                for layers in adaptor:
                    layer = layers.conv
                    if mIdx == 0:  # "constant"
                        if isinstance(layer, nn.Conv2d):
                            nn.init.normal_(layer.weight, std=1e-3)
                            if layer.bias is not None:
                                nn.init.constant_(layer.bias, 0)
                    elif mIdx == 1:  # "xavier"
                        if isinstance(layer, nn.Conv2d):
                            nn.init.normal_(layer.weight, std=1e-3)
                            if layer.bias is not None:
                                nn.init.constant_(layer.bias, 0)
                    elif mIdx == 2:  # "kaiming_normal"
                        if isinstance(layer, nn.Conv2d):
                            nn.init.kaiming_normal_(layer.weight, a=math.sqrt(5))
                            if layer.bias is not None:
                                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                                bound = 1 / math.sqrt(fan_in)
                                nn.init.uniform_(layer.bias, -bound, bound)
                    elif mIdx == 3:  # "kaiming_uniform"
                        if isinstance(layer, nn.Conv2d):
                            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                            if layer.bias is not None:
                                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                                bound = 1 / math.sqrt(fan_in)
                                nn.init.uniform_(layer.bias, -bound, bound)
                    else:  # "normal"
                        if isinstance(layer, nn.Conv2d):
                            nn.init.normal_(layer.weight)
                            if layer.bias is not None:
                                nn.init.normal_(layer.bias)

