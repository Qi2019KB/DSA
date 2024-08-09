# -*- coding: utf-8 -*-
import GLOB as glob
import torch
import torch.nn as nn
from thop import profile, clever_format
from models.classification.wideresnet import build_wideresnet as wideresnet
from models.classification.wideresnet_ms import build_wideresnet_ms as wideresnet_ms
from models.classification.wideresnet_mhe import build_wideresnet_mhe as wideresnet_mhe
from models.classification.wideresnet_acbe import build_wideresnet_acbe as wideresnet_acbe
from models.classification.resnext import build_resnext as resnext
from models.classification.resnext_ms import build_resnext_ms as resnext_ms
from models.classification.vgg import build_vgg as vgg
from models.classification.vgg_ms import build_vgg_ms as vgg_ms
from models.classification.vgg_mhe import build_vgg_mhe as vgg_mhe
from models.classification.vgg_acbe import build_vgg_acbe as vgg_acbe




def create_model(args):
    args = set_parameters(args)
    logger = glob.get_value('logger')
    model, info = None, ''
    # region 1.1 model initialize
    if args.arch == 'WideResNet':
        model = wideresnet(args.model_depth, args.model_width, 0, args.num_classes).to(args.device)
        info = 'WideResNet {}x{}'.format(args.model_depth, args.model_width)
    elif args.arch == 'WideResNet_MS':
        model = wideresnet_ms(args.model_depth, args.model_width, 0, args.num_classes, args.stream_num, args.noisy_factor, args.device).to(args.device)
        info = 'WideResNet_MS {} * {}x{}'.format(args.stream_num, args.model_depth, args.model_width)
    elif args.arch == 'WideResNet_ACBE':
        model = wideresnet_acbe(args.model_depth, args.model_width, 0, args.num_classes, args.stream_num, args.noisy_factor, args.device).to(args.device)
        info = 'WideResNet_ACBE {} * {}x{}'.format(args.stream_num, args.model_depth, args.model_width)
    elif args.arch == 'WideResNet_MHE':
        model = wideresnet_mhe(args.model_depth, args.model_width, 0, args.num_classes, args.stream_num, args.device).to(args.device)
        info = 'WideResNet_MHE {} * {}x{}'.format(args.stream_num, args.model_depth, args.model_width)
    elif args.arch == 'ResNeXt':
        model = resnext(args.model_cardinality, args.model_depth, args.model_width, args.num_classes).to(args.device)
        info = 'ResNeXt {}x{}'.format(args.model_depth+1, args.model_width)
    elif args.arch == 'ResNeXt_MS':
        model = resnext_ms(args.model_cardinality, args.model_depth, args.model_width, args.num_classes, args.stream_num, args.noisy_factor, args.device).to(args.device)
        info = 'ResNeXt_MS {} * {}x{}'.format(args.stream_num, args.model_depth+1, args.model_width)

    elif args.arch.__contains__('VGG'):
        if args.arch.__contains__('_MS'):
            model = vgg_ms(args.arch.split('_MS')[0], args.num_classes, args.stream_num, args.noisy_factor, args.device).to(args.device)
            info = '{} (MS={})'.format(args.arch, args.stream_num)
        elif args.arch.__contains__('_ACBE'):
            model = vgg_acbe(args.arch.split('_ACBE')[0], args.num_classes, args.stream_num, args.noisy_factor, args.device).to(args.device)
            info = '{} (MS={})'.format(args.arch, args.stream_num)
        elif args.arch.__contains__('_MHE'):
            model = vgg_mhe(args.arch.split('_MHE')[0], args.num_classes, args.stream_num, args.noisy_factor, args.device).to(args.device)
            info = '{} (MS={})'.format(args.arch, args.stream_num)
        else:
            model = vgg(args.arch, args.num_classes).to(args.device)
            info = args.arch
    # endregion

    # region 1.2 FLOPs calculate
    input_shape = (3, 32, 32)
    input_tensor = torch.randn(1, *input_shape).to(args.device)
    mac = calculate_mac(model, input=input_tensor)/1000000
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    logger.print('L1', '=> model: {} | params: {} | FLOPs: {} | MAC: {}'.format(info, params, flops, mac))
    # endregion
    return model


def set_parameters(args):
    if args.arch.split('_')[0] == 'ResNeXt':
        if args.dataset == 'CIFAR100':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
        else:
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4
    elif args.arch.split('_')[0] == 'WideResNet':
        if args.dataset == 'CIFAR100':
            args.model_depth = 28
            args.model_width = 8
        else:
            args.model_depth = 28
            args.model_width = 2
    return args


# 计算MAC
def calculate_mac(model, input):
    # 设置模型为eval模式
    model.eval()
    macs = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            # 计算卷积层的MAC
            macs += layer.weight.numel() * input.numel()
        elif isinstance(layer, nn.Linear):
            # 计算全连接层的MAC
            macs += layer.weight.numel() * input.numel()
    return macs