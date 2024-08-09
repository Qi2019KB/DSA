# -*- coding: utf-8 -*-
import GLOB as glob
import datetime
import random
import argparse
import numpy as np
import torch
from comm.base.log import Logger

import os
import cv2
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


class ProjectUtils:
    def __init__(self):
        pass

    @classmethod
    def feature_visualize(cls, feature, save_path, scale_down=0.25):
        c, h, w = feature.shape
        feature_vec = feature.view(c, -1).cpu().data.tolist()
        f, ax = plt.subplots(figsize=(h * w * scale_down, c * scale_down))
        ax = sns.heatmap(feature_vec, linewidths=0.1, linecolor="white", cmap="RdBu_r")  # annot=True, fmt=".2g"
        img = ax.get_figure()

        folderPath = os.path.split(save_path)[0]
        if not os.path.exists(folderPath): os.makedirs(folderPath)
        img.savefig(save_path, bbox_inches='tight')
        plt.close()

    @classmethod
    def feature_visualize2(cls, feature, save_path, scale_down=1):
        c, h, w = feature.shape
        feature_vec = feature[0].cpu().data.tolist()
        f, ax = plt.subplots(figsize=(h * scale_down, w * scale_down))
        ax = sns.heatmap(feature_vec, linewidths=0.1, linecolor="white", cmap="Blues")
        img = ax.get_figure()

        folderPath = os.path.split(save_path)[0]
        if not os.path.exists(folderPath): os.makedirs(folderPath)
        img.savefig(save_path, bbox_inches='tight')
        plt.close()

    @classmethod
    def distribution_visualize(cls, feature, save_path, name_classes=None, scale_down=1):
        h, w = feature.shape
        feature_vec = feature.cpu().data.tolist()
        f, ax = plt.subplots(figsize=(h * scale_down, w * scale_down))
        if name_classes is None:
            ax = sns.heatmap(feature_vec, vmax=1.0, vmin=0.0, linewidths=0.1, linecolor="white", cmap="Blues", annot=True, fmt=".3f")
        else:
            ax = sns.heatmap(feature_vec, vmax=1.0, vmin=0.0, xticklabels=name_classes, yticklabels=name_classes,
                             linewidths=0.1, linecolor="white", cmap="Blues", annot=True, fmt=".3f")
        img = ax.get_figure()
        # plt.title('Prediction Distribution (%)', fontsize=20)
        # plt.xlabel('Predicted categories', fontsize=20)
        # plt.ylabel('Original categories', fontsize=20)

        folderPath = os.path.split(save_path)[0]
        if not os.path.exists(folderPath): os.makedirs(folderPath)
        img.savefig(save_path, bbox_inches='tight')
        plt.close()

    @classmethod
    def project_args_setup(cls, args, params):
        dict_args = vars(args)
        if params is not None:
            for key in params.keys():
                if key in dict_args.keys():
                    dict_args[key] = params[key]
        for key in dict_args.keys():
            if dict_args[key] == 'True': dict_args[key] = True
            if dict_args[key] == 'False': dict_args[key] = False
        return argparse.Namespace(**dict_args)

    @classmethod
    def project_setting(cls, args, mark):
        # region 1. set random seed
        random_seed = args.seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        # endregion

        # region 2. set device
        args.device = torch.device('cuda', args.gpu_id)
        # endregion

        # region 3. set experiment
        args.experiment = '{}(D{}L{})_{}_{}'.format(args.dataset, args.train_num, args.num_labeled, mark, datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
        args.basePath = '{}/{}'.format(glob.expr, args.experiment)
        # endregion

        glob.set_value('logger', Logger(args.experiment, consoleLevel='L1'))
        return args

    @classmethod
    def data_interleave(cls, x, size):
        s = list(x.shape)
        return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    @classmethod
    def data_de_interleave(cls, x, size):
        s = list(x.shape)
        return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    @classmethod
    def data_de_interleave_group(cls, logits, batch_size, args):
        logits = cls.data_de_interleave(logits, 2*args.mu+1)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
        return logits_x, logits_u_w, logits_u_s

    @classmethod
    def data_de_interleave_group2(cls, logits, batch_size, size):
        logits = cls.data_de_interleave(logits, size)
        logits_x = logits[:batch_size]
        logits_u = logits[batch_size:]
        return logits_x, logits_u

    @classmethod
    def _random_set_seed(cls, random_seed):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

    @classmethod
    def _device_set_value(cls, args):
        args.device = torch.device('cuda', args.gpu_id)
        return args

