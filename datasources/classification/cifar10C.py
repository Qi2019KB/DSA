# -*- coding: utf-8 -*-
import GLOB as glob
import os
import math
import numpy as np
from torchvision import datasets
from comm.base.comm import CommUtils as comm


class CIFAR10CData:
    def __init__(self):
        self.name = 'CIFAR10C'
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)
        self.root = 'D:/00Data/Classification/cifar10C/CIFAR-10-C/'
        self.num_classes = 10
        self.name_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def get_data(self, args):
        targets = np.load(os.path.join(self.root, 'labels.npy'))
        labeled_idxs, unlabeled_idxs = self._data_cache(self.name, targets, self.num_classes, args)
        return labeled_idxs, unlabeled_idxs

    def _data_cache(self, dataset, targets, num_classes, args):
        savePath = "{}/datasources/temp_data/{}_{}_{}_{}_{}.json".format(
            glob.root, dataset, args.train_num, args.num_labeled, args.batch_size, args.mu)
        if not comm.file_isfile(savePath):
            labeled_idxs, unlabeled_idxs = self._data_split(targets, num_classes, args)
            comm.json_save([labeled_idxs.tolist(), unlabeled_idxs.tolist()], savePath, isCover=True)
            return labeled_idxs, unlabeled_idxs
        else:
            return comm.json_load(savePath)

    def _data_split(self, labels, num_classes, args):
        label_per_class = args.num_labeled // num_classes
        labels = np.array(labels)
        labeled_idx = []
        # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
        unlabeled_idx = np.array(range(len(labels)))
        for i in range(num_classes):
            idx = np.where(labels == i)[0]
            idx = np.random.choice(idx, label_per_class, False)
            labeled_idx.extend(idx)
        labeled_idx = np.array(labeled_idx)
        assert len(labeled_idx) == args.num_labeled

        if args.expand_labels or args.num_labeled < args.batch_size:
            num_expand_x = math.ceil(args.batch_size * args.eval_step / args.num_labeled)
            labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
        np.random.shuffle(labeled_idx)
        return labeled_idx, unlabeled_idx
