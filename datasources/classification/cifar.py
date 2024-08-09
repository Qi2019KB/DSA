# -*- coding: utf-8 -*-
import GLOB as glob
import math
import numpy as np
from torchvision import datasets
from comm.base.comm import CommUtils as comm


class CIFAR10Data:
    def __init__(self):
        self.name = 'CIFAR10'
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)
        self.root = 'D:/00Data/Classification/cifar10/data'
        self.num_classes = 10
        self.name_classes = None

    def get_data(self, args):
        base_dataset = datasets.CIFAR10(self.root, train=True, download=True)
        if self.name_classes is None: self.name_classes = base_dataset.classes
        labeled_idxs, unlabeled_idxs = self._data_cache(self.name, base_dataset.targets, self.num_classes, args)
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


class CIFAR100Data:
    def __init__(self):
        self.name = 'CIFAR100'
        self.mean = (0.5071, 0.4867, 0.4408)
        self.std = (0.2675, 0.2565, 0.2761)
        self.root = 'D:/00Data/Classification/cifar100/data'
        self.num_classes = 100
        self.name_classes = None

    # 获得全标签数据（用于监督学习）
    def get_data(self, args):
        base_dataset = datasets.CIFAR100(self.root, train=True, download=True)
        if self.name_classes is None: self.name_classes = base_dataset.classes
        labeled_idxs, unlabeled_idxs = self._data_cache(self.name, base_dataset.targets, self.num_classes, args)
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
