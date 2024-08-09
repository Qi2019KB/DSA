# -*- coding: utf-8 -*-
from datasets.classification.cifar10 import get_cifar10 as CIFAR10
from datasets.classification.cifar100 import get_cifar100 as CIFAR100
from datasets.classification.animal10N import get_animal10N as ANIMAL10N
from datasets.classification.cifar10C import get_cifar10C as CIFAR10C
from datasets.pose.pose_dataset import PoseDataset
from datasets.pose.pose_dataset_sw import PoseDataset_SW

__all__ = ('CIFAR10', 'CIFAR100', 'ANIMAL10N', 'CIFAR10C', 'PoseDataset', 'PoseDataset_SW')
