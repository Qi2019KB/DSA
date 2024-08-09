# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from datasets.classification.transform import Transform


def get_cifar10(labeled_idxs, unlabeled_idxs, dataset_root, dataset_mean, dataset_std, mode='sw'):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])
    labeled_dataset = CIFAR10SSL(dataset_root, labeled_idxs, train=True, transform=transform_labeled)
    unlabeled_dataset = CIFAR10SSL(dataset_root, unlabeled_idxs, train=True, transform=Transform(dataset_mean, dataset_std, mode))
    test_dataset = datasets.CIFAR10(dataset_root, train=False, transform=transform_val, download=False)
    return labeled_dataset, unlabeled_dataset, test_dataset


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # return index, img, target
        return img, target
