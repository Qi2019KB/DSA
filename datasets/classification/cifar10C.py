# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import torch
import random
from torchvision import datasets
from torchvision import transforms
from datasets.classification.transform import Transform


def get_cifar10C(labeled_idxs, unlabeled_idxs, dataset_root, dataset_mean, dataset_std, noisy_std=1., mode='sw'):
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
    labeled_dataset = CIFAR10CSSL(dataset_root, labeled_idxs, train=True, transform=transform_labeled, noisy_std = noisy_std)
    unlabeled_dataset = CIFAR10CSSL(dataset_root, unlabeled_idxs, train=True, transform=Transform(dataset_mean, dataset_std, mode), noisy_std = noisy_std)
    test_dataset = datasets.CIFAR10(dataset_root, train=False, transform=transform_val, download=False)
    return labeled_dataset, unlabeled_dataset, test_dataset


class CIFAR10CSSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True, transform=None, target_transform=None, download=False, noisy_std=1.):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.noisy_std = noisy_std

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # region 模拟CIFAR-10N，向各样本添加随机噪声。
        if self.noisy_std > 0:
            # img = self.add_gaussian_noise(img, std=self.noisy_std).squeeze()
            # img = self._noisy_mean(img, nf=self.noisy_std)
            img = self._noisy_gaussian(img, std=self.noisy_std)
        # endregion
        return img, target

    def __len__(self):
        return len(self.targets)

    def _noisy_mean(self, input, prob=0.5, nf=0.2):
        if random.random() <= prob:
            mu = input.mean()
            input = random.uniform(1-nf, 1+nf) * (input - mu) + mu
            input.add_(random.uniform(-nf, nf)).clamp_(0, 1)
        return input

    def _noisy_gaussian(cls, input, prob=0.5, std=0.1):
        if random.random() <= prob:
            mean = 0
            input = input + torch.randn_like(input) * std + mean
            input.clamp_(0, 1)
        return input