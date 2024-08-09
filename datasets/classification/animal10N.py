# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from datasets.classification.transform import Transform_Animal10N
from comm.base.comm import CommUtils as comm
from comm.pose.process import ProcessUtils as proc


def get_animal10N(labeled_idxs, unlabeled_idxs, dataset_root, dataset_mean, dataset_std, mode='sw'):
    train_jsonPathname = "D:/00Data/Classification/Animal-10N/Data/train.json"
    test_jsonPathname = "D:/00Data/Classification/Animal-10N/Data/test.json"

    transform_labeled = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])
    train_data = comm.json_load(train_jsonPathname)
    test_data = comm.json_load(test_jsonPathname)
    labeled_dataset = ANIMAL10NSSL(train_data, labeled_idxs, transform=transform_labeled)
    unlabeled_dataset = ANIMAL10NSSL(train_data, unlabeled_idxs, transform=Transform_Animal10N(dataset_mean, dataset_std, mode))
    test_dataset = ANIMAL10NSSL(test_data, [idx for idx in range(len(test_data['labels']))], transform=transform_val)
    return labeled_dataset, unlabeled_dataset, test_dataset


class ANIMAL10NSSL():
    def __init__(self, data, indexs, transform=None, target_transform=None):
        self.image_ids = [data['image_ids'][idx] for idx in indexs]
        self.image_paths = [data['image_paths'][idx] for idx in indexs]
        self.targets = np.array([data['labels'][idx] for idx in indexs])
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        id, image_path, target = self.image_ids[index], self.image_paths[index], self.targets[index]
        img = proc.image_load(image_path).astype(np.uint8)
        # img = np.transpose(img, (2, 0, 1))
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.targets)
