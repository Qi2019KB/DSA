# -*- coding: utf-8 -*-
from torchvision import transforms
from comm.classification.randaugment import RandAugmentMC


class Transform(object):
    def __init__(self, mean, std, mode='sw'):
        self.mode = mode
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        if self.mode == 'sw':
            weak = self.weak(x)
            strong = self.strong(x)
            return self.normalize(weak), self.normalize(strong)
        elif self.mode == 'mt':
            weak_ema = self.weak(x)
            weak = self.weak(x)
            return self.normalize(weak_ema), self.normalize(weak)


class Transform_Animal10N(object):
    def __init__(self, mean, std, mode='sw'):
        self.mode = mode
        self.weak = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        if self.mode == 'sw':
            weak = self.weak(x)
            strong = self.strong(x)
            return self.normalize(weak), self.normalize(strong)
        elif self.mode == 'mt':
            weak_ema = self.weak(x)
            weak = self.weak(x)
            return self.normalize(weak_ema), self.normalize(weak)
