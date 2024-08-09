# -*- coding: utf-8 -*-
import random
import numpy as np
import cv2
import torch
import skimage
import torch.nn.functional as F
from comm.pose.process import ProcessUtils as proc
from comm.pose.udaap import UDAAP as udaap


class AugmentUtils:
    def __init__(self):
        pass

    @classmethod
    def affine(cls, img, kps, center, scale, sf, angle, rf, matrix_res):
        # scale过小会导致关键点聚在一起，无法复原。且尾巴尖点易跑偏，因此约束，最小值为1.0，即由1.28缩小到1.0 -- wjq
        scale = max(scale * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0], torch.tensor(1.0))
        angle = angle + torch.randn(1).mul_(rf).clamp(- rf, rf)[0] if random.random() <= 1.0 else 0.
        img = cls._affine_image(cls, img, center, scale, matrix_res, angle)
        kps = cls._affine_kps(cls, kps, center, scale, matrix_res, angle)
        return img, kps, scale, angle

    @classmethod
    def affine_back(cls, heatmap, warpmat, is_flip):
        heatmap_back = heatmap.clone()
        # 进行反向仿射变换
        affine_grid = F.affine_grid(warpmat, heatmap_back.size(), align_corners=True)
        heatmap_back = F.grid_sample(heatmap_back, affine_grid, align_corners=True)
        # 进行反向水平翻转
        heatmaps_f = []
        for hIdx in range(len(heatmap_back)):
            heatmap_f = cls.fliplr_back_tensor(heatmap_back[hIdx]) if is_flip[hIdx] else heatmap_back[hIdx]
            heatmaps_f.append(heatmap_f)
        return torch.stack(heatmaps_f, dim=0)

    @classmethod
    def affine_back_single(cls, heatmap, warpmat, is_flip):
        heatmap_back = heatmap.clone().unsqueeze(0)
        # 进行反向仿射变换
        affine_grid = F.affine_grid(warpmat.unsqueeze(0), heatmap_back.size(), align_corners=True)
        heatmap_back = F.grid_sample(heatmap_back, affine_grid, align_corners=True)
        # 进行反向水平翻转
        if is_flip: heatmap_back = cls.fliplr_back_tensor(heatmap_back)
        return heatmap_back

    @classmethod
    def affine_get_warpmat(cls, angle, scale, matrix_res):
        # 根据旋转和比例生成变换矩阵
        M = cv2.getRotationMatrix2D((int(matrix_res[0]/2), int(matrix_res[1]/2)), angle, scale)
        warpmat = cv2.invertAffineTransform(M)
        warpmat[:, 2] = 0
        return torch.Tensor(warpmat)

    @classmethod
    def fliplr(cls, img, kps, center, prob=0.5):
        isflip = False
        if random.random() <= prob:
            img = torch.from_numpy(proc.image_fliplr(img.numpy())).float()
            kps = proc.kps_fliplr(kps, img.size(2))
            center[0] = img.size(2) - center[0]
            isflip = True
        return img, kps, center, torch.tensor(isflip)

    @classmethod
    def fliplr_back_tensor(cls, flip_output):
        # 无论输入数据多少个维度，torch.fliplr()只对第二高的维度进行交换，即x=1的维度。
        if flip_output.ndim == 3:
            return torch.permute(torch.fliplr(torch.permute(flip_output, dims=(0, 2, 1))), dims=(0, 2, 1))
        elif flip_output.ndim == 4:
            return torch.permute(torch.fliplr(torch.permute(flip_output, dims=(0, 3, 2, 1))), dims=(0, 3, 2, 1))

    @classmethod
    def noisy_mean(cls, img, prob=0.5):
        if random.random() <= prob:
            mu = img.mean()
            img = random.uniform(0.8, 1.2) * (img - mu) + mu
            img.add_(random.uniform(-0.2, 0.2)).clamp_(0, 1)
        return img

    def _affine_image(self, img, center, scale, matrix_res, angle=0):
        image = udaap.im_to_numpy(img)  # CxHxW (3, 256, 256) ==> H*W*C (256, 256, 3)
        # Preprocessing for efficient cropping
        ht, wd = image.shape[0], image.shape[1]
        sf = scale * 200.0 / matrix_res[0]
        if sf < 2:
            sf = 1  # 小于2的，取整为1。
        else:
            new_size = int(np.math.floor(max(ht, wd) / sf))  # 取图像最长边的缩放后长度为最新尺寸，int(maxLength/sf)
            new_ht = int(np.math.floor(ht / sf))  # 计算缩放后的height
            new_wd = int(np.math.floor(wd / sf))  # 计算缩放后的width
            if new_size < 2:  # 图像过小（最长边小于2pixels的），设置为256*256*3的0矩阵（h*w*c）。
                return torch.zeros(matrix_res[0], matrix_res[1], image.shape[2]) \
                    if len(image.shape) > 2 else torch.zeros(matrix_res[0], matrix_res[1])
            else:
                # img = scipy.misc.imresize(img, [new_ht, new_wd])
                image = skimage.transform.resize(image, (new_ht, new_wd))  # 依据计算后的height、width，resize图像。
                center = center * 1.0 / sf  # 重新计算中心点
                scale = scale / sf  # 重新计算scale

        # 计算左上角(0, 0)点转换后的点坐标（Upper left point）
        ul = np.array(udaap.transform([0, 0], center, scale, matrix_res, invert=1))
        # 计算右下角(res, res)点转换后的点坐标（Bottom right point）
        br = np.array(udaap.transform(matrix_res, center, scale, matrix_res, invert=1))

        # 填充，当旋转时，适当数量的上下文被包括在内（Padding so that when rotated proper amount of context is included）
        pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)  # 与范式相关。稍后调研。
        if not angle == 0:  # 不旋转时不用padding，旋转时为保证kps不会出图像范围，则添加padding。
            ul -= pad
            br += pad

        new_shape = [br[1] - ul[1], br[0] - ul[0]]
        if len(image.shape) > 2:
            new_shape += [image.shape[2]]  # 添加RGB维度，生成height*3
        new_img = np.zeros(new_shape)

        # 要填充新数组的范围（Range to fill new array）
        new_x = max(0, -ul[0]), min(br[0], image.shape[1]) - ul[0]
        new_y = max(0, -ul[1]), min(br[1], image.shape[0]) - ul[1]
        # 从原始图像到样本的范围（Range to sample from original image）
        old_x = max(0, ul[0]), min(image.shape[1], br[0])
        old_y = max(0, ul[1]), min(image.shape[0], br[1])
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = image[old_y[0]:old_y[1], old_x[0]:old_x[1]]

        if not angle == 0:
            # Remove padding
            # new_img = scipy.misc.imrotate(new_img, rot)
            new_img = skimage.transform.rotate(new_img, angle)
            new_img = new_img[pad:-pad, pad:-pad]
        # new_img = udaap.im_to_torch(scipy.misc.imresize(new_img, res))
        new_img = udaap.im_to_torch(skimage.transform.resize(new_img, tuple(matrix_res)))
        return new_img

    def _affine_kps(self, kps, center, scale, matrix_res, angle=0):
        kps_affined = kps.clone()
        for kIdx, kp in enumerate(kps):
            if kps[kIdx, 1] > 0:  # 坐标中的y值大于0（既该点可见）
                kps_affined[kIdx, 0:2] = torch.from_numpy(udaap.transform(kps[kIdx, 0:2], center, scale, matrix_res, rot=angle))
        return kps_affined
