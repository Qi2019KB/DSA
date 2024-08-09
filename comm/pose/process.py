# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import torch
from comm.pose.udaap import UDAAP as udaap


class ProcessUtils:
    def __init__(self):
        pass

    # return: image (H*W*C)
    @classmethod
    def image_load(cls, pathname):
        return np.array(cv2.imread(pathname), dtype=np.float32)

    @classmethod
    def image_save(cls, img, pathname, compression=0):
        # 创建路径
        folderPath = os.path.split(pathname)[0]
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        cv2.imwrite(pathname, img, [cv2.IMWRITE_PNG_COMPRESSION, compression])  # 压缩值，0为高清，9为最大压缩（压缩时间长），默认3.

    # return: image (H*W*C)
    @classmethod
    def image_resize(cls, img, kps, inpRes):
        h, w, _ = img.shape
        scale = [inpRes / w, inpRes / h]
        img = cv2.resize(img, (inpRes, inpRes))
        kps = [[kp[0] * scale[0], kp[1] * scale[1], kp[2]] for kp in kps]
        return img, kps, scale

    @classmethod
    def image_fliplr(cls, img_np):
        if img_np.ndim == 3:
            img_np = np.transpose(np.fliplr(np.transpose(img_np, (0, 2, 1))), (0, 2, 1))
        elif img_np.ndim == 4:
            for i in range(img_np.shape[0]):
                img_np[i] = np.transpose(np.fliplr(np.transpose(img_np[i], (0, 2, 1))), (0, 2, 1))
        return img_np.astype(float)
        # return np.ascontiguousarray(img_np.astype(float))

    # return: image (H*W*C ==> C*H*W)
    @classmethod
    def image_np2tensor_hwc2chw(cls, img_np):
        img_np = np.transpose(img_np, (2, 0, 1))  # H*W*C ==> C*H*W
        img_tensor = torch.from_numpy(img_np.astype(np.float32))
        if img_tensor.max() > 1:
            img_tensor /= 255
        return img_tensor

    @classmethod
    def image_tensor2np(cls, img_tensor):
        if not torch.is_tensor(img_tensor): return None
        img_np = img_tensor.detach().cpu().numpy()
        if img_np.shape[0] == 1 or img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))  # C*H*W ==> H*W*C
            img_np = np.ascontiguousarray(img_np)
        return img_np

    # input: img (C*H*W)
    @classmethod
    def image_center(cls, img):
        _, h, w = img.shape
        return torch.tensor([int(w / 2), int(h / 2)]).float()

    @classmethod
    def image_color_norm(cls, img, means, stds, useStd=False):
        if img.size(0) == 1:  # 黑白图处理
            img = img.repeat(3, 1, 1)

        for t, m, s in zip(img, means, stds):  # 彩色图处理
            t.sub_(m)  # 去均值，未对方差进行处理。
            if useStd:
                t.div_(s)
        return img

    @classmethod
    def kps_center(cls, kps):
        c, n = [0, 0], 0
        for kp in kps:
            if kp[2] == 0: continue
            c[0] += kp[0]
            c[1] += kp[1]
            n += 1
        return torch.tensor([int(c[0] / n), int(c[1] / n)]).float()

    @classmethod
    def kps_fliplr(cls, kps, img_w):
        kps[:, 0] = img_w - kps[:, 0]
        return kps

    @classmethod
    def kps_from_heatmap(cls, kps_hm, center, scale, res):
        preds = udaap.final_preds(kps_hm, center, scale, res)
        scores = torch.from_numpy(np.max(kps_hm.detach().cpu().numpy(), axis=(2, 3)).astype(np.float32))
        return preds, scores

    @classmethod
    def draw_point(cls, img, coord, color=(0, 95, 191), radius=3, thickness=-1, text=None, textScale=1.0, textColor=(255, 255, 255)):
        img, x, y = img.astype(int), round(coord[0]), round(coord[1])
        if x > 1 and y > 1:
            cv2.circle(img, (x, y), color=color, radius=radius, thickness=thickness)
            if text is not None:
                cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, textScale, textColor, 2)
        return img

    @classmethod
    def draw_points(cls, img, coords, inp_res, color=(0, 95, 191), radius=3, thickness=-1):
        h, w, _ = img.shape
        if type(coords) == torch.Tensor: coords = coords.cpu().data.numpy().tolist()
        for item in coords:
            coord = [item[0] * w / inp_res, item[1] * h / inp_res]
            img = cls.draw_point(img, coord, color=color, radius=radius, thickness=thickness)
        return img

    @classmethod
    def heatmap_from_kps(cls, kps, img_shape, inp_res, out_res, kernel_size=3.0, sigma=1.0):
        _, h, w = img_shape  # C*H*W
        stride = inp_res / out_res
        size_h, size_w = int(h / stride), int(w / stride)  # 计算HeatMap尺寸
        kps_num = len(kps)
        sigma *= kernel_size
        # 将HeatMap大小设置网络最小分辨率
        heatmap = np.zeros((size_h, size_w, kps_num), dtype=np.float32)
        for kIdx in range(kps_num):
            # 检查高斯函数的任意部分是否在范围内
            kp_int = kps[kIdx].to(torch.int32)
            ul = [int(kp_int[0] - sigma), int(kp_int[1] - sigma)]
            br = [int(kp_int[0] + sigma + 1), int(kp_int[1] + sigma + 1)]
            vis = 0 if (br[0] >= w or br[1] >= h or ul[0] < 0 or ul[1] < 0) else 1
            kps[kIdx][2] *= vis

            # 将keypoints转化至指定分辨率下
            x = int(kps[kIdx][0]) * 1.0 / stride
            y = int(kps[kIdx][1]) * 1.0 / stride
            kernel = cls._heatmap_gaussian(cls, size_h, size_w, center=[x, y], sigma=sigma)
            # 边缘修正
            kernel[kernel > 1] = 1
            kernel[kernel < 0.01] = 0
            heatmap[:, :, kIdx] = kernel
        heatmap = torch.from_numpy(np.transpose(heatmap, (2, 0, 1)))
        return kps, heatmap.float()

    def _heatmap_gaussian(self, h, w, center, sigma=3.0):
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        D2 = (grid_x - center[0]) ** 2 + (grid_y - center[1]) ** 2
        return np.exp(-D2 / 2.0 / sigma / sigma)
