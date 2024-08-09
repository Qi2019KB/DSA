# -*- coding: utf-8 -*-
import torch
from comm.pose.process import ProcessUtils as proc
from comm.pose.augment import AugmentUtils as aug


class CommTransform(object):
    def __init__(self, inp_res, out_res):
        self.inp_res = inp_res
        self.out_res = out_res

    def __call__(self, img, kps):
        img, kps, _ = proc.image_resize(img, kps, self.inp_res)  # H*W*C
        img = proc.image_np2tensor_hwc2chw(img)  # ndarry (H*W*C) ==> tensor (C*H*W)
        kps, kps_hm = proc.heatmap_from_kps(torch.tensor(kps).float(), img.shape, self.inp_res, self.out_res)  # heatmap: torch.Size([k, 64, 64])
        center = proc.image_center(img)
        scale = torch.tensor(self.inp_res / 200.0)
        angle = torch.tensor(0.)
        return img, kps, kps_hm, center, scale, angle


class AugTransform(object):
    def __init__(self, inp_res, out_res, sf, rf, use_flip):
        self.inp_res = inp_res
        self.out_res = out_res
        self.use_flip = use_flip
        self.sf = sf
        self.rf = rf
        pass

    def __call__(self, img, kps, center, scale, angle):
        is_flip = self.use_flip
        if self.use_flip:
            img, kps, center, is_flip = aug.fliplr(img, kps, center, prob=0.5)
        img = aug.noisy_mean(img)
        img, kps, scale, angle = aug.affine(img, kps, center, scale, self.sf, angle, self.rf, [self.inp_res, self.inp_res])
        # The initial scale of the keypoint is 1. So, when cal the warpmat should use '1' as the scale.
        kps_warpmat = aug.affine_get_warpmat(-angle, 1, [self.inp_res, self.inp_res])
        # The initial scale of the image is 1.28. So, when cal the warpmat should use 'scale/1.28' as the scale.
        img_warpmat = aug.affine_get_warpmat(-angle, scale/1.28, [self.inp_res, self.inp_res])
        return img, kps, scale, angle, is_flip, kps_warpmat, img_warpmat