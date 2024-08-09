# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data
from comm.pose.process import ProcessUtils as proc
from datasets.pose.transform import CommTransform, AugTransform


class PoseDataset_SW(torch.utils.data.Dataset):
    def __init__(self, targets, item_idxs, means, stds, conf):
        self.targets = targets
        self.item_idxs = item_idxs
        self.means = means
        self.stds = stds
        self.inp_res = conf.inp_res
        self.out_res = conf.out_res
        self.pre = CommTransform(conf.inp_res, conf.out_res)
        self.transform_weak = AugTransform(conf.inp_res, conf.out_res, conf.sf, conf.rf, False)
        self.transform_strong = AugTransform(conf.inp_res, conf.out_res, conf.sf_s, conf.rf_s, False)

    def __getitem__(self, idx):
        target = self.targets[self.item_idxs[idx]]
        is_labeled = torch.tensor(target["is_labeled"] == 1)

        # data pre-processing
        img, kps, _, center, scale, angle = self.pre(proc.image_load(target["image_path"]), target["kps"])
        kps_test = torch.tensor(target["kps_test"])
        img_w, kps_w, scale_w, angle_w, is_flip_w, kps_warpmat_w, img_warpmat_w = self.transform_weak(img, kps, center, scale, angle)
        img_s, kps_s, scale_s, angle_s, is_flip_s, kps_warpmat_s, img_warpmat_s = self.transform_strong(img, kps, center, scale, angle)
        img_w = proc.image_color_norm(img_w, self.means, self.stds)
        img_s = proc.image_color_norm(img_s, self.means, self.stds)
        kps_w, kps_hm_w = proc.heatmap_from_kps(kps_w, img_w.shape, self.inp_res, self.out_res)
        kps_s, kps_hm_s = proc.heatmap_from_kps(kps_s, img_s.shape, self.inp_res, self.out_res)

        meta = {'image_id': target['image_id'], 'image_path': target['image_path'], 'is_labeled': is_labeled,
                'center_w': center, 'scale_w': scale_w, 'angle_w': angle_w, 'is_flip_w': is_flip_w, 'kps_w': kps_w,
                'kps_warpmat_w': kps_warpmat_w, 'img_warpmat_w': img_warpmat_w,
                'center_s': center, 'scale_s': scale_s, 'angle_s': angle_s, 'is_flip_s': is_flip_s, 'kps_s': kps_s,
                'kps_warpmat_s': kps_warpmat_s, 'img_warpmat_s': img_warpmat_s,
                'target': target, 'kps_test': kps_test}
        return img_w, kps_hm_w, img_s, kps_hm_s, meta

    def __len__(self):
        return len(self.item_idxs)
