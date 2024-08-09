# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data
from comm.pose.process import ProcessUtils as proc
from datasets.pose.transform import CommTransform, AugTransform


class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, targets, item_idxs, means, stds, conf, is_aug=True):
        self.targets = targets
        self.item_idxs = item_idxs
        self.means = means
        self.stds = stds
        self.inp_res = conf.inp_res
        self.out_res = conf.out_res
        self.is_aug = is_aug
        self.pre = CommTransform(conf.inp_res, conf.out_res)
        if self.is_aug:
            self.transform = AugTransform(conf.inp_res, conf.out_res, conf.sf, conf.rf, conf.use_flip)

    def __getitem__(self, idx):
        target = self.targets[self.item_idxs[idx]]
        is_labeled = torch.tensor(target["is_labeled"] == 1)

        # data pre-processing
        img, kps, _, center, scale, angle = self.pre(proc.image_load(target["image_path"]), target["kps"])
        kps_test = torch.tensor(target["kps_test"])
        kps_weight = kps[:, 2].clone()
        is_flip, kps_warpmat, img_warpmat = torch.tensor(False), None, None
        # data enhancement
        if self.is_aug:
            img, kps, scale, angle, is_flip, kps_warpmat, img_warpmat = self.transform(img, kps, center, scale, angle)

        # data processing
        img = proc.image_color_norm(img, self.means, self.stds)
        kps, kps_hm = proc.heatmap_from_kps(kps, img.shape, self.inp_res, self.out_res)

        meta = {'image_id': target['image_id'], 'image_path': target['image_path'], 'is_labeled': is_labeled,
                'is_flip': is_flip, 'center': center, 'scale': scale, 'angle': angle, 'kps_weight': kps_weight,
                'kps': kps, 'kps_test': kps_test, 'target': target}
        if self.is_aug:
            meta['kps_warpmat'] = kps_warpmat
            meta['img_warpmat'] = img_warpmat
        return img, kps_hm, meta

    def __len__(self):
        return len(self.item_idxs)
