# -*- coding: utf-8 -*-
import GLOB as glob
import os
import copy
import math
import numpy as np
import scipy.io as io
from comm.base.comm import CommUtils as comm


class FLICData:
    def __init__(self):
        self.name = 'FLIC'
        self.mean = (0.25195965, 0.22432944, 0.20951675)
        self.std = (0.23108867, 0.22090606, 0.22124061)
        self.image_path = "D:/00Data/pose/FLIC/images"
        self.labels_path = "D:/00Data/pose/FLIC/examples.mat"
        self.image_type = "jpg"
        self.inp_res = 256
        self.out_res = 64
        # self.kps_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]； self.pck_ref = [3, 7]
        self.kps_idxs = [0, 2, 3, 5, 9, 10]
        self.kps_num = len(self.kps_idxs)
        self.pck_ref = [1, 2]  # 左肩、右肩
        self.pck_thr = [0.5, 0.3, 0.2, 0.1]

    def get_data(self, args):
        labels = copy.deepcopy(self._data_load())
        labeled_idxs, unlabeled_idxs, valid_idxs = self._data_cache(labels, args)
        labels = self._data_organize(labels, labeled_idxs, valid_idxs)
        return labels, labeled_idxs, unlabeled_idxs, valid_idxs

    # 数据加载
    def _data_load(self):
        matFile = io.loadmat(self.labels_path)['examples']
        labelArray, imgNameArray, kpsArray = [], matFile['filepath'][0], matFile['coords'][0]

        for imgIdx, imgName in enumerate(imgNameArray):
            imgName, kps = imgName[0], [[int(kp[0]), int(kp[1]), 1] for kp in kpsArray[imgIdx].T if math.isnan(kp[0]) == False]
            # 过滤掉关键点不全的图片
            kps_new = []
            for kpIdx, kp in enumerate(kps):
                if kpIdx in self.kps_idxs and kp[2] > 0: kps_new.append([kp[0], kp[1], 1])
            if len(kps_new) < self.kps_num: continue
            id = "im{}".format(str(1000000 + imgIdx + 1)[3:])
            imgID = os.path.splitext(os.path.split(imgName)[1])[0]
            # label组织
            labelArray.append({
                "is_labeled": 1,
                "id": id,
                "image_id": imgID,
                "image_name": imgName,
                "image_path": "{}/{}".format(self.image_path, imgName),
                "kps": kps_new,
                "kps_test": kps_new
            })
        # region 过滤掉多人的样本
        labelArray_new = []
        for cItem in labelArray:
            count = len([item for item in labelArray if item["image_id"] == cItem["image_id"]])
            if count == 1:
                labelArray_new.append(cItem)
        # endregion
        return labelArray_new

    def _data_split(self, labels, args):
        all_idx = np.array(range(len(labels)))
        np.random.shuffle(all_idx)
        valid_idx = np.random.choice(all_idx, args.valid_num, False)
        unlabeled_idx = np.array([item for item in all_idx if item not in valid_idx])[0:args.train_num]
        labeled_idx = np.random.choice(unlabeled_idx, args.num_labeled, False)

        if args.expand_labels:
            num_expand_x = math.ceil(args.batch_size * args.eval_step / args.num_labeled)
            labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
        return labeled_idx, unlabeled_idx, valid_idx

    def _data_organize(self, labels, labeled_idxs, valid_idxs):
        for idx, item in enumerate(labels):
            if idx not in labeled_idxs and idx not in valid_idxs:
                item["is_labeled"] = 0
                item["kps"] = [[0, 0, 0] for _ in range(self.kps_num)]
        return labels

    def _data_cache(self, targets, args):
        save_path = "{}/datasources/temp_data/{}_{}_{}_{}_{}_{}.json".format(
            glob.root, self.name, args.train_num, args.num_labeled, args.valid_num, args.batch_size, args.mu)
        if not comm.file_isfile(save_path):
            labeled_idxs, unlabeled_idxs, valid_idxs = self._data_split(targets, args)
            comm.json_save([labeled_idxs.tolist(), unlabeled_idxs.tolist(), valid_idxs.tolist()], save_path, isCover=True)
            return labeled_idxs, unlabeled_idxs, valid_idxs
        else:
            return comm.json_load(save_path)
