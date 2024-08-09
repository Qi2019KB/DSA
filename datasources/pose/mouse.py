# -*- coding: utf-8 -*-
import GLOB as glob
import math
import numpy as np
from comm.base.comm import CommUtils as comm


class MouseData:
    def __init__(self):
        self.name = 'Mouse'
        self.mean = (0.4921, 0.4921, 0.4921)
        self.std = (0.1663, 0.1663, 0.1663)
        self.labels_path = "D:/00Data/pose/mouse/croppeds_bbox/labels_normal.json"
        self.image_path = "D:/00Data/pose/mouse/croppeds_bbox/images"
        self.image_type = "png"
        self.inp_res = 256
        self.out_res = 64
        self.kps_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.kps_num = len(self.kps_idxs)
        self.pck_ref = [1, 2]  # 左右眼
        self.pck_thr = [0.5, 0.3, 0.2, 0.1]

    def get_data(self, args):
        labels = self._data_load()
        labeled_idxs, unlabeled_idxs, valid_idxs = self._data_cache(labels, args)
        labels = self._data_organize(labels, labeled_idxs, valid_idxs)
        return labels, labeled_idxs, unlabeled_idxs, valid_idxs

    def _data_load(self):
        labels = []
        for annIdx, annObj in enumerate(comm.json_load(self.labels_path)):
            kps = []
            for kpIdx, kp in enumerate(annObj["kps"]):
                if kpIdx in self.kps_idxs:
                    kps.append([kp[0], kp[1], 1])
            ann_id = "im{}".format(str(1000000 + annIdx + 1)[3:])

            labels.append({
                "is_labeled": 1,
                "id": ann_id,
                "image_id": annObj["imageID"],
                "image_name": "{}.{}".format(annObj["imageID"], self.image_type),
                "image_path": "{}/{}".format(self.image_path, "{}.{}".format(annObj["imageID"], self.image_type)),
                "kps": kps,
                "kps_test": kps
            })
        return labels

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
