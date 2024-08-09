# -*- coding: utf-8 -*-
import GLOB as glob
import math
import numpy as np
from torchvision import datasets
from comm.base.comm import CommUtils as comm


class Animal10NData:
    def __init__(self):
        self.name = 'ANIMAL10N'
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.root = ''
        self.train_imgPath = "D:/00Data/Classification/Animal-10N/Data/training"
        self.train_yamlPathname = "D:/00Data/Classification/Animal-10N/Data/train.yaml"
        self.train_jsonPathname = "D:/00Data/Classification/Animal-10N/Data/train.json"
        self.test_imgPath = "D:/00Data/Classification/Animal-10N/Data/testing"
        self.test_yamlPathname = "D:/00Data/Classification/Animal-10N/Data/test.yaml"
        self.test_jsonPathname = "D:/00Data/Classification/Animal-10N/Data/test.json"
        self.image_type = "jpg"
        self.num_classes = 10
        self.name_classes = ['cat', 'lynx', 'wolf', 'coyote', 'cheetah', 'jaguer', 'chimpanzee', 'orangutan', 'hamster', 'guinea pig']

    def get_data(self, args):
        _, _, train_labels = self._save_yaml2json(True, self.train_yamlPathname, self.train_jsonPathname)
        _, _, _ = self._save_yaml2json(False, self.test_yamlPathname, self.test_jsonPathname)
        labeled_idxs, unlabeled_idxs = self._data_cache(self.name, train_labels, self.num_classes, args)
        return labeled_idxs, unlabeled_idxs

    def _save_yaml2json(self, train, yamlPathname, jsonPathname):
        if not comm.file_isfile(jsonPathname):
            image_ids, image_paths, labels = [], [], []
            data = comm.yaml_load(yamlPathname)
            for annObj in data['data']['samples']:
                image_id = comm.path_getFileName(annObj['image'])
                image_ids.append(image_id)
                image_paths.append('{}/{}.{}'.format(self.train_imgPath if train else self.test_imgPath, image_id, self.image_type))
                labels.append(annObj['label']-1)
                # imgMap = proc.image_load('{}/{}.{}'.format(self.train_imgPath if train else self.test_imgPath, image_id, self.image_type))
                # images.append(imgMap.tolist())
            comm.json_save({'image_ids': image_ids, 'image_paths': image_paths, 'labels': labels}, jsonPathname, isCover=True)
            return image_ids, image_paths, labels
        else:
            jsonDict = comm.json_load(jsonPathname)
            return jsonDict['image_ids'], jsonDict['image_paths'], jsonDict['labels']

    def _data_cache(self, dataset, targets, num_classes, args):
        savePath = "{}/datasources/temp_data/{}_{}_{}_{}_{}.json".format(
            glob.root, dataset, args.train_num, args.num_labeled, args.batch_size, args.mu)
        if not comm.file_isfile(savePath):
            labeled_idxs, unlabeled_idxs = self._data_split(targets, num_classes, args)
            comm.json_save([labeled_idxs.tolist(), unlabeled_idxs.tolist()], savePath, isCover=True)
            return labeled_idxs, unlabeled_idxs
        else:
            return comm.json_load(savePath)

    def _data_split(self, labels, num_classes, args):
        label_per_class = args.num_labeled // num_classes
        use_balanced_choice = args.num_labeled < args.train_num
        labels = np.array(labels)
        labeled_idx = []
        # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
        unlabeled_idx = np.array(range(len(labels)))
        for i in range(num_classes):
            idx = np.where(labels == i)[0]
            if use_balanced_choice: idx = np.random.choice(idx, label_per_class, False)
            labeled_idx.extend(idx)
        labeled_idx = np.array(labeled_idx)
        assert len(labeled_idx) == args.num_labeled

        if args.expand_labels or args.num_labeled < args.batch_size:
            num_expand_x = math.ceil(args.batch_size * args.eval_step / args.num_labeled)
            labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
        np.random.shuffle(labeled_idx)
        return labeled_idx, unlabeled_idx
