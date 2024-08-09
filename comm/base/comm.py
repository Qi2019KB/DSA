# -*- coding: utf-8 -*-
import os
import json
import yaml
import struct
import shutil
import torch.utils.data


class CommUtils:
    def __init__(self):
        pass

    @classmethod
    def file_mkdir(cls, path):
        os.makedirs(path)

    @classmethod
    def file_isfile(cls, path):
        return os.path.isfile(path)

    @classmethod
    def file_isdir(cls, path):
        return os.path.isdir(path)

    @classmethod
    def json_load(cls, path):
        with open(path, 'rb') as load_f:
            jsonDict = json.load(load_f)
        return jsonDict

    @classmethod
    def yaml_load(cls, path):
        # 读取YAML文件
        with open(path, 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)
        return data

    @classmethod
    def json_save(cls, content, pathname, isCover=False):
        # 创建路径
        folderPath = os.path.split(pathname)[0]
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        if isCover and os.path.isfile(pathname): os.remove(pathname)
        with open(pathname, 'w') as fileObj:
            json.dump(content, fileObj)

    @classmethod
    def ckpt_save(cls, state, is_best, ckptPath='ckpts'):
        filename = 'checkpoint.pth.tar'
        best_filename = 'checkpoint_best.pth.tar'

        if not cls.file_isdir(ckptPath):
            cls.file_mkdir(ckptPath)

        filepath = os.path.join(ckptPath, filename)
        torch.save(state, '{}/{}'.format(ckptPath, filename))

        if is_best:
            shutil.copyfile(filepath, '{}/{}'.format(ckptPath, best_filename))

    @classmethod
    def path_getFileName(cls, path):
        filename = os.path.basename(path)
        return os.path.splitext(filename)[0]
