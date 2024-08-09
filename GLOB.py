# -*- coding: utf-8 -*-
import os

# Global cache object
global glob_cache
glob_cache = {}


def get_value(key, default=None):
    try:
        return glob_cache[key]
    except KeyError:
        return default


def set_value(key, value):
    glob_cache[key] = value
    return value


root = os.path.abspath(os.path.dirname(__file__))
project = root.split('\\')[-1]

expr = 'E:/00Experiment/expr/DSA'
temp = 'E:/00Experiment/temp/DSA'
stat = 'E:/00Experiment/statistic/DSA'

version = 'DSA_v1.0_20240808.1'
