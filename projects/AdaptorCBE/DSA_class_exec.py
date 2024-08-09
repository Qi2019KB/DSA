# -*- coding: utf-8 -*-
import GLOB as glob

from projects.AdaptorCBE.classification.SSL.DSA_FreeMatch import main as ACBE_FreeMatch


def DSA_SSL_E200():
    epochs = 200
    total_steps = epochs*1024
    for dIdx, dataParam in enumerate([["CIFAR10", 40], ["CIFAR10", 250], ["CIFAR10", 4000], ["CIFAR100", 400], ["CIFAR100", 2500], ["CIFAR100", 10000]]):
        ACBE_FreeMatch('E{}_ACBE_FreeMatch_{}'.format(epochs, glob.version), {'dataset': dataParam[0], 'num_labeled': dataParam[1], 'total_steps': total_steps})


if __name__ == "__main__":
    DSA_SSL_E200()
