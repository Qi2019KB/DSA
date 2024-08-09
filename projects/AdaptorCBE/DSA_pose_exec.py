# -*- coding: utf-8 -*-
import GLOB as glob
from projects.AdaptorCBE.Pose.SSL.DSA_DualPose import main as DSA_DualPose

datasets = [["Mouse", 100, 30, 500], ["Mouse", 200, 60, 500], ["FLIC", 100, 50, 500], ["FLIC", 200, 100, 500], ["LSP", 200, 100, 500], ["LSP", 300, 200, 500]]


def DSA_SSL():
    epochs = 200
    for dIdx, dataParam in enumerate(datasets):
        DSA_DualPose('E{}_DualPose_DSA_{}'.format(epochs, glob.version), {'expand': False, 'lambda_fd': 1.0, 'lambda_mc': 1.0, 'epochs': epochs, 'dataset': dataParam[0], 'train_num': dataParam[1], 'num_labeled': dataParam[2], 'valid_num': dataParam[3]})


if __name__ == "__main__":
    DSA_SSL()
