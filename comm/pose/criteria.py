# -*- coding: utf-8 -*-
import torch
from torch import nn


class AvgCounter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 0. if self.count == 0 else self.sum / self.count


class JointsMSELoss(nn.Module):
    def __init__(self):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, outputs, targets, kps_weight=None):
        preds_hm = outputs.unsqueeze(1) if outputs.ndim == 4 else outputs
        bs, n, k, _, _ = preds_hm.shape
        sel_count = torch.where(kps_weight > 0, 1, 0).sum() if kps_weight is not None else torch.tensor(bs*k, device=outputs.device)
        preds_hm = preds_hm.reshape((bs, n, k, -1))
        targets_hm = targets.reshape((bs, k, -1))
        loss = 0.
        for nIdx in range(n):
            if kps_weight is not None:
                loss += self.criterion(preds_hm[:, nIdx], targets_hm).mean(-1).mul(kps_weight).sum()
            else:
                loss += self.criterion(preds_hm[:, nIdx], targets_hm).mean(-1).sum()
        kps_count = n*sel_count
        loss_avg = loss / kps_count if kps_count > 0 else loss
        return loss_avg, kps_count.item()


class JointsWeightedMSELoss(nn.Module):
    def __init__(self):
        super(JointsWeightedMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, outputs, targets, samples_weight):
        preds_hm = outputs.unsqueeze(1) if outputs.ndim == 4 else outputs
        bs, n, k, _, _ = preds_hm.shape
        sel_count = torch.where(samples_weight > 0, 1, 0).sum()
        preds_hm = preds_hm.reshape((bs, n, k, -1))
        targets_hm = targets.reshape((bs, k, -1))
        loss = 0.
        for nIdx in range(n):
            loss += self.criterion(preds_hm[:, nIdx], targets_hm).mean(-1).mul(samples_weight).sum()
        kps_count = n*sel_count
        loss_avg = loss / kps_count if kps_count > 0 else loss
        return loss_avg, kps_count.item()


class JointDistanceLoss(nn.Module):
    def __init__(self, stack_num=3):
        super(JointDistanceLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.stack_num = stack_num

    def forward(self, outputs, targets):
        bs, k = outputs.size(0), outputs.size(2)
        combined_loss = []
        for nIdx in range(self.stack_num):
            v1 = outputs[:, nIdx].reshape((bs, k, -1))
            v2 = targets.reshape((bs, k, -1))
            loss = self.criterion(v1, v2).mean(-1)
            combined_loss.append(loss)
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss.sum(), self.stack_num*k


class JointsAccuracy(nn.Module):
    def __init__(self):
        super(JointsAccuracy, self).__init__()

    @classmethod
    def pck(cls, preds, gts, pck_ref, pck_thr):
        bs, k, _ = preds.shape
        # 计算各点的相对距离
        dists, dists_ref = cls._calDists(cls, preds, gts, pck_ref)

        # 计算error
        errs, err_sum, err_num = torch.zeros(k + 1), 0, 0
        for kIdx in range(k):
            if errs[kIdx] >= 0:  # 忽略带有-1的值
                errs[kIdx] = dists[kIdx].sum() / len(dists[kIdx])
                err_sum += errs[kIdx]
                err_num += 1
        errs[-1] = err_sum / err_num

        # 根据thr计算accuracy
        accs, acc_sum, acc_num = torch.zeros(k + 1), 0, 0
        for kIdx in range(k):
            accs[kIdx] = cls._counting(cls, dists_ref[kIdx], pck_thr)
            if accs[kIdx] >= 0:  # 忽略带有-1的值
                acc_sum += accs[kIdx]
                acc_num += 1
        if acc_num != 0:
            accs[-1] = acc_sum / acc_num
        return errs, accs

    # 计算各点的相对距离
    def _calDists(self, preds, gts, pck_ref_idxs):
        # 计算参考距离（基于数据集的参考关键点对）
        bs, k, _ = preds.shape
        dists, dists_ref = torch.zeros(k, bs), torch.zeros(k, bs)
        for iIdx in range(bs):
            norm = torch.dist(gts[iIdx, pck_ref_idxs[0], 0:2], gts[iIdx, pck_ref_idxs[1], 0:2])
            for kIdx in range(k):
                if gts[iIdx, kIdx, 0] > 1 and gts[iIdx, kIdx, 1] > 1:
                    dists[kIdx, iIdx] = torch.dist(preds[iIdx, kIdx, 0:2], gts[iIdx, kIdx, 0:2])
                    dists_ref[kIdx, iIdx] = torch.dist(preds[iIdx, kIdx, 0:2], gts[iIdx, kIdx, 0:2]) / norm
                else:
                    dists[kIdx, iIdx] = -1
                    dists_ref[kIdx, iIdx] = -1
        return dists, dists_ref

    # 返回低于阈值的百分比
    def _counting(cls, dists, thr=0.5):
        dists_plus = dists[dists != -1]
        if len(dists_plus) > 0:
            return 1.0 * (dists_plus < thr).sum().item() / len(dists_plus)
        else:
            return -1