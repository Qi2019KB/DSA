# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from comm.pose.process import ProcessUtils as proc
from comm.pose.augment import AugmentUtils as aug


class BusinessUtils:
    def __init__(self):
        pass

    @classmethod
    def prediction_back(cls, logits, warpmat, is_flip, center, scale, args):
        logits_back = aug.affine_back(logits, warpmat, is_flip)
        logits_pred, logits_score = proc.kps_from_heatmap(logits_back.detach().cpu(), center, scale, [args.out_res, args.out_res])
        return logits_back, logits_pred, logits_score

    @classmethod
    def prediction_similarity(cls, ms_logits):
        criterion = nn.MSELoss(reduction='none')
        bs, k = ms_logits.size(1), ms_logits.size(2)
        v1 = ms_logits[0].reshape((bs, k, -1))
        v2 = ms_logits[1].reshape((bs, k, -1))
        loss = criterion(v1, v2).sum()/(bs*k)
        return loss.item(), bs*k

    @classmethod
    def corrcoef_features(cls, inp1, inp2, eta=1):
        # torch.Size([224, 12, 8, 8]) ==> tensor(0.0104, device='cuda:0', grad_fn=<MulBackward0>)
        bs, nstack, c, h, w = inp1.size()  # torch.Size([4, 3, 25, 64, 64]) ==> ;
        corr_val = 0.
        for nIdx in range(nstack):
            f1 = inp1[:, nIdx].contiguous().view(bs, c * h * w).view(-1)
            f2 = inp2[:, nIdx].contiguous().view(bs, c * h * w).view(-1)
            corr_val += torch.abs(torch.corrcoef(torch.stack([f1, f2], dim=0))[0, 1])
        return corr_val*eta, bs*nstack

    @classmethod
    def corrcoef_labeled(cls, logits, targets, args):
        bs, nstack, c, h, w = logits.size()
        corr_val, count = torch.tensor(0., device=logits.device).double(), 0
        for nIdx in range(nstack):
            f1 = logits[:, nIdx].contiguous().view(bs, c* h * w).view(-1)
            f2 = targets.contiguous().view(bs, c* h * w).view(-1)
            corr = torch.abs(torch.corrcoef(torch.stack([f1, f2], dim=0))[0, 1])
            if not torch.isnan(corr):
                corr_val += corr
                count += 1
        if count > 0: corr_val = corr_val/count
        return corr_val, count
