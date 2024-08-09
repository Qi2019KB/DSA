# -*- coding: utf-8 -*-
import torch


class BusinessUtils:
    def __init__(self):
        pass

    @classmethod
    def target_ensemble_mask(cls, ms_logits, args, ms_masks=None):
        flag = ms_masks is None
        if flag:
            ms_masks, ms_max_idx = [], []
            for stIdx in range(args.stream_num):
                logits = ms_logits[stIdx].detach()
                probs_logits = torch.softmax(logits, dim=-1)
                score_logits, max_idx = torch.max(probs_logits, dim=-1)
                ms_masks.append(score_logits.ge(args.score_thr).to(score_logits.dtype))
                ms_max_idx.append(max_idx)
            ms_masks = torch.stack(ms_masks, dim=0)
            ms_max_idx = torch.stack(ms_max_idx, dim=0)
        ms_masks_re = torch.repeat_interleave(ms_masks.unsqueeze(-1), args.num_classes, dim=2)
        targets_softmax = torch.softmax(ms_logits, dim=-1)
        targets_sum = torch.sum(targets_softmax, dim=0)
        mask_sum = torch.sum(ms_masks_re, dim=0)
        targets_mean = torch.where(mask_sum > 0, targets_sum.div(mask_sum), torch.zeros(mask_sum.shape).to(args.device))
        ens_score, ens_preds = torch.max(targets_mean, dim=-1)
        ens_mask = torch.sum(ms_masks, dim=0).ge(args.count_thr).float()
        if flag:
            return ens_preds, ens_score, ens_mask, ms_max_idx, ms_masks
        else:
            return ens_preds, ens_score, ens_mask

    @classmethod
    def prediction_similarity(cls, ms_logits, targets):
        softmax1 = torch.softmax(ms_logits[0].detach(), dim=-1)
        score1, preds1 = torch.max(softmax1, dim=-1)
        res1 = torch.eq(preds1, targets).int()
        softmax2 = torch.softmax(ms_logits[1].detach(), dim=-1)
        score2, preds2 = torch.max(softmax2, dim=-1)
        res2 = torch.eq(preds2, targets).int()
        mask2 = torch.mul(res1, res2)
        mask = torch.where(mask2 > 0, torch.zeros_like(res1), torch.ones_like(res1))
        sim_num = torch.mul(torch.eq(preds1, preds2).int(), mask).int().sum()
        wrong_num = mask.int().sum()
        return sim_num.item(), wrong_num.item()

    @classmethod
    def target_verify(cls, preds, targets, mask=None):
        result = torch.eq(preds, targets).float()
        if mask is None:
            pl_acc = torch.where(result > 0, 1, 0).sum() / result.shape[0]
        else:
            pl_acc = torch.where(result.mul(mask) > 0, 1, 0).sum() / torch.where(mask > 0, 1, 0).sum()
        if torch.isnan(pl_acc): pl_acc = torch.tensor(0.0)
        return pl_acc

    @classmethod
    def target_statistic_train(cls, args, preds_stat, targets_stat, masks_stat=None):
        stat_box = []
        for dIdx in range(args.num_classes):
            if masks_stat is None:
                bincount = torch.bincount(preds_stat[(targets_stat + 1) == (dIdx + 1)], minlength=args.num_classes)
                stat_box.append(bincount / bincount.sum())
            else:
                bincount = torch.bincount(preds_stat[(targets_stat + 1).mul(masks_stat) == (dIdx + 1)], minlength=args.num_classes)
                stat_box.append(bincount / bincount.sum())
        return torch.stack(stat_box, dim=0)

    @classmethod
    def target_statistic_infer(cls, args, preds_stat, targets_stat):
        stat_box = []
        for dIdx in range(args.num_classes):
            bincount = torch.bincount(preds_stat[(targets_stat + 1) == (dIdx + 1)], minlength=args.num_classes)
            stat_box.append(bincount / bincount.sum())
        return torch.stack(stat_box, dim=0)

    @classmethod
    def corrcoef_features(cls, inp1, inp2, eta=1):
        if inp1.ndim == 2:
            bs, c = inp1.size()
            h, w = 1, 1
        else:
            bs, c, h, w = inp1.size()  # torch.Size([224, 12, 8, 8]) ==> 0.0104 ==mean==>
        f1 = inp1.view(bs, c * h * w).view(-1)
        f2 = inp2.view(bs, c * h * w).view(-1)
        corr_val = torch.abs(torch.corrcoef(torch.stack([f1, f2], dim=0))[0, 1])
        return corr_val*eta, bs

    @classmethod
    def corrcoef_labeled(cls, logits, targets, args):
        targets_one_hot = torch.eye(args.num_classes, device=args.device)[targets]
        logits_softmax = torch.softmax(logits, dim=-1)
        targets_vec = targets + 1
        preds_vec = torch.masked_select(logits_softmax, targets_one_hot.ge(0.5))
        preds_vec = preds_vec.mul(targets_vec)
        corr_val = torch.corrcoef(torch.stack([preds_vec, targets_vec], dim=0))[0, 1]
        # if torch.isnan(corr_val): corr_val = torch.tensor(0.0).to(args.device)
        return corr_val, len(targets_vec)

    @classmethod
    def corrcoef_unlabeled(cls, logits, targets, mask, args):
        targets_one_hot = torch.eye(args.num_classes, device=args.device)[targets]
        logits_softmax = torch.softmax(logits, dim=-1)
        targets_vec = targets + 1
        preds_vec = torch.masked_select(logits_softmax, targets_one_hot.ge(0.5))
        sel_targets_vec = torch.masked_select(targets_vec, mask.ge(0.5))
        sel_preds_vec = torch.masked_select(preds_vec, mask.ge(0.5))
        sel_preds_vec = sel_preds_vec.mul(sel_targets_vec)
        corr_val = torch.corrcoef(torch.stack([sel_preds_vec, sel_targets_vec], dim=0))[0, 1]
        # if torch.isnan(corr_val): corr_val = torch.tensor(0.0).to(args.device)
        return corr_val, len(sel_targets_vec)
