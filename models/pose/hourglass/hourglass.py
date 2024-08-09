import torch
from torch import nn
from models.pose.hourglass.base.layers import Conv, Hourglass, Pool, Residual, Merge


class StackedHourglass(nn.Module):
    def __init__(self, k, stack_num):
        super(StackedHourglass, self).__init__()
        self.k = k
        self.stack_num = stack_num

        # data pre-processing
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, 256)
        )

        # 4-Stacked_Hourglass
        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, 256, False, 0)
            ) for sIdx in range(self.stack_num)])

        # feature extraction
        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(256, 256),
                Conv(256, 256, 1, bn=True, relu=True)
            ) for sIdx in range(self.stack_num)])

        # prediction
        self.preds = nn.ModuleList([Conv(256, self.k, 1, relu=False, bn=False) for sIdx in range(self.stack_num)])

        # feature fusion
        self.merge_features = nn.ModuleList(
            [Merge(256, 256) for sIdx in range(self.stack_num - 1)])

        # prediction fusion
        self.merge_preds = nn.ModuleList(
            [Merge(self.k, 256) for sIdx in range(self.stack_num - 1)])

    def forward(self, x):
        x = self.pre(x)  # torch.Size([bs, 256, 64, 64])

        preds_combined = []
        for sIdx in range(self.stack_num):
            # (1) feature extract
            hg = self.hgs[sIdx](x)  # torch.Size([bs, 256, 64, 64])
            feature = self.features[sIdx](hg)  # torch.size([bs, 256, 64, 64])

            # (2) prediction
            preds = self.preds[sIdx](feature)  # torch.Size([bs, k, 64, 64])
            preds_combined.append(preds)

            # (3) fusion (x, feature, pred), torch.Size([bs, 256, 64, 64])
            if sIdx < self.stack_num - 1:
                x = x + self.merge_preds[sIdx](preds) + self.merge_features[sIdx](feature)

        return torch.stack(preds_combined, 1)


def build_hourglass(k, stack_num=3):
    return StackedHourglass(k, stack_num)
