import torch
from torch import nn
from models.pose.hourglass.base.layers import Conv, Hourglass, Pool, Residual, Merge
from models.utils.multiple import MultiStream


class StackedHourglass_MS(nn.Module):
    def __init__(self, device, ms_stream_num, noisy_factor, expand, k, stack_num):
        super(StackedHourglass_MS, self).__init__()
        self.k = k
        self.stack_num = stack_num
        self.noisy_factor = noisy_factor
        self.expand = expand

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

        # prediction fusion, [stack_num]
        self.merge_preds = nn.ModuleList([
            Merge(self.k, 256) for sIdx in range(self.stack_num - 1)])

        # feature fusion, [stack_num]
        self.merge_features = nn.ModuleList([
            Merge(256, 256) for sIdx in range(self.stack_num - 1)])

        self.ms_stream_num = ms_stream_num

        # multiple utils instance, [stack_num, stream_num]
        self.ms = [MultiStream(device, self.ms_stream_num, [256, 64, 64], noisy_factor=self.noisy_factor, is_expand=self.expand) for sIdx in range(self.stack_num)]

        # multiple prediction, [stack_num, stream_num]
        self.ms_fc = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(Conv(256, self.k, 1, bn=False, relu=False)) for stIdx in range(self.ms_stream_num)
            ]) for sIdx in range(self.stack_num)])

    def forward(self, x):
        x = self.pre(x)  # torch.Size([bs, 256, 64, 64])

        # bs, n, c, h, w = inp1.size()
        ms_preds_combined, ms_fs_p_combined = [], []
        for sIdx in range(self.stack_num):
            # (1) feature extract
            hg = self.hgs[sIdx](x)  # torch.Size([bs, 256, 64, 64])
            feature = self.features[sIdx](hg)  # torch.size([bs, 256, 64, 64])

            ms_preds, ms_fs_p = [], []
            feature_pre = self.ms[sIdx].pre(feature.clone())
            for stIdx in range(self.ms_stream_num):
                # (2) multi-forward
                ms_f, ms_f_p = self.ms[sIdx].forward(stIdx, feature_pre.clone())
                ms_fs_p.append(ms_f_p)

                # (3) prediction
                preds = self.ms_fc[sIdx][stIdx](ms_f)  # torch.Size([bs, k, 64, 64])
                ms_preds.append(preds)
            ms_preds = torch.stack(ms_preds, 1)
            ms_fs_p = torch.stack(ms_fs_p, 1)

            # (4) get the mean of multiple predictions
            ms_preds_mix = torch.mean(ms_preds, 1)

            # (5) combine the feature and middle prediction with original image.
            if sIdx < self.stack_num - 1:
                x = x + self.merge_preds[sIdx](ms_preds_mix) + self.merge_features[sIdx](feature)

            ms_preds_combined.append(ms_preds)
            ms_fs_p_combined.append(ms_fs_p)
        return torch.stack(ms_preds_combined, 1), torch.stack(ms_fs_p_combined, 1)


def build_hourglass_ms(stream_num, k, noisy_factor, expand, device, stack_num=3):
    return StackedHourglass_MS(device, stream_num, noisy_factor, expand, k, stack_num)
