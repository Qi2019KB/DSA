import  torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.multiple import MultiStream

cfg = {
    "11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "13": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG_MHE(nn.Module):
    def __init__(self, vgg_type, num_classes, ms_stream_num, noisy_factor, device):
        super(VGG_MHE, self).__init__()
        self.features = self._make_layers(cfg[vgg_type.split('VGG')[1]])

        self.ms_stream_num = ms_stream_num
        self.ms_noisy_factor = noisy_factor
        self.ms = MultiStream(device, self.ms_stream_num, [512, 8, 8], noisy_factor=self.ms_noisy_factor)
        self.ms_fc = nn.ModuleList([nn.Sequential(nn.Linear(512, num_classes)) for stIdx in range(self.ms_stream_num)])

    def forward(self, x):
        out = self.features(x)  # torch.Size([1, 512, 1, 1])

        ms_preds, ms_fs_p = [], []
        for stIdx in range(self.ms_stream_num):
            out = out.clone().view(out.size(0), -1)
            ms_pred = self.ms_fc[stIdx](out)
            ms_preds.append(ms_pred)
        return torch.stack(ms_preds, 0)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3

        for layer_para in cfg:
            if layer_para != "M":
                layers += [
                    nn.Conv2d(in_channels, layer_para, kernel_size=3, padding=1),
                    nn.BatchNorm2d(layer_para),
                    nn.ReLU(inplace=True)
                ]
                in_channels = layer_para
            else:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]        
        return nn.Sequential(*layers)


def build_vgg_mhe(vgg_type, num_classes, stream_num, noisy_factor, device):
    return VGG_MHE(vgg_type, num_classes, stream_num, noisy_factor, device)
