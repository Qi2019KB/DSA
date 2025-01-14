from torch import nn

Pool = nn.MaxPool2d
AvgPool = nn.AvgPool2d

# def model_parameters():
#     nn.init.uniform_(tensor, a=0., b=1.)  # 对张量赋值-均匀分布，默认取值范围(0., 1.)
#     nn.init.constant_(tensor, val)  # 对张量赋值-常量，需要赋一个常数值
#     nn.init.normal_(tensor, mean=0, std=1)  # 对张量赋值-高斯（正太）分布。mean均值，std方差
#     nn.init.xavier_uniform_(tensor, gain=nn.init.calculate_gain('relu'))  # 对张量赋值-xavier初始化。gain可根据激活函数种类获得：nn.init.calculate_gain('relu')
#     nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')  # 对张量赋值-kaiming初始化
#     nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')  # 对张量赋值-kaiming正太分布初始化

def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)


class ViewLinear(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        _, channel, h, w = x.shape
        outChannel = channel * h * w
        x = x.view(-1, outChannel)
        nn.Linear(outChannel, self.L),
        nn.ReLU(),
        return x


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, '{} {}'.format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv2(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu_idx=0):
        super(Conv2, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu_candidates = ['ReLU', 'LReLU', 'PReLU', 'RReLU', 'ELU']
        self.relu = None
        self.bn = None
        self.relu = self.relu_init(self.relu_candidates[relu_idx])
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def relu_init(self, relu_type):
        if relu_type == 'ReLU':
            return nn.ReLU()
        elif relu_type == 'LReLU':
            return nn.LeakyReLU()
        elif relu_type == 'PReLU':
            return nn.PReLU()
        elif relu_type == 'RReLU':
            return nn.RReLU()
        elif relu_type == 'ELU':
            return nn.ELU()

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, '{} {}'.format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n - 1, nf, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        return self.up(x)


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        # 1*1卷积核
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)
