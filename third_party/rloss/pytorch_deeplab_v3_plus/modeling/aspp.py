import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from ..utils import init_params

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        init_params(self)
        # self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.kaiming_normal_(m.bias) if (m.bias is not None) else None
            elif isinstance(m, SynchronizedBatchNorm2d) \
                    or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm, enable_aspp=True, enable_aspp_globalavgpool=True):
        super(ASPP, self).__init__()
        self.enable_aspp = enable_aspp

        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        if enable_aspp:
            self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
            self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
            self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
            self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

            # This author said global average pool will damage cityscapes, so I mark here
            # https://github.com/YudeWang/deeplabv3plus-pytorch/issues/5
            if enable_aspp_globalavgpool:
                self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                     nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                                     BatchNorm(256),
                                                     nn.ReLU())
                conv1_in_c = 1280
            else:
                self.global_avg_pool = None
                conv1_in_c = 1024
        else:
            conv1_in_c = 2048

        self.conv1 = nn.Conv2d(conv1_in_c, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        init_params(self)
        # self._init_weight()

    def forward(self, x):
        if self.enable_aspp:
            x1 = self.aspp1(x)
            x2 = self.aspp2(x)
            x3 = self.aspp3(x)
            x4 = self.aspp4(x)

            if self.global_avg_pool is not None:
                x5 = self.global_avg_pool(x)
                x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
                x = torch.cat((x1, x2, x3, x4, x5), dim=1)
            else:
                x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d) \
                    or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(backbone, output_stride, BatchNorm, enable_aspp=True, enable_aspp_globalavgpool=True):
    return ASPP(backbone, output_stride, BatchNorm, enable_aspp=enable_aspp,
                enable_aspp_globalavgpool=enable_aspp_globalavgpool)