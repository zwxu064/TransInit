import torch
import torch.nn as nn
import torch.nn.functional as F
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .aspp import build_aspp
from .decoder import build_decoder
from .backbone import build_backbone
import matplotlib.pyplot as plt

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet101', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, enable_interpolation=True,
                 pretrained_path=None, norm_layer=nn.BatchNorm2d, enable_aspp=True,
                 enable_aspp_globalavgpool=True):
        super(DeepLab, self).__init__()
        self.enable_aspp = enable_aspp

        if backbone == 'drn':
            output_stride = 8
        BatchNorm = norm_layer
        self.backbone = build_backbone(backbone,
                                       output_stride,
                                       BatchNorm,
                                       pretrained_path=pretrained_path)
        self.aspp = build_aspp(backbone,
                               output_stride,
                               BatchNorm,
                               enable_aspp=self.enable_aspp,
                               enable_aspp_globalavgpool=enable_aspp_globalavgpool)
        self.decoder = build_decoder(num_classes,
                                     backbone,
                                     BatchNorm)
        self.enable_interpolation = enable_interpolation

        # if freeze_bn:
        #     self.freeze_bn()

    def forward(self, input, edge_weights=None):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x, _ = self.decoder(x, low_level_feat)

        if self.enable_interpolation:
            x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], (nn.Conv2d, nn.ConvTranspose2d,
                                     SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], (nn.Conv2d, nn.ConvTranspose2d,
                                     SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


