from .resnet import ResNetSerial
from .mobilenet import MobileNetV2
from .xception import AlignedXception
from .drn import drn_d_54


def build_backbone(backbone, output_stride, BatchNorm, pretrained_path=None):
    if backbone in {'resnet50', 'resnet101', 'resnet152'}:
        return ResNetSerial(backbone, output_stride, BatchNorm,
                            pretrained_path=pretrained_path)
    elif backbone == 'xception':
        return AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
