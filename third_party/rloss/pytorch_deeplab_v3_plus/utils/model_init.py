import torch.nn as nn
import torch
from ..modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


def init_params(models, mode='xv'):
  assert mode in {'xv', 'net2net'}

  if not isinstance(models, list):
    models = [models]

  for model in models:
    for m in model.modules():
      if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if mode == 'net2net':
          m_in_channels = m.in_channels
          assert m_in_channels == m.out_channels and m.kernel_size[0] == 1
          m.weight.data.copy_(torch.eye(m_in_channels).view(m_in_channels, m_in_channels, 1, 1))
        else:
          nn.init.xavier_normal_(m.weight)

        m.bias.data.fill_(0) if (m.bias is not None) else None
      elif isinstance(m, nn.Linear):
        if mode == 'xv':
          nn.init.xavier_normal_(m.weight)
        elif mode == 'net2net':
          m.weight.data.copy_(torch.eye(m.out_features))

        m.bias.data.fill_(0) if (m.bias is not None) else None
      elif isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)