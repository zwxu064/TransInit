import torch
import numpy as np
import torch.nn as nn


def init_params(models, mode='xv'):
  assert mode in {'xv', 'net2net', 'random'}

  if not isinstance(models, list):
    models = [models]

  for model in models:
    for m in model.modules():
      if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if mode == 'net2net':
          m_in_channels = m.in_channels
          assert m_in_channels == m.out_channels and m.kernel_size[0] == 1
          m.weight.data.copy_(torch.eye(m_in_channels).view(m_in_channels, m_in_channels, 1, 1))
        elif mode == 'xv':
          nn.init.xavier_normal_(m.weight)
        elif mode == 'random':
          nn.init.uniform_(m.weight)

        m.bias.data.fill_(0) if (m.bias is not None) else None
      elif isinstance(m, nn.Linear):
        if mode == 'net2net':
          m.weight.data.copy_(torch.eye(m.out_features))
        elif mode == 'xv':
          nn.init.xavier_normal_(m.weight)
        elif mode == 'random':
          nn.init.uniform(m.weight)

        m.bias.data.fill_(0) if (m.bias is not None) else None
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)


class ExtraLayers(nn.Module):
  def __init__(self, args, param_init=None, init_mode='transparent'):
    super(ExtraLayers, self).__init__()
    self.args = args
    self.param_list = []
    self.param_extend_list = []
    self.param_init = param_init
    assert init_mode in {'transparent', 'xv', 'net2net', 'random'}

    # Change the config
    if init_mode == 'transparent' and self.args.enable_activation:
      extend_factor = 2
    else:
      extend_factor = 1

    if init_mode == 'net2net':
      self.args.out_channel_list = [self.args.num_classes] * len(self.args.out_channel_list)

    # Could replace conv2d by linear
    self.layer_first = nn.Conv2d(self.args.num_classes, self.args.out_channel_list[0] * extend_factor,
                                 kernel_size=1, stride=1, bias=self.args.enable_bias)
    self.relu_1 = nn.ReLU()
    self.layer_2 = nn.Conv2d(self.args.out_channel_list[0] * extend_factor,
                             self.args.out_channel_list[1] * extend_factor,
                             kernel_size=1, stride=1, bias=self.args.enable_bias)
    self.relu_2 = nn.ReLU()
    self.layer_last = nn.Conv2d(self.args.out_channel_list[1] * extend_factor,
                                self.args.out_channel_list[2],
                                kernel_size=1, stride=1, bias=self.args.enable_bias)

    self.check_model_validation()

    if init_mode == 'transparent':
      self.run_param_init()
    else:
      init_params(self, mode=init_mode)

  def run_param_init(self):
    self._init_weight()
    self._trans_init_last_layer()

  def dump_param_list(self, enable_squeeze=False):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        if enable_squeeze:
          out_c, in_c = m.weight.size()[0:2]
          self.param_list.append(m.weight.view(out_c, in_c))
        else:
          self.param_list.append(m.weight)

        if m.bias is not None:
          self.param_list.append(m.bias)
        else:
          self.param_list.append(None)

    return self.param_list

  def dump_param_extend_list(self):
    return self.param_extend_list

  def check_model_validation(self):
    for name, m in self.named_modules():
      if isinstance(m, nn.Conv2d):
        if '_last' not in name:
          out_c, in_c = m.weight.size()[0:2]
          assert out_c >= in_c

  def check_param_extend_list(self):
    assert len(self.param_extend_list) != 0

    param_join = self.param_extend_list[0]

    for param_extend in self.param_extend_list[1:-1]:
      param_join = param_join.matmul(param_extend)

    param_join = param_join.matmul(self.param_extend_list[-1])

    return param_join

  def _adjust_param_assign(self, target, source=None):
    if source is None:
      out_c = target.size(0)
      std = 1 / np.sqrt(out_c)

      if len(target.size()) >= 2:
        if self.args.weight_init == 'normal':
          torch.nn.init.normal_(target, mean=0, std=std)
        elif self.args.weight_init == 'orthogonal':
          torch.nn.init.orthogonal_(target)
      else:
        if self.args.bias_init == 'normal':
          torch.nn.init.normal_(target, mean=0, std=std)
        elif self.args.bias_init == 'zero':
          target.data.fill_(0)
    else:
      target.copy_(source)

  def _init_weight(self):
    enable_init = (self.param_init is not None)
    conv_count = 0

    for name, m in self.named_modules():
      if isinstance(m, nn.Conv2d):
        has_bias = (m.bias is not None)
        data_type = m.weight.dtype
        device = m.weight.device
        out_c, in_c = m.weight.size()[0:2]
        out_c_half, in_c_half = out_c // 2, in_c // 2
        weight_init = self.param_init[2 * conv_count].data if enable_init else None
        bias_init = self.param_init[2 * conv_count + 1].data if enable_init else None

        if self.args.enable_activation:
          if '_first' in name:  # first layer
            A = torch.empty((out_c_half, in_c), dtype=data_type, device=device)
            self._adjust_param_assign(A, weight_init)
            AA = torch.cat((A, -A), dim=0)

            if has_bias:
              B = torch.empty((out_c_half), dtype=data_type, device=device)
              self._adjust_param_assign(B, bias_init)
              BB = torch.cat((B, -B), dim=0)
          elif '_last' in name:  # last layer, it does not matter
            A = torch.empty((out_c, in_c_half), dtype=data_type, device=device)
            self._adjust_param_assign(A, weight_init)
            AA = torch.cat((A, -A), dim=1)

            if has_bias:
              BB = torch.empty((out_c), dtype=data_type, device=device)
              self._adjust_param_assign(BB, bias_init)
          else:
            A = torch.empty((out_c_half, in_c_half), dtype=data_type, device=device)
            self._adjust_param_assign(A, weight_init)
            AA = torch.cat((A, -A), dim=0)
            AA = torch.cat((AA, -AA), dim=1)

            if has_bias:
              B = torch.empty((out_c_half), dtype=data_type, device=device)
              self._adjust_param_assign(B, bias_init)
              BB = torch.cat((B, -B), dim=0)

          AA = AA.view(out_c, in_c, 1, 1)  # since it is a fully-connected layer
          m.weight.data.copy_(AA.data)
          m.bias.data.copy_(BB.data) if has_bias else None
        else:
          self._adjust_param_assign(m.weight.data, None)
          self._adjust_param_assign(m.bias.data, None) if has_bias else None

        conv_count += 1
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _trans_init_last_layer(self):
    layer_params = []
    bias_status = []

    # Get AB
    for name, m in self.named_modules():
      if isinstance(m, nn.Conv2d):
        data_type, device = m.weight.dtype, m.weight.device
        out_c, in_c = m.weight.size()[0:2]
        out_c_half, in_c_half = out_c // 2, in_c // 2
        has_bias = (m.bias is not None)
        bias_status.append(has_bias)

        if self.args.enable_activation:
          if '_first' in name:
            A = m.weight[:out_c_half]
          elif '_last' in name:
            A = m.weight[:, :in_c_half]
          else:
            A = m.weight[:out_c_half, :in_c_half]

          if '_last' in name:
            B = m.bias if has_bias else torch.zeros(out_c, dtype=data_type, device=device)
          else:
            B = m.bias[:out_c_half] if has_bias else torch.zeros(out_c_half, dtype=data_type, device=device)
        else:
          A = m.weight
          B = m.bias if has_bias else torch.zeros(out_c, dtype=data_type, device=device)

        B = B.view(B.size(0), 1, 1, 1)
        layer_params.append(torch.cat((A, B), dim=1))  # out_c*(in_c+1)

    # Calculate param values of the last layer via inverse matrix
    num_layers = len(layer_params)

    for idx in range(num_layers):
      params = layer_params[idx].permute(1, 0, 2, 3)  # (in_c+1)*out_c: +1 is bias
      in_c_plus1, out_c = params.size()[0:2]

      params = params.view(in_c_plus1, out_c)
      zero_one_tensor = torch.zeros((in_c_plus1, 1), dtype=params.dtype, device=params.device)
      zero_one_tensor[-1] = 1
      param_extened = torch.cat((params, zero_one_tensor), dim=1)

      if idx == 0:
        param_join = param_extened
      elif idx == num_layers - 1:
        # Note: param_join.matmul(param_last) you will see
        # NOT strictly identity matrix due to precision error
        # Could also consider using Cholesky, LBFGS to get a more accurate or efficient result
        param_last = param_join.pinverse()

        if self.args.num_classes == out_c:
          param_last = param_last[:, :-1]
      else:
        param_join = param_join.matmul(param_extened)

      if idx != num_layers - 1:
        self.param_extend_list.append(param_extened)
      else:
        self.param_extend_list.append(param_last)

    param_last = param_last.permute(1, 0)
    out_c, in_c_plus1 = param_last.size()
    in_c = in_c_plus1 - 1
    A = param_last[:, :in_c].view(out_c, in_c, 1, 1)
    AA = torch.cat((A, -A), dim=1) if self.args.enable_activation else A
    self.layer_last.weight.data.copy_(AA.data)

    if self.layer_last.bias is not None:
      B = param_last[:, in_c].view(out_c)
      self.layer_last.bias.data.copy_(B.data)

  def forward(self, x):
    x_1 = self.layer_first(x)

    if self.args.enable_activation:
      x_1 = self.relu_1(x_1)

    x_2 = self.layer_2(x_1)

    if self.args.enable_activation:
      x_2 = self.relu_2(x_2)

    x_3 = self.layer_last(x_2)

    return x_3