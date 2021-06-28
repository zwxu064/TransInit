import torch.nn as nn
import numpy as np
import torch, sys
import torch.nn.functional as F
sys.path.append('..')
from joint_learning.trans_init import ExtraLayers as TILayer
from third_party.rloss import DeepLab, SynchronizedBatchNorm2d, init_params
from third_party.sp_fcn import SpixelNet1l_bn, shift9pos, update_spixl_map
from third_party.sp_fcn import label2one_hot_torch, build_LABXY_feat


def customized_sp_edges(image, sp_map):
  # 3 directions
  assert len(sp_map.shape) == 2

  # Right
  sp_map_right = sp_map.clone()
  sp_map_right[:, :-1] = (sp_map[:, :-1] - sp_map[:, 1:]).abs()

  # Top
  sp_map_top = sp_map.clone()
  sp_map_top[:-1, :] = (sp_map[:-1, :] - sp_map[1:, :]).abs()

  # Top right
  sp_map_top_right = sp_map.clone()
  sp_map_top_right[1:, :-1] = (sp_map[1:, :-1] - sp_map[:-1, 1:]).abs()

  sp_map_final = ((sp_map_right + sp_map_top + sp_map_top_right) != 0) * 255

  image_merged = image
  image_merged[0, sp_map_final == 255] = 0
  image_merged[1, sp_map_final == 255] = 1
  image_merged[2, sp_map_final == 255] = 1

  return sp_map_final, image_merged


def create_SPID(img_h, img_w, downsize, enable_cuda=True):
  img_h_ = int(np.ceil(img_h / 16.) * 16)
  img_w_ = int(np.ceil(img_w / 16.) * 16)
  n_spixl_h = int(np.floor(img_h_ / downsize))
  n_spixl_w = int(np.floor(img_w_ / downsize))
  n_spixel = int(n_spixl_h * n_spixl_w)
  spix_values = np.int32(np.arange(0, n_spixel).reshape((n_spixl_h, n_spixl_w)))
  spix_idx_tensor_ = shift9pos(spix_values)
  spix_idx_tensor = np.repeat(np.repeat(spix_idx_tensor_, downsize, axis=1),
                              downsize, axis=2)
  spixeIDs = torch.from_numpy(np.tile(spix_idx_tensor, (1, 1, 1, 1))).float()
  spixeIDs = spixeIDs.cuda() if enable_cuda else spixeIDs

  return spixeIDs


class SPNet(nn.Module):
  def __init__(self, args, seg_loss_fn=None, sp_loss_fn=None, denorm=None, enable_trans_activation=True):
    super(SPNet, self).__init__()
    self.args = args
    self.enable_train_sp = (args.slic_loss > 0)
    self.seg_loss_fn = seg_loss_fn
    self.sp_loss_fn = sp_loss_fn
    self.denorm = denorm
    BatchNorm = SynchronizedBatchNorm2d if args.sync_bn else nn.BatchNorm2d
    self.deeplab = DeepLab(num_classes=args.n_classes,
                           backbone=args.deeplab_backbone,
                           output_stride=args.deeplab_outstride,
                           sync_bn=args.deeplab_sync_bn,
                           freeze_bn=args.deeplab_freeze_bn,
                           enable_interpolation=True,
                           pretrained_path=args.resnet_pretrained_path,
                           norm_layer=BatchNorm)
    self.sp_net = SpixelNet1l_bn()
    init_params(self.sp_net)

    if args.enable_ti:
      self.args.num_classes = self.args.n_classes * 2  # unary + repeated superpixel map
      self.args.enable_activation = enable_trans_activation
      self.args.enable_bias = True
      self.args.weight_init = 'normal'
      self.args.bias_init = 'normal'

      if self.args.out_channel_list is None:
        self.args.out_channel_list = [64, 64, self.args.num_classes]  # 21*2 in_channels

      self.ti_net = TILayer(self.args, init_mode=self.args.ti_net_init)

    self.spixelIDs = create_SPID(self.args.crop_size,
                                 self.args.crop_size,
                                 self.args.downsize,
                                 enable_cuda=True)
    num_pixels = self.args.crop_size * self.args.crop_size
    self.num_pixels = num_pixels
    self.a_v, self.a_loc_idx = self.cal_a_v_idx(num_pixels, device='cuda')

  def cal_a_v_idx(self, num_pixels, device='cpu'):
    a_v = torch.ones(num_pixels, dtype=torch.float32, device=device)
    a_loc_idx = torch.arange(0, num_pixels, dtype=torch.float32, device=device)

    return a_v, a_loc_idx

  def get_ti_lr_params(self):
    modules = [self.ti_net]

    for i in range(len(modules)):
      for m in modules[i].modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d,
                          nn.Conv1d, nn.Linear,
                          SynchronizedBatchNorm2d, nn.BatchNorm2d)):
          for p in m.parameters():
            if p.requires_grad:
              yield p

  def get_sp_lr_params(self):
    modules = [self.sp_net]

    for i in range(len(modules)):
      for m in modules[i].modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d,
                          SynchronizedBatchNorm2d, nn.BatchNorm2d)):
          for p in m.parameters():
            if p.requires_grad:
              yield p

  def get_1x_lr_params(self):
    modules = []

    if hasattr(self.deeplab, 'backbone'):
      if not isinstance(self.deeplab.backbone, str):  # patch
        modules += [self.deeplab.backbone]

    if hasattr(self.deeplab, 'pretrained'):
      modules += [self.deeplab.pretrained]

    for i in range(len(modules)):
      for m in modules[i].modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
          for p in m.parameters():
            if p.requires_grad:
              yield p

        # Unnecessary to freeze on single GPU
        if not self.args.deeplab_freeze_bn:
          if isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
            for p in m.parameters():
              if p.requires_grad:
                yield p

  def get_10x_lr_params(self):
    modules = []

    if hasattr(self.deeplab, 'aspp'):
      modules += [self.deeplab.aspp]

    if hasattr(self.deeplab, 'decoder'):
      modules += [self.deeplab.decoder]

    if hasattr(self.deeplab, 'head'):
      modules += [self.deeplab.head]

    if hasattr(self.deeplab, 'auxlayer'):
      modules += [self.deeplab.auxlayer]

    for i in range(len(modules)):
      for m in modules[i].modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
          for p in m.parameters():
            if p.requires_grad:
              yield p

        # Unnecessary to freeze on single GPU
        if not self.args.deeplab_freeze_bn:
          if isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
            for p in m.parameters():
              if p.requires_grad:
                yield p

  def freezebn_modules(self, modules):
    if not isinstance(modules, list):
      modules = [modules]

    for i in range(len(modules)):
      for m in modules[i].modules():
        if isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
          m.eval()  # !!! eval mode will fix running_mean and running_var

  def cal_loss(self, mode, target, XY_feat, image, valid_area, cropping, scores, sp_values):
    celoss = self.seg_loss_fn(scores, target)
    ce_denom = (target != 255).float().sum() if (mode == 'sum') else image.new_ones(1)

    if (self.args.slic_loss > 0) and self.training:
      slic_target = target.clone()
      slic_target[slic_target == 255] = 0
      label_1hot = label2one_hot_torch(slic_target.unsqueeze(1), C=self.args.n_classes)  # Zhiwei set C=50 as SSN does
      LABXY_feat_tensor = build_LABXY_feat(label_1hot, XY_feat)  # B*(50+2)*H*W
      slic_results = self.sp_loss_fn(mode, sp_values, LABXY_feat_tensor,
                                     pos_weight=self.args.sp_pos_weight,
                                     kernel_size=self.args.downsize,
                                     valid_area=valid_area)
      sem_loss = slic_results['sem_loss']
      sem_denom = slic_results['sem_denom']
      pos_loss = slic_results['pos_loss']
      pos_denom = slic_results['pos_denom']
    else:
      sem_loss, sem_denom = image.new_zeros(1), image.new_ones(1)
      pos_loss, pos_denom = image.new_zeros(1), image.new_ones(1)

    results = {'ce_loss': celoss, 'ce_denom': ce_denom,
               'sem_loss': sem_loss, 'sem_denom': sem_denom,
               'pos_loss': pos_loss, 'pos_denom': pos_denom}

    return results

  def merge_score_sp(self, scores, spixel_map, valid_area=None):
    device = scores.device
    valid_area = valid_area.to(device) if (valid_area is not None) else None
    batch, num_classes, img_h, img_w = scores.shape
    scores = scores.view(batch, num_classes, -1)
    spixel_map = spixel_map.view(batch, -1)
    num_pixels = img_h * img_w

    if num_pixels == self.num_pixels:
      a_v, a_loc_idx = self.a_v.to(device), self.a_loc_idx.to(device)
    else:
      a_v, a_loc_idx = self.cal_a_v_idx(num_pixels, device=device)

    num_spixels = int(np.ceil(num_pixels / (self.args.downsize ** 2)))  # len(spixel_map.unique())
    new_scores = []

    for idx in range(batch):
      a_loc = torch.stack((spixel_map[idx], a_loc_idx), dim=0).long()  # 2*(img_h*img_w) xy

      if valid_area is not None:  # encoder to ignore invalid GT and cropping, decoder to include them
        a_v_valid = a_v * valid_area[idx].flatten()
        a_encoder = torch.sparse.FloatTensor(a_loc, a_v_valid, torch.Size([num_spixels, num_pixels])).to(device)
        a_decoder = torch.sparse.FloatTensor(a_loc, a_v, torch.Size([num_spixels, num_pixels])).to(device)
        a_decoder = a_decoder.transpose(0, 1)
      else:
        a_encoder = torch.sparse.FloatTensor(a_loc, a_v, torch.Size([num_spixels, num_pixels])).to(device)
        a_decoder = a_encoder.transpose(0, 1)

      sp_num_each = torch.sparse.sum(a_encoder, dim=1).unsqueeze(1).to_dense()  # number of pixels for each sp
      score = scores[idx].transpose(0, 1)
      sp_score = torch.sparse.mm(a_encoder, score) / (sp_num_each + 1e-5)  # num_spixels*num_classes

      new_score = torch.sparse.mm(a_decoder, sp_score)  # (img_h*img_w)*num_classes
      new_score = new_score.permute(1, 0).view(-1, img_h, img_w)  # num_classes*img_h*img_w
      new_scores.append(new_score)

    new_scores = torch.stack(new_scores, dim=0)  # batch*num_classes*img_h*img_w

    return new_scores

  def forward(self, inputs, valid_area=None, cropping=None, target=None, XY_feat=None):
    # scores:(batch,21,h,w), spixel_map:(batch,1,h,w)
    batch, _, img_h, img_w = inputs.shape      
    scores = self.deeplab(inputs)
    org_scores = scores.clone().detach() if (not self.training) else []
    spixel_values, spixel_map = [], []
    results = {}

    if self.args.enable_sp:
      if self.enable_train_sp:
        spixel_values = self.sp_net(inputs)
      else:
        with torch.no_grad():
          spixel_values = self.sp_net(inputs)

      if (img_h != self.args.crop_size) or (img_w != self.args.crop_size):
        spixelIDs = create_SPID(img_h, img_w, self.args.downsize)
      else:
        spixelIDs = self.spixelIDs

      spixelIDs = spixelIDs.to(inputs.device)
      curr_spixl_map = update_spixl_map(spixelIDs, spixel_values).float()
      spixel_map = F.interpolate(curr_spixl_map, size=(img_h, img_w), mode='nearest')

      if self.args.enable_ti:
        volume = scores.new_zeros((batch, 2 * self.args.n_classes, img_h, img_w))

        for idx in range(self.args.n_classes):
          volume[:, 2 * idx] = scores[:, idx]
          volume[:, 2 * idx + 1] = spixel_map[:, 0]

        new_volume = self.ti_net(volume)
        scores = torch.stack([new_volume[:, 2 * v] for v in range(self.args.n_classes)], dim=1)

      if not self.args.disable_logit_consistency:
        scores = self.merge_score_sp(scores, spixel_map, valid_area=valid_area)

    # Do loss
    if target is not None:
      mode = 'sum' if (self.args.reduction_mode == 'sum') else 'mean'
      loss_groups = self.cal_loss(mode, target, XY_feat, inputs, valid_area,
                                  cropping, scores, spixel_values)

      results.update({'ce_loss': loss_groups['ce_loss'].view(1),
                      'ce_denom': loss_groups['ce_denom'].view(1),
                      'sem_loss': loss_groups['sem_loss'].view(1),
                      'sem_denom': loss_groups['sem_denom'].view(1),
                      'pos_loss': loss_groups['pos_loss'].view(1),
                      'pos_denom': loss_groups['pos_denom'].view(1)})

    if self.training:
      results.update({'scores': scores})
    else:
      results.update({'scores': scores, 'org_scores': org_scores})

      if not isinstance(spixel_map, list):
        results.update({'spixel_map': spixel_map})

    return results