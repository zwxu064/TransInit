import numpy as np
import os, time, torch, tqdm
import scipy as sci

from third_party.rloss import Evaluator, TensorboardSummary, Saver, LR_Scheduler
from third_party.rloss import SegmentationLosses, make_data_loader, DeNorm
from third_party.rloss import DataParallelWithCallback, visualization
from third_party.sp_fcn import compute_semantic_pos_loss, init_spixel_grid, get_spixel_image
from joint_learning.spnet import SPNet, customized_sp_edges
from joint_learning.options import set_config, set_seed, set_config_seg, set_config_sp

if int(sci.__version__.split('.')[0]) == 1:
  from imageio import imwrite as imsave
else:
  from scipy.misc import imsave as imsave


class Trainer(object):
  def __init__(self, args):
    self.args = args

    # Define Saver
    self.saver = Saver(args)
    self.saver.save_experiment_config()

    # Define Tensorboard Summary
    self.summary = TensorboardSummary(self.saver.experiment_dir)
    self.writer = self.summary.create_summary()
    self.denorm = DeNorm(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    # Define Dataloader
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    self.train_loader, self.val_loader, self.test_loader, self.nclass = \
      make_data_loader(args, **kwargs)

    # Define Criterion
    weight = None
    args.reduction_mode = 'sum'
    self.seg_criterion = SegmentationLosses(weight=weight, reduction_mode=args.reduction_mode, cuda=False,
                                            batch_average=False).build_loss(mode=args.loss_type)
    self.sp_criterion = compute_semantic_pos_loss

    # Define network
    args.enable_cuda = True
    args.n_classes = self.nclass

    model = SPNet(args,
                  seg_loss_fn=self.seg_criterion,
                  sp_loss_fn=self.sp_criterion,
                  denorm=self.denorm)

    # Define Optimizer
    if args.optimizer in {'SGD', 'Adam'}:
      if args.ft:
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr}]
      else:
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

      if args.sp_lr is not None:
        train_params += [{'params': model.get_sp_lr_params(), 'lr': args.sp_lr}]

      if args.ti_lr is not None:
        train_params += [{'params': model.get_ti_lr_params(), 'lr': args.ti_lr}]

      if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum, nesterov=args.nesterov)
      else:
        optimizer = torch.optim.Adam(train_params, weight_decay=args.weight_decay)
    else:
      assert False

    self.model, self.optimizer = model, optimizer

    # Define Evaluator
    self.evaluator = Evaluator(self.nclass)
    self.evaluator_single = Evaluator(self.nclass)

    # Define LR scheduler
    if args.lr_scheduler in {'poly', 'step', 'cos'}:
      self.scheduler = LR_Scheduler(
        args.lr_scheduler, [v['lr'] for v in train_params], args.epochs,
        len(self.train_loader), enable_ft=args.ft, warmup_epochs=args.warmup_epochs)
    else:
      self.scheduler = None
      self.lr_scheduler = None

    # Resuming checkpoint
    self.best_pred = 0.0
    self.run_resume()

    # Using CUDA
    if self.args.cuda:
      if self.args.gpu_number > 1:
        self.model = DataParallelWithCallback(self.model, device_ids=self.args.gpu_ids)
      self.model = self.model.cuda()

    # Clear start epoch if fine-tuning
    if args.ft:
      args.start_epoch = 0

  def run_resume(self):
    if self.args.deeplab_resume:
      assert os.path.exists(self.args.deeplab_resume)
      checkpoint = torch.load(self.args.deeplab_resume)
      state_dict = checkpoint['state_dict']
      state_dict = {k.replace('deeplab.', ''): v for k, v in state_dict.items() if k.find('deeplab.') > -1}

      if self.args.n_classes != state_dict['decoder.last_conv.8.weight'].shape[0]:
        model_dict = self.model.deeplab.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k.find('decoder.last_conv.8') <= -1}
        model_dict.update(state_dict)
        self.model.deeplab.load_state_dict(model_dict)
      else:
        self.model.deeplab.load_state_dict(state_dict)

    if self.args.coco_resume:
      assert os.path.exists(self.args.coco_resume)
      checkpoint = torch.load(self.args.coco_resume)
      state_dict = checkpoint['state_dict']

      if self.args.n_classes != state_dict['deeplab.decoder.last_conv.8.weight'].shape[0]:
        model_dict = self.model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if not (k.find('deeplab.decoder.last_conv.8') > -1)}
        model_dict.update(state_dict)
        self.model.load_state_dict(model_dict)
      else:
        self.model.load_state_dict(state_dict)

    if self.args.sp_resume:
      assert os.path.exists(self.args.sp_resume)
      state_dict = torch.load(self.args.sp_resume)['state_dict']
      self.model.sp_net.load_state_dict(state_dict)

    if self.args.resume:
      assert os.path.exists(self.args.resume)
      checkpoint = torch.load(self.args.resume)
      state_dict = checkpoint['state_dict']
      self.args.start_epoch = checkpoint['epoch']
      self.model.load_state_dict(state_dict)
      print("Loaded checkpoint '{}' (epoch {})".format(self.args.resume, checkpoint['epoch']))

      if not self.args.enable_test:
        if not self.args.ft:
          self.optimizer.load_state_dict(checkpoint['optimizer'])

        if 'optimizer' in checkpoint:
          param_groups = checkpoint['optimizer']['param_groups']
          for idx, param_group in enumerate(param_groups):
            print('Finetuned group {}, lr: {}'.format(idx, param_group['lr']))

  def training(self, epoch, plot_count=0):
    train_loss = 0.0
    train_baseloss = 0.0
    train_slicloss = 0.0
    duration = 0

    if hasattr(self.model, 'module'):
      self.model.module.deeplab.train()

      if self.args.slic_loss > 0:
        self.model.module.sp_net.train()
      else:
        self.model.module.sp_net.eval()
    else:
      self.model.deeplab.train()

      if self.args.slic_loss > 0:
        self.model.sp_net.train()
      else:
        self.model.sp_net.eval()

    # Training unary-net but fix batchnorm running_mean and running_var
    # using pretrained models (e.g., resnet-101)
    if self.args.ft or self.args.freeze_bn:
      if hasattr(self.model, 'module'):
        self.model.module.freezebn_modules([self.model.module.deeplab])
      else:
        self.model.freezebn_modules([self.model.deeplab])

    num_img_tr = len(self.train_loader)
    print('LR adjusted', [group['lr'] for group in self.optimizer.param_groups])
    tbar = tqdm.tqdm(self.train_loader)
    spixelID, XY_feat = init_spixel_grid(self.args, b_train=True, batch_size=self.args.batch_size)

    for i, sample in enumerate(tbar):
      image = sample['image']
      target = sample['label'] if ('label' in sample) else None
      cropping, valid_area = None, None

      if target is not None:
        cropping = (target != 254).float()
        target[target == 254] = 255
        valid_area = (target != 255).float()  # use for superpixel to exclude the effect of ambiguous edges in GT and padding areas

      if self.args.cuda:
        image = image.cuda()
        target = target.cuda() if (target is not None) else None
        valid_area = valid_area.cuda() if (valid_area is not None) else None
        cropping = cropping.cuda() if (cropping is not None) else None

      time_start = time.time()

      if self.scheduler:
        self.scheduler(self.optimizer, i, epoch, self.best_pred)

      actual_bz = image.shape[0]

      if actual_bz != self.args.batch_size:
        spixelID, XY_feat = init_spixel_grid(self.args, b_train=True, batch_size=actual_bz)

      self.optimizer.zero_grad()
      outputs = self.model(image, valid_area=valid_area, cropping=cropping, target=target, XY_feat=XY_feat)

      base_loss = outputs['ce_loss'].sum() / outputs['ce_denom'].sum()
      base_loss = base_loss / actual_bz
      slic_loss = outputs['sem_loss'].sum() / outputs['sem_denom'].sum()
      slic_loss = self.args.slic_loss * (slic_loss + outputs['pos_loss'].sum() / outputs['pos_denom'].sum())
      loss = base_loss + slic_loss
      train_baseloss += base_loss.detach().item()
      train_slicloss += slic_loss.detach().item()

      duration += time.time() - time_start

      loss.backward()
      self.optimizer.step()
      loss_scale = loss.item()
      train_loss += loss_scale

      tbar.set_description(
        'Epoch:%d, train loss:%.3f = Base:%.3f + Slic:%.3f'
        % (epoch,
           train_loss / (i + 1),
           train_baseloss / (i + 1),
           train_slicloss / (i + 1)))
      self.writer.add_scalar('train/total_loss_iter', loss_scale, i + num_img_tr * epoch)

    self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
    print('\n[Train epoch {:d}] Loss: {:.4f}, time: {:.4f}s'.format(epoch, train_loss, duration))

    return plot_count

  def validation(self, epoch):
    self.model.eval()
    self.evaluator.reset()
    test_loss = 0.0
    tbar = tqdm.tqdm(self.val_loader, desc='\r')

    for i, sample in enumerate(tbar):
      image, image_name = sample['image'], sample['name']
      target = sample['label'] if ('label' in sample) else None
      valid_area, cropping = None, None

      if target is not None:
        cropping = (target != 254).float()
        target[target == 254] = 255
        valid_area = (target != 255).float()

      if self.args.cuda:
        image = image.cuda()
        target = target.cuda() if (target is not None) else None
        valid_area = valid_area.cuda() if (valid_area is not None) else None
        cropping = cropping.cuda() if (cropping is not None) else None

      with torch.no_grad():
        outputs = self.model(image, valid_area=valid_area, cropping=cropping, target=target)
        scores, org_scores = outputs['scores'], outputs['org_scores']

        if target is not None:
          loss = outputs['ce_loss'].sum() / outputs['ce_denom'].sum()
          loss = loss / image.shape[0]
          target = target.cpu().numpy()
        else:
          loss = 0

        pred = np.argmax(scores.data.cpu().numpy(), axis=1)
        org_pred = np.argmax(org_scores.data.cpu().numpy(), axis=1)

        test_loss += loss
        tbar.set_description('Epoch: %d, valid loss: %.3f.' % (epoch, test_loss / (i + 1)))

      if self.args.enable_sp:
        spixel_map = outputs['spixel_map'].squeeze(1)
        denormalized_image = self.denorm(sample['image']) / 255
        spixel_viz, spixel_label_map = get_spixel_image(denormalized_image.cpu().clamp(0, 1)[0],
                                                        spixel_map[0],
                                                        n_spixels=len(spixel_map.unique()),
                                                        b_enforce_connect=True)

      if self.args.enable_adjust_val:
        image_h, image_w = sample['size'][0].item(), sample['size'][1].item()
        target = target[:, :image_h, :image_w] if (target is not None) else None
        pred = pred[:, :image_h, :image_w]
        org_pred = org_pred[:, :image_h, :image_w]
        denormalized_image = denormalized_image[:, :, :image_h, :image_w] if self.args.enable_sp else None

        if self.args.enable_sp:
          spixel_viz = spixel_viz[:, :image_h, :image_w]

      self.evaluator.add_batch(target, pred) if (target is not None) else None

      if (self.args.val_batch_size == 1) and self.args.enable_save_png \
              and (self.args.dataset == 'pascal'):
        save_dir = 'results/VOC2012/Segmentation'

        if self.saver.experiment_dir.find('coco') > -1:
          save_dir += '/comp6'  # training on any datasets
        else:
          save_dir += '/comp5'  # training only on VOC trainset

        save_dir += '_val_cls'

        save_path = os.path.join(self.saver.experiment_dir, save_dir)
        os.makedirs(save_path) if (not os.path.exists(save_path)) else None

        # Grey
        save_image_name = os.path.join(save_path, '', image_name[0] + '.png')
        imsave(save_image_name, pred.transpose(1, 2, 0).astype(np.uint8))

      # Add batch sample into evaluator
      mIoU_single = []

      if target is not None:
        for idx in range(image.size(0)):
          self.evaluator_single.reset()
          self.evaluator_single.add_batch(target[idx], pred[idx])
          mIoU_single_per = self.evaluator_single.Mean_Intersection_over_Union()
          mIoU_single.append(mIoU_single_per)

      if self.args.enable_save_val and \
              ((i <= 5) or (self.args.enable_test and self.args.val_batch_size == 1)):
        output_directory = os.path.join(self.saver.experiment_dir, 'epoch{}'.format(epoch))
        if not os.path.exists(output_directory):
          os.mkdir(output_directory)

        n_images = pred.shape[0]
        denormalized_image = self.denorm(image) / 255

        for idx in range(n_images):
          image_name_per = image_name[idx]
          image_h, image_w = sample['size'][0][idx].item(), sample['size'][1][idx].item()

          if self.args.enable_adjust_val:
            image = image[:, :, :image_h, :image_w]

          if self.args.enable_sp:
            sp_map_new, spixel_viz = customized_sp_edges(denormalized_image.cpu().clamp(0, 1)[idx],
                                                         spixel_map[idx])
            spixel_viz = spixel_viz.cpu().numpy()

            if self.args.enable_adjust_val:
              spixel_viz = spixel_viz[:, :image_h, :image_w]
          else:
            spixel_viz = None

          visualization(image[idx],
                        pred[idx],
                        target=target[idx] if (target is not None) else None,
                        unary_pred=org_pred[idx],
                        sp_map=spixel_viz,
                        image_name=image_name_per,
                        accuracy=mIoU_single[idx] if (target is not None) else 0,
                        save_dir=output_directory,
                        enable_save_all=self.args.enable_save_all)

    # Fast test during the training
    Acc = self.evaluator.Pixel_Accuracy()
    Acc_class = self.evaluator.Pixel_Accuracy_Class()
    mIoU = self.evaluator.Mean_Intersection_over_Union()
    FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
    self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
    self.writer.add_scalar('val/mIoU', mIoU, epoch)
    self.writer.add_scalar('val/Acc', Acc, epoch)
    self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
    self.writer.add_scalar('val/fwIoU', FWIoU, epoch)

    if mIoU > self.best_pred:
      is_best = True
      self.best_pred = mIoU
      self.best_epoch = epoch + 1
      state_dict_save = self.model.state_dict() \
        if (self.args.gpu_number == 1) else self.model.module.state_dict()
      self.saver.save_checkpoint({
        'epoch': self.best_epoch,
        'state_dict': state_dict_save,
        'optimizer': self.optimizer.state_dict(),
        'best_pred': self.best_pred,
        'current_pred': mIoU},
        is_best, filename='ckpt_{}.pth.tar'.format(str(epoch + 1)))

    print("\n[Val epoch {:d}] Loss: {:.6f}, Acc: {:.6f}, Acc_class: {:.6f}, " \
          "mIoU: {:.6f} (best: {:.6f} at {}th), fwIoU: {:.6f}" \
          .format(epoch, test_loss, Acc, Acc_class, mIoU,
                  self.best_pred, self.best_epoch, FWIoU))


def main():
  args = set_config()

  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.enabled = True

  # Set seed
  set_seed(args.seed)

  # Check multiple GPUs and sync bnorm
  args.cuda = (not args.no_cuda) and torch.cuda.is_available()

  if args.cuda:
    args.sync_bn = (args.gpu_number > 1)

  if args.enable_test:
    if not args.val_batch_size:
      args.val_batch_size = 1
    else:
      print('Set val_batch_size:', args.val_batch_size)
  else:
    if not args.val_batch_size:
      args.val_batch_size = args.batch_size

  if args.enable_adjust_val and (args.val_batch_size != 1):
    assert args.val_batch_size == 1
    print('Enable adjust val size while val batch size is {} (must be 1).'.format(args.val_batch_size))

  # For unary net, set True, Important!!!
  args.freeze_bn = True if (args.gpu_number > 1) else False
  args.batch_size = 1 if args.enable_test else args.batch_size

  if args.output_directory:
    if args.enable_test:
      args.output_directory += '_full'
    else:
      args.output_directory += '_crop{}'.format(args.crop_size)

    if not os.path.exists(args.output_directory):
      os.makedirs(args.output_directory)

  # Default settings for epochs, batch_size and LR
  if not args.epochs:
    epoches = {'coco': 40, 'pascal': 60}
    args.epochs = epoches[args.dataset.lower()]

  if not args.batch_size:
    args.batch_size = 4 * args.gpu_number

  if (args.lr is None) or (args.lr == 0):
    lrs = {'coco': 0.1, 'pascal': 0.007}
    args.lr = lrs[args.dataset.lower()]  / (4 * args.gpu_number) * args.batch_size

  if args.sp_lr is None:
    args.sp_lr = 0.1 * args.lr

  if not args.checkname:
    args.checkname = 'deeplab-' + str(args.backbone) + '-test'

  args = set_config_seg(args)
  args = set_config_sp(args)

  print(args)
  trainer = Trainer(args)
  print('Starting Epoch: {}, Total Epoch: {}'
        .format(trainer.args.start_epoch, trainer.args.epochs))

  if not args.enable_test:
    plot_count = 0

    if trainer.args.start_epoch < trainer.args.epochs:
      for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        set_seed(epoch)
        plot_count = trainer.training(epoch, plot_count=plot_count)
        if not trainer.args.no_val:
          trainer.validation(epoch)
    elif trainer.args.start_epoch == trainer.args.epochs:
      if not trainer.args.no_val:
        trainer.validation(trainer.args.start_epoch)
    else:
      assert False
  else:
    if args.resume and os.path.isdir(args.resume):
      model_root = args.resume

      for idx in range(args.epochs):
        args.resume = os.path.join(model_root, 'ckpt_{}.pth.tar'.format(idx + 1))
        trainer.run_resume()
        trainer.validation(trainer.args.start_epoch)
    else:
      trainer.validation(trainer.args.start_epoch)

  trainer.writer.close()


if __name__ == "__main__":
  main()