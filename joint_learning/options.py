import argparse, random, torch, os
import numpy as np


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)  # affect randomCrop
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  os.environ['PYTHONHASHSEED']=str(seed)


def set_config_seg(args):
  args.deeplab_backbone = args.backbone
  args.deeplab_outstride = args.out_stride
  args.deeplab_sync_bn = args.sync_bn
  args.deeplab_freeze_bn = args.freeze_bn

  return args


def set_config_sp(args):
  args.train_img_height = args.crop_size
  args.train_img_width = args.crop_size
  args.input_img_height = args.base_size
  args.input_img_width = args.base_size

  return args


def set_config():
  parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
  parser.add_argument('--server', type=str, default='039614', choices=['039614', 'data61'], help='039614 or data61')
  parser.add_argument('--backbone', type=str, default='resnet101',
                      choices=['resnet50', 'resnet101', 'xception', 'drn', 'mobilenet', 'resnet152',
                               'resnet101s', 'resnest101'],
                      help='backbone name (default: resnet)')
  parser.add_argument('--out_stride', type=int, default=16, help='network output stride (default: 8)')
  parser.add_argument('--dataset', type=str, default='pascal', choices=['pascal', 'coco'],
                      help='dataset name (default: pascal)')
  parser.add_argument('--use_sbd', action='store_true', default=False, help='whether to use SBD dataset (default: True)')
  parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
  parser.add_argument('--base_size', type=int, default=512, help='base image size')
  parser.add_argument('--crop_size', type=int, default=512, help='crop image size')
  parser.add_argument('--freeze_bn', action='store_true', default=False,
                      help='whether to freeze bn parameters (default: False)')
  parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'focal'], help='loss func type (default: ce)')
  parser.add_argument('--epochs', type=int, default=None, metavar='N', help='number of epochs to train (default: auto)')
  parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
  parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                      help='input batch size for training (default: auto)')
  parser.add_argument('--val_batch_size', type=int, default=16, metavar='N',
                      help='input batch size for training (default: auto)')
  parser.add_argument('--use_balanced_weights', action='store_true', default=False,
                      help='whether to use balanced weights (default: False)')
  parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (default: auto)')
  parser.add_argument('--lr_scheduler', type=str, default='poly', choices=['poly', 'step', 'cos', 'fixed'],
                      help='lr scheduler mode: (default: poly)')
  parser.add_argument('--warmup_epochs', type=float, default=0)
  parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
  parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='M', help='w-decay (default: 5e-4)')
  parser.add_argument('--nesterov', action='store_true', default=False, help='whether use nesterov (default: False)')
  parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
  parser.add_argument('--gpu_ids', type=str, default='0',
                      help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
  parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
  parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
  parser.add_argument('--checkname', type=str, default=None, help='set the checkpoint name')
  parser.add_argument('--ft', action='store_true', default=False, help='finetuning on a different dataset')
  parser.add_argument('--eval_interval', type=int, default=1, help='evaluuation interval (default: 1)')
  parser.add_argument('--no_val', action='store_true', default=False, help='skip validation during training')
  parser.add_argument('--save_interval', type=int, default=None, help='save model interval in epochs')

  parser.add_argument('--data_root', type=str, default=None)
  parser.add_argument('--enable_test', action='store_true', default=False, help='enable test')
  parser.add_argument('--enable_test_full', action='store_true', default=False, help='enable test full size')
  parser.add_argument('--output_directory', type=str, default=None, help='directory to store output images')
  parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='directory to store models')
  parser.add_argument('--mode', type=str, default='fully', choices={'weakly', 'fully'}, help='weakly or fully')
  parser.add_argument('--enable_adjust_val', action='store_true', default=False)
  parser.add_argument('--adjust_val_factor', type=int, default=4)
  parser.add_argument('--enable_save_unary', action='store_true', default=False)
  parser.add_argument('--resnet_pretrained_path', type=str, default=None)
  parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'Adadelta'])
  parser.add_argument('--disable_aspp', action='store_true', default=False)
  parser.add_argument('--enable_sp', action='store_true', default=False)
  parser.add_argument('--deeplab_resume', type=str, default=None)
  parser.add_argument('--sp_resume', type=str, default=None)
  parser.add_argument('--slic_loss', type=float, default=0)
  parser.add_argument('--sp_pos_weight', type=float, default=0.003)
  parser.add_argument('--downsize', type=float, default=16)
  parser.add_argument('--enable_save_val', action='store_true', default=False)
  parser.add_argument('--sp_lr', type=float, default=None, metavar='LR')
  parser.add_argument('--enable_save_all', action='store_true', default=False)
  parser.add_argument('--enable_ti', action='store_true', default=False)
  parser.add_argument('--ti_lr', type=float, default=None)
  parser.add_argument('--enable_save_png', action='store_true', default=False)
  parser.add_argument('--enable_vanilla', action='store_true', default=False)
  parser.add_argument('--coco_resume', type=str, default=None)
  parser.add_argument('--date', type=str, default=None)
  parser.add_argument('--ti_net_init', type=str, default='transparent',
                      choices={'transparent', 'xv', 'net2net', 'random'})
  parser.add_argument('--evaluate_ti', action='store_true', default=False)
  parser.add_argument('--out_channel_list', type=str, default=None)
  parser.add_argument('--model', type=str, default='deeplab')
  parser.add_argument('--aux', action='store_true', default=False)
  parser.add_argument('--se_loss', action='store_true', default=False)
  parser.add_argument('--disable_logit_consistency', action='store_true', default=False)
  args = parser.parse_args()

  args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
  args.gpu_number = len(args.gpu_ids)

  if args.out_channel_list is not None:
    args.out_channel_list = [int(s) for s in args.out_channel_list.split(',')]

  if args.slic_loss > 0:
    args.enable_sp = True

  if args.slic_loss <= 0:
    args.sp_lr = None

  args.val_batch_size = 1 if args.enable_adjust_val else args.val_batch_size
  args.ft = True if (args.deeplab_resume and os.path.exists(args.deeplab_resume)) else args.ft
  args.ti_lr = args.ti_lr if args.enable_ti else None

  return args