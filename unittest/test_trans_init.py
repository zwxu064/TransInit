import torch, argparse, sys
import numpy as np

sys.path.append('..')
from joint_learning.trans_init import ExtraLayers


def Parser():
  parser = argparse.ArgumentParser(description="PyTorch TransInit Training")
  parser.add_argument('--seed', type=int, default=2020)
  parser.add_argument('--num_classes', type=int, default=21)
  parser.add_argument('--weight_init', type=str, default='normal', choices=['normal', 'orthogonal'])
  parser.add_argument('--bias_init', type=str, default='normal', choices=['normal', 'zero'])
  parser.add_argument('--enable_activation', action='store_true', default=True)
  parser.add_argument('--enable_bias', action='store_true', default=True)
  parser.add_argument('--enable_double', action='store_true', default=False)
  args = parser.parse_args()

  return args


class Trainer(object):
  def __init__(self, args, out_channel_list, param_init=None):
    self.args = args
    self.args.out_channel_list = out_channel_list
    self.model = ExtraLayers(self.args, param_init=param_init)
    self.model = self.model.double() if self.args.enable_double else self.model

  def run(self, x):
    return self.model(x)


if __name__ == '__main__':
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.enabled = True

  # Set seeds
  args = Parser()
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)

  # Inputs
  batch_size, h, w = 4, 512, 512
  num_elements = batch_size * h * w
  data_type = torch.float64 if args.enable_double else torch.float32
  x = (torch.rand(size=(batch_size, args.num_classes, h, w), dtype=data_type) - 0.5) * 2 * 10
  x_extend = torch.cat((x.permute(0, 2, 3, 1).reshape(num_elements, args.num_classes),
                        torch.ones((num_elements, 1), dtype=x.dtype, device=x.device)), dim=1)
  out_c_list = [64, 64, args.num_classes]  # 3 layers, [args.num_classes] * 3
  assert out_c_list[0] >= args.num_classes  # ensure right-inverse, i.e., out_c >= in_c

  # No activation
  args.enable_activation = False
  trainer = Trainer(args, out_channel_list=out_c_list)
  results = trainer.run(x)

  if len(results) == 2:
    y = results[0]
    y_inter = results[1]
  else:
    y = results

  param_list = trainer.model.dump_param_list(enable_squeeze=True)
  param_extend_list = trainer.model.dump_param_extend_list()
  join_affine = trainer.model.check_param_extend_list()

  # Activation
  args.enable_activation = True
  param_init = None  # param_list / None
  trainer_act = Trainer(args, out_channel_list=out_c_list, param_init=param_init)
  results = trainer_act.run(x)

  if len(results) == 2:
    y_act = results[0]
    y_inter_act = results[1]
  else:
    y_act = results

  param_list_act = trainer_act.model.dump_param_list(enable_squeeze=True)
  param_extend_list_act = trainer_act.model.dump_param_extend_list()
  join_affine_act = trainer_act.model.check_param_extend_list()
  print('Input size (batch,in_c,h,w): {}, layer size: {} out_channels.\n' \
        .format(list(x.size()), out_c_list))

  if np.equal(x.size(1), y.size(1)):
    y_comp, y_comp_act = y, y_act
  else:
    y_comp, y_comp_act = y[:, :-1], y_act[:, :-1]

  print('Input sum: {}, output sum (no act): {}, output sum (ReLU): {}.\n' \
        .format(x.double().sum(), y_comp.double().sum(), y_comp_act.double().sum()))
  print('Max gap (no act): {}, max gap (ReLU): {}.\n' \
        .format((x - y_comp).abs().max(), (x - y_comp_act).abs().max()))