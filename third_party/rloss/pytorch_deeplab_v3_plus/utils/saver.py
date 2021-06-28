import os, shutil, torch, glob
from collections import OrderedDict


class Saver(object):
    def __init__(self, args):
        self.args = args
        self.directory = os.path.join(args.checkpoint_dir,
                                      args.dataset,
                                      args.checkname)
        if args.date is not None:
            self.directory = os.path.join(self.directory, str(args.date))

        self.directory = os.path.join(self.directory, 'train_batch{}'.format(str(args.batch_size)))
        self.directory += '_GPU{}_epoch{}_slic{}_bz{}_lr{}_splr{}_tilr{}' \
            .format(len(args.gpu_ids),
                    args.epochs,
                    args.slic_loss,
                    args.batch_size,
                    args.lr,
                    args.sp_lr,
                    args.ti_lr)
        self.directory += '_{}_{}_{}'.format(args.optimizer, args.lr_scheduler, args.backbone)

        if args.enable_test:
            if args.enable_adjust_val:
                self.directory += "_adjustval"

        if ((self.args.deeplab_resume) and os.path.exists(self.args.deeplab_resume)) \
            or ((self.args.resume is not None) and (self.args.resume.find('finetune') > -1)):
            self.directory += '_finetune'

        if self.args.enable_vanilla:
            self.directory += '_vanilla'

        if ((self.args.coco_resume is not None) and os.path.exists(self.args.coco_resume)) \
            or ((self.args.resume is not None) and (self.args.resume.find('coco') > -1)) \
            or ((self.args.deeplab_resume is not None) and (self.args.deeplab_resume.find('coco') > -1)):
            self.directory += '_coco'

        self.directory += "_{}".format(self.args.ti_net_init)

        if self.args.disable_logit_consistency:
            self.directory += "_nologitconsistency"

        if self.args.enable_test:
            self.key_word = 'test'
        else:
            self.key_word = 'experiment'

        self.runs = sorted(glob.glob(os.path.join(self.directory, '{}_*'.format(self.key_word))))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0
        self.experiment_dir = os.path.join(self.directory, '{}_{}'.format(self.key_word, str(run_id)))

        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

        if is_best:
            best_pred = state['best_pred']

            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))

            if self.runs:
                previous_miou = [0.0]

                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, '{}_{}'.format(self.key_word, str(run_id)), 'best_pred.txt')

                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue

                max_miou = max(previous_miou)

                if best_pred > max_miou and not self.args.enable_test:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                if not self.args.enable_test:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['datset'] = self.args.dataset
        p['backbone'] = self.args.backbone
        p['out_stride'] = self.args.out_stride
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')

        log_file.close()